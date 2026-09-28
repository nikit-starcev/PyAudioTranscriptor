"""Тонкий клиент к локальному llama.cpp через Vulkan.

Работа с LLM ведётся только локально, без внешних API. Клиент
:class:`LlamaServerClient` поднимает ``llama-server`` как внешний процесс
(GPU через Vulkan) и ходит к нему по OpenAI-совместимому протоколу
``POST /v1/chat/completions``: модель грузится один раз и обслуживает все
запросы конвейера.

Жизненный цикл серверного клиента устроен так, чтобы процесс
``llama-server`` **никогда не оставался висеть** (он держит VRAM):

- клиент поддерживает протокол контекстного менеджера (``with``);
- :meth:`LlamaServerClient.close` идемпотентен и гарантированно снимает
  процесс (SIGTERM, затем SIGKILL);
- все запущенные процессы регистрируются в модульном реестре, который
  очищается через :mod:`atexit` и по сигналам ``SIGINT``/``SIGTERM``;
- перед запуском проверяется, не занят ли порт; зависший собственный
  ``llama-server`` (совпал бинарник, порт и модель) аккуратно останавливается,
  а чужой процесс не трогается — выбирается свободный порт.

Клиент использует только стандартную библиотеку (``urllib``) и подхватывает
``LD_LIBRARY_PATH`` для библиотек llama.cpp — точно так же, как это делает
whisper.cpp в проекте.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from audio_transcriber.config.defaults import DEFAULT_CONTEXT_SIZE
from audio_transcriber.utils.env import with_library_path
from audio_transcriber.utils.exceptions import LlmError
from audio_transcriber.utils.subprocess_registry import (
    StderrReader,
    register_process,
    terminate_process,
)
from audio_transcriber.utils.subprocess_registry import (
    _active_processes as _active_processes,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_TOKENS = 1024
# Сколько слоёв пробуем выгрузить на GPU: полный offload, затем частичный, затем CPU.
DEFAULT_GPU_LAYERS = 99
DEFAULT_PARTIAL_GPU_LAYERS = 16
# Загрузка 7B-модели на Vulkan и прогрев занимают заметное время.
DEFAULT_READY_TIMEOUT = 600.0
DEFAULT_REQUEST_TIMEOUT = 600.0

_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Вспомогательные функции запуска
# ---------------------------------------------------------------------------


def _build_env(library_path: str | None) -> dict[str, str]:
    """Собирает окружение с ``LD_LIBRARY_PATH`` для бинарника llama.cpp."""
    return with_library_path(os.environ, library_path)


def _find_free_port(host: str = DEFAULT_HOST) -> int:
    """Находит свободный TCP-порт на указанном хосте."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _port_in_use(host: str, port: int) -> bool:
    """Проверяет, принимает ли кто-то соединения на ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_port_free(host: str, port: int, *, timeout: float = 5.0) -> bool:
    """Ждёт освобождения порта (после остановки зависшего сервера)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_in_use(host, port):
            return True
        time.sleep(0.1)
    return not _port_in_use(host, port)


def _args_have_option(args: list[str], option: str, value: str) -> bool:
    """Ищет ``--option value`` или ``--option=value`` в списке аргументов."""
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args) and args[index + 1] == value:
            return True
        if arg == f"{option}={value}":
            return True
    return False


def _find_stale_server_pids(port: int, binary: str, model_path: Path) -> list[int]:
    """Ищет зависшие СВОИ llama-server по порту и модели (только Linux).

    Сопоставляем одновременно имя бинарника, ``--port`` и путь к модели — так
    чужой процесс на том же порту мы не тронем.
    """
    if not sys.platform.startswith("linux"):
        return []

    binary_name = Path(binary).name
    model_str = str(model_path)
    found: list[int] = []
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        args = [part for part in raw.decode("utf-8", "replace").split("\0") if part]
        if not args or Path(args[0]).name != binary_name:
            continue
        if not _args_have_option(args, "--port", str(port)):
            continue
        if model_str not in args:
            continue
        found.append(int(entry.name))
    return found


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, *, timeout: float = 5.0) -> None:
    """Останавливает чужой процесс по PID: SIGTERM, затем SIGKILL."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Клиент к llama-server
# ---------------------------------------------------------------------------


class LlamaServerClient:
    """Клиент к ``llama-server``, запускаемому локально как подпроцесс."""

    def __init__(
        self,
        model_path: Path,
        *,
        binary: str = "llama-server",
        library_path: str | None = None,
        gpu: bool = True,
        port: int | None = None,
        host: str = DEFAULT_HOST,
        threads: int | None = None,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
        gpu_layers: int = DEFAULT_GPU_LAYERS,
        partial_gpu_layers: int = DEFAULT_PARTIAL_GPU_LAYERS,
    ) -> None:
        self._model_path = Path(model_path)
        self._binary = binary
        self._library_path = library_path
        self._gpu = gpu
        self._port = port
        self._host = host
        self._threads = threads
        self._context_size = context_size
        self._max_tokens = max_tokens
        self._ready_timeout = ready_timeout
        self._gpu_layers = gpu_layers
        self._partial_gpu_layers = partial_gpu_layers

        self._proc: subprocess.Popen[str] | None = None
        self._stderr: StderrReader | None = None
        self._base_url: str | None = None
        self._actual_port: int | None = None
        self._gpu_layers_used: int | None = None
        self._device: str | None = None

    # -- состояние -----------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def port(self) -> int | None:
        """Порт, на котором фактически слушает сервер."""
        return self._actual_port

    @property
    def device(self) -> str | None:
        """Устройство, на котором реально идёт инференс (после запуска)."""
        return self._device

    @property
    def gpu_layers_used(self) -> int | None:
        """Сколько слоёв выгружено на GPU (``0`` — чистый CPU)."""
        return self._gpu_layers_used

    # -- запуск --------------------------------------------------------
    def _offload_attempts(self) -> list[int]:
        """Последовательность попыток offload: полная → частичная → CPU."""
        if not self._gpu:
            return [0]
        attempts = [self._gpu_layers]
        if self._partial_gpu_layers not in attempts and self._partial_gpu_layers != 0:
            attempts.append(self._partial_gpu_layers)
        if 0 not in attempts:
            attempts.append(0)
        return attempts

    def _resolve_port(self) -> int:
        """Выбирает порт: занятый своим зависшим сервером — освобождает."""
        if self._port is None:
            return _find_free_port(self._host)

        if not _port_in_use(self._host, self._port):
            return self._port

        stale = _find_stale_server_pids(self._port, self._binary, self._model_path)
        for pid in stale:
            logger.warning(
                "Найден зависший llama-server (pid=%d) на порту %d — останавливаю",
                pid,
                self._port,
            )
            _terminate_pid(pid)
        if _wait_port_free(self._host, self._port):
            return self._port

        logger.warning(
            "Порт %d занят посторонним процессом — выбираю свободный порт",
            self._port,
        )
        return _find_free_port(self._host)

    def start(self) -> None:
        """Запускает сервер и ждёт готовности, деградируя при нехватке VRAM."""
        if self.is_running:
            return

        if not self._model_path.is_file():
            raise LlmError(f"Модель LLM не найдена: {self._model_path}")

        port = self._resolve_port()
        env = _build_env(self._library_path)

        errors: list[str] = []
        for gpu_layers in self._offload_attempts():
            try:
                self._spawn(port, gpu_layers, env)
            except FileNotFoundError as exc:
                raise LlmError(f"Бинарник llama-server не найден: {self._binary}") from exc
            except LlmError as exc:
                errors.append(f"offload {gpu_layers} слоёв: {exc}")
                logger.warning(
                    "Не удалось запустить llama-server (offload %d слоёв): %s",
                    gpu_layers,
                    exc,
                )
                self._cleanup_current_process()
                continue

            self._log_device(gpu_layers)
            return

        detail = errors[-1] if errors else "неизвестная ошибка"
        raise LlmError(
            f"Не удалось запустить llama-server ни с GPU, ни на CPU. Последняя ошибка: {detail}"
        )

    def _spawn(self, port: int, gpu_layers: int, env: dict[str, str]) -> None:
        """Поднимает один процесс llama-server с заданным offload и ждёт ready."""
        cmd = [
            self._binary,
            "-m",
            str(self._model_path),
            "--host",
            self._host,
            "--port",
            str(port),
            "-c",
            str(self._context_size),
            "-ngl",
            str(gpu_layers),
            # Подробный лог нужен, чтобы увидеть, сколько слоёв попало на GPU.
            "-v",
        ]
        if self._threads:
            cmd += ["-t", str(self._threads)]

        logger.debug("Запуск llama-server: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        register_process(proc)
        self._proc = proc
        self._actual_port = port
        self._gpu_layers_used = gpu_layers
        assert proc.stderr is not None
        self._stderr = StderrReader(proc.stderr)
        self._stderr.start()
        self._base_url = f"http://{self._host}:{port}"

        try:
            self._wait_ready()
        except LlmError:
            self._cleanup_current_process()
            raise

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._ready_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                detail = self._stderr.tail() if self._stderr else ""
                raise LlmError(
                    f"llama-server завершился с ошибкой (код {self._proc.returncode}): {detail}"
                )
            try:
                with urllib.request.urlopen(f"{self._base_url}/health", timeout=5.0) as resp:
                    if resp.status == 200:
                        return
            except urllib.error.URLError, OSError:
                pass
            time.sleep(0.5)

        detail = self._stderr.tail() if self._stderr else ""
        raise LlmError(f"llama-server не ответил за {self._ready_timeout:.0f} с: {detail}")

    def _detect_device(self) -> str:
        """Определяет по логу, сколько слоёв реально ушло на GPU."""
        text = self._stderr.full_text() if self._stderr else ""
        match = _OFFLOAD_RE.search(text)
        if match and int(match.group(1)) > 0:
            return f"GPU ({match.group(1)}/{match.group(2)} слоёв)"
        return "CPU"

    def _log_device(self, requested_layers: int) -> None:
        self._device = self._detect_device()
        logger.info(
            "llama-server готов: порт=%s, устройство инференса=%s (запрошено слоёв на GPU: %d)",
            self._actual_port,
            self._device,
            requested_layers,
        )
        if requested_layers > 0 and self._device.startswith("CPU"):
            logger.warning(
                "Запрошен GPU (%d слоёв), но модель загружена на CPU. "
                "Проверьте, что доступен Vulkan-бэкенд llama.cpp.",
                requested_layers,
            )

    def _cleanup_current_process(self) -> None:
        proc = self._proc
        self._proc = None
        self._stderr = None
        self._base_url = None
        if proc is not None:
            terminate_process(proc)

    def _ensure_started(self) -> None:
        if not self.is_running:
            self.start()

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Отправляет сообщения в ``/v1/chat/completions`` и возвращает текст."""
        self._ensure_started()
        assert self._base_url is not None

        payload = json.dumps(
            {
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self._max_tokens,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[-1000:]
            hint = self._http_error_hint(exc.code, body)
            raise LlmError(f"llama-server вернул HTTP {exc.code}: {body}{hint}") from exc
        except (urllib.error.URLError, OSError) as exc:
            detail = self._stderr.tail() if self._stderr else ""
            raise LlmError(f"Не удалось обратиться к llama-server: {exc} {detail}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                f"Неожиданный ответ llama-server: {json.dumps(data, ensure_ascii=False)[:500]}"
            ) from exc

    @staticmethod
    def _http_error_hint(code: int, body: str) -> str:
        """Подсказка пользователю при типичных ошибках запроса к LLM.

        HTTP 400 у llama-server почти всегда означает, что промпт превысил
        контекст модели: сообщение об этом содержит ``context``/``token``.
        """
        lowered = body.casefold()
        overflow = any(
            marker in lowered
            for marker in ("context", "token", "too long", "exceed")
        )
        if code == 400 and overflow:
            logger.warning(
                "LLM отклонила запрос: переполнение контекста (HTTP %d). "
                "Уменьшите LLM_CONTEXT или размер входного файла.",
                code,
            )
            return (
                " Подсказка: переполнение контекста — уменьшите LLM_CONTEXT "
                "или размер входного файла."
            )
        return ""

    def close(self) -> None:
        """Останавливает сервер, если он был запущен (идемпотентно)."""
        self._cleanup_current_process()

    def __enter__(self) -> LlamaServerClient:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def create_llm_client(
    *,
    model_path: Path | None,
    binary: str,
    library_path: str | None,
    gpu: bool,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LlamaServerClient | None:
    """Создаёт LLM-клиент по настройкам конфигурации.

    Возвращает ``None``, если модель не задана.
    """
    if model_path is None:
        return None

    return LlamaServerClient(
        model_path,
        binary=binary,
        library_path=library_path,
        gpu=gpu,
        context_size=context_size,
        max_tokens=max_tokens,
    )
