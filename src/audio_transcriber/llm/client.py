"""Клиенты LLM-постобработки: локальный llama.cpp и внешний OpenAI-совместимый API.

Провайдер **по умолчанию — локальный** :class:`LlamaServerClient`: он поднимает
``llama-server`` как внешний процесс (GPU через Vulkan) и ходит к нему по
OpenAI-совместимому протоколу ``POST /v1/chat/completions``; модель грузится
один раз и обслуживает все запросы конвейера. Ничего не меняется для тех, кто
работает «100% локально».

Опционально (``llm_provider="openai"``) используется :class:`OpenAIClient` —
тонкий клиент к внешнему OpenAI-совместимому API (OpenAI, Ollama, vLLM,
LM Studio, OpenRouter и т.п.). В этом случае **текст стенограммы покидает
локальную машину** — пользователь предупреждается в настройках и логе.
:func:`create_llm_client` выбирает провайдера по конфигурации.

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
from typing import TYPE_CHECKING

from audio_transcriber.config.defaults import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_REQUEST_TIMEOUT,
)
from audio_transcriber.llm.base import LlmClient
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

if TYPE_CHECKING:
    from audio_transcriber.config.settings import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_TOKENS = 1024
# Сколько слоёв пробуем выгрузить на GPU: полный offload, затем частичный, затем CPU.
DEFAULT_GPU_LAYERS = 99
DEFAULT_PARTIAL_GPU_LAYERS = 16
# Загрузка 7B-модели на Vulkan и прогрев занимают заметное время.
DEFAULT_READY_TIMEOUT = 600.0
# Таймаут одного запроса к серверу. Значение по умолчанию вынесено в
# ``config.defaults`` (единый источник с AppConfig/CLI); алиас сохранён для
# обратной совместимости импортов.
DEFAULT_REQUEST_TIMEOUT = DEFAULT_LLM_REQUEST_TIMEOUT

#: Сколько раз повторяем запрос к внешней LLM при временных сбоях (5xx/429/сеть).
DEFAULT_MAX_RETRIES = 2
#: Базовая задержка между повторными попытками (секунды), растёт экспоненциально.
DEFAULT_RETRY_BACKOFF = 1.0

_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU", re.IGNORECASE)

#: HTTP-коды, при которых имеет смысл повторить запрос к внешней LLM.
_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def http_error_hint(code: int, body: str) -> str:
    """Подсказка пользователю при типичных ошибках запроса к LLM.

    HTTP 400 у сервера почти всегда означает, что промпт превысил контекст
    модели: сообщение об этом содержит ``context``/``token``.
    """
    lowered = body.casefold()
    overflow = any(marker in lowered for marker in ("context", "token", "too long", "exceed"))
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
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
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
        self._request_timeout = request_timeout
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
            with urllib.request.urlopen(request, timeout=self._request_timeout) as resp:
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
        """Подсказка пользователю при типичных ошибках запроса к LLM."""
        return http_error_hint(code, body)

    def close(self) -> None:
        """Останавливает сервер, если он был запущен (идемпотентно)."""
        self._cleanup_current_process()

    def __enter__(self) -> LlamaServerClient:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()



class OpenAIClient:
    """Клиент к внешнему OpenAI-совместимому API.

    Один-единственный HTTP-эндпоинт ``/chat/completions`` (с префиксом ``/v1``
    или без него — подбирается автоматически). Поддерживает авторизацию
    ``Bearer <api_key>``, таймаут и повторные попытки при временных
    ошибках сети/сервера (5xx, 429). Сетевые ошибки и неожиданные ответы
    поднимаются как :class:`~audio_transcriber.utils.exceptions.LlmError` с
    понятным сообщением, чтобы вызывающий код мог мягко деградировать.

    В отличие от :class:`LlamaServerClient`, ничего не запускает и не держит
    ресурсов: :meth:`close` — no-op.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_key: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ) -> None:
        if not str(base_url).strip():
            raise ValueError("Не задан base_url внешней LLM")
        if not str(model_name).strip():
            raise ValueError("Не задано имя модели внешней LLM")

        self._base_url = str(base_url).strip()
        self._model_name = str(model_name).strip()
        self._api_key = api_key.strip() if isinstance(api_key, str) else None
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = max(0.0, float(retry_backoff))
        self._urls = _chat_completions_urls(self._base_url)

    @property
    def base_url(self) -> str:
        """Базовый URL провайдера (без завершающего слэша)."""
        return self._base_url

    @property
    def model_name(self) -> str:
        """Имя модели, передаваемое в запросе."""
        return self._model_name

    @property
    def urls(self) -> list[str]:
        """Кандидаты URL эндпоинта (основной и запасной без ``/v1``)."""
        return list(self._urls)

    def _request(self, url: str, messages: list[dict[str, str]]) -> urllib.request.Request:
        payload = json.dumps(
            {
                "model": self._model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self._max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return urllib.request.Request(url, data=payload, headers=headers, method="POST")

    def _sleep_before_retry(self, attempt: int) -> None:
        # attempt >= 1: 1-я повторная попытка ждёт base * 2^0 и т.д.
        if self._retry_backoff <= 0:
            return
        time.sleep(self._retry_backoff * (2 ** (attempt - 1)))

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Отправляет сообщения внешнему API и возвращает текст ответа.

        :raises LlmError: при исчерпании повторных попыток, авторизационной
            ошибке, отсутствии модели/эндпоинта или неожиданном формате ответа.
        """
        url_index = 0
        attempt = 0
        while True:
            url = self._urls[url_index]
            request = self._request(url, messages)
            try:
                with urllib.request.urlopen(request, timeout=self._request_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[-1000:]
                # Промах по эндпоинту: сервер ожидает URL без префикса /v1 —
                # пробуем запасной вариант, не расходуя попытки.
                if (
                    exc.code == 404
                    and url_index == 0
                    and len(self._urls) > 1
                ):
                    logger.debug("LLM: %s не найден, пробую %s", url, self._urls[1])
                    url_index = 1
                    continue
                if exc.code in _RETRYABLE_HTTP_STATUS and attempt < self._max_retries:
                    attempt += 1
                    logger.warning(
                        "Внешняя LLM вернула HTTP %d (%s), повтор %d/%d",
                        exc.code,
                        url,
                        attempt,
                        self._max_retries,
                    )
                    self._sleep_before_retry(attempt)
                    continue
                hint = http_error_hint(exc.code, body)
                raise LlmError(
                    f"Внешняя LLM вернула HTTP {exc.code} ({url}): {body}{hint}"
                ) from exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    logger.warning(
                        "Не удалось обратиться к внешней LLM (%s): %s — повтор %d/%d",
                        url,
                        exc,
                        attempt,
                        self._max_retries,
                    )
                    self._sleep_before_retry(attempt)
                    continue
                raise LlmError(
                    f"Не удалось обратиться к внешней LLM {url}: {exc}"
                ) from exc

            return self._parse_response(data)

    def _parse_response(self, data: object) -> str:
        try:
            content = data["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                f"Неожиданный ответ внешней LLM: {json.dumps(data, ensure_ascii=False)[:500]}"
            ) from exc
        if not isinstance(content, str):
            raise LlmError(
                f"Внешняя LLM вернула не текст: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return content

    def close(self) -> None:
        """Ничего не делает: внешний клиент не владеет процессами/сессиями."""

    def __enter__(self) -> OpenAIClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _chat_completions_urls(base_url: str) -> list[str]:
    """Кандидаты URL ``/chat/completions`` для базового URL.

    Поддерживаются оба распространённых варианта:

    - ``https://host`` → ``https://host/v1/chat/completions`` (основной) и
      ``https://host/chat/completions`` (запасной);
    - ``https://host/v1`` → ``https://host/v1/chat/completions`` (основной) и
      ``https://host/chat/completions`` (запасной);
    - уже полный ``.../chat/completions`` → как есть.
    """
    base = base_url.strip().rstrip("/")
    if not base:
        raise LlmError("Не задан base_url внешней LLM")
    if base.endswith("/chat/completions"):
        return [base]
    if base.endswith("/v1"):
        primary = f"{base}/chat/completions"
        fallback = f"{base[:-3]}/chat/completions"
    else:
        primary = f"{base}/v1/chat/completions"
        fallback = f"{base}/chat/completions"
    candidates = [primary]
    if fallback != primary:
        candidates.append(fallback)
    return candidates


def _models_urls(base_url: str) -> list[str]:
    """Кандидаты URL ``/models`` для проверки доступности внешнего сервера."""
    base = base_url.strip().rstrip("/")
    if not base:
        return []
    if base.endswith("/models"):
        return [base]
    if base.endswith("/v1"):
        return [f"{base}/models"]
    return [f"{base}/v1/models", f"{base}/models"]


def _parse_model_ids(raw: str) -> list[str]:
    """Извлекает идентификаторы моделей из ответа ``/models`` (best-effort)."""
    try:
        payload = json.loads(raw)
    except ValueError:
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def probe_openai_server(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> tuple[bool, str, list[str]]:
    """Проверяет доступность OpenAI-совместимого сервера через ``GET /models``.

    Возвращает ``(ok, message, model_ids)``. Это лёгкая проверка для кнопки
    «Проверить доступность» в настройках: сервер считается доступным, если
    отвечает 2xx на список моделей. Сетевые ошибки и не-2xx не выбрасываются —
    это диагностика, а не рабочий вызов.
    """
    urls = _models_urls(base_url)
    if not urls:
        return False, "Не задан base_url", []

    headers = {"Accept": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    last_error = "неизвестная ошибка"
    for url in urls:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and url != urls[-1]:
                last_error = f"HTTP {exc.code} ({url})"
                continue
            return False, f"Сервер ответил HTTP {exc.code} ({url})", []
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return False, f"Не удалось подключиться к {url}: {exc}", []

        models = _parse_model_ids(raw)
        if models:
            preview = ", ".join(models[:3])
            return True, f"Доступно. Моделей: {len(models)} ({preview})", models
        return True, "Сервер доступен (список моделей пуст или в ином формате)", []

    return False, f"Не удалось найти эндпоинт /models: {last_error}", []


def create_llm_client(
    config: AppConfig | None = None,
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    model_path: Path | None = None,
    binary: str = "llama-server",
    library_path: str | None = None,
    gpu: bool = True,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> LlmClient | None:
    """Создаёт LLM-клиент выбранного провайдера.

    Основной способ — передать :class:`AppConfig` первым аргументом: провайдер
    и его параметры берутся из конфигурации. Плоские ключевые аргументы
    сохранены для обратной совместимости (TUI/тесты).

    Возвращает ``None``, если параметров недостаточно: для ``llama`` — нет
    GGUF-модели, для ``openai`` — не задан ``base_url`` или имя модели.
    """
    if config is not None:
        provider = config.llm_provider
        model_path = config.llm_model
        binary = config.llm_binary
        library_path = config.llm_lib_path
        gpu = config.llm_gpu
        context_size = config.llm_context_size
        request_timeout = config.llm_request_timeout
        base_url = config.llm_base_url
        model_name = config.llm_model_name
        api_key = config.llm_api_key

    normalized = (provider or DEFAULT_LLM_PROVIDER).strip().casefold()
    if normalized == "openai":
        if not base_url or not model_name:
            logger.warning(
                "LLM-провайдер openai: не задан base_url или имя модели — "
                "LLM-постобработка будет пропущена"
            )
            return None
        return OpenAIClient(
            base_url,
            model_name,
            api_key=api_key,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
        )

    if model_path is None:
        return None

    return LlamaServerClient(
        model_path,
        binary=binary,
        library_path=library_path,
        gpu=gpu,
        context_size=context_size,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )
