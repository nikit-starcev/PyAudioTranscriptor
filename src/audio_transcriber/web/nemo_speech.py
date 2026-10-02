"""Фоновое скачивание модели Sortformer для nemo-speech из веб-UI.

Модуль — веб-обёртка над ``nemo-speech pull <model>``: запускает команду в
фоновом потоке, разбирает её построчный вывод (фазы загрузки/проверки) и
публикует прогресс в общую шину :class:`~audio_transcriber.web.models.DownloadBus`
(SSE с монотонным ``seq``/``id`` и поддержкой ``Last-Event-ID`` — как у
загрузки моделей HF и установки пакетов).

Реальный запуск ``nemo-speech`` вынесен в ``runner``, который подменяется в
тестах: сторонние команды в тестах не выполняются. Не более одной загрузки
одновременно (проверяется на уровне эндпоинта и менеджера).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from audio_transcriber.config.defaults import DEFAULT_NEMO_SPEECH_MODEL
from audio_transcriber.diarization import nemo_speech_assets as assets
from audio_transcriber.utils.env import with_library_path
from audio_transcriber.web.models import (
    STATUS_DONE,
    STATUS_DOWNLOADING,
    STATUS_ERROR,
    STATUS_IDLE,
    DownloadBus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NemoSpeechModelDownloader",
    "NemoSpeechPullState",
    "PullRunner",
    "run_nemo_speech_pull",
]

#: Запуск команды pull: ``(команда, on_line, окружение) -> код возврата``.
PullRunner = Callable[[list[str], Callable[[str], None], Mapping[str, str]], int]

#: Заголовок фазы скачивания: ``[model] downloading <repo>@<rev> (<role>, 140.3 MiB)``.
_DOWNLOADING_RE = re.compile(
    r"\[model\]\s+downloading\s+\S+@\S+\s+\([^,]+,\s*([\d.]+)\s*"
    r"(B|KiB|MiB|GiB|TiB)\)",
    re.IGNORECASE,
)
#: Отдельные объёмы в строках прогресса curl (если он их печатает).
_BYTES_RE = re.compile(r"([\d.]+)\s*(KiB|MiB|GiB|TiB|B)\b", re.IGNORECASE)
#: Путь к готовому ``.gguf`` (строка ``ready: <path>`` или финальный TSV).
_GGUF_RE = re.compile(r"(\S+\.gguf)\b", re.IGNORECASE)

_UNITS: Mapping[str, int] = {
    "B": 1,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def _to_bytes(value: str, unit: str) -> int:
    try:
        return int(float(value) * _UNITS[unit.upper()])
    except (KeyError, ValueError):
        return 0


def _shorten(text: str, limit: int = 200) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def run_nemo_speech_pull(
    command: list[str],
    on_line: Callable[[str], None],
    env: Mapping[str, str],
) -> int:
    """Реальный запуск ``nemo-speech pull``: stdout+stderr построчно в ``on_line``."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=dict(env),
    )
    try:
        if process.stdout is not None:
            for raw in process.stdout:
                on_line(raw.rstrip("\r\n"))
    finally:
        process.wait()
    return process.returncode


@dataclass(slots=True)
class NemoSpeechPullState:
    """Состояние фоновой загрузки модели nemo-speech."""

    status: str = STATUS_IDLE
    message: str = ""
    fraction: float | None = None
    bytes_done: int = 0
    total: int = 0
    path: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "message": self.message,
            "fraction": round(self.fraction, 4) if self.fraction is not None else None,
            "bytes_done": self.bytes_done,
            "total": self.total,
            "path": self.path,
            "error": self.error,
        }


class NemoSpeechModelDownloader:
    """Скачивание модели Sortformer: один фоновый поток, прогресс в шину."""

    def __init__(
        self,
        *,
        bus: DownloadBus,
        runner: PullRunner | None = None,
        on_success: Callable[[], None] | None = None,
        poll_interval: float | None = 0.5,
    ) -> None:
        self._bus = bus
        self._runner = runner or run_nemo_speech_pull
        self._on_success = on_success
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._state = NemoSpeechPullState()
        self._thread: threading.Thread | None = None

    @property
    def bus(self) -> DownloadBus:
        return self._bus

    def state(self) -> NemoSpeechPullState:
        """Снимок состояния загрузки."""
        with self._lock:
            return replace(self._state)

    def is_running(self) -> bool:
        """Идёт ли загрузка прямо сейчас."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        """Дожидается завершения загрузки (используется в тестах)."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def start(self, *, binary: str, model: str, lib_path: str | None = None) -> bool:
        """Запускает ``nemo-speech pull <model>``; ``False`` — если уже идёт."""
        command = [binary, "pull", model or DEFAULT_NEMO_SPEECH_MODEL]
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._state = NemoSpeechPullState(
                status=STATUS_DOWNLOADING,
                message="Скачивание модели…",
            )
            thread = threading.Thread(
                target=self._run,
                args=(command, model or DEFAULT_NEMO_SPEECH_MODEL, lib_path),
                name="nemo-speech-pull",
                daemon=True,
            )
            self._thread = thread
            event = self._event_locked()
        self._bus.publish(event)
        thread.start()
        return True

    def _event_locked(self) -> dict[str, object]:
        """Событие SSE по текущему состоянию (вызывается под ``self._lock``)."""
        state = self._state
        return {
            "status": state.status,
            "message": state.message,
            "fraction": round(state.fraction, 4) if state.fraction is not None else None,
            "bytes_done": state.bytes_done,
            "total": state.total,
            "path": state.path,
            "error": state.error,
        }

    def _run(self, command: list[str], model: str, lib_path: str | None) -> None:
        env = self._process_env(lib_path)
        monitor = self._start_monitor(model)

        def on_line(line: str) -> None:
            event = self._consume(line)
            if event is not None:
                self._bus.publish(event)

        try:
            code = self._runner(command, on_line, env)
        except Exception as exc:  # noqa: BLE001 — внешний процесс: мягкая ошибка
            logger.warning("nemo-speech pull не запустился: %s", exc)
            self._finish_error(f"Не удалось запустить nemo-speech: {exc}")
            return
        finally:
            if monitor is not None:
                monitor.set()

        if code == 0:
            self._finish_done(model, env)
        else:
            self._finish_error(f"nemo-speech завершился с кодом {code}")

    def _process_env(self, lib_path: str | None) -> dict[str, str]:
        """Окружение запуска: безопасный ``LD_LIBRARY_PATH`` для бандла."""
        return with_library_path(os.environ, assets.safe_library_path(lib_path))

    def _consume(self, line: str) -> dict[str, object] | None:
        """Разбирает одну строку вывода и публикует событие, если что-то изменилось."""
        text = line.strip()
        if not text:
            return None
        lower = text.casefold()
        with self._lock:
            state = self._state
            if "verifying size and sha-256" in lower:
                state.message = "Проверка размера и SHA-256…"
                if state.total:
                    state.fraction = 0.99
                return self._event_locked()
            if lower.startswith("[model] downloading"):
                state.status = STATUS_DOWNLOADING
                state.message = "Скачивание модели…"
                match = _DOWNLOADING_RE.search(text)
                if match is not None:
                    state.total = _to_bytes(match.group(1), match.group(2))
                return self._event_locked()
            path_match = _GGUF_RE.search(text)
            if path_match is not None:
                state.path = path_match.group(1)
                state.message = "Модель скачана, идёт проверка…"
                return self._event_locked()
            bytes_match = _BYTES_RE.search(text)
            if bytes_match is not None:
                done = _to_bytes(bytes_match.group(1), bytes_match.group(2))
                if done > state.bytes_done:
                    state.bytes_done = done
                    if state.total:
                        state.fraction = min(done / state.total, 1.0)
                    state.message = "Скачивание модели…"
                    return self._event_locked()
            if lower.startswith("[model] license"):
                state.message = "Загрузка с Hugging Face…"
                return self._event_locked()
            logger.debug("nemo-speech pull: %s", text)
            return None

    def _start_monitor(self, model: str) -> threading.Event | None:
        """Поток-наблюдатель за размером ``.gguf`` в кэше (реальный прогресс).

        Отключён при ``poll_interval=None`` (тесты): тогда прогресс строится
        только по строкам вывода команды.
        """
        interval = self._poll_interval
        if interval is None or interval <= 0:
            return None
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval):
                size = assets.cached_model_size(model)
                with self._lock:
                    if size <= self._state.bytes_done:
                        continue
                    self._state.bytes_done = size
                    if self._state.total:
                        self._state.fraction = min(size / self._state.total, 1.0)
                    event = self._event_locked()
                self._bus.publish(event)

        thread = threading.Thread(target=loop, name="nemo-speech-size", daemon=True)
        thread.start()
        return stop

    def _finish_done(self, model: str, env: Mapping[str, str]) -> None:
        status = assets.model_status(model, env=env)
        with self._lock:
            state = self._state
            state.status = STATUS_DONE
            state.fraction = 1.0
            state.message = "Готово"
            state.error = None
            state.path = status.path or state.path
            if status.size:
                state.bytes_done = status.size
                state.total = state.total or status.size
            self._thread = None
            event = self._event_locked()
        self._bus.publish(event)
        if self._on_success is not None:
            try:
                self._on_success()
            except Exception as exc:  # noqa: BLE001 — инвалидация кэша не должна ронять поток
                logger.warning("Сбой после успешного скачивания модели nemo-speech: %s", exc)

    def _finish_error(self, message: str) -> None:
        with self._lock:
            state = self._state
            state.status = STATUS_ERROR
            state.message = message
            state.error = message
            self._thread = None
            event = self._event_locked()
        self._bus.publish(event)
