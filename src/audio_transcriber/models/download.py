"""Скачивание моделей каталога с Hugging Face (прогресс, отмена, докачка).

Модуль не зависит от веб-слоя: :class:`ModelDownloadManager` — движок
фоновой загрузки, а :class:`DownloadBus` публикует события прогресса
(веб-слой раздаёт их как SSE, CLI читает состояния напрямую). Сеть делает
только :class:`HfDownloader`; в тестах он подменяется заглушкой.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_transcriber.models.catalog import (
    ModelEntry,
    dir_size,
    local_status,
    primary_path,
    resolve_target,
)

logger = logging.getLogger(__name__)

#: Статусы загрузки.
STATUS_IDLE = "idle"
STATUS_DOWNLOADING = "downloading"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

#: Интервал keep-alive SSE по умолчанию (секунды).
DEFAULT_HEARTBEAT = 15.0

#: Порог публикации прогресса: доля от целого, после которой отправляется событие.
_PROGRESS_STEP = 0.005
#: Минимальный интервал между событиями прогресса (секунды).
_PROGRESS_INTERVAL = 0.25


class DownloadCancelled(Exception):
    """Скачивание прервано пользователем (внутренний сигнал потока загрузки)."""


@runtime_checkable
class Downloader(Protocol):
    """Источник файлов модели (реальный — Hugging Face; в тестах — заглушка)."""

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        """Скачивает один файл в ``destination`` (с докачкой)."""
        ...

    def fetch_snapshot(
        self,
        *,
        repo: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        """Скачивает весь снимок репозитория в ``destination``."""
        ...


def _redact(text: str, token: str | None) -> str:
    if token and token in text:
        return text.replace(token, "***")
    return text


def _shorten(text: str, limit: int = 300) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _progress_tqdm(on_progress: Callable[[int], None]) -> type:
    """Класс tqdm, вызывающий ``on_progress(скачано_байт)`` на каждом шаге."""
    from tqdm.auto import tqdm as base_tqdm

    class _ProgressTqdm(base_tqdm):
        def update(self, n: int | float | None = 1) -> bool | None:
            result = super().update(n)
            if n:
                on_progress(int(n))
            return result

    return _ProgressTqdm


class HfDownloader:
    """Реальный загрузчик через ``huggingface_hub`` (без сети — не вызывается)."""

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        from huggingface_hub import hf_hub_download

        destination.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(destination.parent),
            token=token,
            tqdm_class=_progress_tqdm(on_progress),
        )

    def fetch_snapshot(
        self,
        *,
        repo: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        from huggingface_hub import snapshot_download

        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(destination),
            token=token,
            tqdm_class=_progress_tqdm(on_progress),
        )


@dataclass(slots=True)
class DownloadState:
    """Состояние загрузки одной модели."""

    model_id: str
    status: str = STATUS_IDLE
    fraction: float = 0.0
    bytes_done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fraction": round(self.fraction, 4) if self.status != STATUS_IDLE else None,
            "bytes_done": self.bytes_done,
            "total": self.total,
            "message": self.message,
            "error": self.error,
        }


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[dict[str, object] | None]
    loop: asyncio.AbstractEventLoop


class DownloadBus:
    """Глобальная (по всем моделям) потокобезопасная шина событий прогресса."""

    def __init__(self, *, heartbeat: float = DEFAULT_HEARTBEAT) -> None:
        self._heartbeat = heartbeat
        self._lock = threading.Lock()
        self._subscribers: list[_Subscriber] = []
        self._history: deque[dict[str, object]] = deque(maxlen=500)
        self._seq = 0

    def publish(self, event: Mapping[str, object]) -> None:
        """Отправляет событие подписчикам и запоминает его в истории."""
        payload = dict(event)
        with self._lock:
            self._seq += 1
            payload["seq"] = self._seq
            self._history.append(payload)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, payload)
            except RuntimeError:
                continue

    def history(self, after: int | None = None) -> list[dict[str, object]]:
        """Снимок накопленных событий (для подключившихся с задержкой)."""
        with self._lock:
            events = list(self._history)
        if after is None:
            return events
        result: list[dict[str, object]] = []
        for event in events:
            seq = event.get("seq")
            if isinstance(seq, int) and seq > after:
                result.append(event)
        return result

    def clear(self) -> None:
        """Очищает историю (при остановке сервера/тестах)."""
        with self._lock:
            self._history.clear()

    async def subscribe(self) -> AsyncIterator[dict[str, object] | None]:
        """Живой поток событий; ``None`` — сигнал keep-alive по таймауту."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        subscriber = _Subscriber(queue=queue, loop=loop)
        with self._lock:
            self._subscribers.append(subscriber)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._heartbeat)
                except TimeoutError:
                    yield None
                    continue
                yield item
        finally:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)


@dataclass(slots=True)
class _ProgressTracker:
    """Накопитель прогресса по байтам с троттлингом публикации событий."""

    manager: ModelDownloadManager
    model_id: str
    cancel: threading.Event
    total: int
    done: int = 0
    last_event: float = field(default_factory=time.monotonic)

    def __call__(self, delta: int) -> None:
        if self.cancel.is_set():
            raise DownloadCancelled(self.model_id)
        increment = max(int(delta), 0)
        with self.manager.lock:
            self.done += increment
            state = self.manager.state_locked(self.model_id)
            state.bytes_done = self.done
            state.total = self.total
            if self.total > 0:
                state.fraction = min(self.done / self.total, 1.0)
            now = time.monotonic()
            changed = abs(state.fraction - self._last_fraction) >= _PROGRESS_STEP
            if not changed and (now - self.last_event) < _PROGRESS_INTERVAL:
                return
            self.last_event = now
            self._last_fraction = state.fraction
            event = self.manager.event_locked(state, "Скачивание…")
        self.manager.bus.publish(event)

    _last_fraction: float = 0.0


class ModelDownloadManager:
    """Фоновые загрузки моделей с прогрессом, отменой и удалением файлов."""

    def __init__(
        self,
        *,
        models_root: Path,
        downloader: Downloader,
        resolve_target: Callable[[ModelEntry], Path],
        bus: DownloadBus | None = None,
    ) -> None:
        self._models_root = models_root
        self._downloader = downloader
        self._resolve_target = resolve_target
        self._bus = bus or DownloadBus()
        self.lock = threading.Lock()
        self._states: dict[str, DownloadState] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancels: dict[str, threading.Event] = {}

    @property
    def bus(self) -> DownloadBus:
        return self._bus

    @property
    def models_root(self) -> Path:
        return self._models_root

    def state_locked(self, model_id: str) -> DownloadState:
        """Состояние модели (вызывается под ``self.lock``)."""
        state = self._states.get(model_id)
        if state is None:
            state = DownloadState(model_id=model_id)
            self._states[model_id] = state
        return state

    def state(self, model_id: str) -> DownloadState:
        """Снимок состояния модели."""
        with self.lock:
            return replace(self.state_locked(model_id))

    def is_running(self, model_id: str) -> bool:
        with self.lock:
            thread = self._threads.get(model_id)
            return thread is not None and thread.is_alive()

    def event_locked(
        self, state: DownloadState, message: str | None = None
    ) -> dict[str, object]:
        """Событие SSE по состоянию (вызывается под ``self.lock``)."""
        return {
            "id": state.model_id,
            "status": state.status,
            "fraction": round(state.fraction, 4) if state.status != STATUS_IDLE else None,
            "bytes_done": state.bytes_done,
            "total": state.total,
            "message": message if message is not None else state.message,
        }

    def start(self, entry: ModelEntry, *, token: str | None) -> bool:
        """Запускает загрузку модели; ``False`` — если она уже идёт."""
        with self.lock:
            existing = self._threads.get(entry.id)
            if existing is not None and existing.is_alive():
                return False
            target = self._resolve_target(entry)
            state = self.state_locked(entry.id)
            state.status = STATUS_DOWNLOADING
            state.fraction = 0.0
            state.bytes_done = 0
            state.total = entry.approx_size
            state.message = "Скачивание…"
            state.error = None
            cancel = threading.Event()
            self._cancels[entry.id] = cancel
            thread = threading.Thread(
                target=self._run,
                args=(entry, target, token, cancel),
                name=f"model-download-{entry.id}",
                daemon=True,
            )
            self._threads[entry.id] = thread
            event = self.event_locked(state)
        self._bus.publish(event)
        thread.start()
        return True

    def cancel(self, model_id: str) -> bool:
        """Просит прервать загрузку; ``False`` — если она не выполняется."""
        with self.lock:
            thread = self._threads.get(model_id)
            if thread is None or not thread.is_alive():
                return False
            cancel = self._cancels.get(model_id)
        if cancel is not None:
            cancel.set()
        return True

    def wait(self, model_id: str, *, timeout: float = 10.0) -> None:
        """Дожидается завершения потока загрузки (используется в тестах)."""
        with self.lock:
            thread = self._threads.get(model_id)
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(
        self, entry: ModelEntry, target: Path, token: str | None, cancel: threading.Event
    ) -> None:
        tracker = _ProgressTracker(
            manager=self,
            model_id=entry.id,
            cancel=cancel,
            total=entry.approx_size,
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            for model_file in entry.files:
                if cancel.is_set():
                    raise DownloadCancelled(entry.id)
                self._downloader.fetch(
                    repo=entry.repo,
                    filename=model_file.filename,
                    destination=target / model_file.filename,
                    token=token,
                    on_progress=tracker,
                )
            if entry.snapshot:
                if cancel.is_set():
                    raise DownloadCancelled(entry.id)
                self._downloader.fetch_snapshot(
                    repo=entry.repo,
                    destination=target,
                    token=token,
                    on_progress=tracker,
                )
            if cancel.is_set():
                raise DownloadCancelled(entry.id)
        except DownloadCancelled:
            self._finish(entry.id, STATUS_CANCELLED, message="Скачивание отменено")
            return
        except Exception as exc:  # noqa: BLE001 — внешняя сеть/диск: показываем причину
            message = _shorten(_redact(str(exc), token))
            logger.warning("Загрузка модели %s не удалась: %s", entry.id, message)
            self._finish(entry.id, STATUS_ERROR, message="Ошибка загрузки", error=message)
            return
        self._finish(entry.id, STATUS_DONE, message="Готово", actual_size=dir_size(target))

    def _finish(
        self,
        model_id: str,
        status: str,
        *,
        message: str,
        error: str | None = None,
        actual_size: int | None = None,
    ) -> None:
        with self.lock:
            state = self.state_locked(model_id)
            state.status = status
            state.message = message
            state.error = error
            if status == STATUS_DONE:
                state.fraction = 1.0
                if actual_size is not None:
                    state.bytes_done = actual_size
            event = self.event_locked(state)
            self._threads.pop(model_id, None)
            self._cancels.pop(model_id, None)
        self._bus.publish(event)


def model_payload(
    entry: ModelEntry,
    *,
    models_root: Path,
    settings: object,
    state: DownloadState | None = None,
) -> dict[str, object]:
    """JSON-представление записи каталога с локальным статусом и прогрессом."""
    target = resolve_target(entry, models_root=models_root, settings=settings)
    status = local_status(entry, target)
    primary = primary_path(entry, target)
    payload: dict[str, object] = {
        "id": entry.id,
        "kind": entry.kind,
        "title": entry.title,
        "repo": entry.repo,
        "files": [model_file.filename for model_file in entry.files],
        "approx_size": entry.approx_size,
        "note": entry.note,
        "gated": entry.gated,
        "setting_key": entry.setting_key,
        "target_dir": entry.target_dir,
        "target_path": str(target),
        "primary_path": str(primary),
        "status": {
            "present": status.present,
            "size": status.size,
            "expected_size": status.expected_size,
            "missing_files": list(status.missing),
            "path": str(status.path),
            "partial": status.partial,
        },
    }
    if state is not None:
        payload["download"] = state.as_dict()
    return payload
