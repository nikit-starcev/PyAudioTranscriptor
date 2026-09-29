"""In-process шина событий прогресса задач для SSE.

Один фоновый воркер публикует события прогресса из своего потока, а HTTP-слой
раздаёт их подписчикам через ``text/event-stream``. Мост между потоком воркера
и event loop FastAPI сделан через ``loop.call_soon_threadsafe`` — это не требует
потокоблокирующих очередей на каждый запрос.

Шина хранит историю событий задачи: подписчик, подключившийся с задержкой,
сначала получает накопленные события, а затем живой поток. Если соединение
простаивает дольше ``heartbeat`` секунд, подписчику отправляется комментарий-
пинг (клиент так понимает, что канал жив).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass

from audio_transcriber.web.storage.jobs_db import TERMINAL_STATUSES

#: Тип события прогресса: ``stage``/``fraction``/``message``/``status``.
JobEvent = dict[str, object]

#: Интервал keep-alive по умолчанию (секунды).
DEFAULT_HEARTBEAT = 15.0


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[JobEvent | None]
    loop: asyncio.AbstractEventLoop


class JobEventBus:
    """Потокобезопасная шина событий по ``job_id``."""

    def __init__(self, *, heartbeat: float = DEFAULT_HEARTBEAT) -> None:
        self._heartbeat = heartbeat
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._history: dict[str, list[JobEvent]] = {}

    def publish(self, job_id: str, event: JobEvent) -> None:
        """Публикует событие всем текущим подписчикам задачи.

        Конечное событие (``status`` из ``TERMINAL_STATUSES``) дополнительно
        закрывает поток подписчиков сигналом ``None``.
        """
        with self._lock:
            self._history.setdefault(job_id, []).append(event)
            subscribers = list(self._subscribers.get(job_id, ()))
        terminal = event.get("status") in TERMINAL_STATUSES
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, event)
                if terminal:
                    subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, None)
            except RuntimeError:
                # Event loop уже закрыт (остановка сервера) — событие некому отдать.
                continue

    def history(self, job_id: str) -> list[JobEvent]:
        """Снимок накопленных событий задачи (может быть пустым)."""
        with self._lock:
            return list(self._history.get(job_id, ()))

    def clear(self, job_id: str) -> None:
        """Забывает историю и подписчиков задачи (при удалении)."""
        with self._lock:
            self._history.pop(job_id, None)
            self._subscribers.pop(job_id, None)

    async def subscribe(self, job_id: str) -> AsyncIterator[JobEvent | None]:
        """Отдаёт историю задачи, затем живой поток событий.

        Возвращает ``None`` на таймауте ожидания (сигнал для keep-alive) и
        завершается, когда приходит конечное событие.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[JobEvent | None] = asyncio.Queue()
        subscriber = _Subscriber(queue=queue, loop=loop)
        with self._lock:
            replay = list(self._history.get(job_id, ()))
            already_terminal = bool(replay and replay[-1].get("status") in TERMINAL_STATUSES)
            if not already_terminal:
                self._subscribers.setdefault(job_id, []).append(subscriber)
        try:
            for event in replay:
                yield event
            if already_terminal:
                return
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._heartbeat)
                except TimeoutError:
                    yield None
                    continue
                if item is None:
                    break
                yield item
        finally:
            with self._lock:
                subscribers = self._subscribers.get(job_id)
                if subscribers and subscriber in subscribers:
                    subscribers.remove(subscriber)
                    if not subscribers:
                        self._subscribers.pop(job_id, None)
