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


class _JobStream:
    """Результат атомарной подписки: снимок истории + живой поток.

    Создаётся :meth:`JobEventBus.open`; подписчик уже зарегистрирован в шине,
    поэтому события, опубликованные после возврата ``open``, не теряются.
    ``terminal`` означает, что задача завершилась и живой поток пуст.
    """

    __slots__ = ("_bus", "_job_id", "_subscriber", "history", "terminal")

    def __init__(
        self,
        bus: JobEventBus,
        job_id: str,
        subscriber: _Subscriber | None,
        history: list[JobEvent],
        terminal: bool,
    ) -> None:
        self._bus = bus
        self._job_id = job_id
        self._subscriber = subscriber
        self.history = history
        self.terminal = terminal

    async def events(self) -> AsyncIterator[JobEvent | None]:
        """Живой поток: событие, ``None`` — heartbeat; у конечной задачи пусто."""
        subscriber = self._subscriber
        if self.terminal or subscriber is None:
            return
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        subscriber.queue.get(), timeout=self._bus._heartbeat
                    )
                except TimeoutError:
                    yield None
                    continue
                if item is None:
                    break
                yield item
        finally:
            self._bus._discard_subscriber(self._job_id, subscriber)


class JobEventBus:
    """Потокобезопасная шина событий по ``job_id``."""

    def __init__(self, *, heartbeat: float = DEFAULT_HEARTBEAT) -> None:
        self._heartbeat = heartbeat
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._history: dict[str, list[JobEvent]] = {}
        #: Монотонный (на задачу) номер события: клиент по нему дедуплицирует
        #: историю, повторно отданную при (пере)подключении SSE, а также
        #: сопоставляет ``Last-Event-ID``. Аналог ``DownloadBus._seq`` (#65).
        self._seq: dict[str, int] = {}

    def publish(self, job_id: str, event: JobEvent) -> None:
        """Публикует событие всем текущим подписчикам задачи.

        Конечное событие (``status`` из ``TERMINAL_STATUSES``) дополнительно
        закрывает поток подписчиков сигналом ``None``.
        """
        payload = dict(event)
        with self._lock:
            seq = self._seq.get(job_id, 0) + 1
            self._seq[job_id] = seq
            payload["seq"] = seq
            self._history.setdefault(job_id, []).append(payload)
            subscribers = list(self._subscribers.get(job_id, ()))
        terminal = payload.get("status") in TERMINAL_STATUSES
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, payload)
                if terminal:
                    subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, None)
            except RuntimeError:
                # Event loop уже закрыт (остановка сервера) — событие некому отдать.
                continue

    def history(self, job_id: str, after: int | None = None) -> list[JobEvent]:
        """Снимок накопленных событий задачи (может быть пустым).

        ``after`` задаёт ``Last-Event-ID``: отдаются только события с ``seq``
        строго больше него, чтобы при переподключении не повторять уже
        полученное клиентом (как в :meth:`DownloadBus.history`).
        """
        with self._lock:
            events = self._history_after_locked(job_id, after)
        return events

    def _history_after_locked(self, job_id: str, after: int | None) -> list[JobEvent]:
        """Отфильтрованная история; вызывать под ``self._lock`` (не реентерабельный)."""
        events = list(self._history.get(job_id, ()))
        if after is None:
            return events
        result: list[JobEvent] = []
        for event in events:
            seq = event.get("seq")
            if isinstance(seq, int) and seq > after:
                result.append(event)
        return result

    def _discard_subscriber(self, job_id: str, subscriber: _Subscriber) -> None:
        """Убирает подписчика из шины (идемпотентно)."""
        with self._lock:
            subscribers = self._subscribers.get(job_id)
            if subscribers and subscriber in subscribers:
                subscribers.remove(subscriber)
                if not subscribers:
                    self._subscribers.pop(job_id, None)

    def open(self, job_id: str, *, after: int | None = None) -> _JobStream:
        """Атомарно: снимок истории (``after``) + регистрация живого подписчика.

        Возвращает :class:`_JobStream` с готовым ``history`` и живым потоком
        ``events()``. Снимок истории и регистрация берутся под одним локом —
        события, опубликованные между ними, **не теряются** (issue #85): они
        попадут либо в ``history``, либо в очередь подписчика.

        ``already_terminal`` определяется по **полной** истории (а не по
        отфильтрованной ``after``), чтобы подключение после уже доставленного
        конечного события не «зависло» в ожидании.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[JobEvent | None] = asyncio.Queue()
        subscriber: _Subscriber | None = None
        with self._lock:
            full = list(self._history.get(job_id, ()))
            already_terminal = bool(
                full and full[-1].get("status") in TERMINAL_STATUSES
            )
            history = self._history_after_locked(job_id, after)
            if not already_terminal:
                subscriber = _Subscriber(queue=queue, loop=loop)
                self._subscribers.setdefault(job_id, []).append(subscriber)
        return _JobStream(self, job_id, subscriber, history, already_terminal)

    def clear(self, job_id: str) -> None:
        """Забывает историю, подписчиков и нумерацию задачи.

        Вызывается при удалении задачи и в начале повторного прогона. Новый
        прогон начинается с ``seq=1``: клиент при перезапуске создаёт свежий
        ``EventSource`` (без ``Last-Event-ID`` прошлого прогона), а прежний
        номер нужен был только живому потоку. Нумерацию забываем, чтобы
        ``_seq`` не разрастался по всем когда-либо созданным задачам.
        """
        with self._lock:
            self._history.pop(job_id, None)
            self._subscribers.pop(job_id, None)
            self._seq.pop(job_id, None)

    async def subscribe(
        self, job_id: str, *, replay: bool = True
    ) -> AsyncIterator[JobEvent | None]:
        """История задачи (если ``replay``), затем живой поток событий.

        ``replay=False`` используется HTTP-слоем, который сам отдаёт историю
        ДО свежего снимка состояния: иначе повторно отданные старые события
        откатили бы таймеры клиента. Даже без реплея подписчик получает
        конечное событие, если задача уже завершилась в промежутке между
        снимком истории и подпиской (иначе поток «завис» бы).

        Возвращает ``None`` на таймауте ожидания (сигнал для keep-alive) и
        завершается, когда приходит конечное событие.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[JobEvent | None] = asyncio.Queue()
        subscriber = _Subscriber(queue=queue, loop=loop)
        with self._lock:
            history = list(self._history.get(job_id, ()))
            already_terminal = bool(
                history and history[-1].get("status") in TERMINAL_STATUSES
            )
            if not already_terminal:
                self._subscribers.setdefault(job_id, []).append(subscriber)
        try:
            if replay:
                for event in history:
                    yield event
            elif already_terminal and history:
                # Гонка: задача завершилась после снимка истории, но до подписки.
                yield history[-1]
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
