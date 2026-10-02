"""Прогресс длительных действий веб-интерфейса (этапы + время).

Длительные операции, запускаемые кнопками поверх уже готовой задачи —
«Применить имена» (enrollment, #37), «Применить глоссарий» (#32),
«Проверить текст» (#51) и «Сформировать протокол» (LLM), — выполняются
синхронным HTTP-запросом. Чтобы интерфейс не выглядел «зависшим», сервер
публикует этапы выполнения в отдельную шину событий, а фронтенд подписывается
на неё отдельным SSE-каналом ``GET /api/actions/{action_id}/events``.

Ключ шины — ``action_id``, который генерирует клиент и передаёт заголовком
``X-Action-Id``. Отдельная шина (а не общий ``JobEventBus`` задачи) нужна
потому, что после завершения задачи её SSE-поток уже закрыт, а события
действий не должны смешиваться с прогрессом транскрибации.

Каждое событие имеет форму::

    {
        "action": "enrollment" | "glossary" | "correction" | "protocol",
        "stage":  "samples" | "embeddings" | "matching" | "apply" | ...,
        "message": "Расчёт эмбеддингов образца «Иван» (2/3)",
        "fraction": 0.45 | None,
        "elapsed": 3.21,
        "status": "running" | "done" | "error",
    }

Конечное событие (``status`` из :data:`~audio_transcriber.web.storage.jobs_db.
TERMINAL_STATUSES`) закрывает SSE-поток подписчиков — как и в шине задач.
История последних действий ограничена, чтобы не расти за время жизни сервера.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.web.events import JobEventBus

#: Сколько последних действий хранить (история для запоздавших подписчиков).
DEFAULT_MAX_TRACKED_ACTIONS = 64

#: Максимальная длина идентификатора действия, принимаемого с клиента.
MAX_ACTION_ID_LENGTH = 64

#: Символы, допустимые в ``action_id`` (UUID + безопасные разделители).
_ACTION_ID_EXTRA = frozenset({"-", "_", "."})

#: Идентификаторы видов действий (первое поле события ``action``).
ACTION_ENROLLMENT = "enrollment"
ACTION_GLOSSARY = "glossary"
ACTION_CORRECTION = "correction"
ACTION_PROTOCOL = "protocol"

#: Статус незавершённого действия.
ACTION_RUNNING = "running"


class ActionEventBus:
    """Потокобезопасная шина событий действий с ограниченной историей.

    Обёртка над :class:`~audio_transcriber.web.events.JobEventBus`: ключ —
    ``action_id``. При переполнении самые старые действия забываются
    (``clear``), чтобы история не росла бесконечно.
    """

    def __init__(
        self,
        *,
        heartbeat: float = 15.0,
        max_actions: int = DEFAULT_MAX_TRACKED_ACTIONS,
    ) -> None:
        self._bus = JobEventBus(heartbeat=heartbeat)
        self._order: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._max_actions = max(1, max_actions)

    def publish(self, action_id: str, event: dict[str, object]) -> None:
        """Публикует событие действия и обновляет порядок вытеснения."""
        self._reserve(action_id)
        self._bus.publish(action_id, event)

    def history(self, action_id: str) -> list[dict[str, object]]:
        """Снимок накопленных событий действия (может быть пустым)."""
        return self._bus.history(action_id)

    def subscribe(
        self, action_id: str
    ) -> AsyncIterator[dict[str, object] | None]:
        """Отдаёт историю действия, затем живой поток (``None`` — keep-alive)."""
        return self._bus.subscribe(action_id)

    def _reserve(self, action_id: str) -> None:
        """Отмечает действие как «недавнее», вытесняя самые старые."""
        with self._lock:
            self._order.pop(action_id, None)
            self._order[action_id] = None
            while len(self._order) > self._max_actions:
                oldest, _ = self._order.popitem(last=False)
                self._bus.clear(oldest)


@dataclass(slots=True)
class ActionProgress:
    """Публикует этапы одного действия и ведёт его секундомер.

    Создаётся на время HTTP-запроса действия. ``bus`` — общая шина действий,
    ``action_id`` — идентификатор от клиента, ``kind`` — вид действия
    (:data:`ACTION_ENROLLMENT` и т.д.). Секундомер основан на ``monotonic`` и
    попадает в поле ``elapsed`` каждого события.
    """

    bus: ActionEventBus
    action_id: str
    kind: str
    _started: float = field(default_factory=time.monotonic)
    _finished: bool = False

    def emit(
        self,
        stage: str,
        message: str,
        fraction: float | None = None,
    ) -> None:
        """Публикует событие текущего этапа (``status="running"``)."""
        if self._finished:
            return
        self.bus.publish(
            self.action_id,
            {
                "action": self.kind,
                "stage": stage,
                "message": message,
                "fraction": fraction,
                "elapsed": self.elapsed(),
                "status": ACTION_RUNNING,
            },
        )

    def done(self, message: str = "Готово") -> None:
        """Публикует успешное конечное событие (закрывает SSE-поток)."""
        self._finish("done", "done", message, 1.0)

    def fail(self, message: str) -> None:
        """Публикует конечное событие ошибки (закрывает SSE-поток)."""
        self._finish("error", "error", message, None)

    def elapsed(self) -> float:
        """Секунды с начала действия (с округлением до миллисекунд)."""
        return round(time.monotonic() - self._started, 3)

    def as_callback(self) -> ProgressCallback:
        """Адаптер :class:`ProgressEvent` → события действия.

        Позволяет переиспользовать существующие функции с ``on_progress``
        (enrollment, ``generate_protocol``), не меняя формат их событий.
        """

        def callback(event: ProgressEvent) -> None:
            self.emit(event.stage, event.message, event.fraction)

        return callback

    def _finish(
        self,
        status: str,
        stage: str,
        message: str,
        fraction: float | None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        self.bus.publish(
            self.action_id,
            {
                "action": self.kind,
                "stage": stage,
                "message": message,
                "fraction": fraction,
                "elapsed": self.elapsed(),
                "status": status,
            },
        )


def sanitize_action_id(value: str | None) -> str | None:
    """Проверяет и нормализует ``action_id`` с клиента.

    Допустимы буквы, цифры и символы ``-``, ``_``, ``.`` (UUID и подобные).
    Пустое значение, слишком длинная строка или посторонние символы → ``None``
    (прогресс для такого действия не ведётся).
    """
    if value is None:
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > MAX_ACTION_ID_LENGTH:
        return None
    if not all(char.isalnum() or char in _ACTION_ID_EXTRA for char in candidate):
        return None
    return candidate
