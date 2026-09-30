"""Замер длительности стадий конвейера для веб-интерфейса.

TUI считает времена стадий самостоятельно по ``time.monotonic``, но веб-слой
работает асинхронно (SSE, отдельные HTTP-запросы) и должен уметь отдать уже
посчитанные тайминги через API. :class:`StageTimer` делает это по потоку
событий :class:`~audio_transcriber.progress.ProgressEvent`:

* стадия открывается при первом событии с её ``stage``;
* закрывается, когда начинается следующая стадия или приходит ``done``;
* повторные события той же стадии (например, диаризация: сегментация →
  эмбеддинги → enrollment) не сбрасывают отсчёт, а только уточняют признак
  «из кэша».

Кэшированные стадии, которые конвейер помечает ``detail="из кэша"``,
получают флаг ``cached``. Время монотонное, поэтому длительности не зависят
от перевода системных часов.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from audio_transcriber.progress import ProgressEvent

#: Подстрока в ``detail``, которой конвейер помечает попадание стадии в кэш.
CACHED_DETAIL = "из кэша"

#: Стадия завершения конвейера (закрывает текущую стадию).
DONE_STAGE = "done"


@dataclass(slots=True)
class StageTiming:
    """Длительность одной стадии конвейера."""

    stage: str
    seconds: float
    cached: bool = False

    def as_dict(self) -> dict[str, object]:
        """JSON-представление записи (совместимо с API и БД)."""
        return {"stage": self.stage, "seconds": round(self.seconds, 3), "cached": self.cached}

    @classmethod
    def from_mapping(cls, data: object) -> StageTiming | None:
        """Читает запись из JSON-совместимого словаря; ``None`` — если мусор."""
        if not isinstance(data, dict):
            return None
        stage = data.get("stage")
        seconds = data.get("seconds")
        if not isinstance(stage, str) or not isinstance(seconds, (int, float)):
            return None
        return cls(stage=stage, seconds=float(seconds), cached=bool(data.get("cached", False)))


class StageTimer:
    """Накопитель длительностей стадий по событиям прогресса."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._overall_start = clock()
        self._timings: list[StageTiming] = []
        self._current: str | None = None
        self._started: float = 0.0
        self._cached: bool = False

    def observe(self, event: ProgressEvent) -> None:
        """Учитывает событие стадии (``done`` закрывает текущую стадию)."""
        stage = event.stage
        if stage == DONE_STAGE:
            self.close()
            return
        if stage == self._current:
            if CACHED_DETAIL in event.detail:
                self._cached = True
            return
        self.close()
        self._current = stage
        self._started = self._clock()
        self._cached = CACHED_DETAIL in event.detail

    def close(self) -> None:
        """Закрывает текущую стадию (идемпотентно)."""
        if self._current is None:
            return
        seconds = max(self._clock() - self._started, 0.0)
        self._timings.append(StageTiming(self._current, seconds, self._cached))
        self._current = None
        self._cached = False

    def elapsed(self) -> float:
        """Сколько прошло с момента создания таймера (общее время обработки)."""
        return max(self._clock() - self._overall_start, 0.0)

    def current_elapsed(self) -> float:
        """Сколько длится текущая (незакрытая) стадия."""
        if self._current is None:
            return 0.0
        return max(self._clock() - self._started, 0.0)

    def timings(self) -> list[StageTiming]:
        """Снимок завершённых стадий в порядке выполнения."""
        return list(self._timings)

    def snapshot(self) -> list[dict[str, object]]:
        """Снимок завершённых стадий в виде JSON-совместимых словарей."""
        return [timing.as_dict() for timing in self._timings]
