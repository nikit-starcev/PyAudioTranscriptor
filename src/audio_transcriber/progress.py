"""События прогресса для живого отображения хода транскрибации.

Используются интерфейсом TUI: компоненты конвейера вызывают переданный
колбэк с событиями, а интерфейс отображает их без блокировки.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Одно событие прогресса конвейера.

    ``stage`` — этап: ``asr``, ``diarization``, ``merge``, ``clean``,
    ``correction``, ``llm``, ``export`` или ``done``. ``fraction`` — прогресс
    от 0 до 1, либо ``None`` для неопределённого (анимированного) прогресса.
    """

    stage: str
    message: str = ""
    fraction: float | None = None
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]
