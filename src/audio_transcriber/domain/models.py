"""Доменные модели, описывающие данные, которыми обмениваются компоненты
конвейера: распознавание речи -> диаризация -> объединение -> экспорт.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """Сегмент речи, полученный от движка распознавания (ASR).

    Не содержит информации о говорящем — это добавляется на этапе
    объединения с результатами диаризации.
    """

    start: float
    end: float
    text: str
    avg_logprob: float | None = None


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """Временной интервал, отнесённый диаризацией к конкретному говорящему."""

    start: float
    end: float
    speaker_id: str


@dataclass(frozen=True, slots=True)
class Speaker:
    """Говорящий, участвующий в разговоре."""

    id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """Финальная реплика стенограммы: текст, привязанный к говорящему."""

    start: float
    end: float
    text: str
    speaker: Speaker | None = None


@dataclass(slots=True)
class TranscriptionResult:
    """Итоговый результат работы конвейера для одного аудиофайла."""

    source_path: Path
    language: str | None
    duration: float
    entries: list[TranscriptEntry] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)
