"""Сериализация доменных моделей в JSON-совместимые структуры для кэша.

Формат намеренно простой и явный (словари примитивов), чтобы файлы кэша было
легко читать глазами, а несовместимость ловилась проверками версии и структуры.
Функции разбора бросают :class:`ValueError`, если данные кэша повреждены —
вызывающий код должен воспринимать это как промах и пересчитать стадию.
"""

from __future__ import annotations

from typing import Any

from audio_transcriber.domain.models import (
    SpeakerOverlap,
    SpeakerSegment,
    TranscriptionSegment,
)


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ожидалось число, получено {value!r}")
    return float(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _as_float(value)


def _as_str(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"ожидалась строка, получено {value!r}")
    return value


def _as_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"ожидался список, получено {value!r}")
    return value


def asr_payload(
    segments: list[TranscriptionSegment], language: str | None, duration: float
) -> dict[str, Any]:
    """Упаковывает результат ASR (сегменты с ``avg_logprob``, язык, длительность)."""
    return {
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
            }
            for segment in segments
        ],
        "language": language,
        "duration": duration,
    }


def asr_from_payload(data: dict[str, Any]) -> tuple[list[TranscriptionSegment], str | None, float]:
    """Распаковывает результат ASR; повреждённые данные — :class:`ValueError`."""
    language_raw = data.get("language")
    language = None if language_raw is None else _as_str(language_raw)
    duration = _as_float(data.get("duration", 0.0))
    segments = [
        TranscriptionSegment(
            start=_as_float(item["start"]),
            end=_as_float(item["end"]),
            text=_as_str(item["text"]),
            avg_logprob=_as_optional_float(item.get("avg_logprob")),
        )
        for item in _as_list(data.get("segments"))
    ]
    return segments, language, duration


def diarization_payload(
    segments: list[SpeakerSegment], overlaps: list[SpeakerOverlap]
) -> dict[str, Any]:
    """Упаковывает результат диаризации (сегменты говорящих и зоны наложения)."""
    return {
        "segments": [
            {"start": segment.start, "end": segment.end, "speaker_id": segment.speaker_id}
            for segment in segments
        ],
        "overlaps": [{"start": overlap.start, "end": overlap.end} for overlap in overlaps],
    }


def diarization_from_payload(
    data: dict[str, Any],
) -> tuple[list[SpeakerSegment], list[SpeakerOverlap]]:
    """Распаковывает результат диаризации; повреждённые данные — :class:`ValueError`."""
    segments = [
        SpeakerSegment(
            start=_as_float(item["start"]),
            end=_as_float(item["end"]),
            speaker_id=_as_str(item["speaker_id"]),
        )
        for item in _as_list(data.get("segments"))
    ]
    overlaps = [
        SpeakerOverlap(start=_as_float(item["start"]), end=_as_float(item["end"]))
        for item in _as_list(data.get("overlaps", []))
    ]
    return segments, overlaps
