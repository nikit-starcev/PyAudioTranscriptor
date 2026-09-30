"""Приведение результата конвейера к формату, ожидаемому фронтендом.

Формат намеренно плоский и стабильный (контракт API v1): реплики ссылаются на
говорящих по ``speaker_id`` (основной) и ``extra_speaker_ids`` (дополнительные
участники наложения), а легенда меток приходит полем ``marks``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.export.annotations import (
    DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD,
    is_low_speaker_confidence,
)

#: Легенда меток реплик: ключ → символ и человекочитаемое описание.
MARK_LOW_CONFIDENCE: dict[str, str] = {
    "key": "low_confidence",
    "symbol": "⚠",
    "label": "низкая уверенность",
}
MARK_SPEAKER_UNCERTAIN: dict[str, str] = {
    "key": "speaker_uncertain",
    "symbol": "?",
    "label": "говорящий под вопросом",
}
MARK_OVERLAP: dict[str, str] = {
    "key": "overlap",
    "symbol": "⇄",
    "label": "наложение речи",
}
MARKS: list[dict[str, str]] = [MARK_LOW_CONFIDENCE, MARK_SPEAKER_UNCERTAIN, MARK_OVERLAP]


def _is_low_confidence(entry: TranscriptEntry, threshold: float | None) -> bool:
    """Реплика помечается, если порог задан и логвероятность ниже него."""
    if threshold is None or entry.avg_logprob is None:
        return False
    return entry.avg_logprob < threshold


def entry_to_dict(entry: TranscriptEntry, threshold: float | None) -> dict[str, object]:
    """Одна реплика стенограммы для API."""
    return {
        "start": entry.start,
        "end": entry.end,
        "speaker_id": entry.speaker.id if entry.speaker is not None else None,
        "extra_speaker_ids": [speaker.id for speaker in entry.extra_speakers],
        "speaker_confidence": entry.speaker_confidence,
        "low_speaker_confidence": is_low_speaker_confidence(
            entry, DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD
        ),
        "text": entry.text,
        "low_confidence": _is_low_confidence(entry, threshold),
        "overlap": entry.overlap,
    }


def _result_speakers(result: TranscriptionResult) -> list[Speaker]:
    """Все говорящие результата, включая упомянутых как дополнительные.

    Гарантирует, что ``extra_speaker_ids`` реплик разрешаются фронтендом по
    списку ``speakers``, даже если дополнительный участник почему-то не попал в
    ``result.speakers``. Порядок: основные говорящие, затем недостающие доп.
    """
    speakers: dict[str, Speaker] = {}
    for speaker in result.speakers:
        speakers.setdefault(speaker.id, speaker)
    for entry in result.entries:
        for extra in entry.extra_speakers:
            speakers.setdefault(extra.id, extra)
    return list(speakers.values())


def serialize_result(
    result: TranscriptionResult,
    *,
    samples: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Полный JSON стенограммы для ``GET /api/jobs/{id}/result``.

    ``samples`` — отображение ``speaker_id -> относительный путь`` к образцу
    голоса; используется для флага ``has_sample`` у говорящего.
    """
    samples = samples or {}
    threshold = result.low_confidence_threshold
    return {
        "language": result.language,
        "duration": result.duration,
        "speakers": [
            {
                "id": speaker.id,
                "display_name": speaker.display_name,
                "has_sample": speaker.id in samples,
            }
            for speaker in _result_speakers(result)
        ],
        "entries": [entry_to_dict(entry, threshold) for entry in result.entries],
        "marks": [dict(mark) for mark in MARKS],
        "summary": result.summary,
    }


def load_result_file(path: Path) -> dict[str, object] | None:
    """Читает сохранённый JSON-результат или ``None`` при ошибке/отсутствии."""
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def result_summary(payload: Mapping[str, object]) -> dict[str, object]:
    """Короткая сводка результата для ``GET /api/jobs/{id}``."""
    entries = payload.get("entries")
    speakers = payload.get("speakers")
    samples = payload.get("samples")
    return {
        "language": payload.get("language"),
        "duration": payload.get("duration"),
        "entries": len(entries) if isinstance(entries, list) else 0,
        "speakers": len(speakers) if isinstance(speakers, list) else 0,
        "samples": len(samples) if isinstance(samples, Mapping) else 0,
    }
