"""Приведение результата конвейера к формату, ожидаемому фронтендом.

Формат намеренно плоский и стабильный (контракт API v1): реплики ссылаются на
говорящих по ``speaker_id``, а легенда меток приходит полем ``marks``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from audio_transcriber.domain.models import TranscriptEntry, TranscriptionResult

#: Легенда меток реплик: ключ → символ и человекочитаемое описание.
MARK_LOW_CONFIDENCE: dict[str, str] = {
    "key": "low_confidence",
    "symbol": "⚠",
    "label": "низкая уверенность",
}
MARK_OVERLAP: dict[str, str] = {
    "key": "overlap",
    "symbol": "⇄",
    "label": "наложение речи",
}
MARKS: list[dict[str, str]] = [MARK_LOW_CONFIDENCE, MARK_OVERLAP]


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
        "text": entry.text,
        "low_confidence": _is_low_confidence(entry, threshold),
        "overlap": entry.overlap,
    }


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
            for speaker in result.speakers
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
