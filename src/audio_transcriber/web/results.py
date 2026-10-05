"""Приведение результата конвейера к формату, ожидаемому фронтендом.

Формат намеренно плоский и стабильный (контракт API v1): реплики ссылаются на
говорящих по ``speaker_id`` (основной) и ``extra_speaker_ids`` (дополнительные
участники наложения), а легенда меток приходит полем ``marks``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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


def words_to_dict(entry: TranscriptEntry) -> list[dict[str, object]]:
    """Пословные таймстемпы реплики для API/экспорта (#45).

    Пустой список — стадия выключена или движок не поддерживает; фронтенд
    показывает прежний посегментный вид.
    """
    return [
        {
            "text": word.text,
            "start": word.start,
            "end": word.end,
            "probability": word.probability,
        }
        for word in entry.words
    ]


def entry_to_dict(entry: TranscriptEntry, threshold: float | None) -> dict[str, object]:
    """Одна реплика стенограммы для API.

    Поле ``words`` (#45) добавляется, только если слова есть: при выключенной
    стадии форма реплики остаётся прежней (обратная совместимость контракта).
    """
    payload: dict[str, object] = {
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
        "edited": entry.edited,
        "original_text": entry.original_text,
    }
    words = words_to_dict(entry)
    if words:
        payload["words"] = words
    return payload


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


def apply_transcript_edits(
    payload: Mapping[str, object],
    *,
    edits: Mapping[int, str],
    resets: Sequence[int] = (),
) -> dict[str, object]:
    """Копия payload с ручными правками текста реплик (#26).

    ``edits`` — отображение ``индекс реплики -> новый текст``; ``resets`` —
    индексы реплик, возвращаемых к исходному тексту. Правится только поле
    ``text``: таймкоды и говорящий не меняются. У изменённых реплик
    выставляется ``edited=True`` и запоминается ``original_text`` (первый
    исходный текст); сброс очищает обе пометки. Сброс имеет приоритет над
    правкой для одного и того же индекса.
    """
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("В результате нет реплик")
    reset_set = set(resets)
    updated_entries: list[object] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            updated_entries.append(raw)
            continue
        item = dict(raw)
        if index in reset_set:
            original = item.get("original_text")
            if isinstance(original, str):
                item["text"] = original
            item["edited"] = False
            item["original_text"] = None
        elif index in edits:
            text = edits[index]
            current = item.get("text")
            current_text = current if isinstance(current, str) else ""
            # Правка, совпадающая с текущим текстом, ничего не меняет — не
            # помечаем реплику «изменённой вручную» случайно.
            if text != current_text:
                if not item.get("edited") or not isinstance(item.get("original_text"), str):
                    item["original_text"] = current_text
                item["text"] = text
                item["edited"] = True
        updated_entries.append(item)
    updated = dict(payload)
    updated["entries"] = updated_entries
    return updated


def merge_transcript_edits(
    new_payload: Mapping[str, object], previous_payload: Mapping[str, object] | None
) -> dict[str, object]:
    """Переносит ручные правки текста из прошлого результата в новый.

    Нужен при повторном прогоне задачи (в том числе из ASR-кэша): свежий
    результат содержит исходные тексты, а ручные правки пользователя должны
    сохраниться. Реплики сопоставляются по неизменным таймкодам ``(start,
    end)`` — при правке они не меняются.
    """
    if previous_payload is None:
        return dict(new_payload)
    previous: dict[tuple[float, float], Mapping[str, object]] = {}
    old_entries = previous_payload.get("entries")
    if isinstance(old_entries, list):
        for raw in old_entries:
            if not isinstance(raw, Mapping) or not raw.get("edited"):
                continue
            start = _time_key(raw.get("start"))
            end = _time_key(raw.get("end"))
            if start is not None and end is not None:
                previous[(start, end)] = raw
    if not previous:
        return dict(new_payload)
    entries = new_payload.get("entries")
    if not isinstance(entries, list):
        return dict(new_payload)
    updated_entries: list[object] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            updated_entries.append(raw)
            continue
        start = _time_key(raw.get("start"))
        end = _time_key(raw.get("end"))
        old = previous.get((start, end)) if start is not None and end is not None else None
        item = dict(raw)
        old_text = old.get("text") if old is not None else None
        if old is not None and isinstance(old_text, str):
            item["text"] = old_text
            item["edited"] = True
            item["original_text"] = old.get("original_text")
        updated_entries.append(item)
    updated = dict(new_payload)
    updated["entries"] = updated_entries
    return updated


def _time_key(value: object) -> float | None:
    """Ключ сопоставления реплик: таймкод, округлённый до миллисекунд."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 3)
