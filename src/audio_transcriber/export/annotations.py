"""Общие пометки реплик для экспортёров (уверенность ASR, наложение речи).

Экспортёры txt/docx/json используют одни и те же правила, чтобы пометки в
разных форматах совпадали. Пометки добавляются только тогда, когда есть на то
основание: реплики с неизвестной уверенностью и без наложения остаются ровно
такими же, как раньше, — формат без пометок не меняется.
"""

from __future__ import annotations

from audio_transcriber.domain.models import TranscriptEntry

# Суффиксы-пометки в конце текстовой строки (txt/docx).
LOW_CONFIDENCE_SUFFIX = " ⚠ [низкая уверенность]"
OVERLAP_SUFFIX = " [наложение речи]"


def is_low_confidence(entry: TranscriptEntry, threshold: float | None) -> bool:
    """Ниже ли уверенность реплики заданного порога.

    При ``threshold is None`` или неизвестной уверенности (``avg_logprob is
    None``) реплика низкоуверенной не считается.
    """
    return threshold is not None and entry.avg_logprob is not None and entry.avg_logprob < threshold


def entry_markers(entry: TranscriptEntry, threshold: float | None) -> str:
    """Возвращает суффикс с пометками реплики (может быть пустым)."""
    parts: list[str] = []
    if is_low_confidence(entry, threshold):
        parts.append(LOW_CONFIDENCE_SUFFIX)
    if entry.overlap:
        parts.append(OVERLAP_SUFFIX)
    return "".join(parts)
