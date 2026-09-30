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
SPEAKER_LOW_CONFIDENCE_SUFFIX = " ? [говорящий под вопросом]"

# Порог уверенности привязки говорящего: реплики, у которых
# ``speaker_confidence`` ниже этого значения, помечаются как сомнительные.
# Диаризация/слияние проставляют ``speaker_confidence`` как долю интервала
# реплики, покрытую основным говорящим; ниже 0.5 — привязка ненадёжна.
DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD = 0.5


def is_low_confidence(entry: TranscriptEntry, threshold: float | None) -> bool:
    """Ниже ли уверенность распознавания реплики заданного порога.

    При ``threshold is None`` или неизвестной уверенности (``avg_logprob is
    None``) реплика низкоуверенной не считается.
    """
    return threshold is not None and entry.avg_logprob is not None and entry.avg_logprob < threshold


def is_low_speaker_confidence(
    entry: TranscriptEntry, threshold: float | None = DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD
) -> bool:
    """Ниже ли уверенность привязки говорящего заданного порога.

    При ``threshold is None`` или отсутствии оценки (``speaker_confidence is
    None``, т.е. диаризации не было) реплика сомнительной не считается.
    """
    return (
        threshold is not None
        and entry.speaker_confidence is not None
        and entry.speaker_confidence < threshold
    )


def entry_markers(
    entry: TranscriptEntry,
    threshold: float | None,
    speaker_threshold: float | None = DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD,
) -> str:
    """Возвращает суффикс с пометками реплики (может быть пустым).

    Порядок пометок: низкая уверенность распознавания, сомнительная привязка
    говорящего, наложение речи.
    """
    parts: list[str] = []
    if is_low_confidence(entry, threshold):
        parts.append(LOW_CONFIDENCE_SUFFIX)
    if is_low_speaker_confidence(entry, speaker_threshold):
        parts.append(SPEAKER_LOW_CONFIDENCE_SUFFIX)
    if entry.overlap:
        parts.append(OVERLAP_SUFFIX)
    return "".join(parts)
