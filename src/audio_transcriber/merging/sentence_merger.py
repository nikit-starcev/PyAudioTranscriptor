"""Склейка подряд идущих реплик одного говорящего в предложения.

Движок распознавания (особенно whisper.cpp) выдаёт много коротких сегментов
по несколько слов, и после диаризации каждый такой сегмент становится
отдельной строкой стенограммы. Этот компонент объединяет соседние реплики
одного и того же говорящего в одну, чтобы текст читался как предложения, а
корректор и LLM работали с уже склеенным текстом.
"""

from __future__ import annotations

from dataclasses import replace

from audio_transcriber.domain.models import TranscriptEntry

# Разумная по умолчанию пауза, разрывающая реплику: если между сегментами
# одного говорящего прошло больше секунд — начинаем новую реплику.
DEFAULT_MAX_GAP = 2.0


def _normalize_text(text: str) -> str:
    """Схлопывает любые пробельные последовательности и обрезает края."""
    return " ".join(text.split())


def _join_text(left: str, right: str) -> str:
    """Склеивает два фрагмента через один пробел без дублей пробелов."""
    return _normalize_text(f"{left} {right}")


def _min_logprob(left: float | None, right: float | None) -> float | None:
    """Наихудшая (минимальная) уверенность из двух реплик.

    ``None`` (значение отсутствует) игнорируется; если оба ``None`` — ``None``.
    """
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


class SentenceMerger:
    """Объединяет соседние реплики одного говорящего в одну.

    Склейка выполняется, когда у двух соседних реплик один и тот же говорящий
    (в т.ч. когда обе реплики **без говорящего**, ``speaker is None`` —
    например, при ``--no-diarization``) и пауза между ними не превышает
    ``max_gap``. Реплики без говорящего не присоединяются к именованным и
    наоборот.

    При склейке ``start`` берётся у первой реплики, ``end`` — у последней,
    тексты соединяются через пробел. Уверенность распознавания объединённой
    реплики — наихудшая (минимум ``avg_logprob``, ``None`` игнорируется),
    признак наложения речи — логическое ИЛИ по склеенным репликам.
    """

    def __init__(self, *, max_gap: float = DEFAULT_MAX_GAP) -> None:
        if max_gap < 0:
            raise ValueError("max_gap не может быть отрицательным")
        self._max_gap = max_gap

    def merge(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Возвращает новый список реплик со склеенными соседними репликами."""
        merged: list[TranscriptEntry] = []
        for entry in entries:
            if merged and self._can_merge(merged[-1], entry):
                previous = merged[-1]
                merged[-1] = replace(
                    previous,
                    end=entry.end,
                    text=_join_text(previous.text, entry.text),
                    avg_logprob=_min_logprob(previous.avg_logprob, entry.avg_logprob),
                    overlap=previous.overlap or entry.overlap,
                )
            else:
                merged.append(entry)
        return merged

    def _can_merge(self, left: TranscriptEntry, right: TranscriptEntry) -> bool:
        if left.speaker is None and right.speaker is None:
            # обе реплики без говорящего — склеиваем в один блок
            return (right.start - left.end) <= self._max_gap
        if left.speaker is None or right.speaker is None:
            # не присоединяем реплику без говорящего к именованной и наоборот
            return False
        if left.speaker.id != right.speaker.id:
            return False
        return (right.start - left.end) <= self._max_gap
