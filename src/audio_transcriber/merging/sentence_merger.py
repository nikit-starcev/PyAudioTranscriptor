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


class SentenceMerger:
    """Объединяет соседние реплики одного говорящего в одну.

    Склейка выполняется, только когда у двух соседних реплик один и тот же
    **непустой** говорящий и пауза между ними не превышает ``max_gap``.
    Реплики без говорящего (``speaker is None``) не склеиваются между собой и
    не присоединяются к именованным — они остаются отдельными строками.

    При склейке ``start`` берётся у первой реплики, ``end`` — у последней,
    тексты соединяются через пробел.
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
                )
            else:
                merged.append(entry)
        return merged

    def _can_merge(self, left: TranscriptEntry, right: TranscriptEntry) -> bool:
        if left.speaker is None or right.speaker is None:
            return False
        if left.speaker.id != right.speaker.id:
            return False
        return (right.start - left.end) <= self._max_gap
