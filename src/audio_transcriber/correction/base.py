"""Интерфейс постобработки текста стенограммы."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from audio_transcriber.domain.models import TranscriptEntry


@runtime_checkable
class TextCorrector(Protocol):
    """Контракт компонента, исправляющего слова в репликах по словарю."""

    def correct(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Возвращает копии реплик с исправленным текстом."""
        ...
