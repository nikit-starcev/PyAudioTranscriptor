"""Интерфейс очистки стенограммы от неречевых артефактов.

Whisper (faster-whisper и whisper.cpp) иногда вставляет в текст пометки,
которых нет в речи: ``[АПЛОДИСМЕНТЫ]``, ``(смех)``, ``[BLANK_AUDIO]``,
музыкальные символы ``♪`` и т.п. Компонент очистки удаляет такие пометки,
не затрагивая обычный текст.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from audio_transcriber.domain.models import TranscriptEntry


@runtime_checkable
class ArtifactCleanerProtocol(Protocol):
    """Контракт компонента, удаляющего неречевые артефакты из реплик."""

    def clean(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Возвращает реплики без неречевых пометок.

        Реплики, состоящие только из артефактов, из результата исключаются.
        """
        ...


@runtime_checkable
class RepetitionCleanerProtocol(Protocol):
    """Контракт компонента, схлопывающего подряд повторяющиеся реплики."""

    def clean(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Схлопывает идущие подряд одинаковые/почти одинаковые реплики.

        Из серии повторов остаётся первая реплика (её ``end`` расширяется до
        конца последней); осмысленная короткая повторяющаяся речь сохраняется.
        """
        ...


@runtime_checkable
class TextNormalizerProtocol(Protocol):
    """Контракт компонента безопасной нормализации текста реплик."""

    def normalize(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Возвращает реплики с нормализованными пробелами и пунктуацией.

        Слова не переписываются; текст без отклонений от нормы не меняется.
        """
        ...
