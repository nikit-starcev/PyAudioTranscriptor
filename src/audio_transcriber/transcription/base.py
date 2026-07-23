"""Интерфейс движка распознавания речи.

Любая конкретная реализация (faster-whisper и т.д.) должна соответствовать
протоколу :class:`SpeechRecognizer`, чтобы движок можно было заменить без
изменения остального конвейера.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_transcriber.domain.models import TranscriptionSegment


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Контракт компонента распознавания речи."""

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        """Распознаёт речь в аудиофайле.

        :param audio_path: путь к аудиофайлу.
        :param language: код языка речи; ``None`` — автоопределение.
        :return: сегменты речи, определённый/заданный код языка и
            длительность аудио в секундах.
        """
        ...
