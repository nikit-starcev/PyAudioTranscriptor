"""Интерфейс компонента диаризации.

Любая конкретная реализация (pyannote.audio и т.д.) должна соответствовать
протоколу :class:`SpeakerDiarizer`, чтобы движок диаризации можно было
заменить без изменения остального конвейера.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_transcriber.domain.models import SpeakerSegment


@runtime_checkable
class SpeakerDiarizer(Protocol):
    """Контракт компонента определения говорящих."""

    def diarize(
        self, audio_path: Path, *, num_speakers: int | None = None
    ) -> list[SpeakerSegment]:
        """Определяет говорящих в аудиофайле.

        :param audio_path: путь к аудиофайлу.
        :param num_speakers: точное количество говорящих, если известно;
            ``None`` — определить автоматически.
        :return: список временных интервалов, отнесённых к говорящим.
        """
        ...
