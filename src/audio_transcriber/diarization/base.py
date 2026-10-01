"""Интерфейс компонента диаризации.

Любая конкретная реализация (pyannote.audio и т.д.) должна соответствовать
протоколу :class:`SpeakerDiarizer`, чтобы движок диаризации можно было
заменить без изменения остального конвейера.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment


@runtime_checkable
class SpeakerDiarizer(Protocol):
    """Контракт компонента определения говорящих."""

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: np.ndarray | None = None,
    ) -> list[SpeakerSegment]:
        """Определяет говорящих в аудиофайле.

        :param audio_path: путь к аудиофайлу.
        :param num_speakers: точное количество говорящих, если известно;
            ``None`` — определить автоматически. Имеет приоритет над
            ``min_speakers``/``max_speakers``.
        :param min_speakers: нижняя граница числа говорящих (включительно);
            ``None`` — без ограничения.
        :param max_speakers: верхняя граница числа говорящих (включительно);
            ``None`` — без ограничения.
        :param waveform: уже декодированный моно waveform 16 кГц float32.
            Позволяет переиспользовать результат предыдущей стадии (например,
            шумоподавления) и не декодировать файл повторно. ``None`` —
            декодировать ``audio_path``.
        :return: список временных интервалов, отнесённых к говорящим.
        """
        ...

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """Интервалы наложения речи из последнего вызова ``diarize``.

        Возвращает зоны, где одновременно говорили два и более человек. Если
        движок не поддерживает определение перекрытий, вернётся пустой список
        — потребитель не должен падать.
        """
        ...
