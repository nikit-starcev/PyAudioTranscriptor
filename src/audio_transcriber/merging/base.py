"""Интерфейс компонента объединения сегментов ASR и диаризации."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)


@runtime_checkable
class SegmentMerger(Protocol):
    """Контракт компонента, объединяющего сегменты речи и говорящих."""

    def merge(
        self,
        transcription_segments: list[TranscriptionSegment],
        speaker_segments: list[SpeakerSegment],
        known_speakers: dict[str, str] | None = None,
    ) -> tuple[list[TranscriptEntry], list[Speaker]]:
        """Сопоставляет сегменты речи с говорящими.

        :param transcription_segments: сегменты, полученные от ASR.
        :param speaker_segments: сегменты, полученные от диаризации.
        :param known_speakers: пользовательские имена говорящих
            (идентификатор -> отображаемое имя).
        :return: кортеж из готовых реплик стенограммы и списка говорящих.
        """
        ...
