"""Объединение сегментов ASR и диаризации по максимальному перекрытию во времени."""

from __future__ import annotations

from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)


class OverlapSegmentMerger:
    """Присваивает каждому сегменту речи говорящего с наибольшим перекрытием.

    Реализует протокол ``SegmentMerger``.
    """

    def merge(
        self,
        transcription_segments: list[TranscriptionSegment],
        speaker_segments: list[SpeakerSegment],
        known_speakers: dict[str, str] | None = None,
    ) -> tuple[list[TranscriptEntry], list[Speaker]]:
        known_speakers = known_speakers or {}
        speakers_by_id: dict[str, Speaker] = {}
        entries: list[TranscriptEntry] = []

        for segment in transcription_segments:
            speaker_id = self._best_matching_speaker(segment, speaker_segments)
            speaker = None
            if speaker_id is not None:
                speaker = speakers_by_id.setdefault(
                    speaker_id,
                    Speaker(id=speaker_id, display_name=known_speakers.get(speaker_id, speaker_id)),
                )
            entries.append(
                TranscriptEntry(
                    start=segment.start, end=segment.end, text=segment.text, speaker=speaker
                )
            )

        return entries, list(speakers_by_id.values())

    @staticmethod
    def _best_matching_speaker(
        segment: TranscriptionSegment, speaker_segments: list[SpeakerSegment]
    ) -> str | None:
        best_id: str | None = None
        best_overlap = 0.0

        for speaker_segment in speaker_segments:
            overlap = min(segment.end, speaker_segment.end) - max(segment.start, speaker_segment.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = speaker_segment.speaker_id

        return best_id
