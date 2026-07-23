"""Тесты объединения сегментов ASR и диаризации (``OverlapSegmentMerger``)."""

from __future__ import annotations

from audio_transcriber.domain.models import SpeakerSegment, TranscriptionSegment
from audio_transcriber.merging.aligner import OverlapSegmentMerger


def test_assigns_speaker_with_largest_overlap() -> None:
    segments = [TranscriptionSegment(start=0.0, end=3.0, text="привет")]
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=3.0, speaker_id="SPEAKER_01"),
    ]

    entries, speakers = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].speaker is not None
    assert entries[0].speaker.id == "SPEAKER_01"
    assert [speaker.id for speaker in speakers] == ["SPEAKER_01"]


def test_segment_without_overlap_has_no_speaker() -> None:
    segments = [TranscriptionSegment(start=10.0, end=11.0, text="тишина")]
    speaker_segments = [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    entries, speakers = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].speaker is None
    assert speakers == []


def test_known_speaker_names_are_applied() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="привет")]
    speaker_segments = [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    entries, speakers = OverlapSegmentMerger().merge(
        segments, speaker_segments, known_speakers={"SPEAKER_00": "Иван"}
    )

    assert entries[0].speaker.display_name == "Иван"
    assert speakers[0].display_name == "Иван"


def test_same_speaker_reused_across_segments() -> None:
    segments = [
        TranscriptionSegment(start=0.0, end=1.0, text="привет"),
        TranscriptionSegment(start=1.0, end=2.0, text="как дела"),
    ]
    speaker_segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    entries, speakers = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].speaker is entries[1].speaker
    assert len(speakers) == 1
