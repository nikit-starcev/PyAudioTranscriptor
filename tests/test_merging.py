"""Тесты объединения сегментов ASR и диаризации (``OverlapSegmentMerger``)."""

from __future__ import annotations

import random

from audio_transcriber.domain.models import SpeakerSegment, TranscriptionSegment
from audio_transcriber.merging.aligner import OverlapSegmentMerger


def _reference_speaker_id(
    segment: TranscriptionSegment, speaker_segments: list[SpeakerSegment], max_gap: float
) -> str | None:
    """Эталон: прежний перебор «в лоб» (максимальное перекрытие, затем ближайший)."""
    best_id: str | None = None
    best_overlap = 0.0
    for speaker_segment in speaker_segments:
        overlap = min(segment.end, speaker_segment.end) - max(
            segment.start, speaker_segment.start
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = speaker_segment.speaker_id
    if best_id is not None:
        return best_id

    nearest_id: str | None = None
    nearest_gap: float | None = None
    for speaker_segment in speaker_segments:
        if segment.end < speaker_segment.start:
            gap = speaker_segment.start - segment.end
        elif segment.start > speaker_segment.end:
            gap = segment.start - speaker_segment.end
        else:
            gap = 0.0
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest_id = speaker_segment.speaker_id
    if nearest_id is not None and nearest_gap is not None and nearest_gap <= max_gap:
        return nearest_id
    return None


def _assert_equivalent(
    transcription_segments: list[TranscriptionSegment],
    speaker_segments: list[SpeakerSegment],
    *,
    max_gap: float = 5.0,
) -> None:
    merger = OverlapSegmentMerger(max_gap=max_gap)
    entries, _ = merger.merge(transcription_segments, speaker_segments)

    for segment, entry in zip(transcription_segments, entries, strict=True):
        expected = _reference_speaker_id(segment, speaker_segments, max_gap)
        actual = entry.speaker.id if entry.speaker else None
        assert actual == expected, (segment, expected, actual)


def test_matches_brute_force_on_non_overlapping_segments() -> None:
    rng = random.Random(1234)
    speaker_segments: list[SpeakerSegment] = []
    cursor = 0.0
    for index in range(40):
        start = cursor + rng.uniform(0.0, 3.0)
        end = start + rng.uniform(0.5, 4.0)
        speaker_segments.append(
            SpeakerSegment(start=start, end=end, speaker_id=f"SPEAKER_{index % 3:02d}")
        )
        cursor = end

    transcription_segments = []
    for i in range(60):
        start = rng.uniform(-5.0, cursor + 5.0)
        transcription_segments.append(
            TranscriptionSegment(start=start, end=start + rng.uniform(0.1, 6.0), text=str(i))
        )

    _assert_equivalent(transcription_segments, speaker_segments)


def test_matches_brute_force_on_overlapping_segments() -> None:
    rng = random.Random(99)
    speaker_segments = []
    for index in range(25):
        start = rng.uniform(0.0, 30.0)
        speaker_segments.append(
            SpeakerSegment(
                start=start,
                end=start + rng.uniform(0.2, 8.0),
                speaker_id=f"SPEAKER_{index % 4:02d}",
            )
        )

    transcription_segments = []
    for i in range(50):
        start = rng.uniform(0.0, 35.0)
        transcription_segments.append(
            TranscriptionSegment(start=start, end=start + rng.uniform(0.1, 7.0), text=str(i))
        )

    _assert_equivalent(transcription_segments, speaker_segments)


def test_matches_brute_force_on_empty_speakers() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="текст")]

    _assert_equivalent(segments, [])


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


def test_nearby_speaker_assigned_when_no_overlap() -> None:
    # реплика сразу после интервала говорящего (в пределах зазора 5 с)
    segments = [TranscriptionSegment(start=3.0, end=4.0, text="короткая реплика")]
    speaker_segments = [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    entries, _ = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].speaker is not None
    assert entries[0].speaker.id == "SPEAKER_00"


def test_far_speaker_not_assigned_beyond_gap() -> None:
    segments = [TranscriptionSegment(start=20.0, end=21.0, text="далеко")]
    speaker_segments = [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    entries, _ = OverlapSegmentMerger(max_gap=5.0).merge(segments, speaker_segments)

    assert entries[0].speaker is None


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
