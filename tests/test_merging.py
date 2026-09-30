"""Тесты объединения сегментов ASR и диаризации (``OverlapSegmentMerger``)."""

from __future__ import annotations

import random
import time

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

    # Порядок и таймкоды/текст реплик не меняются — меняется только говорящий.
    assert len(entries) == len(transcription_segments)
    for segment, entry in zip(transcription_segments, entries, strict=True):
        assert (entry.start, entry.end, entry.text) == (segment.start, segment.end, segment.text)
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
    # Реплика накрывает смену говорящего: оба участника должны быть указаны.
    assert [speaker.id for speaker in entries[0].extra_speakers] == ["SPEAKER_00"]
    assert entries[0].overlap is True
    assert [speaker.id for speaker in speakers] == ["SPEAKER_01", "SPEAKER_00"]


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


# --- быстрый путь для произвольных перекрытий (совпадение с перебором) -------


def test_matches_brute_force_on_dense_multiple_overlaps() -> None:
    # Много пересекающихся интервалов (в т.ч. тройные наложения) — старый
    # перебор давал тот же результат, новый быстрый путь обязан совпасть.
    rng = random.Random(2024)
    speaker_segments = []
    for index in range(120):
        start = rng.uniform(0.0, 20.0)
        speaker_segments.append(
            SpeakerSegment(
                start=start,
                end=start + rng.uniform(0.5, 12.0),
                speaker_id=f"SPEAKER_{index % 5:02d}",
            )
        )

    transcription_segments = []
    for i in range(200):
        start = rng.uniform(0.0, 25.0)
        transcription_segments.append(
            TranscriptionSegment(start=start, end=start + rng.uniform(0.1, 10.0), text=str(i))
        )

    _assert_equivalent(transcription_segments, speaker_segments)


def test_matches_brute_force_on_degenerate_intervals() -> None:
    rng = random.Random(5)
    speaker_segments = []
    for index in range(40):
        if rng.random() < 0.3:
            point = rng.uniform(0.0, 10.0)
            start = end = point
        else:
            start = rng.uniform(0.0, 10.0)
            end = start + rng.uniform(0.0, 4.0)
        speaker_segments.append(
            SpeakerSegment(start=start, end=end, speaker_id=f"SPEAKER_{index % 3:02d}")
        )

    transcription_segments = []
    for i in range(60):
        start = rng.uniform(0.0, 12.0)
        transcription_segments.append(
            TranscriptionSegment(start=start, end=start + rng.uniform(0.0, 5.0), text=str(i))
        )

    _assert_equivalent(transcription_segments, speaker_segments)


def test_matches_brute_force_on_exact_overlap_ties() -> None:
    # Целочисленные координаты дают точные тай-брейки по перекрытию.
    rng = random.Random(77)
    speaker_segments = [
        SpeakerSegment(
            start=float(start),
            end=float(start + rng.randint(0, 6)),
            speaker_id=f"SPEAKER_{index % 4:02d}",
        )
        for index, start in enumerate(rng.randint(0, 12) for _ in range(50))
    ]

    transcription_segments = [
        TranscriptionSegment(
            start=float(start),
            end=float(start + rng.randint(0, 6)),
            text=str(i),
        )
        for i, start in enumerate(rng.randint(0, 12) for _ in range(80))
    ]

    _assert_equivalent(transcription_segments, speaker_segments)


def test_large_overlapping_input_is_fast() -> None:
    # 5000×5000 тяжело пересекающихся сегментов: прежний O(N×M) перебор
    # выполнялся бы десятки секунд, быстрый путь укладывается в доли секунды.
    rng = random.Random(123)
    speaker_segments = []
    for index in range(5000):
        start = rng.uniform(0.0, 1000.0)
        speaker_segments.append(
            SpeakerSegment(
                start=start,
                end=start + rng.uniform(0.5, 8.0),
                speaker_id=f"SPEAKER_{index % 50:02d}",
            )
        )

    transcription_segments = []
    for i in range(5000):
        start = rng.uniform(0.0, 1000.0)
        transcription_segments.append(
            TranscriptionSegment(start=start, end=start + rng.uniform(0.1, 7.0), text=str(i))
        )

    started = time.perf_counter()
    entries, _ = OverlapSegmentMerger().merge(transcription_segments, speaker_segments)
    elapsed = time.perf_counter() - started

    assert len(entries) == 5000
    assert elapsed < 2.0, f"слишком медленно: {elapsed:.3f}s"


# --- дополнительные говорящие и уверенность привязки -------------------------


def test_extra_speakers_for_two_overlapping_speakers() -> None:
    segments = [TranscriptionSegment(start=0.0, end=10.0, text="спор")]
    speaker_segments = [
        SpeakerSegment(start=0.0, end=8.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=10.0, speaker_id="SPEAKER_01"),
    ]

    entries, _ = OverlapSegmentMerger().merge(segments, speaker_segments)

    entry = entries[0]
    assert entry.speaker is not None
    assert entry.speaker.id == "SPEAKER_00"  # максимальное перекрытие, тай-брейк по порядку
    assert [speaker.id for speaker in entry.extra_speakers] == ["SPEAKER_01"]
    assert entry.overlap is True
    assert entry.speaker_confidence == 0.8


def test_extra_speakers_ordered_by_descending_overlap() -> None:
    segments = [TranscriptionSegment(start=0.0, end=10.0, text="общий разговор")]
    speaker_segments = [
        SpeakerSegment(start=0.0, end=10.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=6.0, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=4.0, end=10.0, speaker_id="SPEAKER_02"),
    ]

    entries, _ = OverlapSegmentMerger().merge(segments, speaker_segments)

    entry = entries[0]
    assert entry.speaker is not None
    assert entry.speaker.id == "SPEAKER_00"
    # SPEAKER_02 перекрывает 6 с, SPEAKER_01 — 5 с.
    assert [speaker.id for speaker in entry.extra_speakers] == ["SPEAKER_02", "SPEAKER_01"]
    assert entry.speaker_confidence == 1.0


def test_extra_speaker_below_seconds_threshold_ignored() -> None:
    segments = [TranscriptionSegment(start=0.0, end=10.0, text="короткое касание")]
    speaker_segments = [
        SpeakerSegment(start=0.0, end=10.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=0.0, end=0.3, speaker_id="SPEAKER_01"),
    ]

    entries, _ = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].extra_speakers == []
    assert entries[0].overlap is False


def test_extra_speaker_below_fraction_threshold_ignored() -> None:
    segments = [TranscriptionSegment(start=0.0, end=100.0, text="длинная реплика")]
    speaker_segments = [
        SpeakerSegment(start=0.0, end=100.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=0.0, end=4.0, speaker_id="SPEAKER_01"),  # 4 % от 100 с
    ]

    entries, _ = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].extra_speakers == []
    assert entries[0].overlap is False


def test_speaker_confidence_none_without_diarization() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="текст")]

    entries, _ = OverlapSegmentMerger().merge(segments, [])

    assert entries[0].extra_speakers == []
    assert entries[0].speaker_confidence is None


def test_speaker_confidence_partial_coverage() -> None:
    segments = [TranscriptionSegment(start=0.0, end=10.0, text="частично")]
    speaker_segments = [SpeakerSegment(start=0.0, end=6.0, speaker_id="SPEAKER_00")]

    entries, _ = OverlapSegmentMerger().merge(segments, speaker_segments)

    assert entries[0].speaker is not None
    assert entries[0].extra_speakers == []
    assert entries[0].speaker_confidence == 0.6


def test_speaker_confidence_zero_for_nearest_without_overlap() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="рядом")]
    speaker_segments = [SpeakerSegment(start=5.0, end=6.0, speaker_id="SPEAKER_00")]

    entries, _ = OverlapSegmentMerger(max_gap=5.0).merge(segments, speaker_segments)

    assert entries[0].speaker is not None
    assert entries[0].speaker_confidence == 0.0


def test_mark_overlap_disabled_skips_extra_speakers() -> None:
    segments = [TranscriptionSegment(start=0.0, end=10.0, text="спор")]
    speaker_segments = [
        SpeakerSegment(start=0.0, end=8.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=10.0, speaker_id="SPEAKER_01"),
    ]

    entries, _ = OverlapSegmentMerger(mark_overlap=False).merge(segments, speaker_segments)

    assert entries[0].extra_speakers == []
    assert entries[0].overlap is False
    # Уверенность привязки считается независимо от режима пометки.
    assert entries[0].speaker_confidence == 0.8
