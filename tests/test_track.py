"""Тесты дорожки говорящего (#108): интервалы-заглушения и правка waveform.

Аудио не требуется: интервалы строятся по репликам, а ``apply_track`` работает
над синтетическим waveform.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_transcriber.diarization.track import (
    apply_track,
    fingerprint_intervals,
    mute_intervals,
)
from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.utils.audio import SAMPLE_RATE

A = Speaker(id="SPEAKER_00", display_name="Иван")
B = Speaker(id="SPEAKER_01", display_name="Пётр")
C = Speaker(id="SPEAKER_02", display_name="Мария")


def _entry(
    start: float,
    end: float,
    speaker: Speaker | None,
    *,
    extras: list[Speaker] | None = None,
) -> TranscriptEntry:
    return TranscriptEntry(
        start=start,
        end=end,
        text="x",
        speaker=speaker,
        overlap=bool(extras),
        extra_speakers=extras or [],
    )


def test_mute_intervals_only_other_speakers() -> None:
    entries = [
        _entry(0.0, 1.0, A),
        _entry(1.0, 2.0, B),
        _entry(2.0, 3.0, C),
    ]

    assert mute_intervals(entries, "SPEAKER_00") == [(1.0, 3.0)]


def test_mute_intervals_keeps_overlap_participant_by_default() -> None:
    entries = [
        _entry(0.0, 1.0, B, extras=[A]),
        _entry(1.0, 2.0, C),
    ]

    assert mute_intervals(entries, "SPEAKER_00") == [(1.0, 2.0)]
    assert mute_intervals(entries, "SPEAKER_00", include_overlap=False) == [(0.0, 2.0)]


def test_mute_intervals_mutes_entries_without_speaker() -> None:
    entries = [_entry(0.0, 1.0, None), _entry(1.0, 2.0, A)]

    assert mute_intervals(entries, "SPEAKER_00") == [(0.0, 1.0)]


def test_mute_intervals_merges_and_ignores_empty() -> None:
    entries = [
        _entry(0.0, 1.0, B),
        _entry(0.5, 2.0, C),
        _entry(2.0, 2.0, B),
        _entry(3.0, 4.0, B),
    ]

    assert mute_intervals(entries, "SPEAKER_00") == [(0.0, 2.0), (3.0, 4.0)]


def test_mute_intervals_no_mute_when_only_selected() -> None:
    entries = [_entry(0.0, 1.0, A, extras=[B])]

    assert mute_intervals(entries, "SPEAKER_00") == []


def test_apply_track_silences_intervals_and_keeps_rest() -> None:
    waveform = np.ones(4 * SAMPLE_RATE, dtype=np.float32)

    result = apply_track(waveform, intervals=[(1.0, 2.0)], fade_seconds=0.0)

    assert result is not waveform
    assert np.all(result[: SAMPLE_RATE] == 1.0)
    assert np.all(result[2 * SAMPLE_RATE :] == 1.0)
    assert np.all(result[SAMPLE_RATE : 2 * SAMPLE_RATE] == 0.0)


def test_apply_track_fade_smooths_boundaries() -> None:
    waveform = np.ones(3 * SAMPLE_RATE, dtype=np.float32)

    result = apply_track(waveform, intervals=[(1.0, 2.0)], fade_seconds=0.01)
    edge = round(0.01 * SAMPLE_RATE)

    assert result[SAMPLE_RATE] == 1.0
    assert result[SAMPLE_RATE + edge] == 0.0
    assert result[2 * SAMPLE_RATE - edge] == 0.0
    assert 0.0 < result[2 * SAMPLE_RATE - 1] < 1.0
    assert result[2 * SAMPLE_RATE] == 1.0


def test_apply_track_attenuation_partial() -> None:
    waveform = np.ones(2 * SAMPLE_RATE, dtype=np.float32)

    result = apply_track(waveform, intervals=[(0.5, 1.5)], attenuation=0.1, fade_seconds=0.0)

    assert result[SAMPLE_RATE] == pytest.approx(0.1)


def test_apply_track_rejects_invalid_attenuation() -> None:
    with pytest.raises(ValueError):
        apply_track(np.ones(10, dtype=np.float32), intervals=[(0.0, 0.1)], attenuation=1.0)


def test_apply_track_empty_intervals_returns_copy() -> None:
    waveform = np.ones(10, dtype=np.float32)

    result = apply_track(waveform, intervals=[])

    assert result is not waveform
    assert np.array_equal(result, waveform)


def test_apply_track_clamps_out_of_range_intervals() -> None:
    waveform = np.ones(SAMPLE_RATE, dtype=np.float32)

    result = apply_track(waveform, intervals=[(-1.0, 0.5), (0.5, 99.0)], fade_seconds=0.0)

    assert np.all(result == 0.0)


def test_fingerprint_intervals_stable_and_distinct() -> None:
    assert fingerprint_intervals([(1.0, 2.0)]) == fingerprint_intervals([(1.0, 2.0)])
    assert fingerprint_intervals([(1.0, 2.0)]) != fingerprint_intervals([(1.0, 2.5)])
    assert fingerprint_intervals([]) == fingerprint_intervals([])
