"""Тесты декодирования аудио через PyAV (``utils.audio``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform, resample_waveform
from audio_transcriber.utils.exceptions import AudioFileError


def test_load_waveform_decodes_real_audio(jfk_audio_file: Path) -> None:
    waveform = load_waveform(jfk_audio_file)

    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    duration = waveform.shape[0] / SAMPLE_RATE
    assert 10 < duration < 12


def test_load_waveform_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioFileError):
        load_waveform(tmp_path / "does-not-exist.wav")


def test_load_waveform_raises_on_non_audio_file(tmp_path: Path) -> None:
    path = tmp_path / "not-audio.mp3"
    path.write_bytes(b"this is not a valid audio file")

    with pytest.raises(AudioFileError):
        load_waveform(path)


def test_resample_waveform_changes_length_and_keeps_dtype() -> None:
    tone = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)

    resampled = resample_waveform(tone, source_rate=SAMPLE_RATE, target_rate=SAMPLE_RATE * 2)

    assert resampled.dtype == np.float32
    assert abs(resampled.shape[0] - tone.shape[0] * 2) <= 2


def test_resample_waveform_same_rate_returns_input() -> None:
    waveform = np.zeros(100, dtype=np.float32)

    assert resample_waveform(waveform, source_rate=SAMPLE_RATE, target_rate=SAMPLE_RATE) is waveform


def test_resample_waveform_rejects_non_positive_rate() -> None:
    waveform = np.zeros(100, dtype=np.float32)

    with pytest.raises(AudioFileError):
        resample_waveform(waveform, source_rate=0, target_rate=SAMPLE_RATE)

