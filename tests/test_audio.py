"""Тесты декодирования аудио через PyAV (``utils.audio``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
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
