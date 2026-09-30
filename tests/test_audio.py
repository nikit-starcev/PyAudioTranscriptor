"""Тесты декодирования аудио через PyAV (``utils.audio``)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    load_waveform,
    probe_audio,
    resample_waveform,
    write_wav,
)
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


def test_write_wav_load_waveform_round_trip_is_lossless(tmp_path: Path) -> None:
    """s16 → float32 → s16 без потерь: масштаб 32768 и округление к ближайшему."""

    rng = np.random.default_rng(1234)
    raw = rng.integers(-32768, 32768, size=SAMPLE_RATE, dtype=np.int16)
    waveform = raw.astype(np.float32) / 32768.0

    path = tmp_path / "round-trip.wav"
    write_wav(path, waveform)
    loaded = load_waveform(path)

    recovered = np.clip(np.rint(loaded * 32768.0), -32768, 32767).astype(np.int16)
    assert np.array_equal(recovered, raw)


def test_probe_audio_reads_wav_header_without_decoding(tmp_path: Path) -> None:
    path = tmp_path / "probe.wav"
    frames = 8000
    samples = np.zeros(frames, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())

    probe = probe_audio(path)

    assert probe.format_name == "wav"
    assert probe.codec == "pcm_s16le"
    assert probe.sample_rate == SAMPLE_RATE
    assert probe.channels == 1
    assert probe.duration_seconds == pytest.approx(frames / SAMPLE_RATE)


def test_probe_audio_reports_stereo_44k(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    samples = np.zeros(44100, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.repeat(samples[:, None], 2, axis=1).tobytes())

    probe = probe_audio(path)

    assert probe.sample_rate == 44100
    assert probe.channels == 2


def test_probe_audio_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioFileError):
        probe_audio(tmp_path / "does-not-exist.wav")

