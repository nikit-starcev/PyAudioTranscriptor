"""Тесты подготовки эталона голоса (#29): VAD-обрезка, длина, RMS, флаги."""

from __future__ import annotations

import numpy as np
import pytest

from audio_transcriber.diarization.reference import (
    DEFAULT_LOW_ENERGY_DBFS,
    ReferencePrepareOptions,
    assess_reference,
    normalize_rms,
    prepare_reference,
)

SR = 16000


def _speech(seconds: float, value: float = 0.5) -> np.ndarray:
    return np.full(int(seconds * SR), value, dtype=np.float32)


def _with_speech(
    total: float, start: float, end: float, value: float = 0.5
) -> np.ndarray:
    samples = np.zeros(int(total * SR), dtype=np.float32)
    samples[int(start * SR) : int(end * SR)] = value
    return samples


# --- prepare_reference: обрезка тишины и запас ------------------------------


def test_prepare_reference_trims_silence_with_pad() -> None:
    waveform = _with_speech(10.0, 4.0, 6.0, value=0.5)

    prepared = prepare_reference(waveform, SR)

    # Речь 4–6 с плюс запас 0.1 с с каждой стороны.
    assert prepared.start_seconds == pytest.approx(3.9, abs=0.06)
    assert prepared.end_seconds == pytest.approx(6.1, abs=0.06)
    assert prepared.waveform.size == pytest.approx(2.2 * SR, rel=0.05)
    # Внутри окна — речь, снаружи не попала.
    assert float(np.max(np.abs(prepared.waveform))) > 0.0
    assert prepared.quality.speech_seconds == pytest.approx(2.0, abs=0.1)


def test_prepare_reference_caps_max_seconds() -> None:
    waveform = _speech(30.0, value=0.5)

    prepared = prepare_reference(waveform, SR, max_seconds=10.0)

    # Кап — на длину речи; сверху добавляется лишь небольшой запас (пад).
    assert prepared.waveform.size <= int(10.25 * SR)
    assert prepared.waveform.size == pytest.approx(10.0 * SR, rel=0.03)


def test_prepare_reference_handles_no_speech() -> None:
    waveform = np.zeros(5 * SR, dtype=np.float32)

    prepared = prepare_reference(waveform, SR)

    assert prepared.waveform.size == waveform.size
    assert prepared.quality.too_short is True
    assert prepared.quality.mostly_non_speech is True


def test_prepare_reference_disabled_returns_input() -> None:
    waveform = _with_speech(5.0, 1.0, 3.0, value=0.2)
    options = ReferencePrepareOptions(enabled=False)

    prepared = prepare_reference(waveform, SR, options=options)

    assert np.array_equal(prepared.waveform, waveform)
    assert prepared.quality.duration_seconds == pytest.approx(5.0)


# --- RMS-нормализация --------------------------------------------------------


def test_normalize_rms_increase_only_keeps_loud_signal() -> None:
    loud = _speech(3.0, value=0.5)

    normalized = normalize_rms(loud)

    assert np.array_equal(normalized, loud)


def test_normalize_rms_amplifies_quiet_signal_to_target() -> None:
    quiet = _speech(3.0, value=0.01)

    normalized = normalize_rms(quiet, target_dbfs=-30.0)

    rms = float(np.sqrt(np.mean(np.square(normalized.astype(np.float64)))))
    assert 20.0 * np.log10(rms) == pytest.approx(-30.0, abs=0.5)


def test_normalize_rms_limits_gain() -> None:
    tiny = _speech(3.0, value=1e-5)

    normalized = normalize_rms(tiny, max_gain=10.0)

    assert float(np.max(np.abs(normalized))) <= 1e-5 * 10.0 * 1.001


# --- assess_reference: флаги -------------------------------------------------


def test_assess_reference_clean_sample_is_ok() -> None:
    quality = assess_reference(_speech(6.0, value=0.5), SR)

    assert quality.ok is True
    assert quality.warnings() == []
    assert quality.speech_ratio == pytest.approx(1.0, abs=0.05)
    assert quality.trimmed_seconds == pytest.approx(6.0, abs=0.2)


def test_assess_reference_flags_short_clipped_quiet_and_non_speech() -> None:
    short = assess_reference(_speech(1.0, value=0.5), SR)
    assert short.too_short is True
    assert short.clipped is False

    clipped = assess_reference(_speech(6.0, value=1.0), SR)
    assert clipped.clipped is True

    quiet = assess_reference(_speech(6.0, value=1e-3), SR)
    assert quiet.low_energy is True
    assert 20.0 * np.log10(1e-3) < DEFAULT_LOW_ENERGY_DBFS

    mostly_silence = assess_reference(_with_speech(10.0, 0.0, 1.0, value=0.5), SR)
    assert mostly_silence.mostly_non_speech is True


def test_reference_quality_as_dict_contains_warnings() -> None:
    payload = assess_reference(_speech(1.0, value=0.5), SR).as_dict()

    assert payload["too_short"] is True
    assert isinstance(payload["warnings"], list)
    assert any("корот" in message for message in payload["warnings"])
    assert "speech_seconds" in payload


# --- согласованность эталон↔тест --------------------------------------------


def test_reference_and_speaker_window_processed_identically() -> None:
    """Одинаковая обработка эталона и окна говорящего — нет domain mismatch."""
    reference = np.zeros(8 * SR, dtype=np.float32)
    reference[2 * SR : 6 * SR] = 0.3
    speaker_window = np.zeros(8 * SR, dtype=np.float32)
    speaker_window[2 * SR : 6 * SR] = 0.3

    prepared_ref = prepare_reference(reference, SR, max_seconds=5.0)
    prepared_speaker = prepare_reference(speaker_window, SR, max_seconds=5.0)

    assert np.array_equal(prepared_ref.waveform, prepared_speaker.waveform)
    assert prepared_ref.start_seconds == prepared_speaker.start_seconds
    assert prepared_ref.end_seconds == prepared_speaker.end_seconds
