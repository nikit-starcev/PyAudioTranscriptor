"""Тесты локального проигрывания образцов (``utils.playback``)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.utils import playback
from audio_transcriber.utils.audio import SAMPLE_RATE, write_wav


def _write_tone(path: Path, *, loud_seconds: float = 1.0, silent_seconds: float = 1.0) -> None:
    """Создаёт реальный WAV: тон (громкий) + тишина (для теста амплитуды)."""
    rate = SAMPLE_RATE
    samples = np.zeros(round((loud_seconds + silent_seconds) * rate), dtype=np.float32)
    tone_len = round(loud_seconds * rate)
    t = np.arange(tone_len, dtype=np.float32) / rate
    samples[:tone_len] = 0.8 * np.sin(2 * np.pi * 440.0 * t)
    write_wav(path, samples, sample_rate=rate)


def test_find_player_prefers_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"aplay"}

    monkeypatch.setattr(
        playback.shutil, "which", lambda binary: binary if binary in available else None
    )

    assert playback.find_player() == ("aplay", ("-q",))


def test_find_player_none_when_nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playback.shutil, "which", lambda _binary: None)

    assert playback.find_player() is None


def test_play_audio_file_without_player_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(playback.shutil, "which", lambda _binary: None)
    calls: list[object] = []
    monkeypatch.setattr(playback.subprocess, "Popen", lambda *a, **_k: calls.append(a))

    assert playback.play_audio_file(tmp_path / "x.wav") is False
    assert calls == []


def test_play_audio_file_launches_player_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "Иван.wav"
    sample.write_bytes(b"")
    monkeypatch.setattr(
        playback.shutil, "which", lambda binary: binary if binary == "ffplay" else None
    )
    launched: list[list[str]] = []

    def fake_popen(command, **kwargs):
        launched.append(command)
        return object()

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)

    assert playback.play_audio_file(sample) is True
    assert launched == [["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(sample)]]


def test_play_audio_file_degrades_when_launch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        playback.shutil, "which", lambda binary: binary if binary == "paplay" else None
    )

    def exploding_popen(*args, **kwargs):
        raise OSError("нет прав")

    monkeypatch.setattr(playback.subprocess, "Popen", exploding_popen)

    assert playback.play_audio_file(tmp_path / "x.wav") is False


# --- Длительность и амплитуда ----------------------------------------------


def test_read_duration_reads_wav(tmp_path: Path) -> None:
    sample = tmp_path / "voice.wav"
    _write_tone(sample, loud_seconds=2.0, silent_seconds=0.0)

    assert playback.read_duration(sample) == pytest.approx(2.0, abs=0.01)


def test_read_duration_non_wav_or_missing_is_zero(tmp_path: Path) -> None:
    other = tmp_path / "voice.mp3"
    other.write_bytes(b"not audio")

    assert playback.read_duration(other) == 0.0
    assert playback.read_duration(tmp_path / "absent.wav") == 0.0


def test_amplitude_envelope_reflects_loud_then_silent(tmp_path: Path) -> None:
    sample = tmp_path / "voice.wav"
    _write_tone(sample, loud_seconds=1.0, silent_seconds=1.0)

    envelope = playback.amplitude_envelope(sample, columns=50)

    assert len(envelope) == 50
    assert all(0.0 <= value <= 1.0 for value in envelope)
    assert max(envelope) == pytest.approx(1.0)
    assert envelope[0] > 0.1  # начало — громкий тон
    assert envelope[-1] < 0.05  # конец — тишина


def test_amplitude_envelope_silent_sample_is_zeros(tmp_path: Path) -> None:
    sample = tmp_path / "silence.wav"
    write_wav(sample, np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)

    envelope = playback.amplitude_envelope(sample, columns=10)

    assert envelope == [0.0] * 10
    assert max(envelope, default=0.0) < playback.SILENCE_RMS_THRESHOLD


def test_amplitude_envelope_missing_file_is_empty(tmp_path: Path) -> None:
    assert playback.amplitude_envelope(tmp_path / "absent.wav") == []


def test_amplitude_envelope_zero_columns_is_empty(tmp_path: Path) -> None:
    assert playback.amplitude_envelope(tmp_path / "absent.wav", columns=0) == []


# --- PlaybackHandle ---------------------------------------------------------


class _FakeProcess:
    def __init__(self, *, running: bool = True, wait_raises: bool = False) -> None:
        self.terminated = 0
        self.killed = 0
        self.waits = 0
        self._running = running
        self._wait_raises = wait_raises

    def poll(self):
        return None if self._running else 0

    def terminate(self) -> None:
        self.terminated += 1

    def wait(self, timeout=None) -> None:
        self.waits += 1
        if self._wait_raises:
            raise TimeoutError

    def kill(self) -> None:
        self.killed += 1


def test_playback_handle_elapsed_and_is_running() -> None:
    process = _FakeProcess(running=True)
    handle = playback.PlaybackHandle(
        process=process, duration=5.0, started_at=time.monotonic() - 2.0
    )

    assert handle.is_running() is True
    assert handle.elapsed >= 2.0

    process._running = False
    assert handle.is_running() is False


def test_playback_handle_without_process() -> None:
    handle = playback.PlaybackHandle(process=None, duration=0.0, started_at=time.monotonic())

    assert handle.is_running() is False
    handle.stop()  # не должно падать


def test_playback_handle_stop_terminates_process() -> None:
    process = _FakeProcess()
    handle = playback.PlaybackHandle(
        process=process, duration=1.0, started_at=time.monotonic()
    )

    handle.stop()

    assert process.terminated == 1
    assert process.killed == 0


def test_playback_handle_stop_kills_after_wait_timeout() -> None:
    process = _FakeProcess(wait_raises=True)
    handle = playback.PlaybackHandle(
        process=process, duration=1.0, started_at=time.monotonic()
    )

    handle.stop()

    assert process.terminated == 1
    assert process.killed == 1


def test_playback_handle_stop_swallows_errors() -> None:
    class _Exploding:
        def terminate(self) -> None:
            raise OSError("нет прав")

        def wait(self, timeout=None) -> None:
            raise OSError("нет прав")

        def kill(self) -> None:
            raise OSError("нет прав")

    handle = playback.PlaybackHandle(
        process=_Exploding(), duration=1.0, started_at=time.monotonic()
    )

    handle.stop()  # ни одно исключение не должно проброситься


def test_start_playback_returns_handle_with_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "voice.wav"
    _write_tone(sample, loud_seconds=3.0, silent_seconds=0.0)
    monkeypatch.setattr(
        playback.shutil, "which", lambda binary: binary if binary == "ffplay" else None
    )

    def fake_popen(*args, **kwargs):
        return _FakeProcess()

    monkeypatch.setattr(playback.subprocess, "Popen", fake_popen)

    handle = playback.start_playback(sample)

    assert handle is not None
    assert handle.duration == pytest.approx(3.0, abs=0.01)
    assert handle.is_running() is True


def test_start_playback_returns_none_without_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(playback.shutil, "which", lambda _binary: None)

    assert playback.start_playback(tmp_path / "voice.wav") is None
