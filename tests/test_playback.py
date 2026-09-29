"""Тесты локального проигрывания образцов (``utils.playback``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.utils import playback


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
