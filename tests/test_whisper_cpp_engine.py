"""Тесты whisper.cpp: защита процесса реестром и очистка ресурсов.

Реальный бинарник не запускается — ``subprocess.Popen`` подменяется заглушкой.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.transcription import whisper_cpp_engine as whisper_module
from audio_transcriber.transcription.whisper_cpp_engine import WhisperCppRecognizer
from audio_transcriber.utils import subprocess_registry
from audio_transcriber.utils.exceptions import TranscriptionError


class _FakeProc:
    def __init__(self, pid: int, returncode: int) -> None:
        self.pid = pid
        self.returncode = returncode
        self.stderr = iter(["progress = 50%\n"])

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def test_whisper_process_is_unregistered_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")

    def _fake_popen(*_args: object, **_kwargs: object) -> _FakeProc:
        return _FakeProc(pid=4242, returncode=1)

    monkeypatch.setattr(whisper_module, "load_waveform", lambda _path: object())
    monkeypatch.setattr(whisper_module, "write_wav", lambda _path, _waveform: None)
    monkeypatch.setattr(whisper_module.subprocess, "Popen", _fake_popen)

    recognizer = WhisperCppRecognizer(model)

    with pytest.raises(TranscriptionError):
        recognizer.transcribe(audio)

    # Процесс зарегистрирован на время работы и снят в finally.
    assert 4242 not in subprocess_registry._active_processes
