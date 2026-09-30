"""Тесты whisper.cpp: защита процесса реестром и очистка ресурсов.

Реальный бинарник не запускается — ``subprocess.Popen`` подменяется заглушкой.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.transcription import whisper_cpp_engine as whisper_module
from audio_transcriber.transcription.whisper_cpp_engine import WhisperCppRecognizer
from audio_transcriber.utils import subprocess_registry
from audio_transcriber.utils.exceptions import TranscriptionError


class _FakeProc:
    def __init__(
        self,
        pid: int,
        returncode: int,
        *,
        stderr: list[str] | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.stderr = iter(stderr if stderr is not None else ["progress = 50%\n"])

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

    monkeypatch.setattr(
        whisper_module, "load_waveform", lambda _path: np.zeros(16000, dtype=np.float32)
    )
    monkeypatch.setattr(whisper_module, "write_wav", lambda _path, _waveform: None)
    monkeypatch.setattr(whisper_module.subprocess, "Popen", _fake_popen)

    recognizer = WhisperCppRecognizer(model)

    with pytest.raises(TranscriptionError):
        recognizer.transcribe(audio)

    # Процесс зарегистрирован на время работы и снят в finally.
    assert 4242 not in subprocess_registry._active_processes


def _json_payload() -> dict:
    return {
        "result": {"language": "ru"},
        "transcription": [
            {
                "offsets": {"from": 0, "to": 1000},
                "text": " привет",
                "tokens": [{"text": " при", "p": 0.9}],
            },
            {
                "offsets": {"from": 1000, "to": 2500},
                "text": " мир",
                "tokens": [{"text": " мир", "p": 0.8}],
            },
        ],
    }


def _install_fake_popen(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    *,
    captured: list[list[str]] | None = None,
    returncode: int = 0,
    stderr: list[str] | None = None,
) -> None:
    """Подменяет Popen: пишет JSON-результат туда, куда просит ``-of``."""

    def _fake_popen(cmd: list[str], *_args: object, **_kwargs: object) -> _FakeProc:
        if captured is not None:
            captured.append(list(cmd))
        base = Path(cmd[cmd.index("-of") + 1])
        Path(str(base) + ".json").write_text(json.dumps(payload), encoding="utf-8")
        return _FakeProc(pid=777, returncode=returncode, stderr=stderr)

    monkeypatch.setattr(whisper_module, "load_waveform", lambda _path: np.zeros(48000, dtype=np.float32))
    monkeypatch.setattr(whisper_module, "write_wav", lambda _path, _waveform: None)
    monkeypatch.setattr(whisper_module.subprocess, "Popen", _fake_popen)


def test_duration_taken_from_audio_not_last_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    # 3 секунды аудио при конце последнего сегмента 2.5 с (хвостовая тишина).
    captured: list[list[str]] = []
    _install_fake_popen(monkeypatch, _json_payload(), captured=captured)

    recognizer = WhisperCppRecognizer(model)
    segments, language, duration = recognizer.transcribe(tmp_path / "audio.wav")

    assert language == "ru"
    assert duration == pytest.approx(3.0)
    assert duration >= segments[-1].end
    assert [segment.text for segment in segments] == ["привет", "мир"]


def test_vad_disabled_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    captured: list[list[str]] = []
    _install_fake_popen(monkeypatch, _json_payload(), captured=captured)

    WhisperCppRecognizer(model).transcribe(tmp_path / "audio.wav")

    assert "--vad" not in captured[0]


def test_vad_enabled_with_model_matches_faster_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    vad_model = tmp_path / "ggml-silero-vad.bin"
    vad_model.write_bytes(b"vad")

    captured: list[list[str]] = []
    _install_fake_popen(monkeypatch, _json_payload(), captured=captured)

    WhisperCppRecognizer(model, vad_model=vad_model).transcribe(tmp_path / "audio.wav")

    cmd = captured[0]
    assert "--vad" in cmd
    assert cmd[cmd.index("-vm") + 1] == str(vad_model)
    # Условия VAD выровнены с faster_whisper.vad.VadOptions по умолчанию.
    assert cmd[cmd.index("-vt") + 1] == "0.5"
    assert cmd[cmd.index("-vspd") + 1] == "0"
    assert cmd[cmd.index("-vsd") + 1] == "2000"
    assert cmd[cmd.index("-vp") + 1] == "400"


def test_vad_skipped_when_model_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    captured: list[list[str]] = []
    _install_fake_popen(monkeypatch, _json_payload(), captured=captured)

    WhisperCppRecognizer(model, vad_model=tmp_path / "missing.bin").transcribe(
        tmp_path / "audio.wav"
    )

    assert "--vad" not in captured[0]

