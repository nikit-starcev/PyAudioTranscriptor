"""Тесты воркера изоляции (#143) без реальных моделей."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audio_transcriber import isolation
from audio_transcriber.cache.serialization import asr_from_payload, diarization_from_payload
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import (
    SpeakerOverlap,
    SpeakerSegment,
    TranscriptionSegment,
)
from audio_transcriber.isolation import worker
from audio_transcriber.isolation.protocol import KIND_ASR, KIND_DIARIZATION, WorkerRequest
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer


class _FakeRecognizer:
    def __init__(self, options: dict[str, Any], emit: Any) -> None:
        self.options = options
        self.emit = emit
        self.closed = False
        self.language_arg: str | None = "?"
        self.audio_arg: str | None = None

    def transcribe(self, audio_path: Path, *, language: str | None = None):
        self.audio_arg = str(audio_path)
        self.language_arg = language
        self.emit(ProgressEvent("asr", "распознавание", fraction=0.5))
        return (
            [TranscriptionSegment(start=0.0, end=1.0, text="привет", avg_logprob=-0.1)],
            "ru",
            1.0,
        )

    def close(self) -> None:
        self.closed = True


class _FakeDiarizer:
    def __init__(self, options: dict[str, Any], emit: Any) -> None:
        self.options = options
        self.emit = emit
        self.closed = False
        self.calls: dict[str, Any] = {}

    def diarize(self, audio_path: Path, *, num_speakers=None, min_speakers=None,
                max_speakers=None, waveform=None):
        self.calls = {
            "audio": str(audio_path),
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "waveform": waveform,
        }
        self.emit(ProgressEvent("diarization", "шаг", fraction=0.5))
        return [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]

    def overlap_regions(self) -> list[SpeakerOverlap]:
        return [SpeakerOverlap(start=0.2, end=0.4, speaker_ids=("SPEAKER_00", "SPEAKER_01"))]

    def close(self) -> None:
        self.closed = True


def _request_file(tmp_path: Path, kind: str, options: dict[str, Any], **extra: Any) -> Path:
    payload = {"kind": kind, "audio_path": str(tmp_path / "a.wav"), "options": options}
    payload.update(extra)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_execute_asr_builds_engine_and_returns_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: dict[str, Any] = {}

    def _build(options: dict[str, Any], emit: Any) -> _FakeRecognizer:
        engine = _FakeRecognizer(options, emit)
        built["options"] = options
        built["engine"] = engine
        return engine

    monkeypatch.setattr(worker, "_build_recognizer", _build)
    events: list[ProgressEvent] = []
    options = {"model_name": "small", "device": "cpu", "word_timestamps": True}

    request = WorkerRequest(
        kind=KIND_ASR, audio_path=str(tmp_path / "a.wav"), language="ru", options=options
    )
    payload = worker.execute(request, events.append)

    segments, language, duration = asr_from_payload(payload)
    assert language == "ru"
    assert duration == 1.0
    assert [s.text for s in segments] == ["привет"]
    assert built["options"] == options
    assert built["engine"].language_arg == "ru"
    assert built["engine"].closed is True
    assert events and events[0].stage == "asr"


def test_execute_diarization_returns_segments_and_overlaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(worker, "_build_diarizer", _FakeDiarizer)
    options = {
        "device": "cpu",
        "pipeline_name": "pyannote/test",
        "num_speakers": 3,
        "min_speakers": None,
        "max_speakers": None,
    }
    request = WorkerRequest(
        kind=KIND_DIARIZATION, audio_path=str(tmp_path / "a.wav"), options=options
    )

    payload = worker.execute(request, lambda _event: None)

    segments, overlaps = diarization_from_payload(payload)
    assert [s.speaker_id for s in segments] == ["SPEAKER_00"]
    assert [o.speaker_ids for o in overlaps] == [("SPEAKER_00", "SPEAKER_01")]


def test_execute_diarization_loads_waveform_from_npy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingDiarizer(_FakeDiarizer):
        def diarize(self, audio_path, **kwargs):  # type: ignore[override]
            captured["waveform"] = kwargs.get("waveform")
            return super().diarize(audio_path, **kwargs)

    monkeypatch.setattr(worker, "_build_diarizer", _CapturingDiarizer)
    waveform = np.linspace(0.0, 1.0, 16000, dtype=np.float32)
    npy_path = tmp_path / "w.npy"
    np.save(npy_path, waveform)
    request = WorkerRequest(
        kind=KIND_DIARIZATION,
        audio_path=str(tmp_path / "a.wav"),
        waveform_path=str(npy_path),
        options={"device": "cpu"},
    )

    worker.execute(request, lambda _event: None)

    assert captured["waveform"] is not None
    assert np.array_equal(captured["waveform"], waveform)


def test_worker_build_recognizer_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeWhisper:
        def __init__(self, model_name: str, device: Device, **kwargs: Any) -> None:
            captured.update({"model": model_name, "device": device, "kwargs": kwargs})

    monkeypatch.setattr(
        "audio_transcriber.transcription.whisper_engine.WhisperSpeechRecognizer",
        _FakeWhisper,
    )
    worker._build_recognizer(
        {"model_name": "base", "device": "cpu", "word_timestamps": False}, None
    )

    assert captured["model"] == "base"
    assert captured["device"] is Device.CPU
    assert captured["kwargs"]["word_timestamps"] is False


def test_worker_build_diarizer_defaults_pipeline_name(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakePyannote:
        def __init__(self, device: Device, **kwargs: Any) -> None:
            captured.update({"device": device, "kwargs": kwargs})

    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.PyannoteSpeakerDiarizer",
        _FakePyannote,
    )
    worker._build_diarizer({"device": "cpu"}, None)

    # Без явного pipeline_name используется дефолт модели, а не None.
    assert "pipeline_name" not in captured["kwargs"]
    assert captured["device"] is Device.CPU


def test_main_writes_success_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(worker, "_build_recognizer", _FakeRecognizer)
    request_path = _request_file(tmp_path, KIND_ASR, {"model_name": "small", "device": "cpu"})
    result_path = tmp_path / "result.json"

    code = worker.main(["--request", str(request_path), "--result", str(result_path)])

    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["kind"] == KIND_ASR
    assert result["payload"]["language"] == "ru"


def test_main_writes_error_result_on_engine_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(_options: dict[str, Any], _emit: Any) -> None:
        raise RuntimeError("модель не загрузилась")

    monkeypatch.setattr(worker, "_build_recognizer", _boom)
    request_path = _request_file(tmp_path, KIND_ASR, {"model_name": "small", "device": "cpu"})
    result_path = tmp_path / "result.json"

    code = worker.main(["--request", str(request_path), "--result", str(result_path)])

    assert code == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "модель не загрузилась" in result["error"]
    assert result["error_type"] == "RuntimeError"


def test_main_reports_invalid_request(tmp_path: Path) -> None:
    request_path = tmp_path / "bad.json"
    request_path.write_text("не json", encoding="utf-8")
    result_path = tmp_path / "result.json"

    code = worker.main(["--request", str(request_path), "--result", str(result_path)])

    assert code == 2
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ok"] is False


def test_public_engines_are_not_built_in_process() -> None:
    # Обёртки изоляции импортируются без загрузки тяжёлых движков: символы
    # указывают на воркер-версии, а не на in-process классы.
    assert isolation.SubprocessWhisperRecognizer is not WhisperSpeechRecognizer
    assert isolation.SubprocessPyannoteDiarizer is not PyannoteSpeakerDiarizer
