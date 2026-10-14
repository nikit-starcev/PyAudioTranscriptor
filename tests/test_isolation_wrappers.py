"""Тесты обёрток изоляции (#143) с подменённым воркером (без subprocess)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.cache.serialization import asr_payload, diarization_payload
from audio_transcriber.diarization import pyannote_engine
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment, TranscriptionSegment
from audio_transcriber.isolation import diarizer as diarizer_module
from audio_transcriber.isolation import recognizer as recognizer_module
from audio_transcriber.isolation.diarizer import SubprocessPyannoteDiarizer
from audio_transcriber.isolation.protocol import KIND_ASR, KIND_DIARIZATION
from audio_transcriber.isolation.recognizer import SubprocessWhisperRecognizer
from audio_transcriber.isolation.runner import WorkerResult
from audio_transcriber.utils.exceptions import (
    DiarizationError,
    EngineIsolationError,
    TranscriptionError,
)


def test_transcribe_builds_request_and_parses_result(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_worker(request, *, on_progress, timeout):
        captured["request"] = request
        captured["on_progress"] = on_progress
        captured["timeout"] = timeout
        return WorkerResult(
            ok=True,
            payload=asr_payload(
                [TranscriptionSegment(start=0.0, end=1.0, text="hi")], "en", 2.0
            ),
        )

    monkeypatch.setattr(recognizer_module, "run_worker", _fake_run_worker)
    events: list[object] = []
    recognizer = SubprocessWhisperRecognizer(
        "base",
        Device.CPU,
        initial_prompt="prompt",
        hotwords="словарь",
        on_progress=events.append,
        word_timestamps=False,
    )

    segments, language, duration = recognizer.transcribe(Path("/tmp/a.wav"), language="en")

    request = captured["request"]
    assert language == "en"
    assert duration == 2.0
    assert [s.text for s in segments] == ["hi"]
    assert request.kind == KIND_ASR
    assert request.language == "en"
    assert request.audio_path == "/tmp/a.wav"
    assert request.options == {
        "model_name": "base",
        "device": "cpu",
        "initial_prompt": "prompt",
        "hotwords": "словарь",
        "word_timestamps": False,
    }
    assert captured["on_progress"] == events.append


def test_transcribe_wraps_isolation_error_as_transcription_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> WorkerResult:
        raise EngineIsolationError("воркер не ответил")

    monkeypatch.setattr(recognizer_module, "run_worker", _boom)
    recognizer = SubprocessWhisperRecognizer("base", Device.CPU)

    with pytest.raises(TranscriptionError) as excinfo:
        recognizer.transcribe(Path("/tmp/a.wav"))

    assert "воркер не ответил" in str(excinfo.value)


def test_transcribe_raises_on_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        recognizer_module,
        "run_worker",
        lambda *_a, **_k: WorkerResult(ok=False, error="модель упала"),
    )
    recognizer = SubprocessWhisperRecognizer("base", Device.CPU)

    with pytest.raises(TranscriptionError) as excinfo:
        recognizer.transcribe(Path("/tmp/a.wav"))

    assert "модель упала" in str(excinfo.value)


def test_diarizer_passes_waveform_speakers_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_worker(request, *, on_progress, timeout):
        captured["request"] = request
        captured["on_progress"] = on_progress
        return WorkerResult(
            ok=True,
            payload=diarization_payload(
                [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")],
                [SpeakerOverlap(start=0.2, end=0.4, speaker_ids=("SPEAKER_00", "SPEAKER_01"))],
            ),
        )

    monkeypatch.setattr(diarizer_module, "run_worker", _fake_run_worker)
    diarizer = SubprocessPyannoteDiarizer(
        Device.CPU,
        pipeline_name="pyannote/test",
        hf_token="tok",
        min_duration_off=0.7,
        clustering_threshold=0.65,
    )
    waveform = np.linspace(0.0, 1.0, 16000, dtype=np.float32)
    events: list[object] = []
    diarizer._on_progress = events.append

    segments = diarizer.diarize(
        Path("/tmp/a.wav"),
        num_speakers=2,
        min_speakers=1,
        max_speakers=4,
        waveform=waveform,
    )

    request = captured["request"]
    assert [s.speaker_id for s in segments] == ["SPEAKER_00"]
    assert request.kind == KIND_DIARIZATION
    assert request.waveform_path is not None
    # Временный .npy удаляется после стадии.
    assert not Path(request.waveform_path).exists()
    assert request.options["num_speakers"] == 2
    assert request.options["min_speakers"] == 1
    assert request.options["max_speakers"] == 4
    assert request.options["hf_token"] == "tok"
    assert request.options["pipeline_name"] == "pyannote/test"
    assert request.options["min_duration_off"] == 0.7
    assert request.options["clustering_threshold"] == 0.65
    assert captured["on_progress"] == events.append
    assert [o.speaker_ids for o in diarizer.overlap_regions()] == [
        ("SPEAKER_00", "SPEAKER_01")
    ]


def test_diarizer_without_waveform_has_no_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_worker(request, *, on_progress, timeout):
        captured["request"] = request
        return WorkerResult(ok=True, payload=diarization_payload([], []))

    monkeypatch.setattr(diarizer_module, "run_worker", _fake_run_worker)
    diarizer = SubprocessPyannoteDiarizer(Device.CPU)

    diarizer.diarize(Path("/tmp/a.wav"))

    assert captured["request"].waveform_path is None
    assert diarizer.overlap_regions() == []


def test_diarizer_local_model_and_default_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_worker(request, *, on_progress, timeout):
        captured["request"] = request
        return WorkerResult(ok=True, payload=diarization_payload([], []))

    monkeypatch.setattr(diarizer_module, "run_worker", _fake_run_worker)
    diarizer = SubprocessPyannoteDiarizer(Device.CPU, local_model_path="/models/dia")

    diarizer.diarize(Path("/tmp/a.wav"))

    assert captured["request"].options["local_model_path"] == "/models/dia"
    assert captured["request"].options["pipeline_name"] == pyannote_engine.DEFAULT_PIPELINE


def test_diarizer_wraps_isolation_error_as_diarization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> WorkerResult:
        raise EngineIsolationError("таймаут воркера")

    monkeypatch.setattr(diarizer_module, "run_worker", _boom)
    diarizer = SubprocessPyannoteDiarizer(Device.CPU)

    with pytest.raises(DiarizationError) as excinfo:
        diarizer.diarize(Path("/tmp/a.wav"))

    assert "таймаут воркера" in str(excinfo.value)


def test_diarizer_raises_on_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diarizer_module,
        "run_worker",
        lambda *_a, **_k: WorkerResult(ok=False, error="TORCH OOM"),
    )
    diarizer = SubprocessPyannoteDiarizer(Device.CPU)

    with pytest.raises(DiarizationError) as excinfo:
        diarizer.diarize(Path("/tmp/a.wav"))

    assert "TORCH OOM" in str(excinfo.value)


def test_close_is_noop() -> None:
    SubprocessWhisperRecognizer("base", Device.CPU).close()
    SubprocessPyannoteDiarizer(Device.CPU).close()
