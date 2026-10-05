"""Тесты реального движка диаризации pyannote (без запуска модели).

Проверяется в первую очередь контракт хука прогресса: pyannote вызывает
``hook(step_name, artifact, file=..., total=..., completed=...)`` — ключ
``file`` обязателен. Раньше параметр был переименован в ``_file`` и
диаризация падала с ``unexpected keyword argument 'file'``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import audio_transcriber.diarization.pyannote_engine as pyannote_engine
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device
from audio_transcriber.utils.exceptions import DiarizationError


class _FakeAnnotation:
    def itertracks(self, *, yield_label: bool = False):
        yield SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"


class _FakePipeline:
    def __init__(self) -> None:
        # Аргументы числа говорящих, с которыми пайплайн реально вызвали
        # (не указанные ключи отсутствуют — как в вызове движка).
        self.speaker_kwargs: list[dict[str, object]] = []
        self.audio: dict[str, object] | None = None

    def to(self, *args: object, **kwargs: object) -> _FakePipeline:
        return self

    def __call__(self, audio: object, *, hook=None, **kwargs: object):
        # Воспроизводим вызов pyannote: file= передаётся по ключу.
        assert hook is not None
        self.audio = audio  # type: ignore[assignment]
        self.speaker_kwargs.append(dict(kwargs))
        hook("segmentation", None, file={"uri": "test"}, total=2, completed=1)
        hook("embeddings", None, file={"uri": "test"}, total=None, completed=None)
        return _FakeAnnotation()


class _InstantiateRecorder:
    """Мини-пайплайн, запоминающий вызовы ``instantiate`` и умеющий падать."""

    def __init__(self, error: Exception | None = None) -> None:
        self.params: list[dict[str, object]] = []
        self.error = error

    def instantiate(self, params: dict[str, object]) -> None:
        self.params.append(params)
        if self.error is not None:
            raise self.error


def test_diarize_progress_hook_accepts_file_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    diarizer = PyannoteSpeakerDiarizer(
        Device.CPU, on_progress=lambda event: events.append(event)
    )
    pipeline = _FakePipeline()
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: pipeline)
    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        lambda _path, **_kwargs: np.zeros(16000, dtype=np.float32),
    )

    segments = diarizer.diarize(Path("audio.wav"))

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]
    # хук отработал без TypeError и эмитил события прогресса
    assert events
    assert all(event.stage == "diarization" for event in events)


def test_diarize_reuses_provided_waveform_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diarizer = PyannoteSpeakerDiarizer(Device.CPU)
    pipeline = _FakePipeline()
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: pipeline)

    def _unexpected_decode(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("load_waveform не должен вызываться при переданном waveform")

    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        _unexpected_decode,
    )
    waveform = np.zeros(16000, dtype=np.float32)

    segments = diarizer.diarize(Path("audio.wav"), waveform=waveform)

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]
    # Переданный массив ушёл в pyannote как есть, без чтения файла.
    assert pipeline.audio is not None
    assert pipeline.audio["waveform"].shape == (1, 16000)


def test_hyperparameters_default_include_min_duration_off() -> None:
    diarizer = PyannoteSpeakerDiarizer(Device.CPU)

    assert diarizer._hyperparameters() == {"segmentation": {"min_duration_off": 0.5}}


def test_hyperparameters_include_clustering_when_set() -> None:
    diarizer = PyannoteSpeakerDiarizer(
        Device.CPU, clustering_threshold=0.6, clustering_fb=1.5
    )

    assert diarizer._hyperparameters() == {
        "segmentation": {"min_duration_off": 0.5},
        "clustering": {"threshold": 0.6, "Fb": 1.5},
    }


def test_apply_hyperparameters_calls_instantiate() -> None:
    recorder = _InstantiateRecorder()
    diarizer = PyannoteSpeakerDiarizer(
        Device.CPU, min_duration_off=1.0, clustering_threshold=0.6
    )

    diarizer._apply_hyperparameters(recorder)

    assert recorder.params == [
        {"segmentation": {"min_duration_off": 1.0}, "clustering": {"threshold": 0.6}}
    ]


def test_apply_hyperparameters_swallows_unknown_parameter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = _InstantiateRecorder(error=ValueError("parameter 'x' does not exist"))
    diarizer = PyannoteSpeakerDiarizer(Device.CPU)

    with caplog.at_level("WARNING"):
        diarizer._apply_hyperparameters(recorder)  # не бросает исключение

    assert any(
        "гиперпараметры диаризации" in record.message.lower() for record in caplog.records
    )


def test_diarize_passes_speaker_range(monkeypatch: pytest.MonkeyPatch) -> None:
    diarizer = PyannoteSpeakerDiarizer(Device.CPU, on_progress=lambda _event: None)
    pipeline = _FakePipeline()
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: pipeline)
    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        lambda _path, **_kwargs: np.zeros(16000, dtype=np.float32),
    )

    diarizer.diarize(Path("audio.wav"), min_speakers=2, max_speakers=5)

    assert pipeline.speaker_kwargs == [{"min_speakers": 2, "max_speakers": 5}]


def test_diarize_num_speakers_overrides_range(monkeypatch: pytest.MonkeyPatch) -> None:
    diarizer = PyannoteSpeakerDiarizer(Device.CPU, on_progress=lambda _event: None)
    pipeline = _FakePipeline()
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: pipeline)
    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        lambda _path, **_kwargs: np.zeros(16000, dtype=np.float32),
    )

    diarizer.diarize(Path("audio.wav"), num_speakers=3, min_speakers=2, max_speakers=5)

    assert pipeline.speaker_kwargs == [{"num_speakers": 3}]


# --- Issue #76: локальная модель грузится офлайн, онлайн-сбой падает быстро --


def test_load_pipeline_local_model_loads_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub.constants as hf_constants

    local = tmp_path / "speaker-diarization-community-1"
    local.mkdir()
    (local / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")

    calls: list[tuple[str, dict[str, object], bool]] = []

    class _OfflineRecordingPipeline(_FakePipeline):
        @classmethod
        def from_pretrained(cls, checkpoint: object, **_kwargs: object) -> _FakePipeline:
            calls.append((str(checkpoint), dict(_kwargs), hf_constants.HF_HUB_OFFLINE))
            return cls()

    monkeypatch.setattr(
        pyannote_engine, "_import_pipeline_class", lambda: _OfflineRecordingPipeline
    )
    offline_before = hf_constants.HF_HUB_OFFLINE

    diarizer = PyannoteSpeakerDiarizer(Device.CPU, local_model_path=local)
    diarizer._load_pipeline()

    assert calls == [(str(local), {}, True)]
    # Offline-режим восстанавливается после загрузки.
    assert hf_constants.HF_HUB_OFFLINE is offline_before


def test_load_pipeline_local_file_checkpoint_loads_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Путь к самому config.yaml (файлу) — тоже локальная загрузка, без HF."""
    local_dir = tmp_path / "pyannote-local"
    local_dir.mkdir()
    config_file = local_dir / "config.yaml"
    config_file.write_text("pipeline: {}\n", encoding="utf-8")

    calls: list[str] = []

    class _RecordingPipeline(_FakePipeline):
        @classmethod
        def from_pretrained(cls, checkpoint: object, **_kwargs: object) -> _FakePipeline:
            calls.append(str(checkpoint))
            return cls()

    monkeypatch.setattr(pyannote_engine, "_import_pipeline_class", lambda: _RecordingPipeline)

    diarizer = PyannoteSpeakerDiarizer(Device.CPU, local_model_path=config_file)
    diarizer._load_pipeline()

    assert calls == [str(config_file)]


def test_load_pipeline_missing_local_model_fails_without_hf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[object] = []

    class _NeverPipeline:
        @classmethod
        def from_pretrained(cls, checkpoint: object, **_kwargs: object) -> object:
            called.append(checkpoint)
            return object()

    monkeypatch.setattr(pyannote_engine, "_import_pipeline_class", lambda: _NeverPipeline)

    diarizer = PyannoteSpeakerDiarizer(Device.CPU, local_model_path=tmp_path / "missing")
    with pytest.raises(DiarizationError):
        diarizer._load_pipeline()

    # В сеть за gated-моделью не пошли.
    assert called == []


def test_load_pipeline_hf_error_fails_fast_with_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub.constants as hf_constants

    seen: dict[str, object] = {}

    class _Unauthorized(Exception):
        pass

    class _FailingPipeline:
        @classmethod
        def from_pretrained(cls, checkpoint: object, **_kwargs: object) -> object:
            seen["checkpoint"] = checkpoint
            seen["token"] = _kwargs.get("token")
            seen["etag_timeout"] = hf_constants.HF_HUB_ETAG_TIMEOUT
            seen["download_timeout"] = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
            raise _Unauthorized("401 Unauthorized")

    monkeypatch.setattr(pyannote_engine, "_import_pipeline_class", lambda: _FailingPipeline)
    etag_before = hf_constants.HF_HUB_ETAG_TIMEOUT
    download_before = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT

    diarizer = PyannoteSpeakerDiarizer(Device.CPU)
    with pytest.raises(DiarizationError) as excinfo:
        diarizer._load_pipeline()

    message = str(excinfo.value)
    assert "401" in message
    assert "токен" in message or "hf-token" in message.lower()
    # Во время загрузки таймауты были ограничены, чтобы не зависать на ретраях.
    assert seen["etag_timeout"] == pyannote_engine.DEFAULT_HF_TIMEOUT_SECONDS
    assert seen["download_timeout"] == pyannote_engine.DEFAULT_HF_TIMEOUT_SECONDS
    # И восстановлены после попытки.
    assert etag_before == hf_constants.HF_HUB_ETAG_TIMEOUT
    assert download_before == hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
