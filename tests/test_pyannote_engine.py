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

from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device


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
