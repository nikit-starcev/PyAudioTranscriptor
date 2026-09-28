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
    def to(self, *args: object, **kwargs: object) -> _FakePipeline:
        return self

    def __call__(self, audio: object, *, num_speakers=None, hook=None):
        # Воспроизводим вызов pyannote: file= передаётся по ключу.
        assert hook is not None
        hook("segmentation", None, file={"uri": "test"}, total=2, completed=1)
        hook("embeddings", None, file={"uri": "test"}, total=None, completed=None)
        return _FakeAnnotation()


def test_diarize_progress_hook_accepts_file_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    diarizer = PyannoteSpeakerDiarizer(
        Device.CPU, on_progress=lambda event: events.append(event)
    )
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        lambda _path, **_kwargs: np.zeros(16000, dtype=np.float32),
    )

    segments = diarizer.diarize(Path("audio.wav"))

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00"]
    # хук отработал без TypeError и эмитил события прогресса
    assert events
    assert all(event.stage == "diarization" for event in events)
