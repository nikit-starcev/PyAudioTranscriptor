"""Тесты альтернативного движка диаризации на sherpa-onnx (#63).

Реальные модели/VAD не запускаются: VAD, эмбеддер и пути моделей подменяются.
Проверяются нарезка речи на окна, кластеризация в сегменты, склейка,
enrollment-эмбеддер и мягкая деградация без ``sherpa-onnx``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import sherpa_engine
from audio_transcriber.diarization.sherpa_engine import (
    SherpaDiarizationError,
    SherpaSpeakerDiarizer,
    plan_speech_windows,
)
from audio_transcriber.utils.audio import SAMPLE_RATE


class _FakeEmbedder:
    """Эмбеддер, отдающий заданные векторы по порядку вызовов."""

    window_seconds = 5.0

    def __init__(self, vectors: list[list[float]], *, fail: bool = False) -> None:
        self._vectors = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        self._fail = fail
        self.calls = 0

    def embed(self, _waveform: np.ndarray) -> np.ndarray:
        if self._fail:
            raise RuntimeError("embedding failure")
        vector = self._vectors[self.calls % len(self._vectors)]
        self.calls += 1
        return vector


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spans: list[tuple[int, int]],
) -> None:
    monkeypatch.setattr(sherpa_engine, "detect_speech_spans", lambda *_a, **_k: spans)
    monkeypatch.setattr(sherpa_engine, "resolve_vad_model", lambda *_a, **_k: Path("vad.onnx"))


# --- plan_speech_windows ----------------------------------------------------


def test_plan_splits_span_into_windows() -> None:
    spans = [(0, int(6.0 * SAMPLE_RATE))]

    windows = plan_speech_windows(spans, window_seconds=1.5, min_tail_seconds=0.5)

    assert windows == [
        (0, int(1.5 * SAMPLE_RATE)),
        (int(1.5 * SAMPLE_RATE), int(3.0 * SAMPLE_RATE)),
        (int(3.0 * SAMPLE_RATE), int(4.5 * SAMPLE_RATE)),
        (int(4.5 * SAMPLE_RATE), int(6.0 * SAMPLE_RATE)),
    ]


def test_plan_merges_tiny_tail_into_previous_window() -> None:
    spans = [(0, int(3.2 * SAMPLE_RATE))]

    windows = plan_speech_windows(spans, window_seconds=1.5, min_tail_seconds=0.5)

    assert windows == [
        (0, int(1.5 * SAMPLE_RATE)),
        (int(1.5 * SAMPLE_RATE), int(3.2 * SAMPLE_RATE)),
    ]


def test_plan_ignores_empty_spans() -> None:
    assert plan_speech_windows([], window_seconds=1.5) == []
    assert plan_speech_windows([(100, 100)], window_seconds=1.5) == []


# --- diarize ----------------------------------------------------------------


def test_diarize_clusters_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, spans=[(0, int(6.0 * SAMPLE_RATE))])
    embedder = _FakeEmbedder([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
    diarizer = SherpaSpeakerDiarizer(embedder=embedder)

    segments = diarizer.diarize(Path("audio.wav"), waveform=np.zeros(int(6.0 * SAMPLE_RATE), dtype=np.float32))

    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(3.0)
    assert segments[1].start == pytest.approx(3.0)
    assert segments[1].end == pytest.approx(6.0)
    assert diarizer.overlap_regions() == []


def test_diarize_explicit_num_speakers_forces_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch, spans=[(0, int(6.0 * SAMPLE_RATE))])
    embedder = _FakeEmbedder([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    diarizer = SherpaSpeakerDiarizer(embedder=embedder)

    segments = diarizer.diarize(
        Path("audio.wav"),
        num_speakers=2,
        waveform=np.zeros(int(6.0 * SAMPLE_RATE), dtype=np.float32),
    )

    assert len({segment.speaker_id for segment in segments}) == 2


def test_diarize_returns_empty_without_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, spans=[])
    diarizer = SherpaSpeakerDiarizer(embedder=_FakeEmbedder([[1, 0]]))

    assert diarizer.diarize(Path("audio.wav"), waveform=np.zeros(SAMPLE_RATE, dtype=np.float32)) == []


def test_diarize_raises_when_no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime(monkeypatch, spans=[(0, int(3.0 * SAMPLE_RATE))])
    diarizer = SherpaSpeakerDiarizer(embedder=_FakeEmbedder([[1, 0]], fail=True))

    with pytest.raises(SherpaDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32))


def test_diarize_raises_without_sherpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sherpa_engine.embedding_utils, "sherpa_available", lambda: False)
    diarizer = SherpaSpeakerDiarizer()

    with pytest.raises(SherpaDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=np.zeros(SAMPLE_RATE, dtype=np.float32))


# --- enrollment -------------------------------------------------------------


def test_enrollment_engine_returns_provided_embedder() -> None:
    embedder = _FakeEmbedder([[1, 0]])
    diarizer = SherpaSpeakerDiarizer(embedder=embedder)

    assert diarizer.enrollment_engine() is embedder


def test_enrollment_engine_none_without_sherpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sherpa_engine.embedding_utils, "sherpa_available", lambda: False)
    diarizer = SherpaSpeakerDiarizer()

    assert diarizer.enrollment_engine() is None
