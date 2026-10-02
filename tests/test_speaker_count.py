"""Тесты дешёвого оценщика числа говорящих (#64).

Реальные модели/VAD не запускаются: подменяются VAD, эмбеддинги и
кластеризация. Проверяется мягкая деградация (``None``) и распределение
выборки речи по записи.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import speaker_count
from audio_transcriber.utils.audio import SAMPLE_RATE


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    samples: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    windows: list[np.ndarray] | None = None,
    embeddings: np.ndarray | None = None,
    count: int | None = 3,
) -> None:
    if samples is None:
        samples = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    if spans is None:
        spans = [(0, SAMPLE_RATE * 5)]
    if windows is None:
        windows = [np.zeros(SAMPLE_RATE, dtype=np.float32)] * 4
    if embeddings is None:
        embeddings = np.eye(4, dtype=np.float32)

    monkeypatch.setattr(speaker_count, "_sherpa_available", lambda: True)
    monkeypatch.setattr(speaker_count, "_load_samples", lambda *_a, **_k: samples)
    monkeypatch.setattr(
        speaker_count,
        "_resolve_models",
        lambda *_a, **_k: (Path("embedding.onnx"), Path("vad.onnx")),
    )
    monkeypatch.setattr(speaker_count, "_detect_speech", lambda *_a, **_k: spans)
    monkeypatch.setattr(speaker_count, "_sample_windows", lambda *_a, **_k: windows)
    monkeypatch.setattr(speaker_count, "_compute_embeddings", lambda *_a, **_k: embeddings)
    monkeypatch.setattr(speaker_count, "_cluster_embeddings", lambda *_a, **_k: count)


def test_estimate_returns_cluster_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, count=6)

    assert speaker_count.estimate_speaker_count("audio.wav") == 6


def test_estimate_none_without_sherpa(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(speaker_count, "_sherpa_available", lambda: False)

    assert speaker_count.estimate_speaker_count("audio.wav") is None


def test_estimate_none_without_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(speaker_count, "_load_samples", lambda *_a, **_k: None)

    assert speaker_count.estimate_speaker_count(None) is None


def test_estimate_none_without_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(speaker_count, "_resolve_models", lambda *_a, **_k: None)

    assert speaker_count.estimate_speaker_count("audio.wav") is None


def test_estimate_none_without_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, spans=[])

    assert speaker_count.estimate_speaker_count("audio.wav") is None


def test_estimate_single_window_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, windows=[np.zeros(SAMPLE_RATE, dtype=np.float32)])
    called = {"embeddings": False}

    def forbidden(*_a: object, **_k: object) -> np.ndarray:
        called["embeddings"] = True
        return np.eye(1, dtype=np.float32)

    monkeypatch.setattr(speaker_count, "_compute_embeddings", forbidden)

    assert speaker_count.estimate_speaker_count("audio.wav") == 1
    assert called["embeddings"] is False


def test_estimate_none_on_embedding_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch)

    def boom(*_a: object, **_k: object) -> np.ndarray:
        raise RuntimeError("onnx сломался")

    monkeypatch.setattr(speaker_count, "_compute_embeddings", boom)

    assert speaker_count.estimate_speaker_count("audio.wav") is None


def test_estimate_none_on_clustering_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch)

    def boom(*_a: object, **_k: object) -> int:
        raise ValueError("плохие эмбеддинги")

    monkeypatch.setattr(speaker_count, "_cluster_embeddings", boom)

    assert speaker_count.estimate_speaker_count("audio.wav") is None


# --- выборка речи, распределённая по записи ---------------------------------


def test_start_at_finds_offset() -> None:
    cumulative = [
        (0.0, 5.0, 100, 100 + 5 * SAMPLE_RATE),
        (5.0, 10.0, 900, 900 + 5 * SAMPLE_RATE),
    ]

    assert speaker_count._start_at(cumulative, 2.0) == 100 + 2 * SAMPLE_RATE
    assert speaker_count._start_at(cumulative, 7.0) == 900 + 2 * SAMPLE_RATE
    assert speaker_count._start_at(cumulative, 99.0) is None


def test_sample_windows_distributed_across_recording() -> None:
    total = SAMPLE_RATE * 100
    # Значение сэмпла = его индекс: по первому сэмплу окна видно позицию.
    samples = np.arange(total, dtype=np.float32)
    # Три речевых фрагмента: начало, середина, конец записи (всего 30 с речи).
    spans = [
        (SAMPLE_RATE * 0, SAMPLE_RATE * 10),
        (SAMPLE_RATE * 45, SAMPLE_RATE * 55),
        (SAMPLE_RATE * 90, SAMPLE_RATE * 100),
    ]

    windows = speaker_count._sample_windows(samples, spans, max_seconds=3.0)

    assert len(windows) == 2
    starts = [float(window[0]) for window in windows]
    # Цели идут по накопленному времени речи: первое окно — в начале, второе —
    # уже ближе к концу записи (а не оба в первом фрагменте).
    assert starts[0] < SAMPLE_RATE * 20
    assert starts[1] > SAMPLE_RATE * 80


def test_sample_windows_ignores_short_spans() -> None:
    total = SAMPLE_RATE * 10
    samples = np.arange(total, dtype=np.float32)
    spans = [(0, int(SAMPLE_RATE * 0.1)), (SAMPLE_RATE * 5, SAMPLE_RATE * 8)]

    windows = speaker_count._sample_windows(samples, spans, max_seconds=3.0)

    assert len(windows) == 2
    # Короткий фрагмент отброшен — окна берутся из второго (с 5-й секунды).
    assert all(float(window[0]) >= SAMPLE_RATE * 5 for window in windows)


# --- кластеризация и загрузка моделей ---------------------------------------


def test_cluster_embeddings_separates_two_groups() -> None:
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.stack([first, first * 0.99, second, second * 0.99])

    assert speaker_count._cluster_embeddings(embeddings, threshold=0.7) == 2


def test_cluster_embeddings_identical_is_one() -> None:
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.stack([vector, vector, vector])

    assert speaker_count._cluster_embeddings(embeddings, threshold=0.7) == 1


def test_download_file_is_atomic_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    source.write_bytes(b"model-bytes")
    target = tmp_path / "cache" / "model.onnx"

    result = speaker_count._download_file(source.as_uri(), target)

    assert result == target
    assert target.read_bytes() == b"model-bytes"
    assert not list(target.parent.glob(".download-*"))


def test_download_file_missing_source_returns_none(tmp_path: Path) -> None:
    target = tmp_path / "model.onnx"

    assert speaker_count._download_file((tmp_path / "nope.onnx").as_uri(), target) is None
    assert not target.exists()


def test_resolve_named_model_uses_existing_path(tmp_path: Path) -> None:
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"x")

    resolved = speaker_count._resolve_named_model(
        str(model), tmp_path / "cache", default_url="http://invalid", on_progress=None
    )

    assert resolved == model


def test_resolve_named_model_unknown_name_is_none(tmp_path: Path) -> None:
    resolved = speaker_count._resolve_named_model(
        "unknown-model.onnx",
        tmp_path / "cache",
        default_url="http://invalid",
        on_progress=None,
    )

    assert resolved is None
