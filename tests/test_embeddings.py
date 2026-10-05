"""Тесты общего модуля speaker-эмбеддингов (#64).

Реальные модели не запускаются: подменяются ``sherpa-onnx``, доступность и
загрузка файлов. Проверяются L2-нормировка, кластеризация, разрешение модели и
мягкие проверки доступности.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import embeddings


def _install_fake_sherpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет ``sherpa_onnx`` минимальной реализацией эмбеддера."""

    class _Config:
        model = ""
        provider = ""
        num_threads = 0

    class _Stream:
        def accept_waveform(self, _rate: int, _waveform: np.ndarray) -> None:
            return None

        def input_finished(self) -> None:
            return None

    class _Extractor:
        def __init__(self, _config: object) -> None:
            pass

        def create_stream(self) -> _Stream:
            return _Stream()

        def compute(self, _stream: _Stream) -> np.ndarray:
            return np.array([3.0, 4.0], dtype=np.float32)

    fake = types.ModuleType("sherpa_onnx")
    fake.SpeakerEmbeddingExtractorConfig = _Config  # type: ignore[attr-defined]
    fake.SpeakerEmbeddingExtractor = _Extractor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)


# --- нормировка и кластеризация ---------------------------------------------


def test_l2_normalize_unit_length() -> None:
    result = embeddings.l2_normalize(np.array([3.0, 4.0], dtype=np.float32))

    assert float(np.linalg.norm(result)) == pytest.approx(1.0)
    assert result[0] == pytest.approx(0.6)
    assert result[1] == pytest.approx(0.8)


def test_l2_normalize_zero_vector_unchanged() -> None:
    result = embeddings.l2_normalize(np.zeros(3, dtype=np.float32))

    assert np.array_equal(result, np.zeros(3, dtype=np.float32))


def test_cluster_embeddings_separates_two_groups() -> None:
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    matrix = np.stack([first, first * 0.99, second, second * 0.99])

    labels = embeddings.cluster_embeddings(matrix, threshold=0.7)

    assert len(set(labels.tolist())) == 2
    assert embeddings.count_clusters(matrix, threshold=0.7) == 2


def test_cluster_embeddings_n_clusters_hint() -> None:
    vectors = np.eye(4, dtype=np.float32)

    labels = embeddings.cluster_embeddings(vectors, threshold=0.7, n_clusters=2)

    assert len(set(labels.tolist())) == 2


def test_cluster_embeddings_complete_avoids_chain_merging() -> None:
    """average-linkage «сцепляет» цепочку, complete — нет (против chaining)."""
    # Четыре точки единичной окружности с шагом 10°: соседи очень близки
    # (cos-расстояние ≈0.015), но концы цепочки далеки (0°↔30° ≈0.134).
    angles = np.deg2rad([0.0, 10.0, 20.0, 30.0])
    matrix = np.stack(
        [np.array([np.cos(a), np.sin(a)], dtype=np.float32) for a in angles]
    )

    chained = embeddings.cluster_embeddings(
        matrix, threshold=0.08, linkage="average", metric="cosine"
    )
    kept = embeddings.cluster_embeddings(
        matrix, threshold=0.08, linkage="complete", metric="cosine"
    )

    # average через «мостики»-соседей стягивает всю цепочку в один кластер…
    assert len(set(chained.tolist())) == 1
    # …а complete (по максимальному расстоянию между кластерами) не даёт
    # цепочке схлопнуться: цепочка рвётся на две половины.
    assert len(set(kept.tolist())) == 2


def test_cluster_embeddings_defaults_to_complete_cosine() -> None:
    """Без явного linkage/metric работает complete/cosine (против chaining)."""
    angles = np.deg2rad([0.0, 10.0, 20.0, 30.0])
    matrix = np.stack(
        [np.array([np.cos(a), np.sin(a)], dtype=np.float32) for a in angles]
    )

    default = embeddings.cluster_embeddings(matrix, threshold=0.08)
    explicit = embeddings.cluster_embeddings(
        matrix, threshold=0.08, linkage="complete", metric="cosine"
    )

    assert default.tolist() == explicit.tolist()
    assert len(set(default.tolist())) == 2


def test_cluster_embeddings_euclidean_ward_n_clusters() -> None:
    """ward/euclidean собирает ровно заданное число кластеров."""
    matrix = np.eye(4, dtype=np.float32)

    labels = embeddings.cluster_embeddings(
        matrix, threshold=0.7, n_clusters=3, linkage="ward", metric="euclidean"
    )

    assert len(set(labels.tolist())) == 3


def test_cluster_embeddings_single_is_one() -> None:
    labels = embeddings.cluster_embeddings(
        np.array([[1.0, 0.0]], dtype=np.float32), threshold=0.7
    )

    assert labels.tolist() == [0]
    assert embeddings.count_clusters(np.array([[1.0, 0.0]]), threshold=0.7) == 1


# --- доступность и разрешение модели ----------------------------------------


def test_sherpa_available_returns_bool() -> None:
    assert isinstance(embeddings.sherpa_available(), bool)


def test_embedder_available_false_without_sherpa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(embeddings, "sherpa_available", lambda: False)

    assert embeddings.embedder_available("model.onnx", model_dir=tmp_path) is False


def test_embedder_available_true_with_cached_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(embeddings, "sherpa_available", lambda: True)
    (tmp_path / embeddings.DEFAULT_EMBEDDING_MODEL).write_bytes(b"onnx")

    assert (
        embeddings.embedder_available(
            embeddings.DEFAULT_EMBEDDING_MODEL, model_dir=tmp_path
        )
        is True
    )


def test_embedder_available_does_not_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(embeddings, "sherpa_available", lambda: True)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("загрузка не должна запускаться при проверке доступности")

    monkeypatch.setattr(embeddings, "download_file", forbidden)

    assert (
        embeddings.embedder_available(
            embeddings.DEFAULT_EMBEDDING_MODEL, model_dir=tmp_path
        )
        is False
    )


def test_resolve_embedding_model_existing_path(tmp_path: Path) -> None:
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"x")

    resolved = embeddings.resolve_embedding_model(str(model), tmp_path / "cache")

    assert resolved == model


def test_resolve_embedding_model_unknown_name(tmp_path: Path) -> None:
    assert (
        embeddings.resolve_embedding_model("unknown.onnx", tmp_path / "cache") is None
    )


def test_download_file_is_atomic_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    source.write_bytes(b"model-bytes")
    target = tmp_path / "cache" / "model.onnx"

    result = embeddings.download_file(source.as_uri(), target)

    assert result == target
    assert target.read_bytes() == b"model-bytes"
    assert not list(target.parent.glob(".download-*"))


# --- извлечение эмбеддингов --------------------------------------------------


def test_speaker_embedder_normalizes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sherpa(monkeypatch)
    embedder = embeddings.SpeakerEmbedder(Path("model.onnx"))

    vector = embedder.embed(np.zeros(16000, dtype=np.float32))

    assert vector[0] == pytest.approx(0.6)
    assert vector[1] == pytest.approx(0.8)


def test_compute_embeddings_stacks_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_sherpa(monkeypatch)
    windows = [np.zeros(16000, dtype=np.float32)] * 3

    matrix = embeddings.compute_embeddings(windows, Path("model.onnx"))

    assert matrix.shape == (3, 2)
