"""Тесты харнесса калибровки кластеризации sherpa (``scripts/calibrate_sherpa_diarization.py``).

Проверяются только «чистые» функции: разбор linkage, число кластеров и перебор
сетки на синтетических эмбеддингах. Реальные VAD/эмбеддер/аудио не запускаются —
боевые прогоны харнесса выполняются вручную.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_sherpa_diarization import (
    Excerpt,
    cluster_count,
    excerpt_segments,
    parse_linkages,
    plan_observations,
    sweep,
)

#: Два ортогональных «голоса» (косинусное расстояние 1.0) и два близких к первому.
_ORTHOGONAL = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]


def _excerpt(vectors: list[list[float]]) -> Excerpt:
    windows = [(index * 24000, (index + 1) * 24000) for index in range(len(vectors))]
    waveform = np.zeros(len(vectors) * 24000, dtype=np.float32)
    return Excerpt(
        name="sample",
        waveform=waveform,
        windows=windows,
        embeddings=np.asarray(vectors, dtype=np.float32),
    )


def test_parse_linkages_accepts_and_normalizes() -> None:
    assert parse_linkages("WARD, complete") == ("ward", "complete")
    assert parse_linkages("average") == ("average",)


def test_parse_linkages_rejects_unknown_and_empty() -> None:
    with pytest.raises(ValueError):
        parse_linkages("banana")
    with pytest.raises(ValueError):
        parse_linkages(" , ")


def test_plan_observations_pairs_index_with_vector() -> None:
    vectors = np.asarray(_ORTHOGONAL, dtype=np.float32)

    observations = plan_observations(vectors)

    assert [index for index, _speaker, _vector in observations] == [0, 1, 2, 3]
    assert all(speaker == "spk" for _index, speaker, _vector in observations)


def test_cluster_count_respects_threshold() -> None:
    vectors = np.asarray(_ORTHOGONAL, dtype=np.float32)

    # complete/cosine: A-A и B-B расстояние 0, A-B — 1.0.
    assert cluster_count(vectors, threshold=1.5, linkage="complete") == 1
    assert cluster_count(vectors, threshold=0.4, linkage="complete") == 2


def test_cluster_count_empty_is_zero() -> None:
    assert cluster_count(np.zeros((0, 1), dtype=np.float32), threshold=1.0, linkage="complete") == 0


def test_sweep_covers_grid() -> None:
    excerpt = _excerpt(_ORTHOGONAL)

    points = sweep(excerpt, thresholds=(0.4, 1.5), linkages=("complete",))

    assert len(points) == 2
    counts = {(point.threshold, point.linkage): point.n_clusters for point in points}
    assert counts[(0.4, "complete")] == 2
    assert counts[(1.5, "complete")] == 1
    assert all(point.segments >= 1 for point in points)


def test_excerpt_segments_merges_adjacent_same_speaker() -> None:
    excerpt = _excerpt(_ORTHOGONAL)

    segments = excerpt_segments(excerpt, threshold=0.4, linkage="complete")

    assert {turn.speaker for turn in segments} == {"SPEAKER_00", "SPEAKER_01"}
    assert segments[0].start == pytest.approx(0.0)
    assert segments[-1].end == pytest.approx(6.0)
    assert segments[0].end <= segments[1].start
