#!/usr/bin/env python
"""Диагностика гибридной диаризации: воспроизведение наблюдений, анализ «мостов», свип ручек.

Скрипт **ничего не меняет** в движке: он лишь вызывает существующие функции
:mod:`audio_transcriber.diarization.hybrid_engine` и
:mod:`audio_transcriber.diarization.embeddings`, собирает те же эмбеддинги, что
и боевой гибрид, и исследует, как ведёт себя кластеризация при разных
параметрах. Предназначен для ответа на вопрос «почему гибрид перемерживает».

Стадии (все опциональны, задаются через ``--stages``):

* ``summary``     — состав наблюдений (окно/локальный говорящий/вектор) и статистика.
* ``dist``        — распределение косинусных расстояний (внутри/между кластерами).
* ``pairs``       — самые близкие пары наблюдений (кто с кем слипается).
* ``merges``      — порядок слияний агломеративной кластеризации (дендрограмма как данные).
* ``bridges``     — наблюдения-«мосты», связывающие будущие разные кластеры.
* ``sweep``       — свип threshold × linkage × metric (число кластеров и доля топ-кластера).
* ``aggregation`` — покусочно vs усреднение/медоид на локального говорящего (нужно аудио).
* ``window``      — 1–2 медленных прогона движка с другим окном/overload-split (нужен nemo-speech).

Примеры::

    # быстро: переиспользовать ранее собранные наблюдения
    .venv/bin/python scripts/diag_hybrid.py --reuse /tmp/opencode/hybrid_obs.npz \\
        --stages summary,dist,pairs,merges,bridges,sweep --out-dir /tmp/opencode/diag_hybrid

    # полный сбор заново (медленно, nemo-speech)
    .venv/bin/python scripts/diag_hybrid.py --collect --stages all

Метрика качества здесь — **число кластеров и доля топ-кластера по сегментам**, а
не только число кластеров: движок может дать верное ``k``, но всё равно склеить
говорящих в один большой кластер.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_HYBRID_MAX_EMBEDDING_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT,
    DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_NEMO_SPEECH_BINARY,
    DEFAULT_NEMO_SPEECH_DEVICE,
    DEFAULT_NEMO_SPEECH_MODEL,
)
from audio_transcriber.diarization import embeddings as embedding_utils
from audio_transcriber.diarization import hybrid_engine as he
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform

# ---------------------------------------------------------------------------
# Эталон протокола (реплики по говорящим) — для грубой сверки распределения.
# ---------------------------------------------------------------------------
PROTOCOL_TURNS: dict[str, int] = {
    "Artem Vanyan": 114,
    "Шленков": 58,
    "Колупаева": 43,
    "Матвеев": 41,
    "Сафонов": 12,
    "Малюков": 9,
    "Степанов": 9,
    "Иванов": 9,
    "Берстнева": 6,
    "Александр": 2,
    "Терпигорьев": 1,
    "Старцев": 1,
}
PROTOCOL_TOTAL = sum(PROTOCOL_TURNS.values())
#: Доли эталона по убыванию (для сравнения распределений).
PROTOCOL_SHARES = np.array(
    sorted((c / PROTOCOL_TOTAL for c in PROTOCOL_TURNS.values()), reverse=True)
)

TARGET_K = (9, 12)
TARGET_TOP = (0.37, 0.50)


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """Одно наблюдение гибрида: локальный говорящий в единице анализа."""

    unit: int
    speaker: str
    vector: np.ndarray
    speech_seconds: float
    clip_seconds: float
    n_segments: int
    start: float
    end: float
    split_depth: int = 0

    @property
    def key(self) -> tuple[int, str]:
        return (self.unit, self.speaker)


def _unit_depth(unit: object) -> int:
    """Глубина дробления единицы (0 — исходное окно, >0 — подокно overload-split)."""
    return int(getattr(unit, "_diag_depth", 0))


def _segment_metadata(
    window: he.AnalysisWindow, segments: list
) -> tuple[float, int, float, float]:
    """Возвращает ``(speech_seconds, n_segments, start_abs, end_abs)``."""
    speech = sum(max(0.0, s.end - s.start) for s in segments)
    if not segments:
        return 0.0, 0, window.start / SAMPLE_RATE, window.start / SAMPLE_RATE
    start = window.start / SAMPLE_RATE + min(s.start for s in segments)
    end = window.start / SAMPLE_RATE + max(s.end for s in segments)
    return speech, len(segments), start, end


def build_observations(
    samples: np.ndarray,
    units: list,
    unit_segments: list,
    embedder: object,
    *,
    min_speaker_seconds: float,
    max_embedding_seconds: float,
) -> list[Observation]:
    """Считает эмбеддинги локальных говорящих (как :meth:`_collect_embeddings`).

    Отличие от движка — сохраняем метаданные наблюдения (длительность речи,
    число сегментов, абсолютный интервал, глубина дробления), чтобы потом
    связать «мост» с конкретным окном/говорящим и временем.
    """
    observations: list[Observation] = []
    for index, (window, segments) in enumerate(zip(units, unit_segments, strict=True)):
        for speaker, speaker_segments in he._group_by_speaker(segments).items():
            speech = sum(max(0.0, s.end - s.start) for s in speaker_segments)
            if speech < min_speaker_seconds:
                continue
            clip = he._collect_speaker_samples(
                samples, window, speaker_segments, max_seconds=max_embedding_seconds
            )
            if clip.size == 0:
                continue
            try:
                vector = np.asarray(embedder.embed(clip), dtype=np.float32)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — как в движке: мягкая деградация
                print(f"  ! эмбеддинг не посчитан ({index + 1},{speaker}): {exc}")
                continue
            _speech, n_seg, start, end = _segment_metadata(window, speaker_segments)
            observations.append(
                Observation(
                    unit=index,
                    speaker=speaker,
                    vector=vector,
                    speech_seconds=speech,
                    clip_seconds=clip.shape[0] / SAMPLE_RATE,
                    n_segments=n_seg,
                    start=start,
                    end=end,
                    split_depth=_unit_depth(window),
                )
            )
    return observations


def collect_observations(args: argparse.Namespace) -> tuple[list[Observation], list, list]:
    """Прогоняет оконный nemo-speech + эмбеддер (медленно)."""
    started = time.monotonic()
    print(f"[collect] декодирование {args.audio} ...")
    samples = load_waveform(Path(args.audio))
    print(f"[collect] waveform {samples.shape[0] / SAMPLE_RATE:.1f} с за {time.monotonic() - started:.1f} с")

    model_path = embedding_utils.resolve_embedding_model(args.embedding_model)
    if model_path is None:
        raise SystemExit(f"модель эмбеддингов недоступна: {args.embedding_model}")
    embedder = embedding_utils.SpeakerEmbedder(model_path)

    diarizer = he.HybridSpeakerDiarizer(
        args.device,
        binary=args.binary,
        model=args.nemo_model,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        min_speaker_seconds=args.min_speaker_seconds,
        max_embedding_seconds=args.max_embedding_seconds,
        embedding_model=args.embedding_model,
        embedder=embedder,
        overload_split=args.overload_split,
        subwindow_seconds=args.subwindow_seconds,
        max_split_depth=args.max_split_depth,
    )
    windows = he.plan_windows(
        samples.shape[0],
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        samples=samples,
    )
    print(f"[collect] окон: {len(windows)}")
    units, unit_segments = diarizer._resolve_units(Path(args.audio), samples, windows)
    print(f"[collect] единиц после overload-split: {len(units)} ({time.monotonic() - started:.1f} с)")
    observations = build_observations(
        samples,
        units,
        unit_segments,
        embedder,
        min_speaker_seconds=args.min_speaker_seconds,
        max_embedding_seconds=args.max_embedding_seconds,
    )
    print(f"[collect] наблюдений: {len(observations)} ({time.monotonic() - started:.1f} с)")
    return observations, units, unit_segments


def load_npz(path: Path) -> tuple[list[Observation], list, list]:
    """Загружает ранее сохранённые наблюдения/единицы из ``.npz``."""
    data = np.load(path, allow_pickle=True)
    units = list(data["units"])
    unit_segments = [list(seg) for seg in data["unit_segments"]]
    obs_windows = data["obs_windows"]
    obs_speakers = data["obs_speakers"]
    obs_vectors = data["obs_vectors"]

    has_meta = "obs_speech_seconds" in data
    observations: list[Observation] = []
    for i, (w, s, v) in enumerate(
        zip(obs_windows, obs_speakers, obs_vectors, strict=True)
    ):
        window = units[int(w)]
        segments = [seg for seg in unit_segments[int(w)] if seg.speaker_id == str(s)]
        speech, n_seg, start, end = _segment_metadata(window, segments)
        if has_meta:
            speech = float(data["obs_speech_seconds"][i])
            clip_seconds = float(data["obs_clip_seconds"][i])
            n_seg = int(data["obs_n_segments"][i])
            start = float(data["obs_start"][i])
            end = float(data["obs_end"][i])
        else:
            clip_seconds = min(speech, DEFAULT_DIARIZATION_HYBRID_MAX_EMBEDDING_SECONDS)
        observations.append(
            Observation(
                unit=int(w),
                speaker=str(s),
                vector=np.asarray(v, dtype=np.float32),
                speech_seconds=speech,
                clip_seconds=clip_seconds,
                n_segments=n_seg,
                start=start,
                end=end,
                split_depth=0,
            )
        )
    return observations, units, unit_segments


def save_npz(
    path: Path, observations: list[Observation], units: list, unit_segments: list
) -> None:
    """Сохраняет наблюдения и единицы анализа (для быстрых повторных свипов)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        units=np.array(units, dtype=object),
        unit_segments=np.array(unit_segments, dtype=object),
        obs_windows=np.array([o.unit for o in observations], dtype=int),
        obs_speakers=np.array([o.speaker for o in observations], dtype=object),
        obs_vectors=np.stack([o.vector for o in observations]).astype(np.float32)
        if observations
        else np.zeros((0, 0), dtype=np.float32),
        obs_speech_seconds=np.array([o.speech_seconds for o in observations], dtype=np.float32),
        obs_clip_seconds=np.array([o.clip_seconds for o in observations], dtype=np.float32),
        obs_n_segments=np.array([o.n_segments for o in observations], dtype=int),
        obs_start=np.array([o.start for o in observations], dtype=np.float32),
        obs_end=np.array([o.end for o in observations], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Кластеризация / метрики
# ---------------------------------------------------------------------------
def cosine_distance(matrix: np.ndarray) -> np.ndarray:
    """Косинусное расстояние ``1 - cos`` (как в движке)."""
    norm = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    unit = matrix / norm
    distance = 1.0 - unit @ unit.T
    np.fill_diagonal(distance, 0.0)
    return np.clip(distance, 0.0, 2.0)


def euclidean_distance(matrix: np.ndarray) -> np.ndarray:
    """Евклидово расстояние на L2-нормированных векторах."""
    norm = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    unit = matrix / norm
    squared = np.clip(2.0 - 2.0 * unit @ unit.T, 0.0, None)
    np.fill_diagonal(squared, 0.0)
    return np.sqrt(squared)


def cluster_labels(
    matrix: np.ndarray,
    *,
    linkage: str,
    metric: str,
    threshold: float | None = None,
    n_clusters: int | None = None,
) -> np.ndarray:
    """Агломеративная кластеризация: linkage/metric + порог или число кластеров."""
    from sklearn.cluster import AgglomerativeClustering

    norm = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    unit = matrix / norm
    if metric == "cosine":
        distance = cosine_distance(matrix)
        kwargs = {"metric": "precomputed", "linkage": linkage}
        data = distance
    elif metric == "euclidean":
        kwargs = {"metric": "euclidean", "linkage": linkage}
        data = unit
    else:
        raise ValueError(f"неизвестная метрика: {metric}")
    count = int(matrix.shape[0])
    if n_clusters is not None and 1 <= n_clusters < count:
        model = AgglomerativeClustering(n_clusters=int(n_clusters), **kwargs)
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(threshold or 0.0), **kwargs
        )
    return np.asarray(model.fit_predict(data), dtype=int)


def stitch_stats(
    diarizer: he.HybridSpeakerDiarizer,
    units: list,
    unit_segments: list,
    observations: list[Observation],
    labels: np.ndarray,
) -> dict:
    """Прогоняет склейку как движок и считает статистику по кластерам."""
    obs_tuples = [(o.unit, o.speaker, o.vector) for o in observations]
    segments = diarizer._stitch(units, unit_segments, obs_tuples, labels)
    seg_count: Counter = Counter()
    airtime: defaultdict = defaultdict(float)
    for seg in segments:
        seg_count[seg.speaker_id] += 1
        airtime[seg.speaker_id] += max(0.0, seg.end - seg.start)
    total_seg = sum(seg_count.values()) or 1
    total_air = sum(airtime.values()) or 1.0
    sizes = sorted(seg_count.values(), reverse=True)
    return {
        "k": len(seg_count),
        "segments": total_seg,
        "top_seg": (sizes[0] / total_seg) if sizes else 0.0,
        "top_air": (max(airtime.values()) / total_air) if airtime else 0.0,
        "sizes": sizes,
        "seg_count": seg_count,
        "airtime": airtime,
        "segments_list": segments,
    }


def obs_shares(sizes: list[int]) -> np.ndarray:
    arr = np.asarray(sorted(sizes, reverse=True), dtype=float)
    return arr / arr.sum() if arr.size and arr.sum() else arr


def distribution_tv(sizes: list[int]) -> float:
    """Грубая метрика: L1/2 между долями кластеров и долями реплик эталона."""
    shares = obs_shares(sizes)
    length = max(len(shares), len(PROTOCOL_SHARES))
    a = np.zeros(length)
    b = np.zeros(length)
    a[: len(shares)] = shares
    b[: len(PROTOCOL_SHARES)] = PROTOCOL_SHARES
    return float(0.5 * np.abs(a - b).sum())


def target_score(k: int, top_seg: float, tv: float) -> float:
    """Эвристика близости к эталону: k в 9–12, top ~37–50%, малое TV."""
    score = 0.0
    if TARGET_K[0] <= k <= TARGET_K[1]:
        score += 1.0
    elif k < TARGET_K[0]:
        score += max(0.0, 1.0 - (TARGET_K[0] - k) / 5.0)
    else:
        score += max(0.0, 1.0 - (k - TARGET_K[1]) / 8.0)
    if TARGET_TOP[0] <= top_seg <= TARGET_TOP[1]:
        score += 1.0
    else:
        score += max(0.0, 1.0 - abs(top_seg - sum(TARGET_TOP) / 2) / 0.35)
    score += (1.0 - tv)
    return score


# ---------------------------------------------------------------------------
# Стадии
# ---------------------------------------------------------------------------
def stage_summary(observations: list[Observation]) -> None:
    print("\n" + "=" * 78)
    print("СВОДКА НАБЛЮДЕНИЙ")
    print("=" * 78)
    n = len(observations)
    print(f"наблюдений: {n}; единиц анализа: см. лог сбора")
    print("наблюдений по единицам:", dict(sorted(Counter(o.unit for o in observations).items())))
    print("локальные говорящие nemo-speech:", dict(Counter(o.speaker for o in observations)))
    speech = np.array([o.speech_seconds for o in observations])
    clip = np.array([o.clip_seconds for o in observations])
    print(
        f"речь на наблюдение: min={speech.min():.1f} p50={np.median(speech):.1f} "
        f"p90={np.percentile(speech, 90):.1f} max={speech.max():.1f} с"
    )
    print(
        f"клип на наблюдение: min={clip.min():.1f} p50={np.median(clip):.1f} "
        f"max={clip.max():.1f} с"
    )
    print("размерности векторов:", {v.vector.shape for v in observations})
    # Сколько окон имеют все 4 головы заняты (риск head-sharing).
    per_unit: dict[int, set] = defaultdict(set)
    for o in observations:
        per_unit[o.unit].add(o.speaker)
    full = sum(1 for s in per_unit.values() if len(s) >= 4)
    print(f"единиц с >=4 локальными говорящими (риск «прилипания»): {full}/{len(per_unit)}")


def stage_dist(observations: list[Observation], *, threshold: float) -> None:
    print("\n" + "=" * 78)
    print(f"РАСПРЕДЕЛЕНИЕ РАССТОЯНИЙ (average/cosine, порог {threshold})")
    print("=" * 78)
    matrix = np.stack([o.vector for o in observations])
    distance = cosine_distance(matrix)
    labels = cluster_labels(matrix, linkage="average", metric="cosine", threshold=threshold)
    n = len(observations)
    iu = np.triu_indices(n, 1)
    same = labels[iu[0]] == labels[iu[1]]
    intra, inter = distance[iu][same], distance[iu][~same]
    for name, arr in (("внутри кластера", intra), ("между кластерами", inter)):
        if arr.size:
            print(
                f"  {name:18s}: n={arr.size:4d} min={arr.min():.3f} "
                f"p25={np.percentile(arr, 25):.3f} med={np.median(arr):.3f} "
                f"p75={np.percentile(arr, 75):.3f} max={arr.max():.3f}"
            )
    print("гистограмма всех попарных расстояний:")
    counts, edges = np.histogram(distance[iu], bins=np.arange(0.0, 1.21, 0.1))
    for c, lo in zip(counts, edges[:-1], strict=True):
        print(f"  {lo:.1f}–{lo + 0.1:.1f}: {'#' * int(60 * c / max(counts.max(), 1))} {c}")


def stage_pairs(observations: list[Observation], *, top: int) -> None:
    print("\n" + "=" * 78)
    print(f"САМЫЕ БЛИЗКИЕ ПАРЫ НАБЛЮДЕНИЙ (cosine, top-{top})")
    print("=" * 78)
    matrix = np.stack([o.vector for o in observations])
    norm = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    sim = (matrix / norm) @ (matrix / norm).T
    np.fill_diagonal(sim, -1.0)
    n = len(observations)
    iu = np.triu_indices(n, 1)
    order = np.argsort(sim[iu])[::-1][:top]
    print(f"{'cos':>5} | наблюдение A <-> наблюдение B")
    for i in order:
        a, b = observations[iu[0][i]], observations[iu[1][i]]
        cross = "  <<< межоконная" if a.unit != b.unit else ""
        print(f"{sim[iu[0][i], iu[1][i]]:5.3f} | {resp(a)} <-> {resp(b)}{cross}")


def resp(o: Observation) -> str:
    return (
        f"u{o.unit:02d}/{o.speaker} t={o.start / 60:5.1f}m len={o.speech_seconds:4.1f}s "
        f"n={o.n_segments}"
    )


def stage_merges(observations: list[Observation], *, limit: int) -> None:
    print("\n" + "=" * 78)
    print("ПОРЯДОК СЛИЯНИЙ (scipy average/cosine; последние шаги = крупные кластеры)")
    print("=" * 78)
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    matrix = np.stack([o.vector for o in observations])
    distance = cosine_distance(matrix)
    condensed = squareform(distance, checks=False)
    tree = scipy_linkage(condensed, method="average")
    n = len(observations)
    # scipy: первые n-1 строк — слияния; row i соединяет узлы tree[i,0]/tree[i,1].
    # Узел >= n — ранее созданный кластер (индекс = n + номер строки).
    sizes = dict.fromkeys(range(n), 1)
    members = {i: [i] for i in range(n)}
    rows = []
    for step, row in enumerate(tree):
        a, b = int(row[0]), int(row[1])
        sa, sb = sizes[a], sizes[b]
        node = n + step
        sizes[node] = sa + sb
        members[node] = members[a] + members[b]
        rows.append((step, row[2], sa, sb, members[node]))
    print(f"шагов слияния: {len(rows)}; порог по умолчанию 0.55")
    for step, dist, sa, sb, _mem in rows[-(limit):]:
        flag = " <.55" if dist < 0.55 else ""
        print(
            f"  step {step:3d} d={dist:.3f} | {sa:2d}+{sb:2d} -> {sa + sb:3d}{flag}"
        )
    # Крупные слияния: где обе части >= 4 наблюдений.
    print("\nзначимые слияния (обе части >= 4 наблюдений):")
    for step, dist, sa, sb, mem in rows:
        if sa >= 4 and sb >= 4:
            print(f"  step {step:3d} d={dist:.3f} | {sa}+{sb} -> {sa + sb}: {_brief(mem, observations)}")


def _brief(members: list[int], observations: list[Observation], k: int = 8) -> str:
    head = ", ".join(
        f"u{observations[i].unit:02d}/{observations[i].speaker}@{observations[i].start / 60:.1f}m"
        for i in members[:k]
    )
    return head + (f" ... +{len(members) - k}" if len(members) > k else "")


def stage_bridges(
    observations: list[Observation],
    units: list,
    unit_segments: list,
    diarizer: he.HybridSpeakerDiarizer,
    *,
    threshold: float,
    fine_threshold: float,
    top: int,
) -> None:
    print("\n" + "=" * 78)
    print("«МОСТЫ»: наблюдения, связывающие будущие разные кластеры")
    print("=" * 78)
    matrix = np.stack([o.vector for o in observations])
    norm = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    unit = matrix / norm
    labels = cluster_labels(matrix, linkage="average", metric="cosine", threshold=threshold)
    fine = cluster_labels(matrix, linkage="average", metric="cosine", threshold=fine_threshold)
    coarse_stats = stitch_stats(diarizer, units, unit_segments, observations, labels)
    print(
        f"грубый порог {threshold}: k={coarse_stats['k']}, "
        f"топ-кластер {100 * coarse_stats['top_seg']:.0f}% сегментов"
    )
    print(
        f"мелкий порог {fine_threshold}: k={len(set(fine.tolist()))} кластеров "
        "(рассматриваем как «истинные» подгруппы внутри крупного)"
    )

    # Центроиды грубых кластеров и расстояние до двух ближайших.
    centroids: dict[int, np.ndarray] = {}
    for label in set(labels.tolist()):
        members = unit[labels == label]
        centroid = members.mean(axis=0)
        centroids[label] = centroid / max(np.linalg.norm(centroid), 1e-12)
    print("\nнаблюдения с малой дистанцией сразу до двух грубых кластеров (кандидаты-мосты):")
    scored = []
    for i, obs in enumerate(observations):
        dists = sorted(
            (1.0 - float(unit[i] @ centroids[label]), label)
            for label in centroids
            if label != labels[i]
        )
        if labels[i] not in centroids:
            continue
        if len(dists) >= 2:
            d1, l1 = dists[0]
            d2, l2 = dists[1]
            scored.append((d2, d1, l1, l2, i, obs))
    scored.sort(key=lambda item: (item[0], item[1]))
    for d2, d1, l1, l2, i, obs in scored[:top]:
        own = labels[i]
        print(
            f"  d(2-й кластер)={d2:.3f} (к {l2}) d(1-й)={d1:.3f} (к {l1}) | "
            f"свой c{own} | {resp(obs)}"
        )

    # Какие мелкие подгруппы вливаются в каждый крупный кластер и на каком расстоянии.
    print("\nразложение крупного кластера на мелкие подгруппы:")
    for label in sorted(set(labels.tolist()), key=lambda lab: -np.sum(labels == lab)):
        members = np.where(labels == label)[0]
        if len(members) < 4:
            continue
        fine_counts = Counter(fine[i] for i in members)
        spans = []
        for f, _c in fine_counts.most_common():
            idxs = [i for i in members if fine[i] == f]
            t0 = min(observations[i].start for i in idxs) / 60
            t1 = max(observations[i].end for i in idxs) / 60
            spans.append((f, len(idxs), t0, t1))
        print(f"  кластер c{label}: {len(members)} наблюдений, подгрупп {len(fine_counts)}")
        for f, cnt, t0, t1 in sorted(spans, key=lambda s: -s[1]):
            print(f"      подгруппа f{f}: {cnt:2d} набл., время {t0:5.1f}–{t1:5.1f} мин")

    # Реальный эффект объединения: сегменты, которые грубый порог склеивает, а мелкий — нет.
    fine_stats = stitch_stats(diarizer, units, unit_segments, observations, fine)
    print(
        f"\nсегментов: грубо {coarse_stats['segments']}, "
        f"мелко {fine_stats['segments']}; "
        f"топ-кластер грубо {100 * coarse_stats['top_seg']:.0f}%, "
        f"мелко {100 * fine_stats['top_seg']:.0f}%"
    )


def stage_sweep(
    observations: list[Observation],
    units: list,
    unit_segments: list,
    diarizer: he.HybridSpeakerDiarizer,
    *,
    out_dir: Path,
) -> None:
    print("\n" + "=" * 78)
    print("СВИП: порог × linkage × metric (число кластеров и доля топ-кластера по сегментам)")
    print("=" * 78)
    matrix = np.stack([o.vector for o in observations])

    grids = [
        ("cosine", "average", [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]),
        ("cosine", "complete", [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]),
        (
            "euclidean",
            "average",
            [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        ),
        ("euclidean", "complete", [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
        ("euclidean", "ward", [1.10, 1.15, 1.20, 1.25, 1.30, 1.35]),
    ]
    results = []
    print(
        f"{'metric':9} {'linkage':9} {'port':>6} | {'k':>3} "
        f"{'top_seg%':>8} {'top_air%':>8} {'TV':>6} {'min':>4}"
    )
    for metric, linkage, thresholds in grids:
        for threshold in thresholds:
            labels = cluster_labels(
                matrix, linkage=linkage, metric=metric, threshold=threshold
            )
            stats = stitch_stats(diarizer, units, unit_segments, observations, labels)
            tv = distribution_tv(stats["sizes"])
            results.append((metric, linkage, threshold, stats))
            print(
                f"{metric:9} {linkage:9} {threshold:>6.2f} | {stats['k']:>3} "
                f"{100 * stats['top_seg']:>8.0f} {100 * stats['top_air']:>8.0f} {tv:>6.2f} "
                f"{min(stats['sizes'], default=0):>4}"
            )
    # Форсированное число кластеров ward — как ориентир по распределению.
    for k in [8, 9, 10, 11, 12, 13, 14]:
        labels = cluster_labels(matrix, linkage="ward", metric="euclidean", n_clusters=k)
        stats = stitch_stats(diarizer, units, unit_segments, observations, labels)
        tv = distribution_tv(stats["sizes"])
        results.append(("euclidean", "ward", k, stats))
        print(
            f"{'euclidean':9} {'ward':9} {('k=' + str(k)):>6} | {stats['k']:>3} "
            f"{100 * stats['top_seg']:>8.0f} {100 * stats['top_air']:>8.0f} {tv:>6.2f} "
            f"{min(stats['sizes'], default=0):>4}"
        )

    ranked = sorted(
        results,
        key=lambda item: -target_score(
            item[3]["k"], item[3]["top_seg"], distribution_tv(item[3]["sizes"])
        ),
    )
    print("\nЛУЧШИЕ КОНФИГУРАЦИИ (эвристика k≈9–12, top≈37–50%, малое TV):")
    for metric, linkage, param, stats in ranked[:8]:
        print(
            f"  {metric}/{linkage} {param}: k={stats['k']}, "
            f"top={100 * stats['top_seg']:.0f}%, TV={distribution_tv(stats['sizes']):.2f}"
        )

    # Сводка по рекомендованному варианту сохраняется рядом для отчёта.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.txt").write_text(
        "\n".join(
            f"{met}/{lnk}/{prm}\tk={st['k']}\ttop={st['top_seg']:.3f}"
            f"\ttv={distribution_tv(st['sizes']):.3f}"
            for met, lnk, prm, st in results
        ),
        encoding="utf-8",
    )


def _collect_speaker_audio(
    samples: np.ndarray, window: he.AnalysisWindow, segments: list
) -> np.ndarray:
    """Собирает речь говорящего БЕЗ обрезки по max_seconds (для нарезки на куски)."""
    pieces = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = max(0, window.start + int(segment.start * SAMPLE_RATE))
        end = min(samples.shape[0], window.start + int(segment.end * SAMPLE_RATE))
        if end > start:
            pieces.append(samples[start:end])
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def stage_aggregation(
    args: argparse.Namespace,
    observations: list[Observation],
    units: list,
    unit_segments: list,
    diarizer: he.HybridSpeakerDiarizer,
) -> None:
    """Покусочно vs усреднение/медоид на локального говорящего.

    Для каждого локального говорящего речь режется на куски ``chunk-seconds``,
    каждый кусок эмбеддится CAM++ отдельно. Варианты агрегации:

    * ``current`` — как в движке: один вектор со склейки (эталон сравнения);
    * ``mean``    — средний (L2-нормированный) по кускам;
    * ``medoid``  — кусок, наиболее близкий к остальным (устойчив к «фоновым» кускам);
    * ``piece``   — каждый кусок отдельное наблюдение; метки кусков сводятся к
      одному (unit, speaker) большинством голосов, затем склейка как обычно.
    """
    print("\n" + "=" * 78)
    print("АГРЕГАЦИЯ ЭМБЕДДИНГОВ: покусочно vs mean/medoid на локального говорящего")
    print("=" * 78)
    print("[aggregation] декодирование аудио для повторного эмбеддинга ...")
    samples = load_waveform(Path(args.audio))
    model_path = embedding_utils.resolve_embedding_model(args.embedding_model)
    assert model_path is not None
    embedder = embedding_utils.SpeakerEmbedder(model_path)

    chunk_samples = int(args.chunk_seconds * SAMPLE_RATE)
    mean_obs: list[Observation] = []
    medoid_obs: list[Observation] = []

    for o in observations:
        window = units[o.unit]
        segments = [s for s in unit_segments[o.unit] if s.speaker_id == o.speaker]
        audio = _collect_speaker_audio(samples, window, segments)
        if audio.size == 0:
            continue
        clips = [
            audio[i : i + chunk_samples]
            for i in range(0, audio.shape[0], chunk_samples)
            if audio[i : i + chunk_samples].shape[0] >= SAMPLE_RATE // 2
        ] or [audio]
        vectors = []
        for clip in clips:
            with contextlib.suppress(Exception):
                vectors.append(np.asarray(embedder.embed(clip), dtype=np.float32))
        if not vectors:
            continue
        stack = np.stack(vectors)
        mean_vec = stack.mean(axis=0)
        mean_vec /= max(np.linalg.norm(mean_vec), 1e-12)
        sim = stack @ stack.T
        medoid = stack[int(np.argmax(sim.sum(axis=1)))]
        mean_obs.append(_clone(o, mean_vec))
        medoid_obs.append(_clone(o, medoid))

    print(
        f"наблюдений: current={len(observations)}, mean={len(mean_obs)}, "
        f"medoid={len(medoid_obs)} (chunk~{args.chunk_seconds}s)"
    )

    print(f"\n{'variant':8} {'port':>6} | {'k':>3} {'top_seg%':>8} {'TV':>6}")
    for name, obs_list in (
        ("current", list(observations)),
        ("mean", mean_obs),
        ("medoid", medoid_obs),
    ):
        if len(obs_list) < 2:
            continue
        matrix = np.stack([o.vector for o in obs_list])
        for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
            labels = cluster_labels(matrix, linkage="average", metric="cosine", threshold=threshold)
            stats = stitch_stats(diarizer, units, unit_segments, obs_list, labels)
            print(
                f"{name:8} {threshold:>6.2f} | {stats['k']:>3} "
                f"{100 * stats['top_seg']:>8.0f} {distribution_tv(stats['sizes']):>6.2f}"
            )

    # piece: кластеризуем куски, затем большинством голосов сворачиваем к ключу.
    chunk_vectors, chunk_key_map = _piece_chunks(
        observations, units, unit_segments, samples, embedder, chunk_samples
    )
    if len(chunk_vectors) >= 2:
        print(f"кусков для piece: {len(chunk_vectors)}")
        matrix = np.stack(chunk_vectors)
        for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
            chunk_labels = cluster_labels(
                matrix, linkage="average", metric="cosine", threshold=threshold
            )
            reduced_obs, reduced_labels = _reduce_piece(
                observations, chunk_key_map, chunk_labels
            )
            stats = stitch_stats(diarizer, units, unit_segments, reduced_obs, reduced_labels)
            # согласованность: доля ключей, чьи куски в одном кластере.
            per_key: dict[tuple[int, str], set[int]] = defaultdict(set)
            for (key, lab) in zip(chunk_key_map, chunk_labels.tolist(), strict=True):
                per_key[key].add(int(lab))
            consistent = sum(1 for s in per_key.values() if len(s) == 1) / max(len(per_key), 1)
            print(
                f"{'piece':8} {threshold:>6.2f} | {stats['k']:>3} "
                f"{100 * stats['top_seg']:>8.0f} {distribution_tv(stats['sizes']):>6.2f}"
                f"   согласованность ключей={100 * consistent:.0f}%"
            )


def _piece_chunks(
    observations: list[Observation],
    units: list,
    unit_segments: list,
    samples: np.ndarray,
    embedder: object,
    chunk_samples: int,
) -> tuple[list[np.ndarray], list[tuple[int, str]]]:
    """Эмбеддинги отдельных кусков речи + ключ ``(unit, speaker)`` для каждого."""
    vectors: list[np.ndarray] = []
    keys: list[tuple[int, str]] = []
    for o in observations:
        window = units[o.unit]
        segments = [s for s in unit_segments[o.unit] if s.speaker_id == o.speaker]
        audio = _collect_speaker_audio(samples, window, segments)
        if audio.size == 0:
            continue
        clips = [
            audio[i : i + chunk_samples]
            for i in range(0, audio.shape[0], chunk_samples)
            if audio[i : i + chunk_samples].shape[0] >= SAMPLE_RATE // 2
        ] or [audio]
        for clip in clips:
            with contextlib.suppress(Exception):
                vectors.append(np.asarray(embedder.embed(clip), dtype=np.float32))
                keys.append((o.unit, o.speaker))
    return vectors, keys


def _reduce_piece(
    observations: list[Observation],
    chunk_keys: list[tuple[int, str]],
    chunk_labels: np.ndarray,
) -> tuple[list[Observation], np.ndarray]:
    """Сворачивает метки кусков к одному наблюдению на ключ (голосование)."""
    by_key: dict[tuple[int, str], list[int]] = defaultdict(list)
    for key, lab in zip(chunk_keys, chunk_labels.tolist(), strict=True):
        by_key[key].append(int(lab))
    reduced_obs: list[Observation] = []
    reduced_labels: list[int] = []
    for o in observations:
        labels = by_key.get((o.unit, o.speaker))
        if not labels:
            continue
        majority = Counter(labels).most_common(1)[0][0]
        reduced_obs.append(o)
        reduced_labels.append(majority)
    return reduced_obs, np.asarray(reduced_labels, dtype=int)


def _clone(o: Observation, vector: np.ndarray, suffix: str = "") -> Observation:
    return Observation(
        unit=o.unit,
        speaker=o.speaker + suffix,
        vector=np.asarray(vector, dtype=np.float32),
        speech_seconds=o.speech_seconds,
        clip_seconds=o.clip_seconds,
        n_segments=o.n_segments,
        start=o.start,
        end=o.end,
        split_depth=o.split_depth,
    )


def stage_window(args: argparse.Namespace) -> None:
    """1–2 медленных прогона движка с другим окном/overload-split (nemo-speech)."""
    print("\n" + "=" * 78)
    print("ПРОГОНЫ ДВИЖКА: окно / overlap / overload-split")
    print("=" * 78)
    samples = load_waveform(Path(args.audio))
    model_path = embedding_utils.resolve_embedding_model(args.embedding_model)
    assert model_path is not None
    embedder = embedding_utils.SpeakerEmbedder(model_path)
    from collections import Counter as _Counter

    overloads = [
        part.strip().lower() in ("1", "true", "on", "yes")
        for part in args.window_overloads.split(",")
        if part.strip()
    ] or [True]
    for window in [float(x) for x in args.window_runs.split(",") if x]:
        for overload in overloads:
            started = time.monotonic()
            diarizer = he.HybridSpeakerDiarizer(
                args.device,
                binary=args.binary,
                model=args.nemo_model,
                window_seconds=window,
                overlap_seconds=args.overlap_seconds,
                min_speaker_seconds=args.min_speaker_seconds,
                max_embedding_seconds=args.max_embedding_seconds,
                embedding_model=args.embedding_model,
                embedder=embedder,
                overload_split=overload,
                subwindow_seconds=args.subwindow_seconds,
                max_split_depth=args.max_split_depth,
            )
            # Явно задаём порог (дефолт/веб-настройки могут отличаться).
            diarizer._threshold = args.sweep_threshold
            segs = diarizer.diarize(Path(args.audio), waveform=samples)
            cnt = _Counter(s.speaker_id for s in segs)
            total = sum(cnt.values()) or 1
            top = max(cnt.values()) if cnt else 0
            print(
                f"window={window:.0f}s overload={overload} thr={args.sweep_threshold}: "
                f"k={len(cnt)}, top_seg={100 * top / total:.0f}%, "
                f"({time.monotonic() - started:.0f} с)"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path("/home/user/LLM/установка транскрибера и ui")
    parser.add_argument(
        "--audio",
        default=str(base / "PyAudioTranscriptor/web-data/uploads/Отказоустойчивость_28_09.mp4"),
    )
    parser.add_argument("--out-dir", default="/tmp/opencode/diag_hybrid")
    parser.add_argument("--obs-cache", default="")
    parser.add_argument("--reuse", default="", help="готовый npz с observations (без nemo-speech)")
    parser.add_argument("--collect", action="store_true", help="собрать наблюдения заново")
    parser.add_argument(
        "--stages",
        default="summary,dist,pairs,merges,bridges,sweep",
        help="all | список стадий через запятую",
    )
    parser.add_argument("--device", default=DEFAULT_NEMO_SPEECH_DEVICE)
    parser.add_argument("--binary", default=DEFAULT_NEMO_SPEECH_BINARY)
    parser.add_argument("--nemo-model", default=DEFAULT_NEMO_SPEECH_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_DIARIZATION_ESTIMATE_MODEL)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS)
    parser.add_argument("--min-speaker-seconds", type=float, default=DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS)
    parser.add_argument("--max-embedding-seconds", type=float, default=DEFAULT_DIARIZATION_HYBRID_MAX_EMBEDDING_SECONDS)
    parser.add_argument("--subwindow-seconds", type=float, default=DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS)
    parser.add_argument("--max-split-depth", type=int, default=1)
    parser.add_argument("--overload-split", action=argparse.BooleanOptionalAction, default=DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT)
    parser.add_argument("--threshold", type=float, default=0.55, help="рабочий порог для dist/bridges")
    parser.add_argument("--fine-threshold", type=float, default=0.40)
    parser.add_argument("--sweep-threshold", type=float, default=0.55, help="порог для stage=window")
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--pairs-top", type=int, default=25)
    parser.add_argument("--bridges-top", type=int, default=20)
    parser.add_argument("--merges-limit", type=int, default=20)
    parser.add_argument("--window-runs", default="60,120")
    parser.add_argument(
        "--window-overloads",
        default="true",
        help="какие значения overload-split прогонять в stage=window (true,false)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    cache_path = Path(args.obs_cache) if args.obs_cache else out_dir / "obs.npz"

    stages = (
        {"summary", "dist", "pairs", "merges", "bridges", "sweep", "aggregation", "window"}
        if args.stages.strip() == "all"
        else {part.strip() for part in args.stages.split(",") if part.strip()}
    )

    source = args.reuse or (str(cache_path) if cache_path.exists() and not args.collect else "")
    if source and Path(source).exists():
        print(f"[load] наблюдения из {source}")
        observations, units, unit_segments = load_npz(Path(source))
    else:
        observations, units, unit_segments = collect_observations(args)
        save_npz(cache_path, observations, units, unit_segments)
        print(f"[save] {cache_path}")

    if not observations:
        print("нет наблюдений — нечего анализировать", file=sys.stderr)
        return 2

    # Нейтральный экземпляр: только для _stitch (не создаёт subprocess/эмбеддер).
    diarizer = he.HybridSpeakerDiarizer(
        args.device, binary=args.binary, model=args.nemo_model, embedder=object()
    )

    if "summary" in stages:
        stage_summary(observations)
    if "dist" in stages:
        stage_dist(observations, threshold=args.threshold)
    if "pairs" in stages:
        stage_pairs(observations, top=args.pairs_top)
    if "merges" in stages:
        stage_merges(observations, limit=args.merges_limit)
    if "bridges" in stages:
        stage_bridges(
            observations,
            units,
            unit_segments,
            diarizer,
            threshold=args.threshold,
            fine_threshold=args.fine_threshold,
            top=args.bridges_top,
        )
    if "sweep" in stages:
        stage_sweep(observations, units, unit_segments, diarizer, out_dir=out_dir)
    if "aggregation" in stages:
        stage_aggregation(args, observations, units, unit_segments, diarizer)
    if "window" in stages:
        stage_window(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
