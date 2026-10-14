#!/usr/bin/env python
"""Харнесс калибровки кластеризации движка диаризации ``sherpa`` (#133).

Скрипт НЕ является частью пакета ``audio_transcriber`` и не влияет на конвейер:
он отдельно прогоняет конвейер sherpa (silero VAD → окна → эмбеддинги CAM++)
на отрывке аудио, **один раз** считает эмбеддинги окон и затем перебирает
пороги и linkage кластеризации, переиспользуя эти эмбеддинги (кластеризация
дёшева, эмбеддинги — нет). Так калибруется порог
``DIARIZATION_SHERPA_THRESHOLD`` отдельно от порога оценщика числа говорящих.
Запуск из корня проекта (нужен ``sherpa-onnx`` и модель CAM++):

    .venv/bin/python scripts/calibrate_sherpa_diarization.py \\
        --input web-data/uploads/вход.webm --duration 180 --expected-speakers 2 \\
        --thresholds 0.5,0.7,0.9,1.0,1.1 --linkages complete,ward

Прокси-метрики (покрытие речи, доля мелких сегментов) переиспользуются из
:mod:`scripts.tune_diarization`; при известном числе говорящих сводка
показывает ближайшую к эталону точку сетки.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from audio_transcriber.diarization import embeddings as embedding_utils
from audio_transcriber.diarization.hybrid_engine import (
    _assign_global_clusters,
    _merge_adjacent,
)
from audio_transcriber.diarization.sherpa_engine import (
    SHERPA_MIN_TAIL_SECONDS,
    SHERPA_WINDOW_SECONDS,
    plan_speech_windows,
)
from audio_transcriber.diarization.speaker_count import (
    detect_speech_spans,
    resolve_vad_model,
)
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from scripts.tune_diarization import (
    ProxyMetrics,
    Turn,
    compute_proxy_metrics,
    parse_float_grid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Допустимые linkage кластеризации sherpa.
VALID_LINKAGES = ("complete", "ward", "average")


@dataclass(frozen=True, slots=True)
class Excerpt:
    """Отрывок: окна речи и эмбеддинг каждого окна."""

    name: str
    waveform: np.ndarray
    windows: list[tuple[int, int]]
    embeddings: np.ndarray


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """Результат одной точки сетки ``(threshold, linkage)``."""

    threshold: float
    linkage: str
    n_clusters: int
    segments: int
    proxy: ProxyMetrics


def plan_observations(
    vectors: np.ndarray,
) -> list[tuple[int, str, np.ndarray]]:
    """Оборачивает эмбеддинги окон в наблюдения для ``_assign_global_clusters``."""
    matrix = np.asarray(vectors, dtype=np.float32)
    return [(index, "spk", matrix[index]) for index in range(matrix.shape[0])]


def cluster_count(
    vectors: np.ndarray,
    *,
    threshold: float,
    linkage: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> int:
    """Число кластеров у заданного порога/линкажа (без склейки сегментов)."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape[0] == 0:
        return 0
    labels = _assign_global_clusters(
        plan_observations(matrix),
        threshold=threshold,
        num_speakers=num_speakers,
        expected_speakers=None,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        linkage=linkage,
    )
    return len(set(labels.tolist()))


def excerpt_segments(
    excerpt: Excerpt,
    *,
    threshold: float,
    linkage: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Собирает реплики ``(start, end, speaker)`` из окон и меток кластеров."""
    labels = _assign_global_clusters(
        plan_observations(excerpt.embeddings),
        threshold=threshold,
        num_speakers=num_speakers,
        expected_speakers=None,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        linkage=linkage,
    )
    segments = [
        SpeakerSegment(
            start=start / SAMPLE_RATE,
            end=end / SAMPLE_RATE,
            speaker_id=f"SPEAKER_{int(label):02d}",
        )
        for (start, end), label in zip(excerpt.windows, labels.tolist(), strict=True)
    ]
    return [
        Turn(start=segment.start, end=segment.end, speaker=segment.speaker_id)
        for segment in _merge_adjacent(segments)
    ]


def sweep(
    excerpt: Excerpt,
    *,
    thresholds: Sequence[float],
    linkages: Sequence[str],
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SweepPoint]:
    """Перебирает точки сетки, переиспользуя эмбеддинги отрывка."""
    points: list[SweepPoint] = []
    for linkage in linkages:
        for threshold in thresholds:
            segments = excerpt_segments(
                excerpt,
                threshold=threshold,
                linkage=linkage,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            proxy = compute_proxy_metrics(segments, excerpt.waveform, SAMPLE_RATE)
            points.append(
                SweepPoint(
                    threshold=threshold,
                    linkage=linkage,
                    n_clusters=proxy.n_speakers,
                    segments=len(segments),
                    proxy=proxy,
                )
            )
    return points


def load_excerpt(
    path: Path,
    *,
    start: float,
    duration: float | None,
    name: str | None = None,
) -> tuple[str, np.ndarray]:
    """Декодирует файл и возвращает моно-отрывок 16 кГц float32."""
    waveform = load_waveform(path)
    first = max(0, round(start * SAMPLE_RATE))
    if duration is None:
        excerpt = waveform[first:]
    else:
        last = min(waveform.size, first + round(duration * SAMPLE_RATE))
        excerpt = waveform[first:last]
    return (name or path.name, excerpt)


def embed_excerpt(
    name: str,
    waveform: np.ndarray,
    *,
    vad_model: Path,
    embedder: object,
    window_seconds: float = SHERPA_WINDOW_SECONDS,
) -> Excerpt:
    """Считает эмбеддинги окон речи отрывка (дорогая часть — один раз)."""
    spans = detect_speech_spans(waveform, vad_model)
    windows = plan_speech_windows(
        spans, window_seconds=window_seconds, min_tail_seconds=SHERPA_MIN_TAIL_SECONDS
    )
    vectors: list[np.ndarray] = []
    used: list[tuple[int, int]] = []
    for start, end in windows:
        clip = waveform[start:end]
        if clip.size == 0:
            continue
        vectors.append(np.asarray(embedder.embed(clip), dtype=np.float32))  # type: ignore[attr-defined]
        used.append((start, end))
    matrix = (
        np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
    )
    return Excerpt(name=name, waveform=waveform, windows=used, embeddings=matrix)


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов харнесса."""
    parser = argparse.ArgumentParser(
        description="Калибровка порога/linkage кластеризации движка sherpa на отрывке.",
    )
    parser.add_argument("--input", required=True, help="Исходный аудио/видеофайл.")
    parser.add_argument("--start", type=float, default=0.0, help="Начало отрывка, сек.")
    parser.add_argument(
        "--duration", type=float, default=180.0, help="Длительность отрывка, сек (0 — весь файл)."
    )
    parser.add_argument(
        "--expected-speakers",
        type=int,
        default=None,
        help="Эталонное число говорящих (только для сводки).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Форсировать точное число кластеров (жёсткая кластеризация).",
    )
    parser.add_argument(
        "--thresholds",
        type=_grid_argument,
        default=(0.5, 0.7, 0.9, 1.0, 1.1),
        help="Список порогов кластеризации, напр. 0.5,0.9,1.0.",
    )
    parser.add_argument(
        "--linkages",
        default="complete,ward",
        help="Список linkage через запятую: complete, ward, average.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Путь к ONNX-модели эмбеддингов (по умолчанию CAM++ из кэша/релиза).",
    )
    return parser


def _grid_argument(text: str) -> tuple[float, ...]:
    try:
        values = parse_float_grid(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"некорректный список чисел: {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("список чисел пуст")
    return values


def parse_linkages(text: str) -> tuple[str, ...]:
    """Разбирает строку ``"complete,ward"`` в кортеж допустимых linkage."""
    values = tuple(chunk.strip().casefold() for chunk in text.split(",") if chunk.strip())
    if not values:
        raise ValueError("список linkage пуст")
    unknown = [value for value in values if value not in VALID_LINKAGES]
    if unknown:
        raise ValueError(
            f"неизвестный linkage: {', '.join(unknown)} "
            f"(допустимо: {', '.join(VALID_LINKAGES)})"
        )
    return values


def _console() -> object:
    from rich.console import Console

    return Console(width=180)


def _print_table(points: Sequence[SweepPoint], expected: int | None) -> None:
    from rich.table import Table

    table = Table(title="Калибровка кластеризации sherpa")
    table.add_column("linkage")
    table.add_column("threshold", justify="right")
    table.add_column("n_spk", justify="right")
    table.add_column("n_seg", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("short<0.5", justify="right")
    table.add_column("max_gap, s", justify="right")
    for point in points:
        table.add_row(
            point.linkage,
            f"{point.threshold:.2f}",
            str(point.n_clusters),
            str(point.segments),
            f"{point.proxy.speech_coverage:.1%}",
            f"{point.proxy.short_fraction:.1%}",
            f"{point.proxy.max_gap:.2f}",
        )
    console = _console()
    console.print(table)
    if expected is not None and points:
        closest = min(points, key=lambda p: abs(p.n_clusters - expected))
        console.print(
            f"Ближе всего к эталону {expected}: linkage={closest.linkage}, "
            f"threshold={closest.threshold:.2f} → {closest.n_clusters} "
            f"(отклонение {abs(closest.n_clusters - expected)})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа харнесса. Возвращает код выхода процесса."""
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Ошибка: файл не найден: {input_path}", flush=True)
        return 2
    try:
        linkages = parse_linkages(args.linkages)
    except ValueError as exc:
        print(f"Ошибка: {exc}", flush=True)
        return 2

    duration = args.duration if args.duration and args.duration > 0 else None
    name, waveform = load_excerpt(
        input_path, start=args.start, duration=duration
    )

    vad_model = resolve_vad_model(None)
    if vad_model is None:
        print("Ошибка: модель silero VAD недоступна (проверьте кэш/сеть).", flush=True)
        return 2
    model_path = embedding_utils.resolve_embedding_model(args.model or embedding_utils.DEFAULT_EMBEDDING_MODEL)
    if model_path is None:
        print("Ошибка: модель эмбеддингов недоступна (sherpa-onnx/сеть).", flush=True)
        return 2

    console = _console()
    console.print(
        f"[bold]Файл:[/bold] {input_path} | отрывок {args.start:.1f}–"
        f"{waveform.size / SAMPLE_RATE:.1f} с | эталон N={args.expected_speakers}"
    )
    started = time.perf_counter()
    embedder = embedding_utils.SpeakerEmbedder(model_path)
    excerpt = embed_excerpt(name, waveform, vad_model=vad_model, embedder=embedder)
    console.print(
        f"Эмбеддингов: {excerpt.embeddings.shape[0]} окон "
        f"за {time.perf_counter() - started:.1f} с"
    )
    if excerpt.embeddings.shape[0] == 0:
        console.print("[yellow]Речь не найдена — калибровать нечего.[/yellow]")
        return 0

    points = sweep(
        excerpt,
        thresholds=args.thresholds,
        linkages=linkages,
        num_speakers=args.num_speakers,
    )
    _print_table(points, args.expected_speakers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
