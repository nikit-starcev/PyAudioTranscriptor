#!/usr/bin/env python
"""Харнесс для тюнинга гиперпараметров диаризации pyannote (dev-инструмент).

Скрипт НЕ является частью пакета ``audio_transcriber`` и не влияет на конвейер:
он отдельно грузит модель, отдельно вызывает ``pipeline.instantiate(...)`` и
считает метрики на коротком отрывке аудио. Запуск из корня проекта:

    .venv/bin/python scripts/tune_diarization.py \\
        --input ../тест_артефакты.mp4 --start 0 --duration 170 \\
        --thresholds 0.5,0.6,0.7 --fbs 0.5,0.8,1.5

Параметры ``community-1`` (VBx-кластеризация):
  * ``clustering.threshold`` — порог решения «один и тот же говорящий»;
  * ``clustering.Fb``        — регуляризация; выше → меньше «Спикеров»;
  * ``segmentation.min_duration_off`` — склейка коротких пауз внутри реплики.

У powerset-модели ``segmentation.threshold`` отсутствует, поэтому в сетке его нет.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from audio_transcriber.diarization.energy import (
    DEFAULT_ENERGY_THRESHOLD_RATIO,
    energy_threshold,
    median_energy,
    prefix_squares,
)
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.config_env import load_config_env

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Сегмент короче этого считаем «мелким» — прокси дробления на реплики.
DEFAULT_SHORT_SEGMENT_SECONDS = 0.5

#: Длина кадра энергетического VAD-прокси (секунды).
DEFAULT_FRAME_SECONDS = 0.02

#: Значения по умолчанию, соответствующие config.yaml модели community-1.
DEFAULT_THRESHOLD = 0.6
DEFAULT_FB = 0.8


@dataclass(frozen=True)
class Turn:
    """Реплика диаризации: интервал времени и метка говорящего."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class GridPoint:
    """Одна точка сетки гиперпараметров."""

    threshold: float
    fb: float
    min_duration_off: float | None


@dataclass(frozen=True)
class ProxyMetrics:
    """Метрики качества диаризации без эталонной разметки."""

    n_speakers: int
    n_segments: int
    median_duration: float
    min_duration: float
    short_fraction: float
    speech_coverage: float
    max_gap: float
    speech_seconds: float


@dataclass(frozen=True)
class ReferenceScores:
    """Ошибки относительно эталонного RTTM (через ``pyannote.metrics``)."""

    der: float
    miss: float
    confusion: float
    false_alarm: float


@dataclass(frozen=True)
class RunResult:
    """Результат одного прогона пайплайна на отрывке."""

    point: GridPoint
    proxy: ProxyMetrics
    elapsed_seconds: float
    reference: ReferenceScores | None = None


# --------------------------------------------------------------------------- #
# Разбор аргументов и сетки
# --------------------------------------------------------------------------- #
def parse_float_grid(text: str) -> tuple[float, ...]:
    """Разбирает строку вида ``"0.5, 0.6,0.7"`` в кортеж чисел.

    Пустые элементы игнорируются. Бросает ``ValueError`` на нечисловое значение.
    """
    values: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return tuple(values)


def _grid_argument(text: str) -> tuple[float, ...]:
    try:
        values = parse_float_grid(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"некорректный список чисел: {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("список чисел пуст")
    return values


def build_grid(
    thresholds: Sequence[float],
    fbs: Sequence[float],
    min_duration_offs: Sequence[float] = (),
) -> list[GridPoint]:
    """Возвращает декартово произведение параметров.

    Если ``min_duration_offs`` пуст, используется ``None`` (дефолт модели), и
    результат содержит по одной точке на пару ``(threshold, Fb)``.
    """
    mdos: list[float | None] = list(min_duration_offs) if min_duration_offs else [None]
    return [
        GridPoint(threshold=threshold, fb=fb, min_duration_off=mdo)
        for threshold, fb, mdo in itertools.product(thresholds, fbs, mdos)
    ]


def params_for(point: GridPoint) -> dict[str, Any]:
    """Собирает словарь для ``pipeline.instantiate`` из точки сетки.

    ``min_duration_off`` задаётся всегда (по умолчанию 0.0), чтобы один прогон
    не «протекал» в следующий: ``instantiate`` мутирует пайплайн (setattr).
    """
    return {
        "clustering": {"threshold": point.threshold, "Fb": point.fb},
        "segmentation": {"min_duration_off": point.min_duration_off or 0.0},
    }


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов харнесса."""
    parser = argparse.ArgumentParser(
        description="Тюнинг гиперпараметров диаризации pyannote на коротком отрывке.",
    )
    parser.add_argument("--input", required=True, help="Исходный аудио/видеофайл.")
    parser.add_argument("--start", type=float, default=0.0, help="Начало отрывка, сек.")
    parser.add_argument(
        "--duration", type=float, default=120.0, help="Длительность отрывка, сек."
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Путь к локальной модели (по умолчанию PYANNOTE_LOCAL_MODEL из config.env).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Жёстко задать число говорящих для пайплайна.",
    )
    parser.add_argument(
        "--expected-speakers",
        type=int,
        default=None,
        help="Ожидаемое число говорящих (только для сравнения в сводке).",
    )
    parser.add_argument(
        "--thresholds",
        type=_grid_argument,
        default=(DEFAULT_THRESHOLD,),
        help="Список clustering.threshold, напр. 0.5,0.6,0.7.",
    )
    parser.add_argument(
        "--fbs",
        type=_grid_argument,
        default=(DEFAULT_FB,),
        help="Список clustering.Fb, напр. 0.5,0.8,1.5.",
    )
    parser.add_argument(
        "--min-duration-offs",
        type=_grid_argument,
        default=(),
        help="Список segmentation.min_duration_off (по умолчанию — дефолт модели).",
    )
    parser.add_argument(
        "--reference-rttm",
        default=None,
        help="Эталонный RTTM (времена в шкале исходного файла) для DER.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda"), help="Устройство вычислений."
    )
    return parser


# --------------------------------------------------------------------------- #
# Прокси-метрики (без эталона)
# --------------------------------------------------------------------------- #
def speech_frame_mask(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    ratio: float = DEFAULT_ENERGY_THRESHOLD_RATIO,
) -> tuple[np.ndarray, float]:
    """Энергетический VAD-прокси: какие кадры похожи на речь.

    Использует те же примитивы, что и авто-выбор образцов голоса в проекте:
    порог — доля от медианной энергии записи. Возвращает ``(mask, frame_seconds)``,
    где ``mask[i]`` — речь ли в кадре ``i`` (кадры не перекрываются).
    """
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size == 0 or sample_rate <= 0 or frame_seconds <= 0:
        return np.zeros(0, dtype=bool), frame_seconds

    prefix = prefix_squares(values)
    threshold = energy_threshold(
        median_energy(prefix, sample_rate=sample_rate), ratio=ratio
    )
    frame = max(1, round(frame_seconds * sample_rate))
    count = values.size // frame
    if count == 0:
        return np.zeros(0, dtype=bool), frame / sample_rate

    starts = np.arange(count) * frame
    energies = (prefix[starts + frame] - prefix[starts]) / frame
    return energies >= threshold, frame / sample_rate


def covered_frames(
    mask: np.ndarray, frame_seconds: float, segments: Sequence[Turn]
) -> np.ndarray:
    """Побитовая метка: попадает ли кадр хотя бы в один сегмент диаризации.

    Кадр считается покрытым, если пересекается с сегментом (по индексам).
    """
    covered = np.zeros(mask.size, dtype=bool)
    if mask.size == 0 or frame_seconds <= 0:
        return covered
    for turn in segments:
        first = max(0, int(np.floor(turn.start / frame_seconds)))
        last = min(mask.size, int(np.ceil(turn.end / frame_seconds)))
        if last > first:
            covered[first:last] = True
    return covered


def speech_coverage(
    mask: np.ndarray, frame_seconds: float, segments: Sequence[Turn]
) -> float:
    """Доля «речевых» кадров, попавших в какой-либо сегмент диаризации (0..1)."""
    speech = int(mask.sum())
    if speech == 0:
        return 0.0
    covered = covered_frames(mask, frame_seconds, segments)
    return float(np.logical_and(mask, covered).sum()) / speech


def max_gap_in_speech(
    mask: np.ndarray, frame_seconds: float, segments: Sequence[Turn]
) -> float:
    """Максимальная «дырка» (сек) внутри непрерывной речи, не покрытая диаризацией."""
    if mask.size == 0:
        return 0.0
    covered = covered_frames(mask, frame_seconds, segments)
    best = 0
    index = 0
    total = mask.size
    while index < total:
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < total and mask[end]:
            end += 1
        run = 0
        for position in range(index, end):
            run = 0 if covered[position] else run + 1
            best = max(best, run)
        index = end
    return best * frame_seconds


def segment_durations(segments: Sequence[Turn]) -> np.ndarray:
    """Длительности сегментов (сек) как массив float64."""
    return np.asarray([turn.end - turn.start for turn in segments], dtype=np.float64)


def count_speakers(segments: Sequence[Turn]) -> int:
    """Число различных меток говорящих."""
    return len({turn.speaker for turn in segments})


def short_fraction(
    segments: Sequence[Turn], *, short_seconds: float = DEFAULT_SHORT_SEGMENT_SECONDS
) -> float:
    """Доля сегментов короче ``short_seconds`` (прокси дробления)."""
    if not segments:
        return 0.0
    durations = segment_durations(segments)
    return float(np.count_nonzero(durations < short_seconds)) / durations.size


def compute_proxy_metrics(
    segments: Sequence[Turn],
    waveform: np.ndarray,
    sample_rate: int,
    *,
    short_seconds: float = DEFAULT_SHORT_SEGMENT_SECONDS,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    ratio: float = DEFAULT_ENERGY_THRESHOLD_RATIO,
) -> ProxyMetrics:
    """Считает весь набор прокси-метрик без эталонной разметки."""
    mask, frame = speech_frame_mask(
        waveform, sample_rate, frame_seconds=frame_seconds, ratio=ratio
    )
    durations = segment_durations(segments)
    if durations.size:
        median_duration = float(np.median(durations))
        min_duration = float(durations.min())
    else:
        median_duration = 0.0
        min_duration = 0.0
    return ProxyMetrics(
        n_speakers=count_speakers(segments),
        n_segments=len(segments),
        median_duration=median_duration,
        min_duration=min_duration,
        short_fraction=short_fraction(segments, short_seconds=short_seconds),
        speech_coverage=speech_coverage(mask, frame, segments),
        max_gap=max_gap_in_speech(mask, frame, segments),
        speech_seconds=float(mask.size) * frame,
    )


# --------------------------------------------------------------------------- #
# Эталонные метрики (RTTM → DER через pyannote.metrics)
# --------------------------------------------------------------------------- #
def parse_rttm_lines(text: str) -> list[Turn]:
    """Разбирает строки RTTM (только записи ``SPEAKER``) в реплики."""
    turns: list[Turn] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        turns.append(Turn(start=start, end=start + duration, speaker=parts[7]))
    return turns


def shift_and_clip(
    turns: Sequence[Turn], *, start: float, duration: float | None
) -> list[Turn]:
    """Сдвигает эталон в шкалу отрывка (``-start``) и обрезает по его границам."""
    end_limit = None if duration is None else start + duration
    result: list[Turn] = []
    for turn in turns:
        begin = max(turn.start, start)
        finish = turn.end if end_limit is None else min(turn.end, end_limit)
        if finish > begin:
            result.append(Turn(start=begin - start, end=finish - start, speaker=turn.speaker))
    return result


def reference_metrics(
    reference: Sequence[Turn], hypothesis: Sequence[Turn], *, uri: str = "excerpt"
) -> ReferenceScores:
    """DER и его компоненты через ``pyannote.metrics`` (без колларов)."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = Annotation(uri=uri)
    for turn in reference:
        ref[Segment(turn.start, turn.end)] = turn.speaker
    hyp = Annotation(uri=uri)
    for turn in hypothesis:
        hyp[Segment(turn.start, turn.end)] = turn.speaker

    metric = DiarizationErrorRate()
    detailed = metric(ref, hyp, detailed=True)
    total = float(detailed.get("total", 0.0)) or 1.0
    return ReferenceScores(
        der=float(abs(metric)),
        miss=float(detailed.get("missed detection", 0.0)) / total,
        confusion=float(detailed.get("confusion", 0.0)) / total,
        false_alarm=float(detailed.get("false alarm", 0.0)) / total,
    )


# --------------------------------------------------------------------------- #
# Запуск пайплайна
# --------------------------------------------------------------------------- #
def load_excerpt(path: Path, *, start: float, duration: float | None) -> np.ndarray:
    """Декодирует файл и возвращает моно-отрывок 16 кГц float32."""
    waveform = load_waveform(path)
    first = max(0, round(start * SAMPLE_RATE))
    if duration is None:
        return waveform[first:]
    last = min(waveform.size, first + round(duration * SAMPLE_RATE))
    return waveform[first:last]


def _load_pipeline(local_model: Path | None, device: str) -> Any:
    import torch
    from pyannote.audio import Pipeline

    if local_model is not None and local_model.is_dir():
        pipeline = Pipeline.from_pretrained(local_model)
    else:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=os.environ.get("HF_TOKEN")
        )
    if pipeline is None:
        raise RuntimeError("Не удалось загрузить модель диаризации (проверьте путь и токен)")
    pipeline.to(torch.device(device))
    return pipeline


def run_grid(
    pipeline: Any,
    waveform: np.ndarray,
    grid: Sequence[GridPoint],
    *,
    num_speakers: int | None = None,
    reference: Sequence[Turn] | None = None,
    short_seconds: float = DEFAULT_SHORT_SEGMENT_SECONDS,
    on_result: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Прогоняет пайплайн по каждой точке сетки и собирает метрики.

    ``on_result`` вызывается после каждой комбинации (для инкрементального
    прогресса на длинных прогонах).
    """
    import torch

    audio_input = {
        "waveform": torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0),
        "sample_rate": SAMPLE_RATE,
    }
    results: list[RunResult] = []
    for point in grid:
        pipeline.instantiate(params_for(point))
        started = time.perf_counter()
        output = pipeline(audio_input, num_speakers=num_speakers)
        elapsed = time.perf_counter() - started

        exclusive = getattr(output, "exclusive_speaker_diarization", output)
        turns = [
            Turn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in exclusive.itertracks(yield_label=True)
        ]
        proxy = compute_proxy_metrics(
            turns, waveform, SAMPLE_RATE, short_seconds=short_seconds
        )
        scores = reference_metrics(reference, turns) if reference else None
        result = RunResult(
            point=point, proxy=proxy, elapsed_seconds=elapsed, reference=scores
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


# --------------------------------------------------------------------------- #
# Вывод
# --------------------------------------------------------------------------- #
def _console() -> Any:
    """Консоль фиксированной ширины: широкая таблица не должна обрезаться."""
    from rich.console import Console

    return Console(width=220)


def _print_table(results: Sequence[RunResult], has_reference: bool) -> Any:
    from rich.table import Table

    table = Table(title="Тюнинг диаризации pyannote community-1")
    table.add_column("threshold", justify="right")
    table.add_column("Fb", justify="right")
    table.add_column("min_off", justify="right")
    table.add_column("n_spk", justify="right")
    table.add_column("n_seg", justify="right")
    table.add_column("med, s", justify="right")
    table.add_column("min, s", justify="right")
    table.add_column("short<0.5", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("max_gap, s", justify="right")
    table.add_column("time, s", justify="right")
    if has_reference:
        for column in ("DER", "miss", "conf", "FA"):
            table.add_column(column, justify="right")

    for result in results:
        proxy = result.proxy
        row = [
            f"{result.point.threshold:.2f}",
            f"{result.point.fb:.2f}",
            "-" if result.point.min_duration_off is None else f"{result.point.min_duration_off:.2f}",
            str(proxy.n_speakers),
            str(proxy.n_segments),
            f"{proxy.median_duration:.2f}",
            f"{proxy.min_duration:.2f}",
            f"{proxy.short_fraction:.2%}",
            f"{proxy.speech_coverage:.2%}",
            f"{proxy.max_gap:.2f}",
            f"{result.elapsed_seconds:.1f}",
        ]
        if has_reference:
            assert result.reference is not None
            row += [
                f"{result.reference.der:.2%}",
                f"{result.reference.miss:.2%}",
                f"{result.reference.confusion:.2%}",
                f"{result.reference.false_alarm:.2%}",
            ]
        table.add_row(*row)

    console = _console()
    console.print(table)
    return console


def _summarize(results: Sequence[RunResult], expected_speakers: int | None) -> None:
    if not results:
        return
    console = _console()
    best_frag = min(results, key=lambda r: (r.proxy.short_fraction, r.proxy.n_segments))
    best_cov = max(results, key=lambda r: r.proxy.speech_coverage)
    best_gap = min(results, key=lambda r: r.proxy.max_gap)

    console.print("\n[bold]Сводка[/bold]")
    console.print(
        f"  Меньше всего дробления (доля мелких сегментов): "
        f"threshold={best_frag.point.threshold:.2f}, Fb={best_frag.point.fb:.2f} → "
        f"{best_frag.proxy.short_fraction:.2%} ({best_frag.proxy.n_segments} сегментов)"
    )
    console.print(
        f"  Максимальное покрытие речи: "
        f"threshold={best_cov.point.threshold:.2f}, Fb={best_cov.point.fb:.2f} → "
        f"{best_cov.proxy.speech_coverage:.2%}"
    )
    console.print(
        f"  Минимальный максимальный разрыв: "
        f"threshold={best_gap.point.threshold:.2f}, Fb={best_gap.point.fb:.2f} → "
        f"{best_gap.proxy.max_gap:.2f} с"
    )
    if expected_speakers is not None:
        closest = min(
            results, key=lambda r: abs(r.proxy.n_speakers - expected_speakers)
        )
        console.print(
            f"  Ближе всего к ожидаемым {expected_speakers} говорящим: "
            f"threshold={closest.point.threshold:.2f}, Fb={closest.point.fb:.2f} → "
            f"{closest.proxy.n_speakers}"
        )


def _resolve_local_model(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    _, config = load_config_env()
    from_config = config.get("PYANNOTE_LOCAL_MODEL")
    return Path(from_config) if from_config else None


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа харнесса. Возвращает код выхода процесса."""
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Ошибка: файл не найден: {input_path}", flush=True)
        return 2

    local_model = _resolve_local_model(args.local_model)
    grid = build_grid(args.thresholds, args.fbs, args.min_duration_offs)
    reference: list[Turn] | None = None
    if args.reference_rttm:
        rttm_path = Path(args.reference_rttm)
        if not rttm_path.is_file():
            print(f"Ошибка: RTTM не найден: {rttm_path}", flush=True)
            return 2
        reference = shift_and_clip(
            parse_rttm_lines(rttm_path.read_text(encoding="utf-8")),
            start=args.start,
            duration=args.duration,
        )

    console = _console()
    console.print(
        f"[bold]Файл:[/bold] {input_path} | отрывок {args.start:.1f}–"
        f"{args.start + args.duration:.1f} с | модель: {local_model or 'HF hub'}"
    )
    console.print(f"[bold]Сетка:[/bold] {len(grid)} комбинаций")

    waveform = load_excerpt(input_path, start=args.start, duration=args.duration)
    console.print(
        f"[bold]Отрывок декодирован:[/bold] {waveform.size / SAMPLE_RATE:.1f} с"
    )

    load_started = time.perf_counter()
    pipeline = _load_pipeline(local_model, args.device)
    console.print(
        f"[bold]Модель загружена[/bold] за {time.perf_counter() - load_started:.1f} с"
    )

    def _report(result: RunResult) -> None:
        proxy = result.proxy
        console.print(
            f"  · threshold={result.point.threshold:.2f} Fb={result.point.fb:.2f} "
            f"→ спикеров={proxy.n_speakers} сегментов={proxy.n_segments} "
            f"покрытие={proxy.speech_coverage:.1%} "
            f"мелких={proxy.short_fraction:.1%} ({result.elapsed_seconds:.1f} с)"
        )

    results = run_grid(
        pipeline,
        waveform,
        grid,
        num_speakers=args.num_speakers,
        reference=reference,
        on_result=_report,
    )

    _print_table(results, has_reference=reference is not None)
    _summarize(results, args.expected_speakers)

    total = sum(result.elapsed_seconds for result in results)
    console.print(
        f"\n[bold]Время прогонов:[/bold] {total:.1f} с всего, "
        f"{total / len(results):.1f} с в среднем на комбинацию"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
