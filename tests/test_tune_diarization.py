"""Тесты харнесса тюнинга диаризации (``scripts/tune_diarization.py``).

Проверяются только «чистые» функции: разбор сетки, прокси-метрики и парсинг
RTTM. Реальные модели и аудио здесь не используются — прогоны харнесса
выполняются вручную.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tune_diarization import (
    GridPoint,
    Turn,
    build_grid,
    build_parser,
    compute_proxy_metrics,
    count_speakers,
    covered_frames,
    max_gap_in_speech,
    params_for,
    parse_float_grid,
    parse_rttm_lines,
    reference_metrics,
    segment_durations,
    shift_and_clip,
    short_fraction,
    speech_coverage,
    speech_frame_mask,
)

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# Разбор сетки
# --------------------------------------------------------------------------- #
def test_parse_float_grid_trims_and_ignores_empty() -> None:
    assert parse_float_grid(" 0.5, 0.6 ,,0.7 ") == (0.5, 0.6, 0.7)


def test_parse_float_grid_single_value() -> None:
    assert parse_float_grid("1.5") == (1.5,)


def test_parse_float_grid_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        parse_float_grid("0.5,abc")


def test_parser_rejects_empty_grid() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--input", "x", "--thresholds", ",,"])


def test_build_grid_cartesian_product() -> None:
    grid = build_grid((0.5, 0.6), (0.8, 1.5))
    assert grid == [
        GridPoint(0.5, 0.8, None),
        GridPoint(0.5, 1.5, None),
        GridPoint(0.6, 0.8, None),
        GridPoint(0.6, 1.5, None),
    ]


def test_build_grid_with_min_duration_offs() -> None:
    grid = build_grid((0.6,), (0.8,), (0.0, 0.5))
    assert grid == [
        GridPoint(0.6, 0.8, 0.0),
        GridPoint(0.6, 0.8, 0.5),
    ]


def test_params_for_uses_default_min_duration_off() -> None:
    params = params_for(GridPoint(0.5, 1.5, None))
    assert params == {
        "clustering": {"threshold": 0.5, "Fb": 1.5},
        "segmentation": {"min_duration_off": 0.0},
    }


def test_params_for_keeps_explicit_min_duration_off() -> None:
    params = params_for(GridPoint(0.5, 1.5, 0.3))
    assert params["segmentation"] == {"min_duration_off": 0.3}


def test_argparse_grid_argument_type() -> None:
    args = build_parser().parse_args(
        ["--input", "x", "--thresholds", "0.5,0.7", "--fbs", "0.8"]
    )
    assert args.thresholds == (0.5, 0.7)
    assert args.fbs == (0.8,)


# --------------------------------------------------------------------------- #
# Метрики по сегментам
# --------------------------------------------------------------------------- #
def test_segment_durations_and_count_and_short_fraction() -> None:
    segments = [
        Turn(0.0, 2.0, "A"),
        Turn(2.0, 2.3, "A"),  # 0.3 c — мелкий
        Turn(3.0, 4.0, "B"),
    ]
    assert list(segment_durations(segments)) == [2.0, pytest.approx(0.3), 1.0]
    assert count_speakers(segments) == 2
    assert short_fraction(segments) == pytest.approx(1 / 3)


def test_short_fraction_without_segments() -> None:
    assert short_fraction([]) == 0.0


# --------------------------------------------------------------------------- #
# Энергетический VAD-прокси
# --------------------------------------------------------------------------- #
def _tone(seconds: float, *, amplitude: float = 0.5, frequency: float = 200.0) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_speech_frame_mask_separates_silence_and_tone() -> None:
    waveform = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), _tone(1.0)])
    mask, frame_seconds = speech_frame_mask(waveform, SAMPLE_RATE)

    assert frame_seconds == pytest.approx(0.02)
    assert mask.size == 100
    assert not mask[:50].any()
    assert mask[50:].all()


def test_speech_frame_mask_handles_empty_waveform() -> None:
    mask, _ = speech_frame_mask(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    assert mask.size == 0


# --------------------------------------------------------------------------- #
# Покрытие и «дырки»
# --------------------------------------------------------------------------- #
def test_covered_frames_marks_overlapping_frames() -> None:
    mask = np.ones(10, dtype=bool)
    covered = covered_frames(mask, 1.0, [Turn(2.0, 4.0, "A")])
    assert list(covered) == [False, False, True, True, False, False, False, False, False, False]


def test_speech_coverage_partial_and_none() -> None:
    mask = np.array([True, True, True, True, True, False, False, True, True, True])
    assert speech_coverage(mask, 1.0, [Turn(0.0, 3.0, "A")]) == pytest.approx(3 / 8)
    assert speech_coverage(mask, 1.0, []) == 0.0
    assert speech_coverage(mask, 1.0, [Turn(0.0, 10.0, "A")]) == 1.0


def test_max_gap_in_speech_returns_largest_hole_inside_speech() -> None:
    # Речь: кадры 0-4 и 7-9. Сегмент покрывает только 0-2.
    mask = np.array([True, True, True, True, True, False, False, True, True, True])
    gap = max_gap_in_speech(mask, 1.0, [Turn(0.0, 3.0, "A")])
    # В первой речевой зоне дырка 2 с (кадры 3-4), во второй — 3 с (7-9).
    assert gap == pytest.approx(3.0)


def test_max_gap_in_speech_zero_when_fully_covered() -> None:
    mask = np.ones(10, dtype=bool)
    assert max_gap_in_speech(mask, 1.0, [Turn(0.0, 10.0, "A")]) == 0.0


def test_max_gap_in_speech_no_speech() -> None:
    mask = np.zeros(10, dtype=bool)
    assert max_gap_in_speech(mask, 1.0, []) == 0.0


# --------------------------------------------------------------------------- #
# Агрегация прокси
# --------------------------------------------------------------------------- #
def test_compute_proxy_metrics_end_to_end() -> None:
    waveform = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), _tone(1.0)])
    segments = [Turn(0.0, 1.0, "A"), Turn(1.0, 2.0, "B")]

    proxy = compute_proxy_metrics(segments, waveform, SAMPLE_RATE)

    assert proxy.n_speakers == 2
    assert proxy.n_segments == 2
    assert proxy.median_duration == pytest.approx(1.0)
    assert proxy.short_fraction == 0.0
    assert proxy.speech_coverage == pytest.approx(1.0)
    assert proxy.max_gap == 0.0


# --------------------------------------------------------------------------- #
# RTTM и эталонные метрики
# --------------------------------------------------------------------------- #
def test_parse_rttm_lines_keeps_only_speaker_rows() -> None:
    text = (
        "SPEAKER file 1 0.000 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
        "# комментарий\n"
        "SPEAKER file 1 2.000 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
        "\n"
    )
    assert parse_rttm_lines(text) == [
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(2.0, 3.5, "SPEAKER_01"),
    ]


def test_shift_and_clip_offsets_and_trims() -> None:
    # Отрывок: [10, 15). A попадает слева на границу, B — справа, C — до отрывка.
    turns = [Turn(0.0, 5.0, "C"), Turn(9.0, 11.0, "A"), Turn(12.0, 25.0, "B")]
    shifted = shift_and_clip(turns, start=10.0, duration=5.0)
    assert shifted == [Turn(0.0, 1.0, "A"), Turn(2.0, 5.0, "B")]


def test_reference_metrics_identical_is_zero_der() -> None:
    turns = [Turn(0.0, 2.0, "A"), Turn(2.0, 4.0, "B")]
    scores = reference_metrics(turns, turns)
    assert scores.der == pytest.approx(0.0, abs=1e-9)
    assert scores.miss == pytest.approx(0.0, abs=1e-9)


def test_reference_metrics_empty_hypothesis_misses_everything() -> None:
    reference = [Turn(0.0, 2.0, "A")]
    scores = reference_metrics(reference, [])
    assert scores.der == pytest.approx(1.0)
    assert scores.miss == pytest.approx(1.0)
