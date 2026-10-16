"""Юнит-тесты детекта сцен и отбора стабильных кадров (``video.frames``).

Чистые функции проверяются на синтетических кадрах без реального видео.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.video.frames import (
    FrameSample,
    VideoFileError,
    analysis_difference,
    frame_difference,
    has_video_stream,
    iter_occurrences,
    same_frame,
    scan_video,
    to_analysis_image,
)


def _solid(value: int, *, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), float(value), dtype=np.float32)


def _sample(timestamp: float, image: np.ndarray, *, scene_cut: bool = False) -> FrameSample:
    return FrameSample(timestamp=timestamp, analysis=image, image=image, scene_cut=scene_cut)


def test_to_analysis_image_preserves_channels() -> None:
    image = np.zeros((40, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    analysis = to_analysis_image(image, size=8)

    assert analysis.shape == (8, 8, 3)
    assert analysis[..., 0].min() == 255.0
    assert analysis[..., 1].max() == 0.0


def test_frame_difference_identical_is_zero() -> None:
    image = _solid(120, size=64)
    assert frame_difference(image, image) == 0.0
    assert same_frame(image, image, 0.12)


def test_frame_difference_sees_color_not_only_brightness() -> None:
    """Изолуминантные цвета должны различаться (каналы не усредняются)."""
    first = np.zeros((16, 16, 3), dtype=np.uint8)
    first[:, :] = (220, 30, 30)
    second = np.zeros((16, 16, 3), dtype=np.uint8)
    second[:, :] = (30, 220, 30)

    assert frame_difference(first, second) > 0.5


def test_frame_difference_catches_localized_change() -> None:
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    changed = base.copy()
    changed[0:8, :] = 255

    assert frame_difference(base, changed) > 0.5
    assert not same_frame(base, changed, 0.06)


def test_frame_difference_catches_small_template_change() -> None:
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    changed = base.copy()
    changed[0:2, 0:8] = 255

    assert not same_frame(base, changed, 0.06)


def test_frame_difference_tolerates_small_noise() -> None:
    rng = np.random.default_rng(1234)
    base = np.full((64, 64, 3), 100.0, dtype=np.float32)
    noisy = base + rng.normal(0.0, 2.0, base.shape)

    assert same_frame(base, noisy, 0.12)


def test_analysis_difference_shape_mismatch_is_max() -> None:
    assert analysis_difference(_solid(0, size=8), _solid(0, size=16)) == 1.0


def test_iter_occurrences_groups_stable_runs() -> None:
    black = _solid(0)
    white = _solid(255)
    grey = _solid(128)
    samples = [
        _sample(0.0, black),
        _sample(1.0, black),
        _sample(2.0, white),
        _sample(3.0, white),
        _sample(4.0, grey),
    ]

    frames = list(iter_occurrences(samples, change_threshold=0.12, stability_seconds=1.0, duration=5.0))

    assert [(round(f.start, 3), round(f.end, 3), round(f.timestamp, 3)) for f in frames] == [
        (0.0, 2.0, 1.0),
        (2.0, 4.0, 3.0),
    ]


def test_iter_occurrences_drops_short_runs() -> None:
    black = _solid(0)
    white = _solid(255)
    samples = [_sample(0.0, black), _sample(1.0, white)]

    frames = list(iter_occurrences(samples, change_threshold=0.12, stability_seconds=2.0, duration=5.0))

    assert frames == []


def test_iter_occurrences_scene_cut_forces_boundary() -> None:
    black = _solid(0)
    samples = [
        _sample(0.0, black),
        _sample(1.0, black),
        _sample(2.0, black, scene_cut=True),
        _sample(3.0, black),
    ]

    frames = list(iter_occurrences(samples, change_threshold=0.12, stability_seconds=1.0, duration=4.0))

    assert [(round(f.start, 3), round(f.end, 3)) for f in frames] == [(0.0, 2.0), (2.0, 4.0)]


def test_iter_occurrences_last_end_is_none_without_duration() -> None:
    black = _solid(0)
    samples = [_sample(0.0, black), _sample(1.0, black)]

    frames = list(iter_occurrences(samples, change_threshold=0.12, stability_seconds=0.5, duration=None))

    assert len(frames) == 1
    assert frames[0].end is None


def test_iter_occurrences_empty_input() -> None:
    assert list(iter_occurrences([], stability_seconds=1.0)) == []


def test_has_video_stream_missing_file(tmp_path: Path) -> None:
    assert has_video_stream(tmp_path / "absent.mp4") is False


def test_has_video_stream_text_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("not a video", encoding="utf-8")
    assert has_video_stream(path) is False


def test_scan_video_rejects_non_video(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoFileError):
        scan_video(path, tmp_path / "frames")
