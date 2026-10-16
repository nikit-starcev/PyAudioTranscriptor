"""Тесты извлечения слайдов (``video.slides``) на синтетическом видео.

OCR подменяется фейковым движком: реальные Tesseract/RapidOCR не требуются.
Синтетическое видео кодируется PyAV; если у сборки нет кодера ``mpeg4``,
тесты пропускаются.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import av
import numpy as np
import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.video import slides as slides_module
from audio_transcriber.video.frames import VideoFileError
from audio_transcriber.video.ocr import OcrUnavailableError
from audio_transcriber.video.slides import (
    METADATA_NAME,
    build_slides_payload,
    extract_slides,
    read_slides_metadata,
    slide_image_name,
    slides_directory,
)


class FakeOcr:
    """OCR-заглушка: уникальный текст на каждый кадр (стабилен по порядку)."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path) -> str:
        self.calls += 1
        return f"slide number {self.calls}"


def _encode_video(path: Path, colors: tuple[tuple[int, int, int], ...] = ((220, 30, 30), (30, 220, 30), (30, 30, 220))) -> None:
    container = av.open(str(path), "w")
    try:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 160
        stream.height = 120
        stream.pix_fmt = "yuv420p"
        for color in colors:
            frame_array = np.zeros((120, 160, 3), dtype=np.uint8)
            frame_array[:, :] = color
            for _ in range(30):
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "lecture.mp4"
    try:
        _encode_video(path)
    except Exception as exc:  # noqa: BLE001 — нет кодера в сборке PyAV
        pytest.skip(f"PyAV не смог закодировать тестовое видео: {exc}")
    return path


def _config(tmp_path: Path, video: Path, *, use_cache: bool = True) -> AppConfig:
    return AppConfig(
        input_file=video,
        output_dir=tmp_path / "out",
        diarization_enabled=False,
        denoise=False,
        export_speaker_samples=False,
        notifications=False,
        timeline=False,
        protocol_auto=False,
        use_cache=use_cache,
    )


def _extract(config: AppConfig, **kwargs: object):
    params: dict[str, object] = {"interval": 0.5, "stability_seconds": 1.0}
    params.update(kwargs)
    return extract_slides(config, ocr=FakeOcr(), **params)


def test_extract_slides_returns_ordered_slides(tmp_path: Path, synthetic_video: Path) -> None:
    config = _config(tmp_path, synthetic_video)

    artifacts = _extract(config)

    assert [slide.index for slide in artifacts.slides] == [0, 1, 2]
    starts = [slide.start for slide in artifacts.slides]
    assert starts == sorted(starts)
    assert artifacts.slides[0].start == 0.0
    assert artifacts.slides[0].end is not None
    assert artifacts.slides[0].engine == "fake"
    assert artifacts.slides[0].text == "slide number 1"


def test_extract_slides_writes_sidecar(tmp_path: Path, synthetic_video: Path) -> None:
    config = _config(tmp_path, synthetic_video)

    artifacts = _extract(config)

    directory = slides_directory(config.output_dir, synthetic_video)
    assert artifacts.directory == directory
    assert (directory / METADATA_NAME).is_file()
    for slide in artifacts.slides:
        assert (directory / slide.image).is_file()
        assert slide.image == slide_image_name(slide.index)

    metadata = read_slides_metadata(directory)
    assert metadata is not None
    assert metadata["engine"] == "fake"
    assert len(metadata["slides"]) == 3  # type: ignore[arg-type]


def test_extract_slides_reuses_cache(tmp_path: Path, synthetic_video: Path) -> None:
    config = _config(tmp_path, synthetic_video)
    first = _extract(config)

    shutil.rmtree(first.directory)

    second = _extract(config)

    assert [slide.as_dict() for slide in second.slides] == [
        slide.as_dict() for slide in first.slides
    ]
    for slide in second.slides:
        assert (second.directory / slide.image).is_file()


def test_stability_drops_everything(tmp_path: Path, synthetic_video: Path) -> None:
    config = _config(tmp_path, synthetic_video, use_cache=False)

    artifacts = _extract(config, stability_seconds=100.0)

    assert artifacts.slides == ()
    metadata = read_slides_metadata(artifacts.directory)
    assert metadata is not None
    assert metadata["slides"] == []


def test_extract_slides_without_engine_raises(
    tmp_path: Path, synthetic_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, synthetic_video)

    def _no_engine(**_kwargs: object) -> object:
        raise OcrUnavailableError("нет движка")

    monkeypatch.setattr(slides_module, "find_ocr_engine", _no_engine)

    with pytest.raises(OcrUnavailableError):
        extract_slides(config, interval=0.5, stability_seconds=1.0)


def test_extract_slides_non_video_raises(tmp_path: Path) -> None:
    text = tmp_path / "note.txt"
    text.write_text("not a video", encoding="utf-8")
    config = _config(tmp_path, text)

    with pytest.raises(VideoFileError):
        _extract(config)


def test_build_slides_payload() -> None:
    metadata = {
        "version": 1,
        "engine": "tesseract",
        "slides": [
            {"index": 0, "start": 0.0, "end": 5.0, "text": "один", "engine": "tesseract", "image": "slide-000.png"},
            {"index": 1, "start": 5.0, "end": None, "text": "два", "engine": "tesseract", "image": "slide-001.png"},
        ],
    }

    payload = build_slides_payload(metadata, job_id="job-1")

    assert payload["total"] == 2
    slides = payload["slides"]
    assert isinstance(slides, list)
    assert slides[0]["image"] == "/api/jobs/job-1/slides/0/image"
    assert slides[1]["end"] is None


def test_build_slides_payload_skips_malformed() -> None:
    metadata = {"slides": ["мусор", {"index": "x"}, {"index": 2, "start": 1.0}]}

    payload = build_slides_payload(metadata, job_id="job-2")

    assert payload["total"] == 1
    assert payload["slides"][0]["index"] == 2  # type: ignore[index]


def test_read_slides_metadata_missing_and_broken(tmp_path: Path) -> None:
    directory = tmp_path / "empty.slides"
    assert read_slides_metadata(directory) is None

    directory.mkdir()
    (directory / METADATA_NAME).write_text("{broken", encoding="utf-8")
    assert read_slides_metadata(directory) is None

    (directory / METADATA_NAME).write_text(json.dumps({"slides": []}), encoding="utf-8")
    assert read_slides_metadata(directory) == {"slides": []}
