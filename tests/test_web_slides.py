"""Тесты эндпоинтов слайдов (``/api/jobs/{id}/slides``).

Извлечение подменяется фейковой функцией (пишет sidecar с картинками), поэтому
реальный OCR/декодирование не запускаются. Наличие видеодорожки эмулируется
подменой ``has_video_stream``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.video.ocr import OcrUnavailableError
from audio_transcriber.video.slides import Slide, SlidesArtifacts, slides_directory
from audio_transcriber.web import app as app_module
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths

#: Минимальные байты-заглушки PNG (содержимое не проверяется, важен media-type).
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-slide-body"


@pytest.fixture(autouse=True)
def _video_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию исходный файл считается видео (подменяет probe)."""
    monkeypatch.setattr(app_module, "has_video_stream", lambda _path: True)


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        speaker = Speaker(id="SPEAKER_00", display_name="Спикер 1")
        entry = TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[entry],
            speakers=[speaker],
            low_confidence_threshold=-1.0,
        )

    return pipeline


@pytest.fixture
def config_builder(web_paths: WebPaths):
    def build(job_id: str, source_path: Path) -> AppConfig:
        return AppConfig(
            input_file=source_path,
            output_dir=web_paths.results_dir / job_id,
            export_formats=(ExportFormat.TXT,),
            denoise=False,
            diarization_enabled=False,
            export_speaker_samples=False,
            notifications=False,
            timeline=False,
            protocol_auto=False,
            use_cache=False,
        )

    return build


def _make_slides_fn():
    def build(config: AppConfig, *, on_progress=None, **_kwargs: object) -> SlidesArtifacts:
        directory = slides_directory(config.output_dir, config.input_file)
        directory.mkdir(parents=True, exist_ok=True)
        slides = (
            Slide(0, 0.0, 5.0, "первый слайд", "fake", "slide-000.png"),
            Slide(1, 5.0, None, "второй слайд", "fake", "slide-001.png"),
        )
        for slide in slides:
            (directory / slide.image).write_bytes(PNG_BYTES)
        metadata = {
            "version": 1,
            "engine": "fake",
            "slides": [slide.as_dict() for slide in slides],
        }
        (directory / "slides.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        return SlidesArtifacts(slides=slides, directory=directory)

    return build


def _client(web_paths: WebPaths, fake_pipeline, config_builder, slides_fn) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        slides_fn=slides_fn,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client(web_paths: WebPaths, fake_pipeline, config_builder) -> Iterator[TestClient]:
    yield from _client(web_paths, fake_pipeline, config_builder, _make_slides_fn())


@pytest.fixture
def client_no_engine(web_paths: WebPaths, fake_pipeline, config_builder) -> Iterator[TestClient]:
    def failing(config: AppConfig, **kwargs: object) -> SlidesArtifacts:
        raise OcrUnavailableError("Не найден OCR-движок: установите RapidOCR или tesseract")

    yield from _client(web_paths, fake_pipeline, config_builder, failing)


def _enable_slides(client: TestClient) -> None:
    response = client.put("/api/settings", json={"slides_enabled": True})
    assert response.status_code == 200, response.text
    assert response.json()["slides_enabled"] is True


def _upload(client: TestClient, name: str = "sample.mp4") -> dict:
    response = client.post("/api/files/upload", files={"file": (name, b"\x00\x01", "video/mp4")})
    assert response.status_code == 201
    return response.json()


def _run_job(client: TestClient, path: str) -> str:
    created = client.post("/api/jobs", json={"path": path})
    assert created.status_code == 201
    job_id = created.json()["id"]
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        details = client.get(f"/api/jobs/{job_id}").json()
        if details["status"] in {"done", "error"}:
            assert details["status"] == "done", details.get("error")
            return job_id
        time.sleep(0.02)
    raise AssertionError("задача не завершилась за отведённое время")


def test_slides_disabled_by_default(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    assert client.post(f"/api/jobs/{job_id}/slides").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/slides").status_code == 404


def test_slides_post_and_get(client: TestClient) -> None:
    _enable_slides(client)
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    response = client.post(f"/api/jobs/{job_id}/slides")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 2
    assert [slide["index"] for slide in body["slides"]] == [0, 1]
    assert body["slides"][0]["image"] == f"/api/jobs/{job_id}/slides/0/image"
    assert body["slides"][1]["end"] is None

    stored = client.get(f"/api/jobs/{job_id}/slides")
    assert stored.status_code == 200
    assert stored.json() == body


def test_slide_image(client: TestClient) -> None:
    _enable_slides(client)
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    assert client.post(f"/api/jobs/{job_id}/slides").status_code == 200

    image = client.get(f"/api/jobs/{job_id}/slides/0/image")

    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content == PNG_BYTES

    assert client.get(f"/api/jobs/{job_id}/slides/9/image").status_code == 404


def test_get_before_extraction_returns_404(client: TestClient) -> None:
    _enable_slides(client)
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    assert client.get(f"/api/jobs/{job_id}/slides").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/slides/0/image").status_code == 404


def test_unknown_job_returns_404(client: TestClient) -> None:
    _enable_slides(client)

    assert client.post("/api/jobs/unknown/slides").status_code == 404
    assert client.get("/api/jobs/unknown/slides").status_code == 404
    assert client.get("/api/jobs/unknown/slides/0/image").status_code == 404


def test_slides_without_video_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_slides(client)
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    monkeypatch.setattr(app_module, "has_video_stream", lambda _path: False)

    assert client.post(f"/api/jobs/{job_id}/slides").status_code == 404


def test_slides_without_result_returns_404(client: TestClient) -> None:
    _enable_slides(client)
    uploaded = _upload(client)
    created = client.post("/api/jobs", json={"path": uploaded["name"]})
    job_id = created.json()["id"]

    assert client.post(f"/api/jobs/{job_id}/slides").status_code == 404


def test_slides_without_ocr_engine_returns_503(client_no_engine: TestClient) -> None:
    _enable_slides(client_no_engine)
    uploaded = _upload(client_no_engine)
    job_id = _run_job(client_no_engine, uploaded["name"])

    response = client_no_engine.post(f"/api/jobs/{job_id}/slides")

    assert response.status_code == 503
    assert "OCR" in response.json()["detail"]
