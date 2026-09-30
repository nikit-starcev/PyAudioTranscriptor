"""Тесты прямой выгрузки стенограммы (``/api/jobs/{id}/export``).

Экспорт выполняется настоящими экспортёрами из ``export/*`` на текущем JSON
результата задачи. Конвейер подменяется фиктивной функцией, поэтому реальные
модели/LLM/аудио не задействуются.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths

#: Ожидаемый MIME-тип для каждого формата выгрузки.
EXPECTED_MEDIA_TYPES = {
    "txt": "text/plain",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "json": "application/json",
    "srt": "application/x-subrip",
}


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        speaker = Speaker(id="SPEAKER_00", display_name="Спикер 1")
        entry = TranscriptEntry(
            start=0.0, end=1.0, text="привет мир", speaker=speaker
        )
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
            export_formats=(ExportFormat.TXT, ExportFormat.DOCX),
            denoise=False,
            diarization_enabled=False,
            export_speaker_samples=False,
            notifications=False,
            timeline=False,
            protocol_auto=False,
            use_cache=False,
        )

    return build


@pytest.fixture
def client(
    web_paths: WebPaths, fake_pipeline, config_builder
) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str = "sample.mp3") -> dict:
    response = client.post(
        "/api/files/upload", files={"file": (name, b"\x00\x01", "audio/mpeg")}
    )
    assert response.status_code == 201
    return response.json()


def _create_job(client: TestClient, path: str) -> str:
    created = client.post("/api/jobs", json={"path": path})
    assert created.status_code == 201
    return created.json()["id"]


def _run_job(client: TestClient, path: str) -> str:
    job_id = _create_job(client, path)
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        details = client.get(f"/api/jobs/{job_id}").json()
        if details["status"] in {"done", "error"}:
            assert details["status"] == "done", details.get("error")
            return job_id
        time.sleep(0.02)
    raise AssertionError("задача не завершилась за отведённое время")


@pytest.mark.parametrize("fmt", ["txt", "docx", "json", "srt"])
def test_export_returns_non_empty_file_for_each_format(
    client: TestClient, fmt: str
) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": fmt})

    assert response.status_code == 200
    assert EXPECTED_MEDIA_TYPES[fmt] in response.headers["content-type"]
    assert response.content  # непустое тело
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert f"sample.{fmt}" in disposition


def test_export_cyrillic_filename_is_percent_encoded(client: TestClient) -> None:
    uploaded = _upload(client, "Встреча команды.mp3")
    job_id = _run_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "txt"})

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert "filename*=utf-8''" in disposition.lower()
    # Кириллица и пробел корректно закодированы в имени файла.
    assert quote("Встреча команды.txt") in disposition


def test_export_does_not_compute_or_include_summary(
    client: TestClient, web_paths: WebPaths
) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    result_file = web_paths.results_dir / f"{job_id}.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    payload["summary"] = "СЕКРЕТНОЕ РЕЗЮМЕ"
    result_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "txt"})

    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert "СЕКРЕТНОЕ РЕЗЮМЕ" not in body
    assert "привет мир" in body


def test_export_uses_current_renamed_speakers(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_00": "Иван Иванов"}},
    )

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "txt"})

    assert response.status_code == 200
    assert "Иван Иванов" in response.content.decode("utf-8")


def test_export_invalid_format_returns_400(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "pdf"})

    assert response.status_code == 400


def test_export_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/jobs/unknown/export", params={"fmt": "txt"})

    assert response.status_code == 404


def test_export_without_result_returns_404(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _create_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "txt"})

    assert response.status_code == 404
