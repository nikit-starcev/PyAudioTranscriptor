"""Тесты формирования и скачивания протокола (``/api/jobs/{id}/protocol``).

``generate_protocol`` подменяется фиктивной функцией, которая пишет файлы и
возвращает :class:`ProtocolArtifacts`; настоящая LLM/экспорт не запускаются.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.protocol import ProtocolArtifacts
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


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
def captured_results() -> list[TranscriptionResult]:
    return []


@pytest.fixture
def protocol_fn(captured_results: list[TranscriptionResult]):
    def build(config: AppConfig, result: TranscriptionResult, **kwargs) -> ProtocolArtifacts:
        captured_results.append(result)
        config.ensure_output_dir()
        paths: list[Path] = []
        for fmt in ("txt", "docx"):
            target = config.output_dir / f"{config.input_file.stem}.{fmt}"
            target.write_text("тело протокола", encoding="utf-8")
            paths.append(target)
        return ProtocolArtifacts(paths=tuple(paths), summary="Резюме встречи")

    return build


@pytest.fixture
def client(
    web_paths: WebPaths, fake_pipeline, config_builder, protocol_fn
) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        protocol_fn=protocol_fn,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str = "sample.mp3") -> dict:
    response = client.post("/api/files/upload", files={"file": (name, b"\x00\x01", "audio/mpeg")})
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


def test_protocol_generates_and_stores_summary(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    response = client.post(f"/api/jobs/{job_id}/protocol")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Резюме встречи"
    assert len(body["paths"]) == 2
    assert set(body["protocol"]) == {"txt", "docx"}

    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["summary"] == "Резюме встречи"
    assert client.get(f"/api/jobs/{job_id}/summary").json() == {"summary": "Резюме встречи"}


def test_protocol_uses_current_renamed_result(
    client: TestClient, captured_results: list[TranscriptionResult]
) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_00": "Иван Иванов"}},
    )

    assert client.post(f"/api/jobs/{job_id}/protocol").status_code == 200

    assert captured_results
    assert captured_results[-1].speakers[0].display_name == "Иван Иванов"


def test_protocol_download(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    assert client.post(f"/api/jobs/{job_id}/protocol").status_code == 200

    txt = client.get(f"/api/jobs/{job_id}/protocol/download", params={"fmt": "txt"})
    assert txt.status_code == 200
    assert txt.content.decode("utf-8") == "тело протокола"
    assert "attachment" in txt.headers["content-disposition"]
    assert "sample.txt" in txt.headers["content-disposition"]

    docx = client.get(f"/api/jobs/{job_id}/protocol/download", params={"fmt": "docx"})
    assert docx.status_code == 200
    assert "wordprocessingml" in docx.headers["content-type"]

    assert client.get(
        f"/api/jobs/{job_id}/protocol/download", params={"fmt": "pdf"}
    ).status_code == 400


def test_protocol_requires_finished_job(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _create_job(client, uploaded["name"])

    response = client.post(f"/api/jobs/{job_id}/protocol")

    assert response.status_code == 409


def test_download_before_protocol_returns_404(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/protocol/download", params={"fmt": "txt"})

    assert response.status_code == 404


def test_protocol_unknown_job_returns_404(client: TestClient) -> None:
    assert client.post("/api/jobs/unknown/protocol").status_code == 404
    assert client.get("/api/jobs/unknown/protocol/download").status_code == 404
