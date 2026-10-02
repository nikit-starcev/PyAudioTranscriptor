"""Тесты редакторской проверки/исправления текста через API (#51).

Конвейер подменяется. Проверяем dry-run (список предложений без записи),
выборочное применение, сохранение результата и бережное отношение к ручным
правкам (#26).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(start=0.0, end=1.0, text="привет ,мир", speaker=speaker),
            TranscriptEntry(
                start=1.0,
                end=2.0,
                text="Конфиденциальнасть мы не нашли.",
                speaker=speaker,
            ),
            TranscriptEntry(start=2.0, end=3.0, text="пока", speaker=speaker),
        ]
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=3.0,
            entries=entries,
            speakers=[speaker],
        )

    return pipeline


@pytest.fixture
def client(
    web_paths: WebPaths, fake_pipeline
) -> Iterator[TestClient]:
    def config_builder(job_id: str, source_path: Path) -> AppConfig:
        return AppConfig(
            input_file=source_path,
            output_dir=web_paths.results_dir / job_id,
            denoise=False,
            diarization_enabled=False,
            export_speaker_samples=False,
            notifications=False,
            timeline=False,
            protocol_auto=False,
            use_cache=False,
        )

    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _prepared_job(client: TestClient) -> str:
    upload = client.post(
        "/api/files/upload", files={"file": ("sample.mp3", b"\x00\x01", "audio/mpeg")}
    )
    assert upload.status_code == 201
    created = client.post("/api/jobs", json={"path": upload.json()["name"]})
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


def test_dry_run_returns_suggestions_without_writing(client: TestClient) -> None:
    job_id = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/correct-text", json={"dry_run": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied_count"] == 0
    assert body["suggestions"]
    kinds = {item["kind"] for item in body["suggestions"]}
    assert "common" in kinds
    assert "spelling" in kinds
    # Результат не изменился на диске.
    fetched = client.get(f"/api/jobs/{job_id}/result").json()
    assert fetched["entries"][0]["text"] == "привет ,мир"


def test_apply_only_selected_suggestions(client: TestClient) -> None:
    job_id = _prepared_job(client)
    preview = client.post(
        f"/api/jobs/{job_id}/correct-text", json={"dry_run": True}
    ).json()
    common_ids = [
        item["id"] for item in preview["suggestions"] if item["kind"] == "common"
    ]

    response = client.post(
        f"/api/jobs/{job_id}/correct-text",
        json={"selection": common_ids, "check_spelling": False, "fix_common": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["applied_count"] >= 1
    assert body["result"]["entries"][0]["text"] == "привет, мир"
    # Орфографическая реплика не тронута.
    assert (
        body["result"]["entries"][1]["text"] == "Конфиденциальнасть мы не нашли."
    )


def test_apply_all_suggestions_and_persists(client: TestClient) -> None:
    job_id = _prepared_job(client)

    response = client.post(f"/api/jobs/{job_id}/correct-text", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["applied_count"] >= 2
    assert body["result"]["entries"][0]["text"] == "привет, мир"
    assert (
        body["result"]["entries"][1]["text"] == "Конфиденциальность мы не нашли."
    )
    fetched = client.get(f"/api/jobs/{job_id}/result").json()
    assert fetched["entries"][1]["text"] == "Конфиденциальность мы не нашли."


def test_spelling_can_be_disabled(client: TestClient) -> None:
    job_id = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/correct-text",
        json={"dry_run": True, "check_spelling": False},
    )

    assert response.status_code == 200
    kinds = {item["kind"] for item in response.json()["suggestions"]}
    assert kinds == {"common"}


def test_respects_manual_edits_by_default(client: TestClient) -> None:
    job_id = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "привет ,мир!"}]},
    )

    response = client.post(
        f"/api/jobs/{job_id}/correct-text", json={"dry_run": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["skipped_edited"] == 1
    assert all(item["index"] != 0 for item in body["suggestions"])


def test_can_include_manual_edits(client: TestClient) -> None:
    job_id = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "привет ,мир!"}]},
    )

    response = client.post(
        f"/api/jobs/{job_id}/correct-text",
        json={"dry_run": True, "respect_edited": False},
    )

    assert response.status_code == 200
    assert any(item["index"] == 0 for item in response.json()["suggestions"])


def test_no_checks_selected_returns_400(client: TestClient) -> None:
    job_id = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/correct-text",
        json={"fix_common": False, "check_spelling": False},
    )

    assert response.status_code == 400
