"""Тесты применения глоссария к текущей стенограмме по кнопке (#32).

Конвейер подменяется, поэтому GPU/модели не нужны. Источник терминов —
актуальная SQLite-БД глоссария; проверяем число замен, сохранение результата,
идемпотентность и бережное отношение к ручным правкам (#26).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.storage.glossary_db import GlossaryDB
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def glossary_db(tmp_path: Path) -> Path:
    path = tmp_path / "glossary.db"
    with GlossaryDB(path) as db:
        db.add_entry("ОИБ", variant="АИБ")
        db.add_entry("транскрибер")
    return path


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(
                start=0.0, end=1.0, text="АИБ безопасность", speaker=speaker
            ),
            TranscriptEntry(start=1.0, end=2.0, text="пока", speaker=speaker),
        ]
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=2.0,
            entries=entries,
            speakers=[speaker],
        )

    return pipeline


def _client(
    web_paths: WebPaths, fake_pipeline, glossary_db: Path
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
            glossary_db=glossary_db,
        )

    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client(web_paths: WebPaths, fake_pipeline, glossary_db: Path) -> Iterator[TestClient]:
    yield from _client(web_paths, fake_pipeline, glossary_db)


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


def test_apply_glossary_replaces_terms_and_persists(
    client: TestClient, web_paths: WebPaths
) -> None:
    job_id = _prepared_job(client)

    response = client.post(f"/api/jobs/{job_id}/apply-glossary", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["replacements"] == 1
    assert body["terms"] >= 1
    assert body["result"]["entries"][0]["text"] == "ОИБ безопасность"
    assert body["details"][0]["replacements"] == [{"before": "АИБ", "after": "ОИБ"}]
    # Результат сохранён на диск и доступен через GET.
    fetched = client.get(f"/api/jobs/{job_id}/result").json()
    assert fetched["entries"][0]["text"] == "ОИБ безопасность"


def test_apply_glossary_is_idempotent(client: TestClient) -> None:
    job_id = _prepared_job(client)
    client.post(f"/api/jobs/{job_id}/apply-glossary", json={})

    second = client.post(f"/api/jobs/{job_id}/apply-glossary", json={})

    assert second.status_code == 200
    assert second.json()["replacements"] == 0


def test_apply_glossary_skips_manually_edited_by_default(client: TestClient) -> None:
    job_id = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "АИБ остаётся"}]},
    )

    response = client.post(f"/api/jobs/{job_id}/apply-glossary", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["replacements"] == 0
    assert body["skipped_edited"] == 1
    assert body["result"]["entries"][0]["text"] == "АИБ остаётся"


def test_apply_glossary_can_include_edited(client: TestClient) -> None:
    job_id = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "АИБ остаётся"}]},
    )

    response = client.post(
        f"/api/jobs/{job_id}/apply-glossary", json={"respect_edited": False}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["replacements"] == 1
    entry = body["result"]["entries"][0]
    assert entry["text"] == "ОИБ остаётся"
    # Пометка ручной правки не сбрасывается.
    assert entry["edited"] is True
    assert entry["original_text"] == "АИБ безопасность"


def test_apply_glossary_without_result_returns_404(client: TestClient) -> None:
    upload = client.post(
        "/api/files/upload", files={"file": ("empty.mp3", b"\x00", "audio/mpeg")}
    )
    created = client.post("/api/jobs", json={"path": upload.json()["name"]})
    job_id = created.json()["id"]

    response = client.post(f"/api/jobs/{job_id}/apply-glossary", json={})

    assert response.status_code == 404


def test_apply_glossary_empty_db_reports_error(
    web_paths: WebPaths, fake_pipeline, tmp_path: Path
) -> None:
    empty_db = tmp_path / "empty.db"
    with GlossaryDB(empty_db):
        pass
    for test_client in _client(web_paths, fake_pipeline, empty_db):
        job_id = _prepared_job(test_client)
        response = test_client.post(f"/api/jobs/{job_id}/apply-glossary", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["replacements"] == 0
        assert body["error"]
        assert body["result"]["entries"][0]["text"] == "АИБ безопасность"
