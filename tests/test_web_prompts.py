"""Тесты REST API шаблонов промпта резюме (``/api/summary-prompts``, #97).

Проверяются CRUD, выбор активного шаблона и применение его к задаче и к
протоколу по кнопке. LLM/экспорт не запускаются.
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
from audio_transcriber.web.config import build_job_config
from audio_transcriber.web.paths import WebPaths


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


def _create_prompt(client: TestClient, name: str, body: str) -> int:
    response = client.post("/api/summary-prompts", json={"name": name, "body": body})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_defaults_are_seeded(client: TestClient) -> None:
    payload = client.get("/api/summary-prompts").json()
    names = {prompt["name"] for prompt in payload["prompts"]}
    assert "Стандартный" in names
    assert all(prompt["builtin"] for prompt in payload["prompts"])
    active_id = payload["active_id"]
    assert any(prompt["id"] == active_id for prompt in payload["prompts"])


def test_crud_lifecycle(client: TestClient) -> None:
    prompt_id = _create_prompt(client, "Мой формат", "Тело промпта")

    listing = client.get("/api/summary-prompts").json()
    created = next(prompt for prompt in listing["prompts"] if prompt["id"] == prompt_id)
    assert created["name"] == "Мой формат"
    assert created["builtin"] is False

    patched = client.patch(
        f"/api/summary-prompts/{prompt_id}",
        json={"name": "Новый формат", "body": "Новое тело"},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Новый формат"
    assert patched.json()["body"] == "Новое тело"

    deleted = client.delete(f"/api/summary-prompts/{prompt_id}")
    assert deleted.status_code == 200
    assert client.delete(f"/api/summary-prompts/{prompt_id}").status_code == 404


def test_create_rejects_empty_and_duplicate(client: TestClient) -> None:
    assert (
        client.post("/api/summary-prompts", json={"name": "", "body": "x"}).status_code
        == 400
    )
    assert (
        client.post("/api/summary-prompts", json={"name": "x", "body": "  "}).status_code
        == 400
    )
    _create_prompt(client, "Дубль", "тело")
    duplicate = client.post("/api/summary-prompts", json={"name": "Дубль", "body": "другое"})
    assert duplicate.status_code == 409


def test_patch_and_activate_missing_return_404(client: TestClient) -> None:
    assert client.patch("/api/summary-prompts/999999", json={"body": "x"}).status_code == 404
    assert client.post("/api/summary-prompts/999999/activate").status_code == 404


def test_activate_changes_active_prompt(client: TestClient) -> None:
    prompt_id = _create_prompt(client, "Активный", "тело")

    activated = client.post(f"/api/summary-prompts/{prompt_id}/activate")
    assert activated.status_code == 200
    assert activated.json()["active_id"] == prompt_id

    assert client.get("/api/summary-prompts").json()["active_id"] == prompt_id


def test_build_job_config_maps_summary_prompt(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"LLM_SUMMARY_PROMPT": "Составь протокол в формате X"},
    )

    assert config.llm_summary_prompt == "Составь протокол в формате X"


# --- Применение активного шаблона к задаче и кнопке «Сформировать протокол» ---


def _fake_pipeline(captured: list[AppConfig]):
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        captured.append(config)
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


def _upload(client: TestClient, name: str = "sample.mp3") -> str:
    response = client.post(
        "/api/files/upload", files={"file": (name, b"\x00\x01", "audio/mpeg")}
    )
    assert response.status_code == 201
    return response.json()["name"]


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


def test_active_prompt_applied_to_job(web_paths: WebPaths) -> None:
    captured: list[AppConfig] = []
    app = create_app(paths=web_paths, pipeline_fn=_fake_pipeline(captured), heartbeat=0.05)
    with TestClient(app) as client:
        prompt_id = _create_prompt(client, "Задачный", "ПРОМПТ ДЛЯ ЗАДАЧИ")
        assert client.post(f"/api/summary-prompts/{prompt_id}/activate").status_code == 200

        uploaded = _upload(client)
        _run_job(client, uploaded)

    assert captured
    assert captured[-1].llm_summary_prompt == "ПРОМПТ ДЛЯ ЗАДАЧИ"


def test_protocol_prompt_id_overrides_active(web_paths: WebPaths) -> None:
    captured: list[AppConfig] = []

    def protocol_fn(
        config: AppConfig, result: TranscriptionResult, **kwargs
    ) -> ProtocolArtifacts:
        captured.append(config)
        config.ensure_output_dir()
        return ProtocolArtifacts(paths=(), summary="резюме")

    def config_builder(job_id: str, source_path: Path) -> AppConfig:
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

    app = create_app(
        paths=web_paths,
        pipeline_fn=_fake_pipeline([]),
        config_builder=config_builder,
        protocol_fn=protocol_fn,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        prompt_id = _create_prompt(client, "Разовый", "РАЗОВЫЙ ПРОМПТ")
        uploaded = _upload(client)
        job_id = _run_job(client, uploaded)

        response = client.post(
            f"/api/jobs/{job_id}/protocol", json={"prompt_id": prompt_id}
        )
        assert response.status_code == 200, response.text

        missing = client.post(
            f"/api/jobs/{job_id}/protocol", json={"prompt_id": 999999}
        )
        assert missing.status_code == 404

    assert captured
    assert captured[0].llm_summary_prompt == "РАЗОВЫЙ ПРОМПТ"
