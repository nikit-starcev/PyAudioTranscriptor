"""Тесты REST API настроек веб-интерфейса (``/api/settings``).

Проверяются чтение, частичное сохранение и валидация форматов/путей, а также
наложение настроек на ``config.env`` при сборке конфигурации задачи.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.storage.glossary_builder import build_glossary
from audio_transcriber.storage.glossary_db import GlossaryDB
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


def test_get_settings_shape(client: TestClient, web_paths: WebPaths) -> None:
    payload = client.get("/api/settings").json()

    assert {
        "glossary_enabled",
        "glossary_db",
        "voices_dir",
        "export_formats",
        "llm_enabled",
        "llm_summary",
        "denoise",
        "mark_overlap",
        "normalize_text",
        "clean_artifacts",
        "protocol_auto",
        "input_dir",
        "output_dir",
        "glossary_db_path",
        "voices_dir_resolved",
    } <= set(payload)
    assert payload["input_dir"] == str(web_paths.input_dir)
    assert payload["output_dir"] == str(web_paths.results_dir)


def test_put_settings_persists(client: TestClient, web_paths: WebPaths, tmp_path: Path) -> None:
    response = client.put(
        "/api/settings",
        json={
            "glossary_enabled": False,
            "glossary_db": str(tmp_path / "glossary.db"),
            "export_formats": ["txt", "docx"],
            "llm_summary": False,
            "denoise": False,
            "protocol_auto": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["glossary_enabled"] is False
    assert body["export_formats"] == ["txt", "docx"]
    assert body["llm_summary"] is False
    assert body["denoise"] is False
    assert body["protocol_auto"] is True

    saved = client.get("/api/settings").json()
    assert saved["glossary_enabled"] is False
    assert saved["export_formats"] == ["txt", "docx"]
    assert (web_paths.settings_json).is_file()


def test_put_settings_partial_update(client: TestClient) -> None:
    before = client.get("/api/settings").json()
    client.put("/api/settings", json={"denoise": False})
    after = client.get("/api/settings").json()

    assert after["denoise"] is False
    # Незаданные поля не сбрасываются.
    assert after["llm_summary"] == before["llm_summary"]
    assert after["glossary_db"] == before["glossary_db"]


def test_put_settings_rejects_invalid_formats(client: TestClient) -> None:
    response = client.put("/api/settings", json={"export_formats": ["txt", "pdf"]})

    assert response.status_code == 400
    assert "формат" in response.json()["detail"]


def test_put_settings_rejects_empty_formats(client: TestClient) -> None:
    response = client.put("/api/settings", json={"export_formats": []})

    assert response.status_code == 400


def test_put_settings_rejects_missing_directory(
    client: TestClient, tmp_path: Path
) -> None:
    missing = tmp_path / "nope" / "glossary.db"

    response = client.put("/api/settings", json={"glossary_db": str(missing)})

    assert response.status_code == 400
    assert "не существует" in response.json()["detail"]


def test_disabled_glossary_not_applied(audio_file: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "g.db"
    with GlossaryDB(db_path) as db:
        db.add_entry("КИСУСС", variant="кисус", source="test")

    disabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"GLOSSARY_ENABLED": "false", "GLOSSARY_DB": str(db_path)},
    )
    enabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"GLOSSARY_ENABLED": "true", "GLOSSARY_DB": str(db_path)},
    )

    assert disabled.glossary_enabled is False
    assert build_glossary(disabled) is None
    assert enabled.glossary_enabled is True
    assert build_glossary(enabled) is not None


def test_build_job_config_reads_diarization_hyperparameters(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Изолируемся от реального config.env, чтобы проверить именно overrides.
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "DIARIZATION_MIN_DURATION_OFF": "0.9",
            "DIARIZATION_CLUSTERING_THRESHOLD": "0.6",
            "DIARIZATION_CLUSTERING_FB": "1.5",
        },
    )

    assert config.diarization_min_duration_off == pytest.approx(0.9)
    assert config.diarization_clustering_threshold == pytest.approx(0.6)
    assert config.diarization_clustering_fb == pytest.approx(1.5)


def test_build_job_config_diarization_defaults(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
    )

    assert config.diarization_min_duration_off == pytest.approx(0.5)
    assert config.diarization_clustering_threshold is None
    assert config.diarization_clustering_fb is None
