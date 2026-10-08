"""Тесты генерации и управления API-ключом OpenAI-совместимого API (#110).

Проверяются ``GET``/``POST``/``DELETE /api/api-key``: статус, генерация
случайного ключа (с префиксом ``sk-``), приём своего ключа с валидацией
минимальной длины, очистка и приоритет ``web-data/secrets.json`` над
``config.env``. Отдельно проверяется, что сгенерированный ключ начинает
действовать на ``/v1/*`` сразу, без перезапуска сервера.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.web.app import API_KEY_MIN_LENGTH, create_app
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.secrets import SecretsStore, effective_api_key


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def isolate_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует тесты от реального ``config.env`` (там может быть API_KEY)."""
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})


def test_status_when_not_set(client: TestClient) -> None:
    response = client.get("/api/api-key")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body == {
        "set": False,
        "source": None,
        "secret_set": False,
        "key": None,
        "masked": None,
    }


def test_generate_creates_valid_prefixed_key(
    client: TestClient, web_paths: WebPaths
) -> None:
    response = client.post("/api/api-key")

    assert response.status_code == 200
    body = response.json()
    assert body["set"] is True
    assert body["source"] == "secrets"
    assert body["secret_set"] is True
    key = body["key"]
    assert isinstance(key, str)
    assert key.startswith("sk-")
    assert len(key) >= API_KEY_MIN_LENGTH
    assert body["masked"] == f"{key[:3]}…{key[-4:]}"

    stored = SecretsStore(web_paths.secrets_json)
    assert stored.get_api_key() == key
    # Секрет перекрывает значение из config.env (приоритет secrets.json).
    assert effective_api_key(stored, {"API_KEY": "from-env-key-0000"}) == key
    # GET после генерации отдаёт тот же ключ.
    assert client.get("/api/api-key").json()["key"] == key


def test_generate_rotates_key(client: TestClient) -> None:
    first = client.post("/api/api-key").json()["key"]
    second = client.post("/api/api-key").json()["key"]

    assert first != second
    assert client.get("/api/api-key").json()["key"] == second


def test_accept_custom_key(client: TestClient, web_paths: WebPaths) -> None:
    custom = "my-custom-openai-key-123456"
    response = client.post("/api/api-key", json={"key": custom})

    assert response.status_code == 200
    assert response.json()["key"] == custom
    assert SecretsStore(web_paths.secrets_json).get_api_key() == custom


def test_reject_short_and_empty_custom_key(
    client: TestClient, web_paths: WebPaths
) -> None:
    short = client.post("/api/api-key", json={"key": "short"})
    empty = client.post("/api/api-key", json={"key": "   "})

    assert short.status_code == 400
    assert "минимум" in short.json()["detail"]
    assert empty.status_code == 400
    # Ничего не сохранилось.
    assert SecretsStore(web_paths.secrets_json).get_api_key() is None


def test_delete_clears_key(client: TestClient, web_paths: WebPaths) -> None:
    generated = client.post("/api/api-key").json()["key"]
    assert client.get("/api/api-key").json()["set"] is True

    response = client.delete("/api/api-key")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "set": False,
        "source": None,
        "secret_set": False,
        "key": None,
        "masked": None,
    }
    assert client.get("/api/api-key").json()["set"] is False
    assert SecretsStore(web_paths.secrets_json).get_api_key() is None
    assert generated


def test_env_key_status_and_clear_keeps_env(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "audio_transcriber.web.app.env_defaults",
        lambda: {"API_KEY": "env-only-key-123456"},
    )

    assert client.get("/api/api-key").json()["source"] == "env"

    stored = client.post("/api/api-key").json()
    assert stored["source"] == "secrets"

    cleared = client.delete("/api/api-key").json()
    assert cleared["set"] is True
    assert cleared["source"] == "env"
    assert cleared["secret_set"] is False
    assert cleared["key"] == "env-only-key-123456"


def test_generated_key_authenticates_v1_immediately(web_paths: WebPaths) -> None:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        assert test_client.get("/v1/models").status_code == 401

        key = test_client.post("/api/api-key").json()["key"]
        headers = {"Authorization": f"Bearer {key}"}
        assert test_client.get("/v1/models", headers=headers).status_code == 200

        test_client.delete("/api/api-key")
        assert test_client.get("/v1/models", headers=headers).status_code == 401


def test_generated_key_not_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        key = client.post("/api/api-key").json()["key"]

    assert key not in caplog.text
    assert key[:8] not in caplog.text
