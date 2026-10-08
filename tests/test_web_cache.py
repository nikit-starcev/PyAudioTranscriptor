"""Тесты ручной очистки постадийного кэша через веб-API (#100).

Кэш — общий каталог ``web-data/cache``; эндпоинты отдают его размер/состав и
чистят его через :meth:`StageCache.clear`. Пока воркер ведёт задачу, очистка по
умолчанию отклоняется (409), а с ``force=true`` — выполняется. Сеть и конвейер
не нужны.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


def _seed_cache(web_paths: WebPaths, files: dict[str, bytes]) -> None:
    web_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (web_paths.cache_dir / name).write_bytes(data)


def test_cache_info_reports_files_and_bytes(web_paths: WebPaths, client: TestClient) -> None:
    _seed_cache(
        web_paths,
        {"asr-abc.json": b"12345", "denoise-def.wav": b"abc", "notes.txt": b"x"},
    )

    response = client.get("/api/cache")

    assert response.status_code == 200
    payload = response.json()
    assert payload["directory"] == str(web_paths.cache_dir)
    assert payload["files"] == 3
    assert payload["bytes"] == 9
    assert payload["active_jobs"] == []


def test_cache_clear_removes_all_files(web_paths: WebPaths, client: TestClient) -> None:
    _seed_cache(web_paths, {"asr-abc.json": b"1", "asr-abc.tmp": b"22"})

    response = client.post("/api/cache/clear", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["removed"] == 2
    assert payload["forced"] is False
    assert list(web_paths.cache_dir.iterdir()) == []

    assert client.get("/api/cache").json()["files"] == 0


def test_cache_clear_on_empty_is_ok(client: TestClient) -> None:
    response = client.post("/api/cache/clear")

    assert response.status_code == 200
    assert response.json()["removed"] == 0


def test_cache_clear_refused_while_job_active(
    web_paths: WebPaths, client: TestClient
) -> None:
    _seed_cache(web_paths, {"asr-abc.json": b"1"})
    client.app.state.runner._active.add("job-1")

    refused = client.post("/api/cache/clear", json={})
    assert refused.status_code == 409
    assert "Идёт обработка" in refused.json()["detail"]
    # Кэш не тронут, а информация показывает активную задачу.
    info = client.get("/api/cache").json()
    assert info["files"] == 1
    assert info["active_jobs"] == ["job-1"]

    forced = client.post("/api/cache/clear", json={"force": True})
    assert forced.status_code == 200
    assert forced.json()["removed"] == 1
    assert forced.json()["forced"] is True
    assert forced.json()["active_jobs"] == ["job-1"]
