"""Тесты безопасности веб-слоя: пути, лимиты загрузок, CSRF (#86/#87).

Проверяются: ограничение явных путей образцов разрешённым корнем (включая
``..`` и симлинки), изоляция ``result_path`` каталогом результатов, потоковая
запись загрузок с лимитом размера (413), а также мягкая CSRF-защита по
``Origin``/``Sec-Fetch-Site``. Реальный конвейер/сеть не задействуются.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from audio_transcriber.web import app as app_module
from audio_transcriber.web.app import (
    _UPLOAD_CHUNK_SIZE,
    _explicit_references,
    _reference_roots,
    _result_path,
    _stream_upload,
    create_app,
)
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.storage.jobs_db import (
    STATUS_RUNNING,
    Job,
)


class _FakeUpload:
    """Мини-двойник ``UploadFile``: потоковое чтение чанками."""

    def __init__(self, data: bytes, *, size: int | None = None) -> None:
        self._data = data
        self.size = len(data) if size is None else size
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if not self._data:
            return b""
        chunk = self._data[:size]
        self._data = self._data[size:]
        return chunk


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


# --- #86: явные образцы голоса внутри разрешённого корня -------------------


def test_explicit_references_allows_inside_root(web_paths: WebPaths) -> None:
    samples = web_paths.data_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    allowed = samples / "a.wav"
    allowed.write_bytes(b"x")
    roots = _reference_roots(web_paths, web_paths.data_dir)

    result = _explicit_references({"Иван": str(allowed)}, roots=roots)

    assert result == {"Иван": [allowed.resolve()]}


def test_explicit_references_rejects_absolute_outside_root(
    web_paths: WebPaths, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"x")
    roots = _reference_roots(web_paths, web_paths.data_dir)

    with pytest.raises(HTTPException) as excinfo:
        _explicit_references({"Иван": str(secret)}, roots=roots)

    assert excinfo.value.status_code == 400


def test_explicit_references_rejects_symlink_escape(
    web_paths: WebPaths, tmp_path: Path
) -> None:
    samples = web_paths.data_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")
    link = samples / "link.wav"
    link.symlink_to(outside)
    roots = _reference_roots(web_paths, web_paths.data_dir)

    with pytest.raises(HTTPException) as excinfo:
        _explicit_references({"Иван": str(link)}, roots=roots)

    assert excinfo.value.status_code == 400


def test_explicit_references_rejects_dotdot_escape(web_paths: WebPaths) -> None:
    roots = _reference_roots(web_paths, web_paths.data_dir)
    escaping = str(web_paths.data_dir / "samples" / ".." / ".." / "etc_passwd")

    with pytest.raises(HTTPException) as excinfo:
        _explicit_references({"Иван": escaping}, roots=roots)

    assert excinfo.value.status_code == 400


# --- #86: result_path не читается/пишется вне каталога результатов ---------


def test_result_path_falls_back_when_outside_results_dir(
    web_paths: WebPaths, tmp_path: Path
) -> None:
    job = Job(
        id="job1",
        source_path="/tmp/x.mp3",
        status=STATUS_RUNNING,
        created_at="2026-01-01T00:00:00+00:00",
        result_path=str(tmp_path / "evil.json"),
    )

    assert _result_path(web_paths, job) == web_paths.results_dir / "job1.json"


def test_result_path_keeps_inside_result(web_paths: WebPaths) -> None:
    inside = web_paths.results_dir / "job1.json"
    job = Job(
        id="job1",
        source_path="/tmp/x.mp3",
        status=STATUS_RUNNING,
        created_at="2026-01-01T00:00:00+00:00",
        result_path=str(inside),
    )

    assert _result_path(web_paths, job) == inside.resolve()


# --- #86: потоковая запись и лимиты загрузок -------------------------------


def test_stream_upload_writes_in_chunks(tmp_path: Path) -> None:
    payload = b"x" * (3 * _UPLOAD_CHUNK_SIZE + 17)
    fake = _FakeUpload(payload)
    target = tmp_path / "upload.bin"

    written = asyncio.run(_stream_upload(fake, target, max_bytes=0))

    assert written == len(payload)
    assert target.read_bytes() == payload
    # Все чтения — фиксированным чанком: файл не тянется в память целиком.
    assert fake.read_sizes
    assert set(fake.read_sizes) == {_UPLOAD_CHUNK_SIZE}


def test_stream_upload_rejects_declared_oversize(tmp_path: Path) -> None:
    target = tmp_path / "upload.bin"
    fake = _FakeUpload(b"x", size=app_module.DEFAULT_MAX_UPLOAD_BYTES + 1)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            _stream_upload(
                fake, target, max_bytes=app_module.DEFAULT_MAX_UPLOAD_BYTES
            )
        )

    assert excinfo.value.status_code == 413
    assert not target.exists()


def test_stream_upload_rejects_streaming_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_UPLOAD_CHUNK_SIZE", 16)
    target = tmp_path / "upload.bin"
    fake = _FakeUpload(b"y" * 100, size=0)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_stream_upload(fake, target, max_bytes=32))

    assert excinfo.value.status_code == 413
    assert not target.exists()


def test_media_upload_limit_returns_413(
    client: TestClient, web_paths: WebPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEB_MAX_UPLOAD_MB", "0.001")
    payload = b"z" * 8192

    response = client.post(
        "/api/files/upload", files={"file": ("big.mp3", payload, "audio/mpeg")}
    )

    assert response.status_code == 413
    assert list(web_paths.input_dir.glob("*")) == []


def test_voice_upload_limit_returns_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEB_MAX_SAMPLE_UPLOAD_MB", "0.001")
    payload = b"z" * 8192

    response = client.post(
        "/api/voices",
        files={"file": ("sample.wav", payload, "audio/wav")},
        data={"name": "Иван"},
    )

    assert response.status_code == 413


# --- #87: CSRF (Origin / Sec-Fetch-Site) -----------------------------------


def _small_upload() -> dict:
    return {"file": ("s.mp3", b"\x00\x01", "audio/mpeg")}


def test_csrf_blocks_cross_origin_post(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload",
        files=_small_upload(),
        headers={"Origin": "http://evil.example"},
    )

    assert response.status_code == 403


def test_csrf_blocks_cross_site_sec_fetch(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload",
        files=_small_upload(),
        headers={"Sec-Fetch-Site": "cross-site"},
    )

    assert response.status_code == 403


def test_csrf_allows_same_origin_post(client: TestClient) -> None:
    response = client.post(
        "/api/files/upload",
        files=_small_upload(),
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 201


def test_csrf_allows_post_without_headers(client: TestClient) -> None:
    response = client.post("/api/files/upload", files=_small_upload())

    assert response.status_code == 201


def test_csrf_does_not_affect_get(client: TestClient) -> None:
    response = client.get("/api/health", headers={"Origin": "http://evil.example"})

    assert response.status_code == 200
