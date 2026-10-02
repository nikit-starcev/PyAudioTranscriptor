"""Тесты каталога моделей и фонового скачивания (без сети).

Реальный загрузчик подменяется заглушкой, которая пишет файлы на диск и шлёт
прогресс — поэтому тесты быстрые и полностью локальные. Проверяются формат
``/api/models``, запуск/отмена/удаление, ошибки (нет токена, мало места) и
события SSE.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.web import app as web_app
from audio_transcriber.web import models as web_models
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.settings import WebSettings


class FakeDownloader:
    """Заглушка загрузчика: пишет файлы и уведомляет о прогрессе."""

    def __init__(self, *, block: threading.Event | None = None) -> None:
        self.block = block
        self.calls: list[tuple[str, str]] = []

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress,
    ) -> None:
        self.calls.append((repo, filename))
        if self.block is not None:
            self.block.wait(timeout=5)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = b"x" * 4096
        destination.write_bytes(data)
        on_progress(len(data))

    def fetch_snapshot(
        self,
        *,
        repo: str,
        destination: Path,
        token: str | None,
        on_progress,
    ) -> None:
        self.calls.append((repo, "<snapshot>"))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.yaml").write_bytes(b"ok")
        (destination / "embedding").mkdir(exist_ok=True)
        (destination / "embedding" / "model.bin").write_bytes(b"y" * 128)
        on_progress(130)


class LoopingDownloader:
    """Заглушка, которая «висит», пока её не отменят через on_progress."""

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(400):
            time.sleep(0.01)
            on_progress(1024)

    def fetch_snapshot(self, *, repo, destination, token, on_progress) -> None:  # pragma: no cover
        raise AssertionError("не используется")


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует тесты от реального ``config.env`` (в нём могут быть пути моделей)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    monkeypatch.setattr(
        "audio_transcriber.web.doctor_api.load_config_env", lambda: (None, {})
    )


def _make_client(web_paths: WebPaths, downloader) -> TestClient:
    app = create_app(paths=web_paths, downloader=downloader, heartbeat=0.05)
    return TestClient(app)


@pytest.fixture
def downloader() -> FakeDownloader:
    return FakeDownloader()


@pytest.fixture
def client(web_paths: WebPaths, downloader: FakeDownloader) -> Iterator[TestClient]:
    with _make_client(web_paths, downloader) as test_client:
        yield test_client


def _manager(client: TestClient) -> web_models.ModelDownloadManager:
    return client.app.state.downloads  # type: ignore[attr-defined]


def test_catalog_shape_and_uniqueness() -> None:
    ids = [entry.id for entry in web_models.MODEL_CATALOG]
    assert len(ids) == len(set(ids))
    kinds = {entry.kind for entry in web_models.MODEL_CATALOG}
    assert kinds == {
        web_models.KIND_WHISPER,
        web_models.KIND_LLM,
        web_models.KIND_PYANNOTE,
        web_models.KIND_GIGAAM,
    }
    pyannote = web_models.find_model("pyannote-community-1")
    assert pyannote is not None
    assert pyannote.gated is True
    assert pyannote.snapshot is True
    assert pyannote.setting_key == "pyannote_local_model"
    for entry in web_models.MODEL_CATALOG:
        assert entry.approx_size > 0
        assert entry.target_dir
        assert entry.repo
        assert entry.setting_key


def test_local_status_reports_partial(tmp_path: Path) -> None:
    entry = web_models.find_model("whisper-large-v3-turbo")
    assert entry is not None
    target = tmp_path / "whisper-models"
    target.mkdir(parents=True)
    (target / "ggml-large-v3-turbo.bin").write_bytes(b"z" * 10)

    status = web_models.local_status(entry, target)

    assert status.present is True
    assert status.size == 10
    assert status.missing == ()


def test_local_status_missing_file(tmp_path: Path) -> None:
    entry = web_models.find_model("qwen2.5-7b-instruct-q4_k_m")
    assert entry is not None
    target = tmp_path / "llama-models"
    target.mkdir(parents=True)
    (target / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf").write_bytes(b"a" * 5)

    status = web_models.local_status(entry, target)

    assert status.present is False
    assert status.partial is True
    assert status.missing == ("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",)


def test_resolve_target_uses_configured_path(tmp_path: Path) -> None:
    entry = web_models.find_model("pyannote-community-1")
    assert entry is not None
    settings = WebSettings(pyannote_local_model="/models/some/pyannote")
    assert (
        web_models.resolve_target(entry, models_root=tmp_path, settings=settings)
        == Path("/models/some/pyannote")
    )

    whisper = web_models.find_model("whisper-small")
    assert whisper is not None
    settings2 = WebSettings(whisper_cpp_model="/models/whisper-models/ggml-small.bin")
    assert (
        web_models.resolve_target(whisper, models_root=tmp_path, settings=settings2)
        == Path("/models/whisper-models")
    )
    assert (
        web_models.primary_path(
            whisper, Path("/models/whisper-models")
        ).name
        == "ggml-small.bin"
    )


def test_gigaam_snapshot_entry(tmp_path: Path) -> None:
    entry = web_models.find_model("gigaam-v3-onnx")
    assert entry is not None
    assert entry.kind == web_models.KIND_GIGAAM
    assert entry.snapshot is True
    assert entry.gated is False
    assert entry.repo == "istupakov/gigaam-v3-onnx"
    assert entry.setting_key == "gigaam_model_path"
    assert entry.target_dir == "gigaam-models/gigaam-v3-onnx"
    assert entry.approx_size > 0

    # Путь по умолчанию — каталог моделей; настроенный — используется как есть.
    assert web_models.resolve_target(entry, models_root=tmp_path, settings=WebSettings()) == (
        tmp_path / "gigaam-models" / "gigaam-v3-onnx"
    )
    configured = WebSettings(gigaam_model_path="/models/gigaam")
    assert (
        web_models.resolve_target(entry, models_root=tmp_path, settings=configured)
        == Path("/models/gigaam")
    )

    # Статус снимка: пусто/только .cache — нет; содержательный файл — есть.
    target = tmp_path / "gigaam-models" / "gigaam-v3-onnx"
    assert web_models.local_status(entry, target).present is False
    (target / ".cache").mkdir(parents=True)
    (target / ".cache" / "tmp").write_bytes(b"cache")
    assert web_models.local_status(entry, target).present is False
    (target / "v3_e2e_rnnt_encoder.onnx").write_bytes(b"z" * 64)
    status = web_models.local_status(entry, target)
    assert status.present is True
    assert status.size >= 64

    # Удаление снимка работает без спец-кода (ограничено каталогом модели).
    assert web_models.delete_model_files(entry, target) is True
    assert not target.exists()
    assert web_models.delete_model_files(entry, target) is False


def test_get_models_endpoint_shape(client: TestClient) -> None:
    payload = client.get("/api/models").json()

    assert set(payload) == {"models", "disk"}
    assert payload["disk"]["models_dir"] == str(client.app.state.models_dir)  # type: ignore[attr-defined]
    by_id = {model["id"]: model for model in payload["models"]}
    assert set(by_id) == {entry.id for entry in web_models.MODEL_CATALOG}
    sample = by_id["whisper-small"]
    assert set(sample) >= {
        "id",
        "kind",
        "title",
        "repo",
        "files",
        "approx_size",
        "gated",
        "setting_key",
        "target_dir",
        "target_path",
        "primary_path",
        "status",
        "download",
    }
    assert sample["status"]["present"] is False
    assert sample["status"]["missing_files"] == ["ggml-small.bin"]
    assert sample["download"]["status"] == "idle"
    assert sample["download"]["fraction"] is None


def test_download_writes_files_and_marks_present(
    client: TestClient, web_paths: WebPaths, downloader: FakeDownloader
) -> None:
    response = client.post("/api/models/whisper-small/download")

    assert response.status_code == 202
    assert response.json()["status"] == "downloading"

    _manager(client).wait("whisper-small")

    target = web_paths.models_dir / "whisper-models" / "ggml-small.bin"
    assert target.is_file()
    assert downloader.calls == [("ggerganov/whisper.cpp", "ggml-small.bin")]

    model = _find(client, "whisper-small")
    assert model["status"]["present"] is True
    assert model["download"]["status"] == "done"
    assert model["download"]["fraction"] == 1.0

    events = client.app.state.download_bus.history()  # type: ignore[attr-defined]
    statuses = [event["status"] for event in events if event["id"] == "whisper-small"]
    assert "downloading" in statuses
    assert statuses[-1] == "done"


def test_download_snapshot_pyannote_with_token(
    client: TestClient, web_paths: WebPaths, downloader: FakeDownloader
) -> None:
    client.put("/api/settings", json={"hf_token": "hf_secret_token"})

    assert client.post("/api/models/pyannote-community-1/download").status_code == 202
    _manager(client).wait("pyannote-community-1")

    target = web_paths.models_dir / "pyannote-models" / "speaker-diarization-community-1"
    assert (target / "config.yaml").is_file()
    assert _find(client, "pyannote-community-1")["status"]["present"] is True


def test_download_snapshot_gigaam(
    client: TestClient, web_paths: WebPaths, downloader: FakeDownloader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Снимок GigaAM большой — не упираемся в реальное свободное место на диске.
    monkeypatch.setattr(web_app, "free_space", lambda _path: 10**12)

    assert client.post("/api/models/gigaam-v3-onnx/download").status_code == 202
    _manager(client).wait("gigaam-v3-onnx")

    target = web_paths.models_dir / "gigaam-models" / "gigaam-v3-onnx"
    assert (target / "config.yaml").is_file()
    assert downloader.calls == [("istupakov/gigaam-v3-onnx", "<snapshot>")]
    assert _find(client, "gigaam-v3-onnx")["status"]["present"] is True


def test_download_gated_without_token_is_rejected(client: TestClient) -> None:
    response = client.post("/api/models/pyannote-community-1/download")

    assert response.status_code == 400
    assert "токен" in response.json()["detail"].lower()


def test_download_insufficient_space(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_app, "free_space", lambda _path: 10)

    response = client.post("/api/models/whisper-large-v3-turbo/download")

    assert response.status_code == 400
    assert "места" in response.json()["detail"]


def test_download_unknown_model_404(client: TestClient) -> None:
    assert client.post("/api/models/does-not-exist/download").status_code == 404


def test_download_twice_conflicts(web_paths: WebPaths) -> None:
    block = threading.Event()
    downloader = FakeDownloader(block=block)
    with _make_client(web_paths, downloader) as test_client:
        assert test_client.post("/api/models/whisper-small/download").status_code == 202
        second = test_client.post("/api/models/whisper-small/download")
        assert second.status_code == 409
        block.set()
        _manager(test_client).wait("whisper-small")


def test_cancel_download(web_paths: WebPaths) -> None:
    with _make_client(web_paths, LoopingDownloader()) as test_client:
        assert test_client.post("/api/models/whisper-small/download").status_code == 202
        assert test_client.post("/api/models/whisper-small/cancel").status_code == 200
        _manager(test_client).wait("whisper-small")

        model = _find(test_client, "whisper-small")
        assert model["download"]["status"] == "cancelled"

        # Отмена на неактивной модели — 409.
        assert test_client.post("/api/models/whisper-small/cancel").status_code == 409


def test_delete_model(client: TestClient, web_paths: WebPaths) -> None:
    client.post("/api/models/whisper-small/download")
    _manager(client).wait("whisper-small")
    target = web_paths.models_dir / "whisper-models" / "ggml-small.bin"
    assert target.is_file()

    response = client.delete("/api/models/whisper-small")

    assert response.status_code == 200
    assert response.json() == {"deleted": "whisper-small", "removed": True}
    assert not target.exists()
    assert _find(client, "whisper-small")["status"]["present"] is False

    # Повторное удаление — нечего удалять.
    assert client.delete("/api/models/whisper-small").json()["removed"] is False


def test_delete_unknown_model_404(client: TestClient) -> None:
    assert client.delete("/api/models/nope").status_code == 404


def test_download_bus_history_and_clear() -> None:
    bus = web_models.DownloadBus(heartbeat=0.01)
    bus.publish({"id": "m", "status": "downloading"})
    bus.publish({"id": "m", "status": "done"})

    assert [event["status"] for event in bus.history()] == ["downloading", "done"]
    bus.clear()
    assert bus.history() == []


def test_download_bus_live_subscription() -> None:
    bus = web_models.DownloadBus(heartbeat=0.01)

    async def scenario() -> list[str]:
        agen = bus.subscribe()
        consumer = asyncio.ensure_future(anext(agen))
        await asyncio.sleep(0.005)
        bus.publish({"id": "m", "status": "downloading"})
        event = await asyncio.wait_for(consumer, timeout=1.0)
        while event is None:  # keep-alive по heartbeat — ждём настоящее событие
            event = await asyncio.wait_for(anext(agen), timeout=1.0)
        await agen.aclose()
        return [str(event["status"])]

    assert asyncio.run(scenario()) == ["downloading"]


def _find(client: TestClient, model_id: str) -> dict[str, object]:
    for model in client.get("/api/models").json()["models"]:
        if model["id"] == model_id:
            return model
    raise AssertionError(f"модель {model_id} не найдена")
