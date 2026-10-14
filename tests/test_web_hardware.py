"""Тесты эндпоинта автоопределения железа и профилей настроек (#146)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber import hardware
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


def _patch_hardware(
    monkeypatch: pytest.MonkeyPatch,
    *,
    total_ram_gb: float,
    gpu: hardware.GpuProfile | None = None,
    binaries: frozenset[str] = frozenset({"whisper-cli"}),
) -> None:
    profile = hardware.HardwareProfile(
        os_name="Linux",
        cpu=hardware.CpuProfile(name="Test", logical_cores=8, physical_cores=4),
        total_ram_gb=total_ram_gb,
        available_ram_gb=total_ram_gb,
        gpu=gpu or hardware.GpuProfile(),
        free_disk_gb=100.0,
    )
    monkeypatch.setattr(
        hardware, "detect_hardware", lambda *, disk_path=None: profile  # noqa: ARG005
    )
    monkeypatch.setattr(
        "audio_transcriber.web.app.binary_available", lambda binary: binary in binaries
    )


def test_get_hardware_endpoint_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hardware(monkeypatch, total_ram_gb=8.0)

    payload = client.get("/api/hardware").json()

    assert set(payload) >= {"hardware", "recommendation", "available_models"}
    assert payload["hardware"]["total_ram_gb"] == 8.0
    rec = payload["recommendation"]
    assert set(rec) >= {
        "profile",
        "asr_backend",
        "device",
        "diarization_engine",
        "denoise",
        "llm_enabled",
        "engine_isolation",
        "found",
        "proposed",
        "why",
        "apply",
    }
    assert rec["found"].startswith("нашли:")
    assert rec["apply"]["asr_backend"] == rec["asr_backend"]


def test_hardware_recommends_weak_on_low_ram(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_hardware(monkeypatch, total_ram_gb=4.0)

    rec = client.get("/api/hardware").json()["recommendation"]

    assert rec["profile"] == "weak"
    assert rec["diarization_engine"] == "sherpa"
    assert rec["apply"]["engine_isolation"] == "off"


def test_hardware_recommends_cuda_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu = hardware.GpuProfile(present=True, vendor="nvidia", vram_gb=12.0, accelerators=("cuda",))
    _patch_hardware(monkeypatch, total_ram_gb=32.0, gpu=gpu, binaries=frozenset({"llama-server"}))

    rec = client.get("/api/hardware").json()["recommendation"]

    assert rec["asr_backend"] == "faster-whisper"
    assert rec["device"] == "cuda"
    assert rec["llm_enabled"] is True


def test_settings_expose_engine_isolation_and_all_presets(client: TestClient) -> None:
    payload = client.get("/api/settings").json()

    assert payload["engine_isolation"] == "off"
    presets = {item["id"] for item in payload["hardware_presets"]}
    assert presets == {"weak", "balanced", "performance"}


def test_put_settings_applies_balanced_preset(client: TestClient) -> None:
    response = client.put("/api/settings", json={"preset": "balanced"})

    assert response.status_code == 200
    body = response.json()
    assert body["asr_backend"] == "whisper-cpp"
    assert body["diarization_engine"] == "auto"
    assert body["denoise"] is True
    assert body["llm_enabled"] is False
    assert body["engine_isolation"] == "off"


def test_put_settings_applies_performance_preset(client: TestClient) -> None:
    response = client.put("/api/settings", json={"preset": "performance"})

    assert response.status_code == 200
    body = response.json()
    assert body["diarization_hybrid_enabled"] is True
    assert body["llm_enabled"] is True
    assert body["engine_isolation"] == "subprocess"


def test_put_settings_engine_isolation_roundtrip(client: TestClient) -> None:
    response = client.put("/api/settings", json={"engine_isolation": "subprocess"})

    assert response.status_code == 200
    assert response.json()["engine_isolation"] == "subprocess"
    assert client.get("/api/settings").json()["engine_isolation"] == "subprocess"


def test_put_settings_rejects_unknown_engine_isolation(client: TestClient) -> None:
    response = client.put("/api/settings", json={"engine_isolation": "docker"})

    assert response.status_code == 400
    assert "изоляц" in response.json()["detail"]
