"""Тесты индикатора устройства ASR (#72).

Проверяют поля ``GET /api/asr/device``: определение GPU Vulkan для whisper.cpp
(с подменой проб :mod:`audio_transcriber.doctor`), CPU-подписи для
GigaAM/faster-whisper, мягкую деградацию при неизвестном устройстве и пометку,
что денойз/диаризация всегда идут на CPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber import doctor
from audio_transcriber.web import asr_device as asr_module
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


def _select_backend(client: TestClient, **fields: object) -> None:
    """Сохраняет настройки веб-интерфейса (они перекрывают ``config.env``)."""
    response = client.put("/api/settings", json=fields)
    assert response.status_code == 200, response.text


def _no_vulkan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет все Vulkan-пробы на «ничего не найдено»."""
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    monkeypatch.setattr(doctor, "_vulkan_devices", lambda _binary, _lib: None)
    monkeypatch.setattr(doctor, "_vulkan_library_present", lambda _lib, _binary: False)
    monkeypatch.setattr(doctor, "_supports_gpu_flags", lambda _help: False)
    monkeypatch.setattr(doctor, "_whisper_help", lambda _binary, _lib: None)
    monkeypatch.setattr(doctor, "_vulkaninfo_devices", lambda: None)


def test_asr_device_faster_whisper_cpu(client: TestClient) -> None:
    """faster-whisper на CPU — подпись без GPU."""
    _select_backend(client, asr_backend="faster-whisper", device="cpu")

    payload = client.get("/api/asr/device").json()

    assert payload["backend"] == "faster-whisper"
    assert payload["device"] == "cpu"
    assert payload["accelerator"] is None
    assert payload["label"] == "CPU (faster-whisper)"
    assert "CPU" in payload["note"]


def test_asr_device_note_marks_cpu_only_stages(client: TestClient) -> None:
    _select_backend(client, asr_backend="faster-whisper", device="cpu")

    payload = client.get("/api/asr/device").json()

    assert "денойз" in payload["note"]
    assert "диаризация" in payload["note"]
    assert "CPU" in payload["note"]


def test_asr_device_whisper_cpp_gpu_vulkan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_backend(client, asr_backend="whisper-cpp", whisper_cpp_binary="whisper-cli")
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    monkeypatch.setattr(
        doctor, "_vulkan_devices", lambda _binary, _lib: ["Vulkan0: AMD Radeon RX 590"]
    )

    payload = client.get("/api/asr/device").json()

    assert payload["backend"] == "whisper-cpp"
    assert payload["device"] == "gpu"
    assert payload["accelerator"] == "vulkan"
    assert payload["name"] == "AMD Radeon RX 590"
    assert payload["label"] == "whisper.cpp · GPU Vulkan0 (AMD Radeon RX 590)"


def test_asr_device_whisper_cpp_cpu_without_vulkan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_backend(client, asr_backend="whisper-cpp", whisper_cpp_binary="whisper-cli")
    _no_vulkan(monkeypatch)

    payload = client.get("/api/asr/device").json()

    assert payload["backend"] == "whisper-cpp"
    assert payload["device"] == "cpu"
    assert payload["accelerator"] is None
    assert payload["label"] == "whisper.cpp · CPU"


def test_asr_device_whisper_cpp_vulkaninfo_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--list-devices`` не поддержан, но Vulkan собран — имя даёт vulkaninfo."""
    _select_backend(client, asr_backend="whisper-cpp", whisper_cpp_binary="whisper-cli")
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    monkeypatch.setattr(doctor, "_vulkan_devices", lambda _binary, _lib: None)
    monkeypatch.setattr(doctor, "_vulkan_library_present", lambda _lib, _binary: True)
    monkeypatch.setattr(doctor, "_supports_gpu_flags", lambda _help: True)
    monkeypatch.setattr(doctor, "_whisper_help", lambda _binary, _lib: "--device")
    monkeypatch.setattr(
        doctor,
        "_vulkaninfo_devices",
        lambda: ["AMD Radeon RX 590 Series (RADV POLARIS10)"],
    )

    payload = client.get("/api/asr/device").json()

    assert payload["device"] == "gpu"
    assert payload["accelerator"] == "vulkan"
    assert "AMD Radeon RX 590 Series" in payload["label"]


def test_asr_device_whisper_cpp_unknown_when_binary_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Нет бинарника — устройство неизвестно, мягкая деградация."""
    _select_backend(client, asr_backend="whisper-cpp", whisper_cpp_binary="whisper-cli")
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: False)

    payload = client.get("/api/asr/device").json()

    assert payload["device"] == "unknown"
    assert "неизвестно" in payload["label"]
    assert payload["details"]


def test_asr_device_gigaam_cpu(client: TestClient) -> None:
    _select_backend(client, asr_backend="gigaam", device="cpu")

    payload = client.get("/api/asr/device").json()

    assert payload["backend"] == "gigaam"
    assert payload["device"] == "cpu"
    assert payload["label"] == "CPU (GigaAM)"


def test_asr_device_gigaam_cuda_falls_back_without_onnx(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_backend(client, asr_backend="gigaam", device="cuda")
    monkeypatch.setattr(asr_module, "_onnx_cuda_available", lambda: False)

    payload = client.get("/api/asr/device").json()

    assert payload["device"] == "cpu"
    assert payload["label"] == "CPU (GigaAM)"
    assert any("CUDA" in item for item in payload["details"])


def test_asr_device_gigaam_cuda_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_backend(client, asr_backend="gigaam", device="cuda")
    monkeypatch.setattr(asr_module, "_onnx_cuda_available", lambda: True)

    payload = client.get("/api/asr/device").json()

    assert payload["device"] == "gpu"
    assert payload["accelerator"] == "cuda"
    assert payload["label"] == "GigaAM · GPU CUDA"


def test_asr_device_faster_whisper_cuda(client: TestClient) -> None:
    _select_backend(client, asr_backend="faster-whisper", device="cuda")

    payload = client.get("/api/asr/device").json()

    assert payload["device"] == "gpu"
    assert payload["accelerator"] == "cuda"
    assert payload["label"] == "GPU CUDA (faster-whisper)"
