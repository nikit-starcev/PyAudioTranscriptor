"""Тесты разрешения вычислительного устройства (CPU/CUDA)."""

from __future__ import annotations

import pytest

from audio_transcriber.domain.enums import Device
from audio_transcriber.utils import device as device_module
from audio_transcriber.utils.exceptions import DeviceNotAvailableError


def test_cpu_is_always_resolved_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "is_cuda_available", lambda: True)

    assert device_module.resolve_device(Device.CPU) is Device.CPU


def test_auto_resolves_to_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "is_cuda_available", lambda: True)

    assert device_module.resolve_device(Device.AUTO) is Device.CUDA


def test_auto_resolves_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(device_module, "is_cuda_available", lambda: False)

    assert device_module.resolve_device(Device.AUTO) is Device.CPU


def test_explicit_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "is_cuda_available", lambda: False)

    with pytest.raises(DeviceNotAvailableError):
        device_module.resolve_device(Device.CUDA)


def test_explicit_cuda_succeeds_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "is_cuda_available", lambda: True)

    assert device_module.resolve_device(Device.CUDA) is Device.CUDA


def test_is_cuda_available_reflects_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    # Функция кэширует результат — сбрасываем кэш между проверками.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device_module.is_cuda_available.cache_clear()
    assert device_module.is_cuda_available() is True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device_module.is_cuda_available.cache_clear()
    assert device_module.is_cuda_available() is False
