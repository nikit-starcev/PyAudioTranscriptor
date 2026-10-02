"""Определение фактического устройства ASR для индикатора в UI (#72).

Модуль отвечает на вопрос «на чём пойдёт распознавание речи» и отдаёт
человекочитаемую подпись для шапки/панели прогресса. Для whisper.cpp устройство
определяется по собранному Vulkan-бэкенду и перечисленным GPU (тот же механизм,
что и в :mod:`audio_transcriber.doctor`); для GigaAM/faster-whisper — по
настройке ``DEVICE``. Денойз и диаризация в проекте всегда выполняются на CPU:
об этом сообщает поле ``note``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from audio_transcriber import doctor
from audio_transcriber.domain.enums import AsrBackend, Device

logger = logging.getLogger(__name__)

#: Стадии, которые всегда считаются на CPU (денойз/диаризация не используют GPU).
CPU_ONLY_NOTE = "денойз и диаризация выполняются на CPU"


@dataclass(frozen=True, slots=True)
class AsrDeviceInfo:
    """Устройство распознавания речи для индикатора в UI.

    ``device`` — обобщённый класс (``gpu``/``cpu``/``unknown``); ``label`` —
    готовая строка вида «whisper.cpp · GPU Vulkan0 (AMD Radeon RX 590)»;
    ``accelerator``/``name`` — детали для подсказки; ``details`` — сырые строки
    проб (для tooltip); ``note`` — пометка о стадиях, всегда идущих на CPU.
    """

    backend: str
    device: str
    label: str
    accelerator: str | None = None
    name: str | None = None
    note: str = CPU_ONLY_NOTE
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API."""
        return {
            "backend": self.backend,
            "device": self.device,
            "label": self.label,
            "accelerator": self.accelerator,
            "name": self.name,
            "note": self.note,
            "details": list(self.details),
        }


def _backend(env: Mapping[str, str]) -> str:
    """Эффективный бэкенд ASR из окружения (неизвестный → faster-whisper)."""
    raw = str(env.get("ASR_BACKEND", "")).strip().casefold()
    try:
        return AsrBackend(raw).value
    except ValueError:
        return AsrBackend.FASTER_WHISPER.value


def _device_setting(env: Mapping[str, str]) -> Device:
    """Запрошенное устройство ``DEVICE`` (неизвестное → ``auto``)."""
    raw = str(env.get("DEVICE", "")).strip().casefold()
    try:
        return Device(raw)
    except ValueError:
        return Device.AUTO


def _format_vulkan_device(line: str) -> tuple[str, str | None]:
    """«Vulkan0: AMD Radeon RX 590» → шильдик «Vulkan0 (AMD Radeon RX 590)».

    Возвращает готовую подпись и имя устройства (``None``, если строка не
    разобрана на «индекс: имя»).
    """
    index, separator, name = line.partition(":")
    if separator and name.strip():
        return f"{index.strip()} ({name.strip()})", name.strip()
    return line.strip(), None


def _whisper_cpp_info(env: Mapping[str, str]) -> AsrDeviceInfo:
    """Устройство whisper.cpp: реальный GPU Vulkan либо честный CPU."""
    binary = str(env.get("WHISPER_CPP_BINARY", "")).strip() or doctor.DEFAULT_WHISPER_BINARY
    lib_path = str(env.get("WHISPER_CPP_LIB_PATH", "")).strip() or None
    backend = AsrBackend.WHISPER_CPP.value
    if not doctor._binary_available(binary):
        return AsrDeviceInfo(
            backend=backend,
            device="unknown",
            label="whisper.cpp · устройство неизвестно",
            details=["whisper-cli не найден — устройство определить нельзя"],
        )

    # 1. Устройства, если билд поддерживает ``--list-devices``.
    devices = doctor._vulkan_devices(binary, lib_path)
    if devices:
        badge, name = _format_vulkan_device(devices[0])
        return AsrDeviceInfo(
            backend=backend,
            device="gpu",
            accelerator="vulkan",
            name=name,
            label=f"whisper.cpp · GPU {badge}",
            details=list(devices[:3]),
        )

    # 2. Собран ли Vulkan вообще (libggml-vulkan рядом или GPU-флаги в --help).
    vulkan_built = doctor._vulkan_library_present(
        lib_path, binary
    ) or doctor._supports_gpu_flags(doctor._whisper_help(binary, lib_path))
    if vulkan_built:
        # 3. whisper-device не перечислил — пробуем vulkaninfo.
        info_devices = doctor._vulkaninfo_devices()
        if info_devices:
            return AsrDeviceInfo(
                backend=backend,
                device="gpu",
                accelerator="vulkan",
                name=info_devices[0],
                label=f"whisper.cpp · GPU Vulkan ({info_devices[0]})",
                details=list(info_devices[:3]),
            )
        return AsrDeviceInfo(
            backend=backend,
            device="cpu",
            label="whisper.cpp · CPU (Vulkan собран, GPU не обнаружен)",
            details=["Vulkan-бэкенд собран, но устройство не перечислено"],
        )
    return AsrDeviceInfo(backend=backend, device="cpu", label="whisper.cpp · CPU")


def _onnx_cuda_available() -> bool:
    """Есть ли в onnxruntime рабочий CUDA-провайдер (для GigaAM)."""
    try:
        from audio_transcriber.transcription.gigaam_engine import resolve_onnx_providers
    except Exception:  # noqa: BLE001 — onnx-asr необязателен: мягкая деградация
        return False
    try:
        return resolve_onnx_providers(Device.CUDA) is not None
    except Exception:  # noqa: BLE001 — onnxruntime может быть недоступен
        return False


def _gigaam_info(env: Mapping[str, str]) -> AsrDeviceInfo:
    """Устройство GigaAM: CUDA при доступном провайдере, иначе CPU."""
    backend = AsrBackend.GIGAAM.value
    requested = _device_setting(env)
    if requested is Device.CUDA and _onnx_cuda_available():
        return AsrDeviceInfo(
            backend=backend,
            device="gpu",
            accelerator="cuda",
            label="GigaAM · GPU CUDA",
        )
    details: list[str] = []
    if requested is Device.CUDA:
        details.append("CUDA недоступна в onnxruntime — используется CPU")
    return AsrDeviceInfo(
        backend=backend, device="cpu", label="CPU (GigaAM)", details=details
    )


def _faster_whisper_info(env: Mapping[str, str]) -> AsrDeviceInfo:
    """Устройство faster-whisper по настройке ``DEVICE`` (CUDA или CPU)."""
    backend = AsrBackend.FASTER_WHISPER.value
    requested = _device_setting(env)
    if requested is Device.CUDA:
        return AsrDeviceInfo(
            backend=backend,
            device="gpu",
            accelerator="cuda",
            label="GPU CUDA (faster-whisper)",
        )
    details: list[str] = []
    if requested is Device.AUTO:
        details.append("DEVICE=auto — при наличии CUDA выбирается GPU")
    return AsrDeviceInfo(
        backend=backend, device="cpu", label="CPU (faster-whisper)", details=details
    )


def describe_asr_device(env: Mapping[str, str]) -> AsrDeviceInfo:
    """Определяет устройство ASR по эффективным настройкам окружения.

    Провайдеры: whisper.cpp — по Vulkan/GPU (пробы ``doctor``); GigaAM и
    faster-whisper — по ``DEVICE``. Неизвестный бэкенд деградирует к ``unknown``.
    """
    backend = _backend(env)
    if backend == AsrBackend.WHISPER_CPP.value:
        return _whisper_cpp_info(env)
    if backend == AsrBackend.GIGAAM.value:
        return _gigaam_info(env)
    if backend == AsrBackend.FASTER_WHISPER.value:
        return _faster_whisper_info(env)
    return AsrDeviceInfo(
        backend=backend,
        device="unknown",
        label=f"{backend} · устройство неизвестно",
        details=[f"неизвестный бэкенд: {backend}"],
    )
