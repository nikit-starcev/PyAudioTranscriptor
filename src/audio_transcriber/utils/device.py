"""Определение и разрешение вычислительного устройства (CPU/CUDA).

``torch`` импортируется лениво внутри функций, чтобы команды CLI,
не требующие фактических вычислений (например, ``--help``), запускались
быстро и не зависели от наличия PyTorch на этапе импорта пакета.
"""

from __future__ import annotations

import logging

from audio_transcriber.domain.enums import Device
from audio_transcriber.utils.exceptions import DeviceNotAvailableError

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Проверяет доступность CUDA через PyTorch."""

    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolve_device(requested: Device) -> Device:
    """Приводит запрошенное устройство к фактически используемому.

    При ``Device.AUTO`` выбирает CUDA, если она доступна, иначе — CPU.
    При явном запросе CUDA без доступного GPU выбрасывает исключение.
    """

    if requested is Device.CUDA:
        if not is_cuda_available():
            raise DeviceNotAvailableError(
                "Запрошено устройство CUDA, но совместимый GPU/драйверы CUDA "
                "не найдены. Укажите --device cpu или --device auto."
            )
        return Device.CUDA

    if requested is Device.CPU:
        return Device.CPU

    resolved = Device.CUDA if is_cuda_available() else Device.CPU
    logger.debug("Автоматически выбрано устройство: %s", resolved.value)
    return resolved
