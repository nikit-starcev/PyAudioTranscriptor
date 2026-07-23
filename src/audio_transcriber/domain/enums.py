"""Перечисления, используемые доменной моделью, конфигурацией и CLI."""

from __future__ import annotations

from enum import Enum


class Device(str, Enum):
    """Устройство, на котором должны выполняться вычисления."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ExportFormat(str, Enum):
    """Поддерживаемые форматы экспорта стенограммы."""

    TXT = "txt"
    DOCX = "docx"
    JSON = "json"
    SRT = "srt"
