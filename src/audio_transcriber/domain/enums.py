"""Перечисления, используемые доменной моделью, конфигурацией и CLI."""

from __future__ import annotations

from enum import StrEnum


class Device(StrEnum):
    """Устройство, на котором должны выполняться вычисления."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class AsrBackend(StrEnum):
    """Движок распознавания речи."""

    FASTER_WHISPER = "faster-whisper"
    WHISPER_CPP = "whisper-cpp"


class ExportFormat(StrEnum):
    """Поддерживаемые форматы экспорта стенограммы."""

    TXT = "txt"
    DOCX = "docx"
    JSON = "json"
    SRT = "srt"
