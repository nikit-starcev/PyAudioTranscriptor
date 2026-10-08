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
    #: GigaAM v3 (RU) через onnx-asr/ONNX Runtime — без torch. Опциональная
    #: зависимость: при её отсутствии движок мягко деградирует с явной ошибкой.
    GIGAAM = "gigaam"


class ExportFormat(StrEnum):
    """Поддерживаемые форматы экспорта стенограммы."""

    TXT = "txt"
    DOCX = "docx"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    MARKDOWN = "md"
    PDF = "pdf"
