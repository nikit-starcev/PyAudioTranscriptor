"""Опции выпадающих списков TUI и их нормализация по ``config.env``."""

from __future__ import annotations

from audio_transcriber.domain.enums import ExportFormat

_FORMAT_OPTIONS = [
    ("txt", "txt"),
    ("txt + docx", "txt,docx"),
    ("txt + docx + json + srt", "txt,docx,json,srt"),
]

_LANG_OPTIONS = [
    ("авто", ""),
    ("ru", "ru"),
    ("en", "en"),
]

_BACKEND_OPTIONS = [
    ("whisper-cpp (Vulkan)", "whisper-cpp"),
    ("faster-whisper", "faster-whisper"),
    ("gigaam (onnx-asr, RU)", "gigaam"),
]

_DEVICE_OPTIONS = [
    ("auto", "auto"),
    ("cpu", "cpu"),
    ("cuda", "cuda"),
]

_VALID_FORMATS = {fmt.value for fmt in ExportFormat}


def _format_select(raw: str) -> tuple[list[tuple[str, str]], str]:
    """Строит опции форматов и выбранное значение из строки ``FORMATS``.

    Если в ``config.env`` задан набор, которого нет среди стандартных, он
    добавляется отдельным пунктом — настройка не теряется.
    """
    normalized = ",".join(part.strip() for part in raw.split(",") if part.strip())
    parts = normalized.split(",") if normalized else []
    if parts and all(part in _VALID_FORMATS for part in parts):
        for _, value in _FORMAT_OPTIONS:
            if value == normalized:
                return _FORMAT_OPTIONS, normalized
        label = " + ".join(parts)
        return [*_FORMAT_OPTIONS, (label, normalized)], normalized
    return _FORMAT_OPTIONS, ExportFormat.TXT.value


def _select_value(raw: str, options: list[tuple[str, str]], default: str) -> str:
    """Возвращает ближайшее допустимое значение выпадающего списка."""
    values = {value for _, value in options}
    candidate = (raw or "").strip()
    if candidate in values:
        return candidate
    for value in values:
        if value.lower() == candidate.lower():
            return value
    return default
