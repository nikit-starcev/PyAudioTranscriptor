"""Иерархия исключений приложения.

Каждый компонент конвейера выбрасывает своё собственное исключение,
унаследованное от :class:`AudioTranscriberError`, чтобы CLI мог
единообразно перехватывать ошибки и показывать понятные сообщения.
"""

from __future__ import annotations


class AudioTranscriberError(Exception):
    """Базовое исключение приложения."""


class ConfigurationError(AudioTranscriberError):
    """Некорректные параметры конфигурации или аргументы CLI."""


class AudioFileError(AudioTranscriberError):
    """Проблемы с исходным аудиофайлом (не найден, не читается и т.п.)."""


class DeviceNotAvailableError(AudioTranscriberError):
    """Запрошенное устройство (например, CUDA) недоступно в системе."""


class TranscriptionError(AudioTranscriberError):
    """Ошибка на этапе распознавания речи."""


class DiarizationError(AudioTranscriberError):
    """Ошибка на этапе определения говорящих."""


class MergeError(AudioTranscriberError):
    """Ошибка при объединении результатов распознавания и диаризации."""


class ExportError(AudioTranscriberError):
    """Ошибка при экспорте результата в выбранный формат."""
