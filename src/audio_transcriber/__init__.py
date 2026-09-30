"""AudioTranscriptor — локальная утилита для транскрибации аудиозаписей
с разделением говорящих (speaker diarization).
"""

from importlib.metadata import PackageNotFoundError, version as _package_version

# Единственный источник версии — ``pyproject.toml``; здесь читаем метаданные
# установленного пакета, чтобы версия не дублировалась в двух местах.
try:
    __version__ = _package_version("audio-transcriber")
except PackageNotFoundError:  # исходники без установки (например, запуск из src)
    __version__ = "0.1.0"
