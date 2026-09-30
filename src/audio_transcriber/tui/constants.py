"""Константы TUI: этапы прогресса, расширения медиафайлов и метки статусов."""

from __future__ import annotations

from rich.text import Text

#: Этапы конвейера в порядке выполнения (ключ стадии, подпись для интерфейса).
STAGES: list[tuple[str, str]] = [
    ("denoise", "Шумоподавление"),
    ("asr", "Распознавание речи"),
    ("diarization", "Определение говорящих"),
    ("merge", "Объединение сегментов"),
    ("clean", "Очистка артефактов"),
    ("correction", "Автоисправление опечаток"),
    ("llm", "LLM-постобработка"),
    ("export", "Экспорт"),
]

# Расширения аудио- и видеофайлов, которые показываются в дереве файлов.
MEDIA_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".ape",
    ".amr",
    ".mp2",
    ".mpga",
    ".wv",
    ".mp4",
    ".m4v",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".ts",
    ".mts",
    ".m2ts",
    ".3gp",
    ".3g2",
    ".ogv",
    ".vob",
}

#: Метка статуса строки очереди для таблицы.
_STATUS_MARK = {
    "pending": Text("·", style="dim"),
    "running": Text("→", style="bold cyan"),
    "done": Text("✓", style="green"),
    "error": Text("✗", style="red"),
}
