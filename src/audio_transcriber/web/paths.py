"""Пути данных веб-интерфейса (``web-data/`` по умолчанию).

Все пользовательские данные (загруженные файлы, БД задач, результаты,
образцы голоса) лежат в одном каталоге данных, чтобы его можно было целиком
исключить из git. Каталог можно переопределить переменной окружения
``AUDIO_TRANSCRIBER_WEB_DATA``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Каталог данных веб-интерфейса по умолчанию (относительно рабочего каталога).
DEFAULT_DATA_DIR = "web-data"

#: Переменная окружения для переопределения каталога данных.
DATA_DIR_ENV = "AUDIO_TRANSCRIBER_WEB_DATA"

#: Переменная окружения для переопределения каталога скачиваемых моделей.
MODELS_DIR_ENV = "AUDIO_TRANSCRIBER_MODELS_DIR"

#: Каталог собранной статики SPA внутри пакета.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True, slots=True)
class WebPaths:
    """Производные пути внутри каталога данных веб-интерфейса."""

    data_dir: Path
    #: Переопределение каталога моделей; ``None`` — ``<data_dir>/models``
    #: (или значение ``AUDIO_TRANSCRIBER_MODELS_DIR``). Используется в тестах.
    models_dir_override: Path | None = None

    @classmethod
    def default(cls) -> WebPaths:
        """Каталог данных из окружения или ``./web-data``."""
        raw = os.environ.get(DATA_DIR_ENV, "").strip()
        models_raw = os.environ.get(MODELS_DIR_ENV, "").strip()
        return cls(
            Path(raw) if raw else Path(DEFAULT_DATA_DIR),
            Path(models_raw) if models_raw else None,
        )

    @property
    def models_dir(self) -> Path:
        """Каталог скачанных моделей по умолчанию."""
        return self.models_dir_override or (self.data_dir / "models")

    @property
    def bin_dir(self) -> Path:
        """Каталог автоустановленных внешних бинарников (#98)."""
        return self.data_dir / "bin"

    @property
    def jobs_db(self) -> Path:
        """SQLite-база задач."""
        return self.data_dir / "jobs.db"

    @property
    def settings_json(self) -> Path:
        """JSON с редактируемыми настройками веб-интерфейса."""
        return self.data_dir / "settings.json"

    @property
    def prompts_db(self) -> Path:
        """SQLite-база пользовательских шаблонов промпта резюме (#97)."""
        return self.data_dir / "prompts.db"

    @property
    def chat_db(self) -> Path:
        """SQLite-база истории чата по стенограмме (#54/#96)."""
        return self.data_dir / "chat.db"

    @property
    def secrets_json(self) -> Path:
        """Секреты (токен Hugging Face) с правами доступа только владельцу."""
        return self.data_dir / "secrets.json"

    @property
    def input_dir(self) -> Path:
        """Каталог загруженных/выбираемых аудиофайлов."""
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        """Каталог JSON-результатов и постадийного вывода конвейера."""
        return self.data_dir / "results"

    @property
    def samples_dir(self) -> Path:
        """Каталог образцов голоса (WAV) для прослушивания."""
        return self.data_dir / "samples"

    @property
    def cache_dir(self) -> Path:
        """Общий постадийный кэш конвейера."""
        return self.data_dir / "cache"

    def ensure(self) -> None:
        """Создаёт все каталоги данных, если их ещё нет."""
        for directory in (
            self.data_dir,
            self.input_dir,
            self.results_dir,
            self.samples_dir,
            self.cache_dir,
            self.models_dir,
            self.bin_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
