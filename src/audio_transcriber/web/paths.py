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

#: Каталог собранной статики SPA внутри пакета.
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True, slots=True)
class WebPaths:
    """Производные пути внутри каталога данных веб-интерфейса."""

    data_dir: Path

    @classmethod
    def default(cls) -> WebPaths:
        """Каталог данных из окружения или ``./web-data``."""
        raw = os.environ.get(DATA_DIR_ENV, "").strip()
        return cls(Path(raw) if raw else Path(DEFAULT_DATA_DIR))

    @property
    def jobs_db(self) -> Path:
        """SQLite-база задач."""
        return self.data_dir / "jobs.db"

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
        ):
            directory.mkdir(parents=True, exist_ok=True)
