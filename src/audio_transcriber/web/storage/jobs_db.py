"""SQLite-хранилище задач транскрибации (stdlib ``sqlite3``).

Каждый вызов открывает короткое соединение — это безопасно при обращении из
потока воркера и потоков HTTP-запросов FastAPI. Схема намеренно плоская:
одна таблица ``jobs`` с состоянием, стадией и путём к JSON-результату.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: Статусы жизненного цикла задачи.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

#: Статусы, после которых задача больше не меняется.
TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED})

#: Поля, которые разрешено менять через :meth:`JobsDB.update` (защита от SQL-инъекций).
_UPDATABLE_FIELDS = frozenset(
    {
        "status",
        "started_at",
        "finished_at",
        "language",
        "duration",
        "error",
        "result_path",
        "stage",
        "fraction",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    language TEXT,
    duration REAL,
    error TEXT,
    result_path TEXT,
    stage TEXT,
    fraction REAL
)
"""


def utc_now_iso() -> str:
    """Текущее время UTC в формате ISO-8601 (секунды, без микросекунд)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class Job:
    """Запись задачи (строка таблицы ``jobs``)."""

    id: str
    source_path: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    language: str | None = None
    duration: float | None = None
    error: str | None = None
    result_path: str | None = None
    stage: str | None = None
    fraction: float | None = None

    @property
    def name(self) -> str:
        """Имя исходного файла без каталогов."""
        return Path(self.source_path).name

    @property
    def is_terminal(self) -> bool:
        """Задача в конечном статусе (``done``/``error``/``cancelled``)."""
        return self.status in TERMINAL_STATUSES

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответов API."""
        return {
            "id": self.id,
            "name": self.name,
            "source_path": self.source_path,
            "status": self.status,
            "stage": self.stage,
            "fraction": self.fraction,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "language": self.language,
            "duration": self.duration,
            "error": self.error,
            "result_path": self.result_path,
        }


class JobsDB:
    """Доступ к SQLite-БД задач."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Путь к файлу БД."""
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def initialize(self) -> None:
        """Создаёт каталог и таблицу задач, если их ещё нет."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_SCHEMA)

    def create(
        self, job_id: str, source_path: str | Path, *, language: str | None = None
    ) -> Job:
        """Создаёт задачу в статусе ``queued`` и возвращает её."""
        created_at = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, source_path, status, created_at, language) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, str(source_path), STATUS_QUEUED, created_at, language),
            )
        job = self.get(job_id)
        assert job is not None  # только что вставили
        return job

    def get(self, job_id: str) -> Job | None:
        """Возвращает задачу по id или ``None``."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row is not None else None

    def list(self) -> list[Job]:
        """Все задачи, новые сверху."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC")
            return [_row_to_job(row) for row in rows.fetchall()]

    def update(self, job_id: str, **fields: object) -> Job | None:
        """Обновляет перечисленные поля задачи; неизвестные поля игнорируются."""
        allowed = {key: value for key, value in fields.items() if key in _UPDATABLE_FIELDS}
        if allowed:
            assignments = ", ".join(f"{key} = ?" for key in allowed)
            values = [*allowed.values(), job_id]
            with self._connect() as connection:
                connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
        return self.get(job_id)

    def delete(self, job_id: str) -> bool:
        """Удаляет задачу; возвращает ``True``, если строка существовала."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cursor.rowcount > 0


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=str(row["id"]),
        source_path=str(row["source_path"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        language=row["language"],
        duration=row["duration"],
        error=row["error"],
        result_path=row["result_path"],
        stage=row["stage"],
        fraction=row["fraction"],
    )
