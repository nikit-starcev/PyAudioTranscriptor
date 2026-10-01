"""SQLite-хранилище задач транскрибации (stdlib ``sqlite3``).

Каждый вызов открывает короткое соединение — это безопасно при обращении из
потока воркера и потоков HTTP-запросов FastAPI. Схема намеренно плоская:
одна таблица ``jobs`` с состоянием, стадией и путём к JSON-результату.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from audio_transcriber.web.timings import StageTiming

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
        "num_speakers",
        "min_speakers",
        "max_speakers",
        "stage_started_at",
        "stage_times",
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
    fraction REAL,
    num_speakers INTEGER,
    min_speakers INTEGER,
    max_speakers INTEGER,
    stage_started_at TEXT,
    stage_times TEXT
)
"""


def utc_now_iso() -> str:
    """Текущее время UTC в формате ISO-8601 (секунды, без микросекунд)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    """Разбирает ISO-8601 из БД; ``None`` при пустом/некорректном значении."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _serialize_stage_times(value: object) -> str | None:
    """Приводит тайминги к JSON-строке для колонки ``stage_times``.

    Принимает список :class:`StageTiming` (штатный путь воркера) или список
    словарей (совместимость); ``None`` очищает колонку.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        return None
    payload: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, StageTiming):
            payload.append(item.as_dict())
        elif isinstance(item, Mapping):
            payload.append(dict(item))
    return json.dumps(payload, ensure_ascii=False)


def _parse_stage_times(raw: object) -> list[StageTiming]:
    """Читает тайминги из колонки ``stage_times`` (терпимо к мусору)."""
    if not isinstance(raw, str) or not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    timings: list[StageTiming] = []
    for item in data:
        timing = StageTiming.from_mapping(item)
        if timing is not None:
            timings.append(timing)
    return timings


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
    #: Ожидаемое число говорящих; ``None`` — автоопределение (pyannote сам решает).
    num_speakers: int | None = None
    #: Нижняя/верхняя граница числа говорящих; ``None`` — без ограничения.
    #: Игнорируются, если задано точное ``num_speakers``.
    min_speakers: int | None = None
    max_speakers: int | None = None
    #: Когда началась текущая стадия (ISO); ``None`` — стадия ещё не сообщалась.
    stage_started_at: str | None = None
    #: Длительности завершённых стадий в порядке выполнения.
    stage_times: list[StageTiming] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Имя исходного файла без каталогов."""
        return Path(self.source_path).name

    @property
    def is_terminal(self) -> bool:
        """Задача в конечном статусе (``done``/``error``/``cancelled``)."""
        return self.status in TERMINAL_STATUSES

    @property
    def total_seconds(self) -> float | None:
        """Общее время обработки: ``finished_at − started_at`` или текущее.

        Для незапущенной задачи (нет ``started_at``) — ``None``; для идущей
        задачи — время с момента старта до текущего момента.
        """
        start = _parse_iso(self.started_at)
        if start is None:
            return None
        end = _parse_iso(self.finished_at) or datetime.now(UTC)
        return round(max((end - start).total_seconds(), 0.0), 3)

    @property
    def stage_elapsed(self) -> float | None:
        """Сколько уже длится текущая стадия (для живого таймера); ``None`` — нет."""
        if self.is_terminal:
            return None
        start = _parse_iso(self.stage_started_at)
        if start is None:
            return None
        return round(max((datetime.now(UTC) - start).total_seconds(), 0.0), 3)

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
            "num_speakers": self.num_speakers,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "stage_started_at": self.stage_started_at,
            "stage_times": [timing.as_dict() for timing in self.stage_times],
            "total_seconds": self.total_seconds,
            "stage_elapsed": self.stage_elapsed,
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
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Добавляет недостающие колонки в уже существующую таблицу.

        ``CREATE TABLE IF NOT EXISTS`` не меняет старую схему, поэтому для баз,
        созданных до появления ``num_speakers``/``min_speakers``/
        ``max_speakers``/``stage_started_at``/``stage_times``, колонки добавляем
        отдельно.
        """
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
        if "num_speakers" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN num_speakers INTEGER")
        if "min_speakers" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN min_speakers INTEGER")
        if "max_speakers" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN max_speakers INTEGER")
        if "stage_started_at" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN stage_started_at TEXT")
        if "stage_times" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN stage_times TEXT")

    def create(
        self,
        job_id: str,
        source_path: str | Path,
        *,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Job:
        """Создаёт задачу в статусе ``queued`` и возвращает её."""
        created_at = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs "
                "(id, source_path, status, created_at, language, num_speakers, "
                "min_speakers, max_speakers) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    str(source_path),
                    STATUS_QUEUED,
                    created_at,
                    language,
                    num_speakers,
                    min_speakers,
                    max_speakers,
                ),
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
        allowed = {
            key: _serialize_stage_times(value) if key == "stage_times" else value
            for key, value in fields.items()
            if key in _UPDATABLE_FIELDS
        }
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
        num_speakers=row["num_speakers"],
        min_speakers=row["min_speakers"],
        max_speakers=row["max_speakers"],
        stage_started_at=row["stage_started_at"],
        stage_times=_parse_stage_times(row["stage_times"]),
    )
