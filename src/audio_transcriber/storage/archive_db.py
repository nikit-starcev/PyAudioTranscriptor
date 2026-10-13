"""Локальное SQLite-хранилище архива стенограмм (issue #99).

Архив привязывает готовые стенограммы задач к датам и встречам: заголовок,
дата (редактируемая), участники/теги и свободные заметки. Поверх текстов
стенограмм строится полнотекстовый поиск с фильтрами по говорящему и дате.

Модуль, как и :mod:`audio_transcriber.storage.prompts_db` и
:mod:`audio_transcriber.storage.chat_db`, работает только на стандартной
библиотеке (:mod:`sqlite3`) и не обращается к сети.

Схема::

    archive_meta(job_id PK, title, meeting_date, participants, notes,
                 created_at, updated_at)
    archive_index_state(job_id PK, result_mtime, entries, indexed_at)
    archive_fts(text, job_id UNINDEXED, entry_index UNINDEXED, speaker UNINDEXED)
        -- виртуальная таблица FTS5, если доступна

Если FTS5-модуль недоступен (сборка SQLite без него), используется
индексированный LIKE-фолбэк::

    archive_entries(job_id, entry_index, speaker, text, text_lower,
                    PRIMARY KEY(job_id, entry_index))

``participants`` хранится JSON-массивом строк. Версия схемы фиксируется в
``PRAGMA user_version`` — идемпотентная миграция по версии.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Версия схемы БД архива (``PRAGMA user_version``).
SCHEMA_VERSION = 1

#: Ограничение длины заголовка записи архива.
MAX_TITLE_LENGTH = 300
#: Ограничение длины заметки (защита от мусора/переполнения).
MAX_NOTES_LENGTH = 20000
#: Ограничения участников: длина одного имени и общее число.
MAX_PARTICIPANT_LENGTH = 200
MAX_PARTICIPANTS = 200

#: Значения по умолчанию и потолок для выдачи поиска/списка.
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 200

#: Радиус окна сниппета вокруг найденного токена (символы).
_SNIPPET_RADIUS = 90

#: Разбор пользовательского запроса на слова (кириллица/латиница/цифры).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Сентинел «поле не передано» для частичного обновления записи.
_UNSET: object = object()


def _now_iso() -> str:
    """Текущее время в формате ISO-8601 (UTC, секунды)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean_date(value: str) -> str | None:
    """Нормализует дату к ``YYYY-MM-DD``; пустая строка/``None`` — очистка."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Некорректная дата: {value!r} (ожидается ГГГГ-ММ-ДД)") from exc
    return parsed.isoformat()


def _clean_title(value: str) -> str | None:
    """Нормализует заголовок; пустая строка очищает поле."""
    text = (value or "").strip()
    if not text:
        return None
    if len(text) > MAX_TITLE_LENGTH:
        raise ValueError(f"Заголовок длиннее {MAX_TITLE_LENGTH} символов")
    return text


def _clean_notes(value: str) -> str | None:
    """Нормализует заметки; пустая строка очищает поле."""
    text = (value or "").strip()
    if not text:
        return None
    if len(text) > MAX_NOTES_LENGTH:
        raise ValueError(f"Заметка длиннее {MAX_NOTES_LENGTH} символов")
    return text


def _clean_participants(values: object) -> tuple[str, ...]:
    """Приводит участников к упорядоченному кортежу без дублей и пустых."""
    if values is None:
        return ()
    if isinstance(values, str):
        raw_items: list[object] = [values]
    elif isinstance(values, Sequence):
        raw_items = list(values)
    else:
        raise ValueError("Участники должны быть списком строк")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, str):
            raise ValueError("Участники должны быть списком строк")
        name = " ".join(item.split())
        if not name:
            continue
        if len(name) > MAX_PARTICIPANT_LENGTH:
            raise ValueError(f"Имя участника длиннее {MAX_PARTICIPANT_LENGTH} символов")
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)
    if len(cleaned) > MAX_PARTICIPANTS:
        raise ValueError(f"Участников больше {MAX_PARTICIPANTS}")
    return tuple(cleaned)


def _parse_participants(raw: object) -> tuple[str, ...]:
    """Читает участников из JSON-колонки (терпимо к мусору)."""
    if not isinstance(raw, str) or not raw:
        return ()
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return ()
    if not isinstance(data, list):
        return ()
    return tuple(item for item in data if isinstance(item, str))


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    """Метаданные записи архива (привязка стенограммы к дате/встрече)."""

    job_id: str
    title: str | None
    meeting_date: str | None
    participants: tuple[str, ...]
    notes: str | None
    created_at: str | None
    updated_at: str | None

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответа API."""
        return {
            "job_id": self.job_id,
            "title": self.title,
            "meeting_date": self.meeting_date,
            "participants": list(self.participants),
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Одна реплика стенограммы, найденная полнотекстовым поиском."""

    job_id: str
    entry_index: int
    speaker: str | None
    snippet: str
    score: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответа API."""
        return {
            "job_id": self.job_id,
            "entry_index": self.entry_index,
            "speaker": self.speaker,
            "snippet": self.snippet,
            "score": self.score,
        }


class ArchiveDB:
    """SQLite-хранилище метаданных архива и индекса текстов стенограмм.

    Создание файла и схемы идемпотентно. Полнотекстовый поиск использует
    FTS5, если модуль доступен у сборки SQLite, иначе — индексированный
    LIKE-фолбэк. Поддерживается использование как контекстного менеджера
    (``with ArchiveDB(path) as db: ...``).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            logger.warning("Не удалось включить WAL для %s", path)
        self.fts5 = False
        self._create_schema()

    # ------------------------------------------------------------------
    # Жизненный цикл и схема
    # ------------------------------------------------------------------
    def _create_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_meta (
                    job_id TEXT PRIMARY KEY,
                    title TEXT,
                    meeting_date TEXT,
                    participants TEXT,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS archive_index_state (
                    job_id TEXT PRIMARY KEY,
                    result_mtime INTEGER,
                    entries INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS archive_entries (
                    job_id TEXT NOT NULL,
                    entry_index INTEGER NOT NULL,
                    speaker TEXT,
                    text TEXT NOT NULL,
                    text_lower TEXT NOT NULL,
                    PRIMARY KEY (job_id, entry_index)
                );

                CREATE INDEX IF NOT EXISTS idx_archive_entries_job
                    ON archive_entries(job_id);
                """
            )
            self._migrate()
        self.fts5 = self._try_create_fts()

    def _migrate(self) -> None:
        """Идемпотентно доводит схему до :data:`SCHEMA_VERSION`."""
        version = self.schema_version()
        if version < 1:
            # Базовые таблицы уже созданы выше (``IF NOT EXISTS``) — v1 считаем
            # применённой и лишь фиксируем версию.
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _try_create_fts(self) -> bool:
        """Создаёт FTS5-таблицу; ``False`` — модуль FTS5 недоступен."""
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5("
                    "text, job_id UNINDEXED, entry_index UNINDEXED, "
                    "speaker UNINDEXED, tokenize='unicode61')"
                )
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 недоступен, поиск по архиву через LIKE: %s", exc)
            return False
        return True

    def schema_version(self) -> int:
        """Версия схемы БД (``PRAGMA user_version``)."""
        row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row is not None else 0

    def close(self) -> None:
        """Закрывает соединение с базой."""
        self._conn.close()

    def __enter__(self) -> ArchiveDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Метаданные (CRUD)
    # ------------------------------------------------------------------
    def get_record(self, job_id: str) -> ArchiveRecord | None:
        """Запись архива по id задачи или ``None``."""
        row = self._conn.execute(
            "SELECT * FROM archive_meta WHERE job_id = ?", (job_id,)
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list_records(self) -> list[ArchiveRecord]:
        """Все записи архива, свежие по дате встречи сверху."""
        rows = self._conn.execute(
            "SELECT * FROM archive_meta "
            "ORDER BY COALESCE(meeting_date, '') DESC, updated_at DESC, job_id"
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def ensure_record(self, job_id: str, *, default_date: str | None = None) -> ArchiveRecord:
        """Возвращает запись, создавая её при отсутствии с датой по умолчанию."""
        record = self.get_record(job_id)
        if record is not None:
            return record
        return self.upsert_record(job_id, meeting_date=default_date)

    def upsert_record(
        self,
        job_id: str,
        *,
        title: object = _UNSET,
        meeting_date: object = _UNSET,
        participants: object = _UNSET,
        notes: object = _UNSET,
    ) -> ArchiveRecord:
        """Создаёт/обновляет запись архива; ``_UNSET`` — поле не трогать.

        Для ``title``/``meeting_date``/``notes`` пустая строка очищает поле, а
        ``None`` тоже очищает (для частичного PATCH вызывающий передаёт лишь
        те поля, что пришли). ``participants`` — список строк, ``[]`` очищает.
        """
        now = _now_iso()
        current = self.get_record(job_id)
        if current is None:
            new_title = _clean_title(title) if isinstance(title, str) else None
            new_date = _clean_date(meeting_date) if isinstance(meeting_date, str) else None
            new_participants = _clean_participants(participants) if participants is not _UNSET else ()
            new_notes = _clean_notes(notes) if isinstance(notes, str) else None
            with self._conn:
                self._conn.execute(
                    "INSERT INTO archive_meta "
                    "(job_id, title, meeting_date, participants, notes, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        new_title,
                        new_date,
                        json.dumps(list(new_participants), ensure_ascii=False),
                        new_notes,
                        now,
                        now,
                    ),
                )
            created = self.get_record(job_id)
            assert created is not None
            return created

        fields: dict[str, object] = {}
        if title is not _UNSET:
            fields["title"] = _clean_title(title) if isinstance(title, str) else None
        if meeting_date is not _UNSET:
            fields["meeting_date"] = (
                _clean_date(meeting_date) if isinstance(meeting_date, str) else None
            )
        if participants is not _UNSET:
            fields["participants"] = json.dumps(
                list(_clean_participants(participants)), ensure_ascii=False
            )
        if notes is not _UNSET:
            fields["notes"] = _clean_notes(notes) if isinstance(notes, str) else None
        if fields:
            fields["updated_at"] = now
            assignments = ", ".join(f"{key} = ?" for key in fields)
            with self._conn:
                self._conn.execute(
                    f"UPDATE archive_meta SET {assignments} WHERE job_id = ?",
                    [*fields.values(), job_id],
                )
        updated = self.get_record(job_id)
        assert updated is not None  # запись существовала выше
        return updated

    def delete_record(self, job_id: str) -> bool:
        """Удаляет метаданные записи архива; ``False`` — записи не было."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM archive_meta WHERE job_id = ?", (job_id,)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Индекс текста стенограмм
    # ------------------------------------------------------------------
    def index_state(self, job_id: str) -> tuple[bool, int | None, int]:
        """Состояние индекса задачи: ``(индексирован, mtime, число реплик)``."""
        row = self._conn.execute(
            "SELECT result_mtime, entries FROM archive_index_state WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return False, None, 0
        mtime = row["result_mtime"]
        return True, int(mtime) if mtime is not None else None, int(row["entries"] or 0)

    def is_current(self, job_id: str, result_mtime: int) -> bool:
        """Индекс задачи актуален для указанного времени изменения результата."""
        indexed, mtime, _ = self.index_state(job_id)
        return indexed and mtime == result_mtime

    def index_job(
        self,
        job_id: str,
        entries: Sequence[tuple[int, str | None, str]],
        *,
        result_mtime: int,
    ) -> int:
        """Переиндексирует текст стенограммы задачи (замена предыдущих строк)."""
        with self._conn:
            if self.fts5:
                self._conn.execute("DELETE FROM archive_fts WHERE job_id = ?", (job_id,))
                self._conn.executemany(
                    "INSERT INTO archive_fts (text, job_id, entry_index, speaker) "
                    "VALUES (?, ?, ?, ?)",
                    [(text, job_id, index, speaker) for index, speaker, text in entries],
                )
            else:
                self._conn.execute(
                    "DELETE FROM archive_entries WHERE job_id = ?", (job_id,)
                )
                self._conn.executemany(
                    "INSERT INTO archive_entries "
                    "(job_id, entry_index, speaker, text, text_lower) VALUES (?, ?, ?, ?, ?)",
                    [
                        (job_id, index, speaker, text, text.casefold())
                        for index, speaker, text in entries
                    ],
                )
            self._conn.execute(
                "INSERT INTO archive_index_state "
                "(job_id, result_mtime, entries, indexed_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET "
                "result_mtime = excluded.result_mtime, entries = excluded.entries, "
                "indexed_at = excluded.indexed_at",
                (job_id, result_mtime, len(entries), _now_iso()),
            )
        return len(entries)

    def remove_job(self, job_id: str) -> None:
        """Удаляет метаданные, индекс и состояние индекса задачи."""
        with self._conn:
            if self.fts5:
                self._conn.execute("DELETE FROM archive_fts WHERE job_id = ?", (job_id,))
            else:
                self._conn.execute(
                    "DELETE FROM archive_entries WHERE job_id = ?", (job_id,)
                )
            self._conn.execute(
                "DELETE FROM archive_index_state WHERE job_id = ?", (job_id,)
            )
            self._conn.execute("DELETE FROM archive_meta WHERE job_id = ?", (job_id,))

    def indexed_job_ids(self) -> set[str]:
        """id задач, текст которых уже проиндексирован."""
        rows = self._conn.execute("SELECT job_id FROM archive_index_state").fetchall()
        return {str(row["job_id"]) for row in rows}

    def speakers(self) -> list[str]:
        """Уникальные имена говорящих во всём индексе (для фильтра поиска)."""
        if self.fts5:
            rows = self._conn.execute(
                "SELECT DISTINCT speaker FROM archive_fts "
                "WHERE speaker IS NOT NULL AND speaker <> '' ORDER BY speaker"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT speaker FROM archive_entries "
                "WHERE speaker IS NOT NULL AND speaker <> '' ORDER BY speaker"
            ).fetchall()
        return [str(row["speaker"]) for row in rows]

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        job_ids: Sequence[str] | None = None,
        speaker: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        offset: int = 0,
    ) -> tuple[list[SearchHit], int]:
        """Полнотекстовый поиск по текстам стенограмм.

        ``job_ids`` ограничивает поиск набором задач (например, по диапазону
        дат); ``speaker`` — точное имя говорящего; ``query`` — слова, которые
        ищутся по префиксам (регистронезависимо, включая кириллицу). Возвращает
        ``(хиты, общее число совпадений)``.
        """
        tokens = _TOKEN_RE.findall(query or "")
        if not tokens:
            return [], 0
        if job_ids is not None and not job_ids:
            return [], 0
        clean_speaker = (speaker or "").strip() or None
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        offset = max(0, int(offset))

        if self.fts5:
            hits, total = self._search_fts(tokens, job_ids, clean_speaker, limit, offset)
        else:
            hits, total = self._search_like(tokens, job_ids, clean_speaker, limit, offset)
        return hits, total

    def _search_fts(
        self,
        tokens: Sequence[str],
        job_ids: Sequence[str] | None,
        speaker: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[SearchHit], int]:
        match = " ".join(f'"{token}"*' for token in tokens)
        clauses = ["archive_fts MATCH ?"]
        params: list[object] = [match]
        if job_ids is not None:
            placeholders = ", ".join("?" for _ in job_ids)
            clauses.append(f"job_id IN ({placeholders})")
            params.extend(job_ids)
        if speaker is not None:
            clauses.append("speaker = ?")
            params.append(speaker)
        where = " AND ".join(clauses)
        total_row = self._conn.execute(
            f"SELECT COUNT(*) AS c FROM archive_fts WHERE {where}", params
        ).fetchone()
        total = int(total_row["c"]) if total_row is not None else 0
        rows = self._conn.execute(
            "SELECT job_id, entry_index, speaker, text, bm25(archive_fts) AS rank "
            f"FROM archive_fts WHERE {where} "
            "ORDER BY rank, job_id, entry_index LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        hits = [
            SearchHit(
                job_id=str(row["job_id"]),
                entry_index=int(row["entry_index"]),
                speaker=row["speaker"],
                snippet=_snippet(str(row["text"]), tokens),
                score=float(row["rank"]) if row["rank"] is not None else None,
            )
            for row in rows
        ]
        return hits, total

    def _search_like(
        self,
        tokens: Sequence[str],
        job_ids: Sequence[str] | None,
        speaker: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[SearchHit], int]:
        clauses: list[str] = []
        params: list[object] = []
        for token in tokens:
            clauses.append("text_lower LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(token.casefold())}%")
        if job_ids is not None:
            placeholders = ", ".join("?" for _ in job_ids)
            clauses.append(f"job_id IN ({placeholders})")
            params.extend(job_ids)
        if speaker is not None:
            clauses.append("speaker = ?")
            params.append(speaker)
        where = " AND ".join(clauses)
        total_row = self._conn.execute(
            f"SELECT COUNT(*) AS c FROM archive_entries WHERE {where}", params
        ).fetchone()
        total = int(total_row["c"]) if total_row is not None else 0
        rows = self._conn.execute(
            "SELECT job_id, entry_index, speaker, text FROM archive_entries "
            f"WHERE {where} ORDER BY job_id, entry_index LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        hits = [
            SearchHit(
                job_id=str(row["job_id"]),
                entry_index=int(row["entry_index"]),
                speaker=row["speaker"],
                snippet=_snippet(str(row["text"]), tokens),
            )
            for row in rows
        ]
        return hits, total


def _escape_like(value: str) -> str:
    """Экранирует ``%``/``_``/``\\`` для LIKE с ``ESCAPE '\\'``."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _snippet(text: str, tokens: Sequence[str], radius: int = _SNIPPET_RADIUS) -> str:
    """Фрагмент текста вокруг первого найденного токена (для выдачи поиска)."""
    lowered = text.casefold()
    position = -1
    for token in tokens:
        found = lowered.find(token.casefold())
        if found >= 0 and (position < 0 or found < position):
            position = found
    if position < 0:
        return text.strip()[: radius * 2]
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _row_to_record(row: sqlite3.Row) -> ArchiveRecord:
    return ArchiveRecord(
        job_id=str(row["job_id"]),
        title=row["title"],
        meeting_date=row["meeting_date"],
        participants=_parse_participants(row["participants"]),
        notes=row["notes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
