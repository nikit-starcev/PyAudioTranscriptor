"""Локальное SQLite-хранилище истории чата по стенограмме (issue #54/#96).

Диалог с LLM по активной задаче сохраняется, чтобы вопросы и ответы не терялись
при перезагрузке страницы и при переключении задач. Хранилище, как и
:mod:`audio_transcriber.storage.prompts_db`, работает только на стандартной
библиотеке (:mod:`sqlite3`) и не обращается к сети.

Схема::

    chat_messages(id, job_id, role, content, citations, created_at)
    INDEX (job_id, id)

``citations`` — JSON-массив ссылок на реплики (индекс/таймкоды/говорящий/текст),
пустой для сообщений пользователя. Сообщения читаются в порядке добавления.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Версия схемы БД чата (``PRAGMA user_version``).
SCHEMA_VERSION = 1

#: Допустимые роли сообщений.
VALID_ROLES = frozenset({"user", "assistant"})

#: Максимальная длина одного сообщения (защита от мусора/переполнения контекста).
MAX_MESSAGE_LENGTH = 20000


def _now_iso() -> str:
    """Текущее время в формате ISO-8601 (UTC, секунды)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Одно сообщение чата (вопрос пользователя или ответ LLM)."""

    id: int
    job_id: str
    role: str
    content: str
    citations: tuple[dict[str, object], ...] = field(default_factory=tuple)
    created_at: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответа API."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "citations": [dict(item) for item in self.citations],
            "created_at": self.created_at,
        }


class ChatDB:
    """SQLite-хранилище истории чата по задачам.

    Создание файла и схемы идемпотентно; поддерживается использование как
    контекстного менеджера (``with ChatDB(path) as db: ...``).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chat_job
                    ON chat_messages(job_id, id);
                """
            )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        """Закрывает соединение с базой."""
        self._conn.close()

    def __enter__(self) -> ChatDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def add_message(
        self,
        job_id: str,
        role: str,
        content: str,
        *,
        citations: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    ) -> int:
        """Добавляет сообщение и возвращает его id.

        :raises ValueError: неизвестная роль или пустое/слишком длинное сообщение.
        """
        clean_role = (role or "").strip()
        if clean_role not in VALID_ROLES:
            raise ValueError(f"Неизвестная роль сообщения: {role!r}")
        clean_content = (content or "").strip()
        if not clean_content:
            raise ValueError("Сообщение не может быть пустым")
        if len(clean_content) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Сообщение длиннее {MAX_MESSAGE_LENGTH} символов")
        payload = json.dumps(list(citations or []), ensure_ascii=False)
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO chat_messages (job_id, role, content, citations, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, clean_role, clean_content, payload, _now_iso()),
            )
        return int(cursor.lastrowid or 0)

    def list_messages(self, job_id: str) -> list[ChatMessage]:
        """Сообщения задачи в порядке добавления."""
        rows = self._conn.execute(
            "SELECT * FROM chat_messages WHERE job_id = ? ORDER BY id",
            (job_id,),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    def count(self, job_id: str | None = None) -> int:
        """Число сообщений (всего или по конкретной задаче)."""
        if job_id is None:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM chat_messages").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return int(row["c"]) if row is not None else 0

    def clear(self, job_id: str) -> int:
        """Удаляет историю задачи; возвращает число удалённых сообщений."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM chat_messages WHERE job_id = ?", (job_id,)
            )
        return int(cursor.rowcount or 0)


def _row_to_message(row: sqlite3.Row) -> ChatMessage:
    raw = row["citations"]
    citations: tuple[dict[str, object], ...] = ()
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = []
        if isinstance(parsed, list):
            citations = tuple(item for item in parsed if isinstance(item, dict))
    return ChatMessage(
        id=int(row["id"]),
        job_id=str(row["job_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        citations=citations,
        created_at=row["created_at"],
    )
