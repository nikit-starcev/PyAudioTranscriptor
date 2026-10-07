"""Локальное SQLite-хранилище пользовательских шаблонов промпта резюме (#97).

Раньше системный промпт резюме был зашит в :mod:`audio_transcriber.llm.summary`.
Теперь пользователь может завести свои шаблоны (формат протокола, стиль,
язык, акценты, длина), отредактировать их и выбрать активный. Активный шаблон
подставляется в резюме при постановке задачи и по кнопке «Сформировать
протокол».

Модуль, как и :mod:`audio_transcriber.storage.glossary_db`, работает только на
стандартной библиотеке (:mod:`sqlite3`) и не обращается к сети.

Схема::

    prompts(id, name UNIQUE, body, builtin, created_at, updated_at)
    meta(key PRIMARY KEY, value)

Активный шаблон хранится в ``meta`` под ключом :data:`ACTIVE_KEY` как id
записи. При открытии пустой БД засеваются дефолтные шаблоны
(:data:`~audio_transcriber.llm.summary.DEFAULT_SUMMARY_TEMPLATES`), а активным
делается «Стандартный».
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from audio_transcriber.llm.summary import (
    DEFAULT_SUMMARY_TEMPLATE_NAME,
    DEFAULT_SUMMARY_TEMPLATES,
)

logger = logging.getLogger(__name__)

#: Версия схемы БД шаблонов (``PRAGMA user_version``).
SCHEMA_VERSION = 1

#: Ключ активного шаблона в таблице ``meta``.
ACTIVE_KEY = "active_summary_prompt_id"

#: Максимальная длина тела шаблона (защита от мусора/переполнения контекста).
MAX_PROMPT_LENGTH = 20000


def _now_iso() -> str:
    """Текущее время в формате ISO-8601 (UTC, секунды)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Prompt:
    """Шаблон промпта резюме."""

    id: int
    name: str
    body: str
    builtin: bool
    created_at: str | None
    updated_at: str | None


class PromptsDB:
    """SQLite-хранилище шаблонов промпта резюме.

    Создание файла и схемы идемпотентно; при первом открытии БД засеваются
    дефолтные шаблоны. Поддерживается использование как контекстного менеджера
    (``with PromptsDB(path) as db: ...``).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        self.seed_defaults()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def _create_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    body TEXT NOT NULL,
                    builtin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        """Закрывает соединение с базой."""
        self._conn.close()

    def __enter__(self) -> PromptsDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Дефолтные шаблоны
    # ------------------------------------------------------------------
    def seed_defaults(self) -> None:
        """Идемпотентно добавляет дефолтные шаблоны и делает один активным.

        Существующие одноимённые записи не трогаются (пользователь мог их
        отредактировать). Активным становится первый дефолт, если активный ещё
        не выбран.
        """
        with self._conn:
            for name, body in DEFAULT_SUMMARY_TEMPLATES:
                self._conn.execute(
                    "INSERT OR IGNORE INTO prompts "
                    "(name, body, builtin, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
                    (name, body, _now_iso(), _now_iso()),
                )
        if self.active_id() is None:
            default = self.get_by_name(DEFAULT_SUMMARY_TEMPLATE_NAME)
            if default is not None:
                self.set_active(default.id)

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def list_prompts(self) -> list[Prompt]:
        """Все шаблоны: сначала встроенные, затем пользовательские, по имени."""
        rows = self._conn.execute(
            "SELECT * FROM prompts ORDER BY builtin DESC, name"
        ).fetchall()
        return [_row_to_prompt(row) for row in rows]

    def get_prompt(self, prompt_id: int) -> Prompt | None:
        """Шаблон по id или ``None``."""
        row = self._conn.execute(
            "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
        ).fetchone()
        return _row_to_prompt(row) if row is not None else None

    def get_by_name(self, name: str) -> Prompt | None:
        """Шаблон по имени или ``None``."""
        row = self._conn.execute(
            "SELECT * FROM prompts WHERE name = ?", (name,)
        ).fetchone()
        return _row_to_prompt(row) if row is not None else None

    def count(self) -> int:
        """Общее число шаблонов."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM prompts").fetchone()
        return int(row["c"]) if row is not None else 0

    def active_id(self) -> int | None:
        """id активного шаблона или ``None``, если он ещё не выбран."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (ACTIVE_KEY,)
        ).fetchone()
        if row is None:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def active_prompt(self) -> Prompt | None:
        """Активный шаблон; если он удалён — первый доступный."""
        active = self.active_id()
        if active is not None:
            prompt = self.get_prompt(active)
            if prompt is not None:
                return prompt
        prompts = self.list_prompts()
        return prompts[0] if prompts else None

    # ------------------------------------------------------------------
    # Изменение
    # ------------------------------------------------------------------
    def add_prompt(self, name: str, body: str, *, builtin: bool = False) -> int:
        """Добавляет шаблон и возвращает его id.

        :raises ValueError: пустое имя/тело или слишком длинное тело.
        :raises sqlite3.IntegrityError: имя уже занято.
        """
        clean_name = _clean_name(name)
        clean_body = _clean_body(body)
        now = _now_iso()
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO prompts (name, body, builtin, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (clean_name, clean_body, 1 if builtin else 0, now, now),
            )
        return int(cursor.lastrowid or 0)

    def update_prompt(
        self,
        prompt_id: int,
        *,
        name: str | None = None,
        body: str | None = None,
    ) -> bool:
        """Обновляет имя/тело шаблона. ``False`` — записи не было.

        ``None`` означает «поле не трогать». Пустые/слишком длинные значения —
        :class:`ValueError`; конфликт имени — :class:`sqlite3.IntegrityError`.
        """
        fields: dict[str, object] = {}
        if name is not None:
            fields["name"] = _clean_name(name)
        if body is not None:
            fields["body"] = _clean_body(body)
        if not fields:
            return self.get_prompt(prompt_id) is not None
        fields["updated_at"] = _now_iso()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._conn:
            cursor = self._conn.execute(
                f"UPDATE prompts SET {assignments} WHERE id = ?",
                [*fields.values(), prompt_id],
            )
        return cursor.rowcount > 0

    def delete_prompt(self, prompt_id: int) -> bool:
        """Удаляет шаблон. ``False`` — записи не было."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
        if cursor.rowcount > 0 and self.active_id() == prompt_id:
            # Активный удалён — переключаемся на первый доступный.
            remaining = self.list_prompts()
            if remaining:
                self.set_active(remaining[0].id)
            else:
                with self._conn:
                    self._conn.execute("DELETE FROM meta WHERE key = ?", (ACTIVE_KEY,))
        return cursor.rowcount > 0

    def set_active(self, prompt_id: int) -> bool:
        """Делает шаблон активным. ``False`` — шаблона не существует."""
        if self.get_prompt(prompt_id) is None:
            return False
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (ACTIVE_KEY, str(prompt_id)),
            )
        return True


def _clean_name(name: str) -> str:
    """Нормализует имя шаблона, проверяя непустоту."""
    clean = " ".join((name or "").split())
    if not clean:
        raise ValueError("Имя шаблона не может быть пустым")
    return clean


def _clean_body(body: str) -> str:
    """Нормализует тело шаблона, проверяя непустоту и длину."""
    clean = (body or "").strip()
    if not clean:
        raise ValueError("Тело шаблона не может быть пустым")
    if len(clean) > MAX_PROMPT_LENGTH:
        raise ValueError(f"Тело шаблона длиннее {MAX_PROMPT_LENGTH} символов")
    return clean


def _row_to_prompt(row: sqlite3.Row) -> Prompt:
    return Prompt(
        id=int(row["id"]),
        name=str(row["name"]),
        body=str(row["body"]),
        builtin=bool(row["builtin"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
