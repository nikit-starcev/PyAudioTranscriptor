"""Локальное SQLite-хранилище глоссария терминов.

Исторически глоссарий хранился в текстовых файлах: по одному термину (или
явной паре ``"ошибочная форма = канон"``) в строке, ``#`` — комментарий.
Такие файлы поддерживаются по-прежнему (см.
:mod:`audio_transcriber.llm.glossary`), но объёмный глоссарий удобнее держать
в локальной базе SQLite: несколько источников (файлы ТЗ, пользовательский
список, выгрузка из Confluence), включение/отключение источника или отдельной
записи, поиск и дополнение без перезаписи файла.

Модуль работает только на стандартной библиотеке (:mod:`sqlite3`) и не
обращается к сети: файл БД лежит рядом с рабочим каталогом.

Схема::

    sources(id, name UNIQUE, kind, path, enabled, imported_at)
    entries(id, canonical, variant, category, note, source_id → sources.id,
            enabled, created_at, updated_at, UNIQUE(canonical, variant, source_id))

``canonical`` — канонический термин, ``variant`` — ошибочная форма, которую
слышит ASR (может быть пустой). Для матчера пары отдаются строкой
``"variant = canonical"``, одиночные термины — каноном (см.
:meth:`GlossaryDB.terms_for_matcher`).
"""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Разделитель явной пары «ошибочная форма = канон».
_PAIR_SEPARATOR = "="

#: Версия схемы БД глоссария (``PRAGMA user_version``). v1 — таблицы
#: ``sources``/``entries``; v2 — служебная таблица ``meta`` для отметок
#: миграций. Схема создаётся/дополняется идемпотентно до этой версии.
SCHEMA_VERSION = 2

#: Ключ в ``meta`` с версией схемы, на которой выполнялась миграция старых
#: текстовых глоссариев (``GLOSSARY_PATH``) в БД. Сама миграция идемпотентна
#: по имени источника, поэтому отметка нужна лишь для наблюдаемости.
LEGACY_GLOSSARY_MIGRATION_KEY = "legacy_glossary_migration"

#: Версия схемы БД глоссария (``PRAGMA user_version``). v1 — таблицы
#: ``sources``/``entries``; v2 — служебная таблица ``meta`` для отметок
#: миграций. Схема создаётся/дополняется идемпотентно до этой версии.
SCHEMA_VERSION = 2

#: Ключ в ``meta`` с версией схемы, на которой выполнялась миграция старых
#: текстовых глоссариев (``GLOSSARY_PATH``) в БД. Сама миграция идемпотентна
#: по имени источника, поэтому отметка нужна лишь для наблюдаемости.
LEGACY_GLOSSARY_MIGRATION_KEY = "legacy_glossary_migration"

# Кодировка чтения текстовых глоссариев (utf-8-sig съедает BOM, если он есть).
_TEXT_ENCODING = "utf-8-sig"

# Распознаваемые заголовки CSV (в нижнем регистре) → смысловая колонка.
_HEADER_ALIASES: dict[str, str] = {
    "term": "term",
    "термин": "term",
    "wrong": "variant",
    "ошибка": "variant",
    "вариант": "variant",
    "ошибочная": "variant",
    "ошибочная форма": "variant",
    "canonical": "canonical",
    "канон": "canonical",
    "правильно": "canonical",
    "правильная форма": "canonical",
    "note": "note",
    "заметка": "note",
    "примечание": "note",
    "комментарий": "note",
    "category": "category",
    "категория": "category",
}

# Разделители CSV, которые пробуем распознать автоматически.
_CSV_DELIMITERS = ",;\t"

# Тип разобранной записи: (canonical, variant, note, category).
_Row = tuple[str, str | None, str | None, str | None]


def _now_iso() -> str:
    """Текущее время в формате ISO-8601 (UTC, секунды)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize_spaces(value: str) -> str:
    """Схлопывает любые пробельные последовательности к одному пробелу."""
    return " ".join(value.split())


def _split_pair(value: str) -> tuple[str, str] | None:
    """Разбирает строку ``"ошибочная форма = канон"``.

    Возвращает ``(variant, canonical)`` либо ``None``, если разделителя нет
    или одна из частей пуста.
    """
    if _PAIR_SEPARATOR not in value:
        return None
    wrong, _, canonical = value.partition(_PAIR_SEPARATOR)
    wrong, canonical = wrong.strip(), canonical.strip()
    if not wrong or not canonical:
        return None
    return wrong, canonical


@dataclass(frozen=True, slots=True)
class Source:
    """Источник глоссария (файл, ручной список, выгрузка)."""

    id: int
    name: str
    kind: str
    path: str | None
    enabled: bool
    imported_at: str | None


@dataclass(frozen=True, slots=True)
class Entry:
    """Запись глоссария: канон и, опционально, ошибочная форма."""

    id: int
    canonical: str
    variant: str | None
    category: str | None
    note: str | None
    source_id: int | None
    enabled: bool
    created_at: str | None
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class ImportReport:
    """Итог импорта одного источника."""

    source: str
    kind: str
    added: int
    skipped: int
    total: int
    replaced: bool = False


class GlossaryDB:
    """SQLite-хранилище глоссария.

    Создание файла и схемы идемпотентно: повторный вызов с тем же путём
    ничего не теряет. Поддерживается использование как контекстного
    менеджера (``with GlossaryDB(path) as db: ...``) — соединение закрывается
    автоматически.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL ускоряет параллельное чтение и переживает аварийное завершение;
        # для in-memory базы SQLite сам оставит режим memory.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            logger.warning("Не удалось включить WAL для %s — используется режим по умолчанию", path)
        self._create_schema()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def _create_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                kind TEXT NOT NULL,
                path TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                imported_at TEXT
            );

            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY,
                canonical TEXT NOT NULL,
                variant TEXT,
                category TEXT,
                note TEXT,
                source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(canonical, variant, source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_entries_canonical ON entries(canonical);
            CREATE INDEX IF NOT EXISTS idx_entries_source_id ON entries(source_id);
            """
            )
            self._apply_schema_migrations()

    def _apply_schema_migrations(self) -> None:
        """Идемпотентно доводит схему до :data:`SCHEMA_VERSION`.

        ``CREATE TABLE IF NOT EXISTS`` не меняет уже существующую БД, поэтому
        версия схемы хранится в ``PRAGMA user_version``: сюда добавляются
        шаги для баз, созданных более ранними версиями. Повторный вызов на
        актуальной БД ничего не делает.
        """
        version = self.schema_version()
        if version < 1:
            # Базовые таблицы уже созданы выше (IF NOT EXISTS) — v1 считаем
            # применённой и лишь фиксируем версию.
            version = 1
        if version < 2:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            version = 2
        if self.schema_version() != version:
            self._conn.execute(f"PRAGMA user_version = {version}")

    def schema_version(self) -> int:
        """Версия схемы БД (``PRAGMA user_version``)."""
        row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row is not None else 0

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def legacy_migration_version(self) -> int | None:
        """Версия схемы, на которой выполнена миграция старых глоссариев.

        ``None`` — миграция ещё не выполнялась. См.
        :data:`LEGACY_GLOSSARY_MIGRATION_KEY`.
        """
        raw = self._get_meta(LEGACY_GLOSSARY_MIGRATION_KEY)
        return int(raw) if raw is not None and raw.isdigit() else None

    def close(self) -> None:
        """Закрывает соединение с базой."""
        self._conn.close()

    def __enter__(self) -> GlossaryDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Импорт
    # ------------------------------------------------------------------
    def import_txt(
        self, path: Path | str, *, source: str | None = None, replace: bool = True
    ) -> ImportReport:
        """Импортирует текстовый глоссарий (по записи в строке).

        Комментарии (``#``) и пустые строки пропускаются; строка вида
        ``"ошибочная форма = канон"`` даёт пару, иначе — одиночный термин.
        ``source`` по умолчанию — имя файла; ``replace=True`` заменяет
        существующий одноимённый источник.
        """
        file_path = Path(path)
        text = file_path.read_text(encoding=_TEXT_ENCODING)
        rows = _parse_txt(text)
        name = source or file_path.name
        return self._import_rows(name, "txt", str(file_path), rows, replace=replace)

    def import_csv(
        self, path: Path | str, *, source: str | None = None, replace: bool = True
    ) -> ImportReport:
        """Импортирует CSV-глоссарий с гибким разбором заголовка.

        Заголовок необязателен; распознаются колонки ``term/термин``,
        ``wrong/ошибка/вариант``, ``canonical/канон``, ``note/заметка``,
        ``category/категория``. Разделитель (``,`` ``;`` табуляция)
        определяется автоматически. Без заголовка первый столбец считается
        термином либо парой ``"="``.
        """
        file_path = Path(path)
        text = file_path.read_text(encoding=_TEXT_ENCODING)
        rows = _parse_csv(text)
        name = source or file_path.name
        return self._import_rows(name, "csv", str(file_path), rows, replace=replace)

    def add_entry(
        self,
        canonical: str,
        *,
        variant: str | None = None,
        source: str = "manual",
        note: str | None = None,
        category: str | None = None,
    ) -> int:
        """Добавляет одну запись и возвращает её id (существующей — тоже).

        Источник ``source`` создаётся при необходимости. Дубликат по
        ``(canonical, variant, source)`` обновляет заметку/категорию и
        включается обратно, а не добавляется второй раз.
        """
        canonical_value = _normalize_spaces(canonical.strip()) if canonical else ""
        if not canonical_value:
            raise ValueError("Канонический термин не может быть пустым")
        variant_value = _normalize_spaces(variant.strip()) if variant else None
        note_value = note.strip() if note else None
        category_value = category.strip() if category else None
        name = source.strip() or "manual"

        with self._conn:
            source_id = self._ensure_source(name, "manual", None)
            existing_id = self._find_entry_id(source_id, canonical_value, variant_value)
            now = _now_iso()
            if existing_id is not None:
                self._conn.execute(
                    "UPDATE entries SET enabled = 1, "
                    "note = COALESCE(?, note), category = COALESCE(?, category), "
                    "updated_at = ? WHERE id = ?",
                    (note_value, category_value, now, existing_id),
                )
                return existing_id
            cursor = self._conn.execute(
                "INSERT INTO entries "
                "(canonical, variant, category, note, source_id, enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (canonical_value, variant_value, category_value, note_value, source_id, now, now),
            )
            return int(cursor.lastrowid or 0)

    def _import_rows(
        self,
        name: str,
        kind: str,
        path: str | None,
        rows: Sequence[_Row],
        *,
        replace: bool,
    ) -> ImportReport:
        """Записывает разобранные строки в источник ``name``."""
        with self._conn:
            source_id = self._prepare_source(name, kind, path, replace=replace)
            existing_keys = self._existing_keys(source_id)
            seen: set[tuple[str, str]] = set()
            added = 0
            skipped = 0
            now = _now_iso()
            for canonical, variant, note, category in rows:
                canonical_value = _normalize_spaces(canonical.strip())
                variant_value = _normalize_spaces(variant.strip()) if variant else None
                if not canonical_value:
                    skipped += 1
                    continue
                key = (canonical_value.casefold(), (variant_value or "").casefold())
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                if key in existing_keys:
                    self._conn.execute(
                        "UPDATE entries SET enabled = 1, "
                        "note = COALESCE(?, note), category = COALESCE(?, category), "
                        "updated_at = ? WHERE source_id = ? AND canonical = ? "
                        "AND variant IS ?",
                        (
                            note.strip() if note else None,
                            category.strip() if category else None,
                            now,
                            source_id,
                            canonical_value,
                            variant_value,
                        ),
                    )
                    skipped += 1
                    continue
                self._conn.execute(
                    "INSERT INTO entries "
                    "(canonical, variant, category, note, source_id, enabled, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        canonical_value,
                        variant_value,
                        category.strip() if category else None,
                        note.strip() if note else None,
                        source_id,
                        now,
                        now,
                    ),
                )
                added += 1
            total_row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM entries WHERE source_id = ?", (source_id,)
            ).fetchone()
        return ImportReport(
            source=name,
            kind=kind,
            added=added,
            skipped=skipped,
            total=int(total_row["c"]) if total_row is not None else 0,
            replaced=replace,
        )

    def migrate_from_paths(self, paths: Sequence[Path]) -> list[ImportReport]:
        """Импортирует текстовые глоссарии, которых ещё нет среди источников.

        Идемпотентно: повторный вызов не создаёт дублей и не перезаписывает
        уже импортированный источник. Несуществующие файлы пропускаются с
        предупреждением. Версия схемы, на которой выполнена миграция, пишется
        в ``meta`` (:data:`LEGACY_GLOSSARY_MIGRATION_KEY`).
        """
        reports: list[ImportReport] = []
        for raw_path in paths:
            file_path = Path(raw_path)
            if not file_path.is_file():
                logger.warning("Глоссарий не найден, пропущен: %s", file_path)
                continue
            if self._source_by_name(file_path.name) is not None:
                logger.info("Источник глоссария уже импортирован: %s", file_path.name)
                continue
            reports.append(self.import_txt(file_path, source=file_path.name, replace=False))
        self._set_meta(LEGACY_GLOSSARY_MIGRATION_KEY, str(SCHEMA_VERSION))
        return reports

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def list_sources(self) -> list[Source]:
        """Все источники, отсортированные по имени."""
        rows = self._conn.execute("SELECT * FROM sources ORDER BY name").fetchall()
        return [_row_to_source(row) for row in rows]

    def list_entries(
        self,
        *,
        source: str | int | None = None,
        enabled_only: bool = False,
        search: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entry]:
        """Записи с фильтрами по источнику, включённости и подстроке.

        Поиск регистронезависимый (в т.ч. по кириллице) и выполняется по
        канону, ошибочной форме, заметке и категории. Поиск по подстроке
        делается на стороне Python (SQLite ``LIKE``/``lower`` не умеют
        кириллицу), поэтому при ``search`` пагинация применяется после
        фильтрации. Без поиска ``limit``/``offset`` уходят в SQL — листание
        больших глоссариев не выгружает всю таблицу.
        """
        clauses, params = self._entry_clauses(source, enabled_only)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        base_sql = (
            "SELECT e.* FROM entries e LEFT JOIN sources s ON s.id = e.source_id"
            f"{where} ORDER BY e.canonical, e.variant, e.id"
        )
        needle = search.casefold() if search and search.strip() else ""

        if needle:
            rows = self._conn.execute(base_sql, params).fetchall()
            entries = [
                entry
                for entry in (_row_to_entry(row) for row in rows)
                if needle in _entry_haystack(entry)
            ]
            start = max(offset, 0)
            if start:
                entries = entries[start:]
            if limit is not None:
                entries = entries[: max(limit, 0)]
            return entries

        # Без поиска пагинация выполняется в SQL: LIMIT -1 означает «без лимита».
        sql = base_sql
        page_params: list[object] = list(params)
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            page_params.extend((max(limit, 0), max(offset, 0)))
        elif offset > 0:
            sql += " LIMIT -1 OFFSET ?"
            page_params.append(max(offset, 0))
        rows = self._conn.execute(sql, page_params).fetchall()
        return [_row_to_entry(row) for row in rows]

    def count_entries(
        self,
        *,
        source: str | int | None = None,
        enabled_only: bool = False,
        search: str | None = None,
    ) -> int:
        """Число записей, удовлетворяющих тем же фильтрам, что и ``list_entries``.

        Нужно для пагинации: ``list_entries`` с ``limit``/``offset`` отдаёт одну
        страницу, а общий размер считает этот метод (без загрузки всех записей в
        объекты ``Entry``, когда поиск не используется).
        """
        if search and search.strip():
            return len(
                self.list_entries(
                    source=source, enabled_only=enabled_only, search=search
                )
            )
        clauses, params = self._entry_clauses(source, enabled_only)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM entries e "
            f"LEFT JOIN sources s ON s.id = e.source_id{where}",
            params,
        ).fetchone()
        return int(row["c"]) if row is not None else 0

    def _entry_clauses(
        self, source: str | int | None, enabled_only: bool
    ) -> tuple[list[str], list[object]]:
        """Общие SQL-фильтры ``list_entries``/``count_entries``."""
        clauses: list[str] = []
        params: list[object] = []
        if source is not None:
            if isinstance(source, int):
                clauses.append("e.source_id = ?")
                params.append(source)
            else:
                clauses.append("s.name = ?")
                params.append(source)
        if enabled_only:
            clauses.append("e.enabled = 1")
        return clauses, params

    def get_entry(self, entry_id: int) -> Entry | None:
        """Возвращает запись по id или ``None``."""
        row = self._conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return _row_to_entry(row) if row is not None else None

    def count(self) -> int:
        """Общее число записей глоссария."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM entries").fetchone()
        return int(row["c"]) if row is not None else 0

    def count_enabled(self) -> int:
        """Число включённых записей глоссария."""
        row = self._conn.execute("SELECT COUNT(*) AS c FROM entries WHERE enabled = 1").fetchone()
        return int(row["c"]) if row is not None else 0

    def entry_counts(self) -> dict[str, int]:
        """Число записей по каждому источнику (имя → количество)."""
        rows = self._conn.execute(
            "SELECT s.name AS name, COUNT(e.id) AS c FROM sources s "
            "LEFT JOIN entries e ON e.source_id = s.id GROUP BY s.id"
        ).fetchall()
        return {str(row["name"]): int(row["c"]) for row in rows}

    def terms_for_matcher(
        self, *, sources: Iterable[str] | None = None, enabled_only: bool = True
    ) -> list[str]:
        """Строки для :class:`~audio_transcriber.llm.glossary.Glossary`.

        Пары отдаются как ``"ошибочная форма = канон"``, одиночные термины —
        каноном. ``sources=None`` — все включённые источники (если
        ``enabled_only``). Порядок детерминированный, дубликаты убраны.
        """
        clauses: list[str] = []
        params: list[object] = []
        if enabled_only:
            clauses.append("e.enabled = 1")
        if sources is not None:
            names = [name for name in sources if name]
            if not names:
                return []
            placeholders = ", ".join("?" for _ in names)
            clauses.append(f"s.name IN ({placeholders})")
            params.extend(names)
        elif enabled_only:
            # Записи без источника (откреплённые) тоже считаем включёнными.
            clauses.append("(s.enabled = 1 OR e.source_id IS NULL)")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT e.canonical AS canonical, e.variant AS variant FROM entries e "
            f"LEFT JOIN sources s ON s.id = e.source_id{where} "
            "ORDER BY e.canonical, e.variant, e.id"
        )
        rows = self._conn.execute(sql, params).fetchall()
        result: list[str] = []
        seen: set[str] = set()
        for row in rows:
            canonical = str(row["canonical"])
            variant = row["variant"]
            term = f"{variant} = {canonical}" if variant else canonical
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(term)
        return result

    # ------------------------------------------------------------------
    # Изменение источников и записей
    # ------------------------------------------------------------------
    def set_source_enabled(self, name: str, enabled: bool) -> bool:
        """Включает/отключает источник. ``False`` — источник не найден."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE sources SET enabled = ? WHERE name = ?", (1 if enabled else 0, name)
            )
        return cursor.rowcount > 0

    def set_entry_enabled(self, entry_id: int, enabled: bool) -> bool:
        """Включает/отключает отдельную запись. ``False`` — записи не было."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE entries SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, _now_iso(), entry_id),
            )
        return cursor.rowcount > 0

    def update_entry(
        self,
        entry_id: int,
        *,
        canonical: str | None = None,
        variant: str | None = None,
        note: str | None = None,
        category: str | None = None,
        enabled: bool | None = None,
    ) -> bool:
        """Обновляет переданные поля записи. ``False`` — записи не было.

        ``None`` означает «поле не трогать»; пустой ``variant``/``note``/
        ``category`` очищает соответствующее поле. Пустой ``canonical``
        недопустим (``ValueError``). Конфликт уникальности ``(canonical,
        variant, source)`` поднимает :class:`sqlite3.IntegrityError`.
        """
        fields: dict[str, object] = {}
        if canonical is not None:
            canonical_value = _normalize_spaces(canonical.strip()) if canonical else ""
            if not canonical_value:
                raise ValueError("Канонический термин не может быть пустым")
            fields["canonical"] = canonical_value
        if variant is not None:
            variant_value = _normalize_spaces(variant.strip())
            fields["variant"] = variant_value or None
        if note is not None:
            fields["note"] = note.strip() or None
        if category is not None:
            fields["category"] = category.strip() or None
        if enabled is not None:
            fields["enabled"] = 1 if enabled else 0
        if not fields:
            return self.get_entry(entry_id) is not None
        fields["updated_at"] = _now_iso()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._conn:
            cursor = self._conn.execute(
                f"UPDATE entries SET {assignments} WHERE id = ?",
                [*fields.values(), entry_id],
            )
        return cursor.rowcount > 0

    def delete_entry(self, entry_id: int) -> bool:
        """Удаляет запись по id. ``False`` — записи не было."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        return cursor.rowcount > 0

    def delete_source(self, name: str, *, cascade: bool = True) -> int:
        """Удаляет источник и возвращает число затронутых записей.

        ``cascade=True`` удаляет и записи источника (``ON DELETE CASCADE``),
        ``cascade=False`` — открепляет их (``source_id = NULL``), сохраняя
        сами записи в глоссарии.
        """
        source = self._source_by_name(name)
        if source is None:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM entries WHERE source_id = ?", (source.id,)
        ).fetchone()
        affected = int(row["c"]) if row is not None else 0
        with self._conn:
            if cascade:
                self._conn.execute("DELETE FROM sources WHERE id = ?", (source.id,))
            else:
                self._conn.execute(
                    "UPDATE entries SET source_id = NULL WHERE source_id = ?", (source.id,)
                )
                self._conn.execute("DELETE FROM sources WHERE id = ?", (source.id,))
        return affected

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _prepare_source(self, name: str, kind: str, path: str | None, *, replace: bool) -> int:
        """Возвращает id источника, создавая/пересоздавая его.

        При ``replace=True`` одноимённый источник удаляется вместе с
        записями (каскад), затем создаётся заново.
        """
        existing = self._source_by_name(name)
        if existing is not None and replace:
            self._conn.execute("DELETE FROM sources WHERE id = ?", (existing.id,))
            existing = None
        now = _now_iso()
        if existing is None:
            cursor = self._conn.execute(
                "INSERT INTO sources (name, kind, path, enabled, imported_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (name, kind, path, now),
            )
            return int(cursor.lastrowid or 0)
        self._conn.execute(
            "UPDATE sources SET kind = ?, path = ?, enabled = 1, imported_at = ? WHERE id = ?",
            (kind, path, now, existing.id),
        )
        return existing.id

    def _ensure_source(self, name: str, kind: str, path: str | None) -> int:
        """Возвращает id источника, создавая его при отсутствии."""
        existing = self._source_by_name(name)
        if existing is not None:
            return existing.id
        cursor = self._conn.execute(
            "INSERT INTO sources (name, kind, path, enabled, imported_at) "
            "VALUES (?, ?, ?, 1, ?)",
            (name, kind, path, _now_iso()),
        )
        return int(cursor.lastrowid or 0)

    def _source_by_name(self, name: str) -> Source | None:
        row = self._conn.execute("SELECT * FROM sources WHERE name = ?", (name,)).fetchone()
        return _row_to_source(row) if row is not None else None

    def _find_entry_id(self, source_id: int, canonical: str, variant: str | None) -> int | None:
        target = (canonical.casefold(), (variant or "").casefold())
        rows = self._conn.execute(
            "SELECT id, canonical, variant FROM entries WHERE source_id = ?", (source_id,)
        ).fetchall()
        for row in rows:
            key = (str(row["canonical"]).casefold(), (row["variant"] or "").casefold())
            if key == target:
                return int(row["id"])
        return None

    def _existing_keys(self, source_id: int) -> set[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT canonical, variant FROM entries WHERE source_id = ?", (source_id,)
        ).fetchall()
        return {
            (str(row["canonical"]).casefold(), str(row["variant"] or "").casefold())
            for row in rows
        }


# ----------------------------------------------------------------------
# Разбор строк глоссария
# ----------------------------------------------------------------------
def _parse_txt(text: str) -> list[_Row]:
    """Разбирает текстовый глоссарий в список записей."""
    rows: list[_Row] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pair = _split_pair(line)
        if pair is not None:
            variant, canonical = pair
            rows.append((canonical, variant, None, None))
        else:
            rows.append((line, None, None, None))
    return rows


def _parse_csv(text: str) -> list[_Row]:
    """Разбирает CSV-глоссарий в список записей (с заголовком или без)."""
    sample = text[:4096]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    header_map = _header_map(rows[0])
    data = rows[1:] if header_map is not None else rows
    parsed: list[_Row] = []
    for row in data:
        if header_map is not None:
            entry = _row_from_mapping(row, header_map)
        else:
            entry = _row_without_header(row)
        if entry is not None:
            parsed.append(entry)
    return parsed


def _header_map(header_row: Sequence[str]) -> dict[str, int] | None:
    """Сопоставляет смысловые колонки с индексами, если это заголовок."""
    mapping: dict[str, int] = {}
    for index, cell in enumerate(header_row):
        group = _HEADER_ALIASES.get(cell.strip().casefold())
        if group is not None and group not in mapping:
            mapping[group] = index
    return mapping or None


def _cell(row: Sequence[str], mapping: dict[str, int], group: str) -> str:
    index = mapping.get(group)
    if index is None or index >= len(row):
        return ""
    return row[index].strip()


def _row_from_mapping(row: Sequence[str], mapping: dict[str, int]) -> _Row | None:
    """Собирает запись из строки с распознанным заголовком."""
    canonical = _cell(row, mapping, "canonical")
    variant = _cell(row, mapping, "variant")
    term = _cell(row, mapping, "term")
    note = _cell(row, mapping, "note") or None
    category = _cell(row, mapping, "category") or None

    if canonical:
        if not variant and term and term.casefold() != canonical.casefold():
            variant = term
        return (canonical, variant or None, note, category)
    if term:
        pair = _split_pair(term)
        if pair is not None:
            wrong, canon = pair
            return (canon, wrong, note, category)
        return (term, variant or None, note, category)
    return None


def _row_without_header(row: Sequence[str]) -> _Row | None:
    """Собирает запись, когда заголовка нет: первый столбец — термин/пара."""
    first = row[0].strip() if row else ""
    if not first:
        return None
    pair = _split_pair(first)
    if pair is not None:
        wrong, canonical = pair
        return (canonical, wrong, None, None)
    return (first, None, None, None)


def _entry_haystack(entry: Entry) -> str:
    """Строка для регистронезависимого поиска по записи."""
    parts = [entry.canonical, entry.variant, entry.note, entry.category]
    return " ".join(part for part in parts if part).casefold()


def _row_to_source(row: sqlite3.Row) -> Source:
    return Source(
        id=int(row["id"]),
        name=str(row["name"]),
        kind=str(row["kind"]),
        path=row["path"],
        enabled=bool(row["enabled"]),
        imported_at=row["imported_at"],
    )


def _row_to_entry(row: sqlite3.Row) -> Entry:
    return Entry(
        id=int(row["id"]),
        canonical=str(row["canonical"]),
        variant=row["variant"],
        category=row["category"],
        note=row["note"],
        source_id=row["source_id"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
