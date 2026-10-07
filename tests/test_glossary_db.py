"""Тесты локального SQLite-хранилища глоссария.

Проверяются схема, импорт .txt/.csv, дедупликация, включение/отключение
источников и записей, формат строк для матчера, миграция старых текстовых
глоссариев и сборка :class:`Glossary` через :func:`build_glossary`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.llm.glossary import Glossary
from audio_transcriber.storage.glossary_builder import build_glossary
from audio_transcriber.storage.glossary_db import SCHEMA_VERSION, GlossaryDB


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- Схема -----------------------------------------------------------------
def test_schema_created_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "glossary.db"

    with GlossaryDB(db_path) as db:
        assert db.list_sources() == []
        assert db.count() == 0

    # Повторное открытие той же базы не роняет создание схемы.
    with GlossaryDB(db_path) as db:
        assert db.count() == 0

    assert db_path.is_file()


def test_schema_version_is_current_and_idempotent(tmp_path: Path) -> None:
    """Схема доводится до ``SCHEMA_VERSION`` и переоткрытие её не ломает."""
    db_path = tmp_path / "glossary.db"

    with GlossaryDB(db_path) as db:
        assert db.schema_version() == SCHEMA_VERSION

    with GlossaryDB(db_path) as db:
        assert db.schema_version() == SCHEMA_VERSION


# --- Импорт .txt -----------------------------------------------------------
def test_import_txt_terms_pairs_and_comments(tmp_path: Path) -> None:
    txt = _write(
        tmp_path / "glossary.txt",
        "# комментарий\nОИБ\nАИБ = ОИБ\n\nАРМ\n",
    )

    with GlossaryDB(tmp_path / "glossary.db") as db:
        report = db.import_txt(txt)
        terms = db.terms_for_matcher()
        sources = db.list_sources()

    assert report.source == "glossary.txt"
    assert report.kind == "txt"
    assert report.added == 3
    assert report.total == 3
    assert "ОИБ" in terms
    assert "АИБ = ОИБ" in terms
    assert "АРМ" in terms
    assert [src.kind for src in sources] == ["txt"]


def test_import_txt_deduplicates_case_insensitively(tmp_path: Path) -> None:
    txt = _write(tmp_path / "glossary.txt", "ОИБ\nОИБ\nоиб\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        report = db.import_txt(txt)

    assert report.added == 1
    assert report.skipped == 2
    assert report.total == 1


def test_import_txt_replace_and_append(tmp_path: Path) -> None:
    txt = _write(tmp_path / "glossary.txt", "ОИБ\n")
    with GlossaryDB(tmp_path / "glossary.db") as db:
        first = db.import_txt(txt)
        second = db.import_txt(txt, replace=False)  # дубликат — не добавится
        third = db.import_txt(txt, replace=True)  # перезапись
        total = db.count()

    assert first.added == 1
    assert second.added == 0
    assert second.skipped == 1
    assert third.added == 1
    assert third.replaced is True
    assert total == 1


# --- Импорт .csv -----------------------------------------------------------
def test_import_csv_with_header_semicolon(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "terms.csv",
        "term;wrong;note;category\nОИБ;АИБ;основной;аббревиатура\nАРМ;;;\n",
    )

    with GlossaryDB(tmp_path / "glossary.db") as db:
        report = db.import_csv(csv_path)
        entries = {entry.canonical: entry for entry in db.list_entries()}

    assert report.kind == "csv"
    assert report.added == 2
    assert entries["ОИБ"].variant == "АИБ"
    assert entries["ОИБ"].note == "основной"
    assert entries["ОИБ"].category == "аббревиатура"
    assert entries["АРМ"].variant is None


def test_import_csv_canonical_column(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "terms.csv", "wrong,canonical\nАИБ,ОИБ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.import_csv(csv_path)
        entries = db.list_entries()

    assert len(entries) == 1
    assert entries[0].canonical == "ОИБ"
    assert entries[0].variant == "АИБ"


def test_import_csv_without_header(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "terms.csv", "АИБ = ОИБ\nАРМ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        report = db.import_csv(csv_path)
        terms = db.terms_for_matcher()

    assert report.added == 2
    assert "АИБ = ОИБ" in terms
    assert "АРМ" in terms


# --- Записи ----------------------------------------------------------------
def test_add_entry_deduplicates_and_deletes(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        first = db.add_entry("ОИБ", variant="АИБ", source="manual", note="n", category="c")
        second = db.add_entry("ОИБ", variant="АИБ", source="manual")
        assert first == second
        assert db.count() == 1

        assert db.delete_entry(first) is True
        assert db.delete_entry(first) is False
        assert db.count() == 0


def test_add_entry_requires_canonical(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db, pytest.raises(ValueError):
        db.add_entry("   ")


# --- Включение/отключение --------------------------------------------------
def test_disable_source_hides_terms(tmp_path: Path) -> None:
    txt = _write(tmp_path / "glossary.txt", "ОИБ\nАРМ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.import_txt(txt)
        assert set(db.terms_for_matcher()) == {"ОИБ", "АРМ"}

        assert db.set_source_enabled("glossary.txt", False) is True
        assert db.terms_for_matcher() == []

        assert db.set_source_enabled("glossary.txt", True) is True
        assert set(db.terms_for_matcher()) == {"ОИБ", "АРМ"}

        assert db.set_source_enabled("missing.txt", True) is False


def test_disable_entry_hides_only_it(tmp_path: Path) -> None:
    txt = _write(tmp_path / "glossary.txt", "ОИБ\nАРМ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.import_txt(txt)
        target = next(entry for entry in db.list_entries() if entry.canonical == "АРМ")

        assert db.set_entry_enabled(target.id, False) is True
        assert db.terms_for_matcher() == ["ОИБ"]
        assert [entry.canonical for entry in db.list_entries(enabled_only=True)] == ["ОИБ"]
        assert db.set_entry_enabled(99999, False) is False


# --- terms_for_matcher -----------------------------------------------------
def test_terms_for_matcher_pair_format(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.add_entry("ОИБ", variant="АИБ", source="a")
        db.add_entry("АРМ", source="a")

        terms = db.terms_for_matcher(sources=["a"])

    assert "АИБ = ОИБ" in terms
    assert "АРМ" in terms


def test_terms_for_matcher_filters_sources(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.add_entry("ОИБ", source="a")
        db.add_entry("АРМ", source="b")

        only_a = db.terms_for_matcher(sources=["a"])
        only_b = db.terms_for_matcher(sources=["b"])
        unknown = db.terms_for_matcher(sources=["nope"])

    assert only_a == ["ОИБ"]
    assert only_b == ["АРМ"]
    assert unknown == []


# --- list_entries: поиск и лимит -------------------------------------------
def test_list_entries_search_is_case_insensitive(tmp_path: Path) -> None:
    txt = _write(tmp_path / "glossary.txt", "ОИБ\nАИБ = ОИБ\nАРМ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.import_txt(txt)
        by_variant = db.list_entries(search="аиБ")
        by_canonical = db.list_entries(search="арм")

    assert [entry.canonical for entry in by_variant] == ["ОИБ"]
    assert [entry.canonical for entry in by_canonical] == ["АРМ"]


def test_list_entries_offset_and_limit(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        for term in ("А", "Б", "В", "Г"):
            db.add_entry(term, source="s")

        page = db.list_entries(limit=2, offset=1)

    assert [entry.canonical for entry in page] == ["Б", "В"]


def test_list_entries_sql_pagination_matches_full_slice(tmp_path: Path) -> None:
    """SQL-пагинация без поиска даёт ровно тот же срез, что и срез полного списка."""
    with GlossaryDB(tmp_path / "glossary.db") as db:
        for index in range(25):
            db.add_entry(f"Термин {index:02d}", source="s")

        everything = db.list_entries()
        for offset in (0, 5, 24, 40):
            for limit in (None, 1, 7):
                page = db.list_entries(limit=limit, offset=offset)
                expected = everything[offset:]
                if limit is not None:
                    expected = expected[:limit]
                assert [entry.id for entry in page] == [entry.id for entry in expected]


def test_count_entries_respects_filters(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.add_entry("А", source="a")
        db.add_entry("Б", source="a")
        db.add_entry("В", source="b")
        db.set_entry_enabled(db.list_entries(search="В")[0].id, False)

        assert db.count_entries() == 3
        assert db.count_entries(source="a") == 2
        assert db.count_entries(enabled_only=True) == 2
        assert db.count_entries(search="а") == 1
        assert db.count_entries(source="b", enabled_only=True) == 0


# --- Удаление источника ----------------------------------------------------
def test_delete_source_cascade(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.add_entry("ОИБ", source="a")
        db.add_entry("АРМ", source="a")
        db.add_entry("СУБД", source="b")

        affected = db.delete_source("a", cascade=True)

        assert affected == 2
        assert db.count() == 1
        assert [src.name for src in db.list_sources()] == ["b"]
        assert db.delete_source("a") == 0


def test_delete_source_detach_keeps_entries(tmp_path: Path) -> None:
    with GlossaryDB(tmp_path / "glossary.db") as db:
        db.add_entry("ОИБ", source="a")
        affected = db.delete_source("a", cascade=False)

        entries = db.list_entries()
        terms = db.terms_for_matcher()
        sources = db.list_sources()

    assert affected == 1
    assert len(entries) == 1
    assert entries[0].source_id is None
    assert sources == []
    assert terms == ["ОИБ"]


# --- Миграция --------------------------------------------------------------
def test_migrate_from_paths_is_idempotent(tmp_path: Path) -> None:
    first = _write(tmp_path / "one.txt", "ОИБ\n")
    second = _write(tmp_path / "two.txt", "АРМ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        reports = db.migrate_from_paths([first, second])
        after_first = db.count()
        again = db.migrate_from_paths([first, second])
        after_second = db.count()

    assert [report.source for report in reports] == ["one.txt", "two.txt"]
    assert after_first == 2
    assert again == []
    assert after_second == 2


def test_migrate_from_paths_skips_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.txt"

    with GlossaryDB(tmp_path / "glossary.db") as db:
        reports = db.migrate_from_paths([missing])

    assert reports == []


def test_migrate_from_paths_records_schema_version(tmp_path: Path) -> None:
    """Итог миграции отмечается версией схемы в ``meta`` (issue #90)."""
    txt = _write(tmp_path / "glossary.txt", "ОИБ\n")

    with GlossaryDB(tmp_path / "glossary.db") as db:
        assert db.legacy_migration_version() is None
        db.migrate_from_paths([txt])
        assert db.legacy_migration_version() == SCHEMA_VERSION


def test_build_glossary_migrates_txt_with_existing_db_entries(
    tmp_path: Path, audio_file: Path
) -> None:
    """Текстовый глоссарий импортируется, даже если БД уже непуста (issue #90).

    Раньше миграция запускалась только при ``count() == 0``, поэтому при
    наличии ручных записей ``GLOSSARY_PATH`` молча игнорировался.
    """
    txt = _write(tmp_path / "extra.txt", "КИСУСС\n")
    db_path = tmp_path / "glossary.db"
    with GlossaryDB(db_path) as db:
        db.add_entry("ОИБ", source="manual")

    config = AppConfig(input_file=audio_file, glossary_db=db_path, glossary_path=(txt,))
    glossary = build_glossary(config)

    assert isinstance(glossary, Glossary)
    assert "ОИБ" in glossary
    assert "КИСУСС" in glossary
    with GlossaryDB(db_path) as db:
        assert "extra.txt" in [src.name for src in db.list_sources()]
        assert db.count() == 2

    # Повторная сборка не дублирует источник (идемпотентно).
    build_glossary(config)
    with GlossaryDB(db_path) as db:
        assert db.count() == 2


# --- build_glossary --------------------------------------------------------
def test_build_glossary_disabled_returns_none(tmp_path: Path, audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        glossary_db=tmp_path / "glossary.db",
        glossary_enabled=False,
    )

    assert build_glossary(config) is None
    assert not (tmp_path / "glossary.db").exists()


def test_build_glossary_reads_terms_from_db(tmp_path: Path, audio_file: Path) -> None:
    db_path = tmp_path / "glossary.db"
    with GlossaryDB(db_path) as db:
        db.add_entry("ОИБ", variant="АИБ", source="manual")
        db.add_entry("АРМ", source="manual")

    config = AppConfig(input_file=audio_file, glossary_db=db_path)
    glossary = build_glossary(config)

    assert isinstance(glossary, Glossary)
    assert "ОИБ" in glossary
    assert "АРМ" in glossary
    # пара «АИБ = ОИБ» применяется к тексту
    corrected, _ = glossary.correct_text("Обсуждали АИБ.")
    assert "ОИБ" in corrected


def test_build_glossary_migrates_txt_once(tmp_path: Path, audio_file: Path) -> None:
    txt = _write(tmp_path / "glossary.txt", "ОИБ\nАИБ = ОИБ\n")
    db_path = tmp_path / "glossary.db"
    config = AppConfig(input_file=audio_file, glossary_db=db_path, glossary_path=(txt,))

    first = build_glossary(config)
    with GlossaryDB(db_path) as db:
        first_count = db.count()
        first_sources = [src.name for src in db.list_sources()]
    second = build_glossary(config)
    with GlossaryDB(db_path) as db:
        second_count = db.count()

    assert isinstance(first, Glossary)
    assert "ОИБ" in first
    assert first_count == 2
    assert first_sources == ["glossary.txt"]
    assert isinstance(second, Glossary)
    assert second_count == first_count


def test_resolved_glossary_db_default_points_to_isolated_path(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.resolved_glossary_db() == tmp_path / "glossary.db"
