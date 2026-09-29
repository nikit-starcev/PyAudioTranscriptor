"""Тесты CLI-подкоманды ``glossary`` (локальная SQLite-БД глоссария)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from audio_transcriber.cli.app import app

runner = CliRunner()


def _run(*args: str):
    return runner.invoke(app, ["glossary", *args])


def test_glossary_import_sources_and_list(tmp_path: Path) -> None:
    txt = tmp_path / "glossary.txt"
    txt.write_text("ОИБ\nАИБ = ОИБ\n", encoding="utf-8")
    db = tmp_path / "glossary.db"

    imported = _run("import", str(txt), "--db", str(db))
    assert imported.exit_code == 0
    assert "добавлено 2" in imported.stdout

    sources = _run("sources", "--db", str(db))
    assert sources.exit_code == 0
    assert "glossary.txt" in sources.stdout
    assert "2" in sources.stdout

    listed = _run("list", "--db", str(db), "--search", "аиб")
    assert listed.exit_code == 0
    assert "ОИБ" in listed.stdout
    assert "АИБ" in listed.stdout


def test_glossary_add_enable_disable_remove(tmp_path: Path) -> None:
    db = tmp_path / "glossary.db"

    added = _run("add", "АИБ = ОИБ", "--source", "manual", "--db", str(db))
    assert added.exit_code == 0
    assert "АИБ = ОИБ" in added.stdout

    disabled = _run("disable", "manual", "--db", str(db))
    assert disabled.exit_code == 0
    assert "отключён" in disabled.stdout

    enabled = _run("enable", "manual", "--db", str(db))
    assert enabled.exit_code == 0
    assert "включён" in enabled.stdout

    removed = _run("remove", "manual", "--db", str(db))
    assert removed.exit_code == 0
    assert "удалён" in removed.stdout

    missing = _run("remove", "manual", "--db", str(db))
    assert missing.exit_code == 1


def test_glossary_import_unknown_kind_fails(tmp_path: Path) -> None:
    txt = tmp_path / "glossary.txt"
    txt.write_text("ОИБ\n", encoding="utf-8")

    result = _run("import", str(txt), "--kind", "xml", "--db", str(tmp_path / "g.db"))

    assert result.exit_code == 1
    assert "тип" in (result.stderr or "")


def test_glossary_list_empty_message(tmp_path: Path) -> None:
    result = _run("list", "--db", str(tmp_path / "empty.db"))

    assert result.exit_code == 0
    assert "не найдены" in result.stdout
