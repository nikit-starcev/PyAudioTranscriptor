"""Тесты загрузки словаря терминов (:mod:`audio_transcriber.utils.vocabulary`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.utils.exceptions import ConfigurationError
from audio_transcriber.utils.vocabulary import build_hotwords, load_vocabulary_terms


def test_load_vocabulary_terms_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    vocabulary_file = tmp_path / "vocabulary.txt"
    vocabulary_file.write_text(
        "# Участники\nИванов\n\nПетров\n  # ещё комментарий\nвзаимопонимание  \n",
        encoding="utf-8",
    )

    terms = load_vocabulary_terms(vocabulary_file)

    assert terms == ["Иванов", "Петров", "взаимопонимание"]


def test_load_vocabulary_terms_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises(ConfigurationError):
        load_vocabulary_terms(missing)


def test_build_hotwords_merges_terms_and_extra() -> None:
    hotwords, dropped = build_hotwords(["Иванов", "Петров"], "юрист Смирнова")

    assert hotwords == "Иванов, Петров, юрист Смирнова"
    assert dropped == []


def test_build_hotwords_splits_extra_by_comma() -> None:
    hotwords, dropped = build_hotwords(["Иванов"], "Петров, юрист Смирнова")

    assert hotwords == "Иванов, Петров, юрист Смирнова"
    assert dropped == []


def test_build_hotwords_without_extra() -> None:
    hotwords, dropped = build_hotwords(["Иванов"], None)

    assert hotwords == "Иванов"
    assert dropped == []


def test_build_hotwords_empty_returns_none() -> None:
    assert build_hotwords([], None) == (None, [])
    assert build_hotwords([], "   ") == (None, [])


def test_build_hotwords_drops_terms_over_the_limit() -> None:
    terms = ["Иванов", "Петров", "очень-длинный-термин-который-не-влезет-в-лимит"]

    hotwords, dropped = build_hotwords(terms, max_length=20)

    assert hotwords == "Иванов, Петров"
    assert dropped == ["очень-длинный-термин-который-не-влезет-в-лимит"]
