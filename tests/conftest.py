"""Общие фикстуры для тестов."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Пустой файл-заглушка аудиозаписи для тестов, не требующих реального декодирования."""

    path = tmp_path / "sample.mp3"
    path.write_bytes(b"")
    return path


@pytest.fixture(autouse=True)
def isolate_glossary_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Изолирует SQLite-БД глоссария по умолчанию в ``tmp_path`` каждого теста.

    Без этого тесты, создающие ``AppConfig``, делили бы один ``glossary.db``
    в рабочем каталоге и влияли друг на друга через накопленные термины.
    """

    from audio_transcriber.config import defaults as config_defaults

    db_path = tmp_path / "glossary.db"
    monkeypatch.setattr(config_defaults, "DEFAULT_GLOSSARY_DB", str(db_path))
    return db_path


@pytest.fixture
def jfk_audio_file() -> Path:
    """Короткая (~11с) реальная запись речи, используемая в интеграционных тестах."""

    path = Path(__file__).parent / "tests_jfk.flac"
    if not path.exists():
        pytest.skip(f"Тестовый аудиофайл не найден: {path}")
    return path
