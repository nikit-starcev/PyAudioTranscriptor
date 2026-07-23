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


@pytest.fixture
def jfk_audio_file() -> Path:
    """Короткая (~11с) реальная запись речи, используемая в интеграционных тестах."""

    path = Path(__file__).parent / "tests_jfk.flac"
    if not path.exists():
        pytest.skip(f"Тестовый аудиофайл не найден: {path}")
    return path
