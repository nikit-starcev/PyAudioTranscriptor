"""Тесты SQLite-хранилища шаблонов промпта резюме (#97)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from audio_transcriber.llm.summary import DEFAULT_SUMMARY_TEMPLATE_NAME
from audio_transcriber.storage.prompts_db import MAX_PROMPT_LENGTH, PromptsDB

EXPECTED_DEFAULTS = 3


def test_seeds_default_templates(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "prompts.db") as db:
        prompts = db.list_prompts()
        assert len(prompts) == EXPECTED_DEFAULTS
        assert {prompt.name for prompt in prompts} == {
            DEFAULT_SUMMARY_TEMPLATE_NAME,
            "Кратко (3–5 пунктов)",
            "Постановка задач",
        }
        assert all(prompt.builtin for prompt in prompts)
        # Активным становится «Стандартный».
        active = db.active_prompt()
        assert active is not None
        assert active.name == DEFAULT_SUMMARY_TEMPLATE_NAME


def test_seed_defaults_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "prompts.db"
    with PromptsDB(path) as db:
        db.add_prompt("Мой", "мой промпт")
        count = db.count()
    with PromptsDB(path) as db:
        assert db.count() == count  # дефолты не добавились повторно
        assert db.get_by_name("Мой") is not None


def test_crud_roundtrip(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "p.db") as db:
        prompt_id = db.add_prompt("Формат А", "Тело А")
        prompt = db.get_prompt(prompt_id)
        assert prompt is not None
        assert prompt.name == "Формат А"
        assert prompt.body == "Тело А"
        assert prompt.builtin is False

        assert db.update_prompt(prompt_id, name="Формат Б", body="Тело Б")
        updated = db.get_prompt(prompt_id)
        assert updated is not None
        assert (updated.name, updated.body) == ("Формат Б", "Тело Б")

        assert db.delete_prompt(prompt_id)
        assert db.get_prompt(prompt_id) is None
        assert db.delete_prompt(prompt_id) is False


def test_add_duplicate_name_raises(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "p.db") as db:
        db.add_prompt("Дубль", "тело")
        with pytest.raises(sqlite3.IntegrityError):
            db.add_prompt("Дубль", "другое")


def test_empty_name_or_body_rejected(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "p.db") as db:
        with pytest.raises(ValueError):
            db.add_prompt("   ", "тело")
        with pytest.raises(ValueError):
            db.add_prompt("имя", "   ")
        prompt_id = db.add_prompt("имя", "тело")
        with pytest.raises(ValueError):
            db.update_prompt(prompt_id, body="  ")


def test_too_long_body_rejected(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "p.db") as db, pytest.raises(ValueError):
        db.add_prompt("имя", "а" * (MAX_PROMPT_LENGTH + 1))


def test_set_active_and_active_id(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "p.db") as db:
        first = db.add_prompt("Первый", "тело 1")
        second = db.add_prompt("Второй", "тело 2")

        assert db.set_active(second)
        assert db.active_id() == second
        active = db.active_prompt()
        assert active is not None
        assert active.id == second

        assert db.set_active(first)
        assert db.active_id() == first
        assert not db.set_active(999_999)


def test_delete_active_falls_back(tmp_path: Path) -> None:
    with PromptsDB(tmp_path / "p.db") as db:
        prompt_id = db.add_prompt("Временный", "тело")
        assert db.set_active(prompt_id)
        assert db.delete_prompt(prompt_id)
        # Активный удалён — выбирается первый доступный (дефолтный).
        active = db.active_prompt()
        assert active is not None
        assert active.id != prompt_id


def test_persists_between_opens(tmp_path: Path) -> None:
    path = tmp_path / "p.db"
    with PromptsDB(path) as db:
        custom = db.add_prompt("Сохранённый", "долгий промпт")
        db.set_active(custom)
    with PromptsDB(path) as db:
        assert db.active_id() == custom
        prompt = db.get_prompt(custom)
        assert prompt is not None
        assert prompt.body == "долгий промпт"
