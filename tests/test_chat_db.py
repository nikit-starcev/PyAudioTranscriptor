"""Тесты SQLite-хранилища истории чата по стенограмме (#54/#96)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.storage.chat_db import MAX_MESSAGE_LENGTH, ChatDB


def test_add_and_list_roundtrip(tmp_path: Path) -> None:
    with ChatDB(tmp_path / "chat.db") as db:
        user_id = db.add_message("job-1", "user", "о чём договорились?")
        citations = [
            {"index": 1, "start": 65.0, "end": 70.0, "speaker": "Мария", "text": "Срок — пятница"}
        ]
        assistant_id = db.add_message(
            "job-1", "assistant", "Срок — пятница [1]", citations=citations
        )
        messages = db.list_messages("job-1")
    assert [m.id for m in messages] == [user_id, assistant_id]
    assert messages[0].role == "user"
    assert messages[0].citations == ()
    assert messages[1].citations[0]["index"] == 1
    assert messages[1].as_dict()["citations"] == citations


def test_scoped_by_job(tmp_path: Path) -> None:
    with ChatDB(tmp_path / "chat.db") as db:
        db.add_message("job-1", "user", "первый")
        db.add_message("job-2", "user", "второй")
        assert db.count() == 2
        assert db.count("job-1") == 1
        assert [m.content for m in db.list_messages("job-2")] == ["второй"]


def test_clear_returns_count(tmp_path: Path) -> None:
    with ChatDB(tmp_path / "chat.db") as db:
        db.add_message("job-1", "user", "раз")
        db.add_message("job-1", "assistant", "два")
        assert db.clear("job-1") == 2
        assert db.list_messages("job-1") == []
        assert db.clear("job-1") == 0


def test_persists_between_opens(tmp_path: Path) -> None:
    path = tmp_path / "chat.db"
    with ChatDB(path) as db:
        db.add_message("job-1", "user", "вопрос")
    with ChatDB(path) as db:
        assert [m.content for m in db.list_messages("job-1")] == ["вопрос"]


def test_invalid_role_and_empty_rejected(tmp_path: Path) -> None:
    with ChatDB(tmp_path / "chat.db") as db:
        with pytest.raises(ValueError):
            db.add_message("job-1", "system", "нельзя")
        with pytest.raises(ValueError):
            db.add_message("job-1", "user", "   ")
        with pytest.raises(ValueError):
            db.add_message("job-1", "user", "а" * (MAX_MESSAGE_LENGTH + 1))
