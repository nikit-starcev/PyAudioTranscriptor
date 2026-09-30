"""Тесты SQLite-хранилища задач веб-интерфейса."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from audio_transcriber.web.storage.jobs_db import (
    STATUS_DONE,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
    JobsDB,
)


def _make_db(tmp_path: Path) -> JobsDB:
    db = JobsDB(tmp_path / "jobs.db")
    db.initialize()
    return db


def test_create_and_get_job(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    job = db.create("job-1", tmp_path / "uploads" / "audio.mp3")

    assert job.id == "job-1"
    assert job.status == STATUS_QUEUED
    assert job.name == "audio.mp3"
    assert job.created_at
    assert job.num_speakers is None
    fetched = db.get("job-1")
    assert fetched is not None
    assert fetched.source_path == str(tmp_path / "uploads" / "audio.mp3")


def test_create_job_with_num_speakers(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    job = db.create("job-1", tmp_path / "a.mp3", num_speakers=4)

    assert job.num_speakers == 4
    fetched = db.get("job-1")
    assert fetched is not None
    assert fetched.num_speakers == 4
    assert fetched.as_dict()["num_speakers"] == 4


def test_update_num_speakers(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "a.mp3", num_speakers=2)

    updated = db.update("job-1", num_speakers=None)

    assert updated is not None
    assert updated.num_speakers is None


def test_migration_adds_num_speakers_to_existing_table(tmp_path: Path) -> None:
    """Старая база без колонки ``num_speakers`` аккуратно мигрируется."""
    path = tmp_path / "jobs.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs ("
            "id TEXT PRIMARY KEY, source_path TEXT NOT NULL, status TEXT NOT NULL, "
            "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, "
            "language TEXT, duration REAL, error TEXT, result_path TEXT, "
            "stage TEXT, fraction REAL)"
        )
        connection.execute(
            "INSERT INTO jobs (id, source_path, status, created_at) "
            "VALUES ('old', '/tmp/old.mp3', 'done', '2020-01-01T00:00:00+00:00')"
        )

    db = JobsDB(path)
    db.initialize()

    migrated = db.get("old")
    assert migrated is not None
    assert migrated.num_speakers is None  # колонка есть, у старой записи NULL

    assert db.create("new", tmp_path / "new.mp3", num_speakers=3).num_speakers == 3

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert "num_speakers" in columns


def test_get_missing_job_returns_none(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    assert db.get("nope") is None


def test_list_jobs_newest_first(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("a", tmp_path / "a.mp3")
    db.create("b", tmp_path / "b.mp3")

    jobs = db.list()

    assert {job.id for job in jobs} == {"a", "b"}


def test_update_changes_fields_and_ignores_unknown(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "audio.mp3")

    updated = db.update(
        "job-1",
        status=STATUS_RUNNING,
        stage="asr",
        fraction=0.5,
        language="ru",
        not_a_column="ignored",
    )

    assert updated is not None
    assert updated.status == STATUS_RUNNING
    assert updated.stage == "asr"
    assert updated.fraction == 0.5
    assert updated.language == "ru"


def test_delete_job(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "audio.mp3")

    assert db.delete("job-1") is True
    assert db.get("job-1") is None
    assert db.delete("job-1") is False


def test_job_helpers_and_dict(tmp_path: Path) -> None:
    job = Job(id="x", source_path="/tmp/out/file.wav", status=STATUS_DONE, created_at="now")

    assert job.is_terminal is True
    assert job.name == "file.wav"
    assert job.as_dict()["status"] == STATUS_DONE
