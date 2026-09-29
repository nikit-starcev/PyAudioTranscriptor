"""Тесты SQLite-хранилища задач веб-интерфейса."""

from __future__ import annotations

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
    fetched = db.get("job-1")
    assert fetched is not None
    assert fetched.source_path == str(tmp_path / "uploads" / "audio.mp3")


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
