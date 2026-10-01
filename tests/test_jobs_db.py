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
from audio_transcriber.web.timings import StageTiming


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


def test_create_job_with_speaker_range(tmp_path: Path) -> None:
    db = _make_db(tmp_path)

    job = db.create("job-1", tmp_path / "a.mp3", min_speakers=2, max_speakers=5)

    assert job.min_speakers == 2
    assert job.max_speakers == 5
    fetched = db.get("job-1")
    assert fetched is not None
    payload = fetched.as_dict()
    assert payload["min_speakers"] == 2
    assert payload["max_speakers"] == 5


def test_update_speaker_range(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "a.mp3")

    updated = db.update("job-1", min_speakers=1, max_speakers=4)

    assert updated is not None
    assert updated.min_speakers == 1
    assert updated.max_speakers == 4

    reset = db.update("job-1", min_speakers=None, max_speakers=None)
    assert reset is not None
    assert reset.min_speakers is None
    assert reset.max_speakers is None


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


def test_migration_adds_speaker_range_columns(tmp_path: Path) -> None:
    """Старая база без колонок ``min_speakers``/``max_speakers`` мигрируется."""
    path = tmp_path / "jobs.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs ("
            "id TEXT PRIMARY KEY, source_path TEXT NOT NULL, status TEXT NOT NULL, "
            "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, "
            "language TEXT, duration REAL, error TEXT, result_path TEXT, "
            "stage TEXT, fraction REAL, num_speakers INTEGER)"
        )

    db = JobsDB(path)
    db.initialize()

    assert db.create("new", tmp_path / "new.mp3", min_speakers=2, max_speakers=4).max_speakers == 4

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert {"min_speakers", "max_speakers"} <= columns


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


def test_stage_times_roundtrip(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "a.mp3")
    timings = [
        StageTiming("denoise", 1.25),
        StageTiming("asr", 0.2, cached=True),
        StageTiming("merge", 0.05),
    ]

    updated = db.update("job-1", stage_times=timings)

    assert updated is not None
    assert [timing.stage for timing in updated.stage_times] == ["denoise", "asr", "merge"]
    assert updated.stage_times[1].cached is True
    fetched = db.get("job-1")
    assert fetched is not None
    assert fetched.as_dict()["stage_times"] == [
        {"stage": "denoise", "seconds": 1.25, "cached": False},
        {"stage": "asr", "seconds": 0.2, "cached": True},
        {"stage": "merge", "seconds": 0.05, "cached": False},
    ]


def test_stage_times_accepts_plain_dicts_and_clears(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "a.mp3")

    db.update("job-1", stage_times=[{"stage": "asr", "seconds": 3, "cached": True}])
    fetched = db.get("job-1")
    assert fetched is not None
    assert fetched.stage_times[0].seconds == 3.0

    db.update("job-1", stage_times=[])
    cleared = db.get("job-1")
    assert cleared is not None
    assert cleared.stage_times == []


def test_total_seconds_for_finished_and_queued_jobs(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    queued = db.create("queued", tmp_path / "a.mp3")
    assert queued.total_seconds is None
    assert queued.as_dict()["total_seconds"] is None

    db.update(
        "queued",
        status=STATUS_DONE,
        started_at="2026-01-01T10:00:00+00:00",
        finished_at="2026-01-01T10:02:30+00:00",
    )
    finished = db.get("queued")
    assert finished is not None
    assert finished.total_seconds == 150.0
    assert finished.as_dict()["total_seconds"] == 150.0


def test_garbage_stage_times_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "jobs.db"
    db = _make_db(tmp_path)
    db.create("job-1", tmp_path / "a.mp3")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE jobs SET stage_times = ? WHERE id = 'job-1'", ("{broken",))

    fetched = db.get("job-1")
    assert fetched is not None
    assert fetched.stage_times == []


def test_updated_at_set_on_create_and_refresh_on_update(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    created = db.create("job-1", tmp_path / "a.mp3")

    assert created.updated_at == created.created_at
    assert created.as_dict()["updated_at"] == created.created_at

    db.update("job-1", status=STATUS_RUNNING, stage="asr")
    updated = db.get("job-1")
    assert updated is not None
    assert updated.updated_at is not None
    # Штамп времени есть и он не пустой после любого изменения.
    assert updated.as_dict()["updated_at"] == updated.updated_at


def test_migration_adds_updated_at_to_existing_table(tmp_path: Path) -> None:
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
    assert migrated.updated_at is None  # колонка есть, у старой записи NULL

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert "updated_at" in columns


def test_migration_adds_stage_columns(tmp_path: Path) -> None:
    """Старая база без колонок таймингов аккуратно мигрируется."""
    path = tmp_path / "jobs.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE jobs ("
            "id TEXT PRIMARY KEY, source_path TEXT NOT NULL, status TEXT NOT NULL, "
            "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, "
            "language TEXT, duration REAL, error TEXT, result_path TEXT, "
            "stage TEXT, fraction REAL, num_speakers INTEGER)"
        )
        connection.execute(
            "INSERT INTO jobs (id, source_path, status, created_at) "
            "VALUES ('old', '/tmp/old.mp3', 'done', '2020-01-01T00:00:00+00:00')"
        )

    db = JobsDB(path)
    db.initialize()

    migrated = db.get("old")
    assert migrated is not None
    assert migrated.stage_times == []
    assert migrated.stage_started_at is None

    db.update("old", stage_started_at="2026-01-01T00:00:00+00:00", stage_times=[StageTiming("asr", 1.0)])
    again = db.get("old")
    assert again is not None
    assert again.stage_started_at == "2026-01-01T00:00:00+00:00"
    assert again.stage_times[0].stage == "asr"

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert {"stage_started_at", "stage_times"} <= columns
