"""Хранилище веб-интерфейса (SQLite-БД задач)."""

from __future__ import annotations

from audio_transcriber.web.storage.jobs_db import Job, JobsDB

__all__ = ["Job", "JobsDB"]
