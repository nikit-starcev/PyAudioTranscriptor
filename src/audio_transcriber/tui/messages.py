"""Сообщения TUI между фоновым воркером и интерфейсом."""

from __future__ import annotations

from pathlib import Path

from textual.message import Message

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.progress import ProgressEvent


class ProgressUpdate(Message):
    def __init__(self, event: ProgressEvent) -> None:
        super().__init__()
        self.event = event


class FileStarted(Message):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


class FileDone(Message):
    def __init__(
        self, path: Path, result: TranscriptionResult | None = None, error: str | None = None
    ) -> None:
        super().__init__()
        self.path = path
        self.result = result
        self.error = error


class QueueDone(Message):
    pass
