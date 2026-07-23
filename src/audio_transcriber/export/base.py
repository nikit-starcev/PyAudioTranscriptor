"""Интерфейс экспортёра результата транскрибации в конкретный формат."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_transcriber.domain.models import TranscriptionResult


@runtime_checkable
class ResultExporter(Protocol):
    """Контракт экспортёра результата в конкретный формат вывода."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        """Сохраняет результат транскрибации по указанному пути."""
        ...
