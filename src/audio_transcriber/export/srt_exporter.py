"""Экспорт стенограммы в формат субтитров SubRip (.srt)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.timestamps import format_srt_timestamp
from audio_transcriber.utils.exceptions import ExportError


class SrtExporter:
    """Реализует протокол ``ResultExporter`` для формата SRT."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        blocks = [
            f"{index}\n"
            f"{format_srt_timestamp(entry.start)} --> {format_srt_timestamp(entry.end)}\n"
            f"{entry.speaker.display_name + ': ' if entry.speaker else ''}{entry.text}\n"
            for index, entry in enumerate(result.entries, start=1)
        ]

        try:
            output_path.write_text("\n".join(blocks), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
