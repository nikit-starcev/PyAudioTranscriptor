"""Экспорт стенограммы в текстовый формат (.txt)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.annotations import entry_markers
from audio_transcriber.export.timestamps import format_timestamp
from audio_transcriber.utils.exceptions import ExportError


class TxtExporter:
    """Реализует протокол ``ResultExporter`` для формата TXT."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        lines: list[str] = []

        if result.participants:
            lines.append("Участники:")
            lines.extend(f"  {participant}" for participant in result.participants)
            lines.append("")

        threshold = result.low_confidence_threshold
        lines.extend(
            f"[{format_timestamp(entry.start)}] "
            f"{entry.speaker.display_name if entry.speaker else '?'}: "
            f"{entry.text}{entry_markers(entry, threshold)}"
            for entry in result.entries
        )

        try:
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
