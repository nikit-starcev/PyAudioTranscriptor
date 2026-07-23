"""Экспорт стенограммы в документ Microsoft Word (.docx)."""

from __future__ import annotations

from pathlib import Path

from docx import Document

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.timestamps import format_timestamp
from audio_transcriber.utils.exceptions import ExportError


class DocxExporter:
    """Реализует протокол ``ResultExporter`` для формата DOCX."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        document = Document()
        document.add_heading(result.source_path.name, level=1)

        for entry in result.entries:
            speaker_label = entry.speaker.display_name if entry.speaker else "?"
            paragraph = document.add_paragraph()
            paragraph.add_run(f"[{format_timestamp(entry.start)}] {speaker_label}: ").bold = True
            paragraph.add_run(entry.text)

        try:
            document.save(str(output_path))
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
