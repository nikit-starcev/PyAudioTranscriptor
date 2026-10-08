"""Экспорт стенограммы в Markdown (.md)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.annotations import entry_markers
from audio_transcriber.export.timestamps import format_timestamp
from audio_transcriber.utils.exceptions import ExportError

#: Служебные символы Markdown, которые экранируются в тексте реплик.
_MARKDOWN_SPECIALS = "\\`*_{}[]<>#|"


def escape_markdown(text: str) -> str:
    """Экранирует служебные символы Markdown обратным слешем."""

    return "".join(f"\\{char}" if char in _MARKDOWN_SPECIALS else char for char in text)


class MarkdownExporter:
    """Реализует протокол ``ResultExporter`` для формата Markdown."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        lines: list[str] = [f"# {escape_markdown(result.source_path.name)}", ""]

        if result.language is not None:
            lines.append(f"- Язык: {escape_markdown(result.language)}")
        lines.append(f"- Длительность: {format_timestamp(result.duration)}")
        lines.append("")

        if result.participants:
            lines.append("## Участники")
            lines.append("")
            lines.extend(f"- {escape_markdown(participant)}" for participant in result.participants)
            lines.append("")

        if result.summary:
            lines.append("## Резюме встречи")
            lines.append("")
            lines.append(result.summary)
            lines.append("")

        lines.append("## Расшифровка")
        lines.append("")

        threshold = result.low_confidence_threshold
        for entry in result.entries:
            marker = entry_markers(entry, threshold)
            lines.append(
                f"[{format_timestamp(entry.start)}] "
                f"**{escape_markdown(entry.speaker_label)}:** "
                f"{escape_markdown(entry.text)}{marker}"
            )

        try:
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
