"""Экспорт стенограммы в формат субтитров WebVTT (.vtt)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import TranscriptEntry, TranscriptionResult
from audio_transcriber.export.timestamps import format_vtt_timestamp
from audio_transcriber.export.words import escape_vtt_text, has_words, vtt_cue_text
from audio_transcriber.utils.exceptions import ExportError


class VttExporter:
    """Реализует протокол ``ResultExporter`` для формата WebVTT."""

    def __init__(self, *, highlight_words: bool = False) -> None:
        self._highlight_words = highlight_words

    def _cue_text(self, entry: TranscriptEntry) -> str:
        if self._highlight_words and has_words(entry) and not entry.edited:
            body = vtt_cue_text(entry.words, highlight=True)
        else:
            body = escape_vtt_text(entry.text)
        if entry.speaker is None and not entry.extra_speakers:
            return body
        return f"<v {escape_vtt_text(entry.speaker_label)}>{body}"

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        blocks: list[str] = ["WEBVTT", ""]
        for index, entry in enumerate(result.entries, start=1):
            blocks.append(str(index))
            blocks.append(
                f"{format_vtt_timestamp(entry.start)} --> {format_vtt_timestamp(entry.end)}"
            )
            blocks.append(self._cue_text(entry))
            blocks.append("")

        try:
            output_path.write_text("\n".join(blocks), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
