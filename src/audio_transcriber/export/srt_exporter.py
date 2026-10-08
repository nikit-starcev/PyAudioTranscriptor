"""Экспорт стенограммы в формат субтитров SubRip (.srt)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import TranscriptEntry, TranscriptionResult
from audio_transcriber.export.timestamps import format_srt_timestamp
from audio_transcriber.export.words import has_words, srt_cue_text
from audio_transcriber.utils.exceptions import ExportError


class SrtExporter:
    """Реализует протокол ``ResultExporter`` для формата SRT."""

    def __init__(self, *, highlight_words: bool = False) -> None:
        self._highlight_words = highlight_words

    def _entry_text(self, entry: TranscriptEntry) -> str:
        if self._highlight_words and has_words(entry) and not entry.edited:
            return srt_cue_text(entry.words, highlight=True)
        return entry.text

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        blocks: list[str] = []
        for index, entry in enumerate(result.entries, start=1):
            # Как и раньше, реплики без говорящих выводятся без подписи.
            prefix = (
                ""
                if entry.speaker is None and not entry.extra_speakers
                else f"{entry.speaker_label}: "
            )
            blocks.append(
                f"{index}\n"
                f"{format_srt_timestamp(entry.start)} --> {format_srt_timestamp(entry.end)}\n"
                f"{prefix}{self._entry_text(entry)}\n"
            )

        try:
            output_path.write_text("\n".join(blocks), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
