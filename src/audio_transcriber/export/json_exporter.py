"""Экспорт стенограммы в формат JSON (.json)."""

from __future__ import annotations

import json
from pathlib import Path

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.utils.exceptions import ExportError


class JsonExporter:
    """Реализует протокол ``ResultExporter`` для формата JSON."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        payload = {
            "source_path": str(result.source_path),
            "language": result.language,
            "duration": result.duration,
            "participants": result.participants,
            "speakers": [
                {"id": speaker.id, "display_name": speaker.display_name}
                for speaker in result.speakers
            ],
            "entries": [
                {
                    "start": entry.start,
                    "end": entry.end,
                    "text": entry.text,
                    "speaker": entry.speaker.id if entry.speaker else None,
                }
                for entry in result.entries
            ],
        }

        try:
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
