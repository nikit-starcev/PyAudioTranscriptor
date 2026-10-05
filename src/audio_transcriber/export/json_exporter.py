"""Экспорт стенограммы в формат JSON (.json)."""

from __future__ import annotations

import json
from pathlib import Path

from audio_transcriber.domain.models import TranscriptEntry, TranscriptionResult
from audio_transcriber.export.annotations import is_low_confidence
from audio_transcriber.utils.exceptions import ExportError


class JsonExporter:
    """Реализует протокол ``ResultExporter`` для формата JSON."""

    @staticmethod
    def _entry_payload(entry: TranscriptEntry, threshold: float | None) -> dict[str, object]:
        """Поля одной реплики; ``words`` (#45) — только при наличии слов.

        Так при выключенной стадии форма реплики остаётся прежней (обратная
        совместимость), а при включённой появляется поле ``words``.
        """
        payload: dict[str, object] = {
            "start": entry.start,
            "end": entry.end,
            "text": entry.text,
            "speaker": entry.speaker.id if entry.speaker else None,
            "extra_speakers": [speaker.id for speaker in entry.extra_speakers],
            "speaker_confidence": entry.speaker_confidence,
            "avg_logprob": entry.avg_logprob,
            "low_confidence": is_low_confidence(entry, threshold),
            "overlap": entry.overlap,
        }
        words = [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in entry.words
        ]
        if words:
            payload["words"] = words
        return payload

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        threshold = result.low_confidence_threshold
        payload = {
            "source_path": str(result.source_path),
            "language": result.language,
            "duration": result.duration,
            "participants": result.participants,
            "summary": result.summary,
            "speakers": [
                {"id": speaker.id, "display_name": speaker.display_name}
                for speaker in result.speakers
            ],
            "entries": [
                self._entry_payload(entry, threshold) for entry in result.entries
            ],
        }

        try:
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
