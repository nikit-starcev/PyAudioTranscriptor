"""Тесты сборки конвейера (``run_pipeline``) с фиктивными компонентами."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)
from audio_transcriber.pipeline import run_pipeline


class FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return (
            [TranscriptionSegment(start=0.0, end=1.0, text="привет")],
            "ru",
            1.0,
        )


class FakeDiarizer:
    def diarize(self, audio_path: Path, *, num_speakers: int | None = None):
        return [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]


class FakeMerger:
    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)]
        return entries, [speaker]


def test_run_pipeline_wires_components_and_exports(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT, ExportFormat.JSON),
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
    )

    assert result.language == "ru"
    assert result.duration == 1.0
    assert len(result.entries) == 1
    assert result.entries[0].speaker.display_name == "Иван"

    stem = audio_file.stem
    assert (output_dir / f"{stem}.txt").exists()
    assert (output_dir / f"{stem}.json").exists()
