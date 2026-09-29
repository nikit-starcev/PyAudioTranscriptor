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


class FakeCorrector:
    def correct(self, entries):
        return [
            TranscriptEntry(
                start=entry.start,
                end=entry.end,
                text=entry.text.upper(),
                speaker=entry.speaker,
            )
            for entry in entries
        ]


class MultiSegmentMerger:
    """Возвращает несколько коротких реплик одного говорящего."""

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker),
            TranscriptEntry(start=1.1, end=2.0, text="мир", speaker=speaker),
        ]
        return entries, [speaker]


class RecordingCorrector:
    """Корректор, запоминающий входные реплики, чтобы проверить порядок этапов."""

    def __init__(self) -> None:
        self.seen: list[TranscriptEntry] = []

    def correct(self, entries):
        self.seen = list(entries)
        return entries


class ArtifactMerger:
    """Возвращает реплику-артефакт и обычную реплику одного говорящего."""

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(start=0.0, end=1.0, text="[АПЛОДИСМЕНТЫ]", speaker=speaker),
            TranscriptEntry(start=1.1, end=2.0, text="привет", speaker=speaker),
        ]
        return entries, [speaker]


class RecordingCleaner:
    """Очистка, запоминающая входные реплики, чтобы проверить порядок этапов."""

    def __init__(self) -> None:
        self.seen: list[TranscriptEntry] | None = None

    def clean(self, entries):
        self.seen = list(entries)
        return entries


def test_run_pipeline_merges_sentences_before_correction(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        enable_correction=True,
    )
    corrector = RecordingCorrector()

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=MultiSegmentMerger(),
        corrector=corrector,
    )

    # Корректор получил уже склеенную реплику.
    assert len(corrector.seen) == 1
    assert corrector.seen[0].text == "привет мир"
    assert len(result.entries) == 1
    assert result.entries[0].text == "привет мир"


def test_run_pipeline_applies_text_corrector(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        enable_correction=True,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
        corrector=FakeCorrector(),
    )

    assert result.entries[0].text == "ПРИВЕТ"


class ExplodingDiarizer:
    """Диаризатор, который обязан не вызываться при отключённой диаризации."""

    def diarize(self, *args, **kwargs):
        raise AssertionError("диаризация не должна вызываться, когда она отключена")


def test_run_pipeline_skips_diarization_when_disabled(
    audio_file: Path, tmp_path: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=ExplodingDiarizer(),
    )

    # Реплики без говорящего, список говорящих пуст.
    assert len(result.entries) == 1
    assert result.entries[0].speaker is None
    assert result.speakers == []


def test_run_pipeline_skips_correction_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        enable_correction=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        diarizer=FakeDiarizer(),
        merger=FakeMerger(),
    )

    assert result.entries[0].text == "привет"


def test_run_pipeline_removes_artifact_entries_by_default(
    audio_file: Path, tmp_path: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=ArtifactMerger(),
    )

    assert [entry.text for entry in result.entries] == ["привет"]


def test_run_pipeline_keeps_artifacts_when_cleaning_disabled(
    audio_file: Path, tmp_path: Path
) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        clean_artifacts=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=ArtifactMerger(),
    )

    # Очистка выключена: пометка остаётся в тексте (склейка предложений
    # объединяет обе реплики одного говорящего).
    assert [entry.text for entry in result.entries] == ["[АПЛОДИСМЕНТЫ] привет"]


def test_run_pipeline_cleans_before_sentence_merger(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )
    cleaner = RecordingCleaner()

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=MultiSegmentMerger(),
        artifact_cleaner=cleaner,
    )

    # Очистка получила отдельные реплики (до склейки предложений).
    assert cleaner.seen is not None
    assert [entry.text for entry in cleaner.seen] == ["привет", "мир"]
    assert len(result.entries) == 1
    assert result.entries[0].text == "привет мир"


def test_run_pipeline_emits_clean_progress_event(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
    )
    events = []

    run_pipeline(
        config,
        device=Device.CPU,
        recognizer=FakeRecognizer(),
        merger=FakeMerger(),
        on_progress=events.append,
    )

    assert any(event.stage == "clean" for event in events)

