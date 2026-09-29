"""Тесты пометки наложения речи: вычисление зон, модель и интеграция."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.overlap import compute_overlap_regions
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    SpeakerOverlap,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)
from audio_transcriber.merging.overlap import mark_overlap_entries
from audio_transcriber.pipeline import run_pipeline

# --- вычисление зон ----------------------------------------------------------


def test_compute_overlap_regions_finds_shared_interval() -> None:
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=5.0, speaker_id="SPEAKER_01"),
    ]

    assert compute_overlap_regions(segments) == [SpeakerOverlap(start=2.0, end=3.0)]


def test_compute_overlap_regions_empty_without_overlap() -> None:
    segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
    ]

    assert compute_overlap_regions(segments) == []


def test_compute_overlap_regions_three_speakers() -> None:
    segments = [
        SpeakerSegment(start=0.0, end=4.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=5.0, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=2.0, end=3.0, speaker_id="SPEAKER_02"),
    ]

    assert compute_overlap_regions(segments) == [SpeakerOverlap(start=1.0, end=4.0)]


def test_compute_overlap_regions_ignores_degenerate() -> None:
    segments = [SpeakerSegment(start=1.0, end=1.0, speaker_id="SPEAKER_00")]

    assert compute_overlap_regions(segments) == []


# --- пометка реплик ----------------------------------------------------------


def test_mark_overlap_entries_marks_intersecting_entry() -> None:
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="вне"),
        TranscriptEntry(start=1.0, end=2.0, text="внутри"),
    ]
    regions = [SpeakerOverlap(start=0.5, end=1.5)]

    result = mark_overlap_entries(entries, regions)

    assert [entry.overlap for entry in result] == [True, True]
    assert result[0].text == "вне"


def test_mark_overlap_entries_leaves_non_intersecting() -> None:
    entries = [TranscriptEntry(start=5.0, end=6.0, text="тишина")]

    result = mark_overlap_entries(entries, [SpeakerOverlap(start=0.0, end=1.0)])

    assert result[0].overlap is False


def test_mark_overlap_entries_returns_same_list_without_regions() -> None:
    entries = [TranscriptEntry(start=0.0, end=1.0, text="текст")]

    result = mark_overlap_entries(entries, [])

    assert result is entries


# --- движок pyannote ---------------------------------------------------------


class _FakeTurns:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, *, yield_label: bool = False):
        for start, end, speaker in self._turns:
            yield SimpleNamespace(start=start, end=end), None, speaker


class _FakeDiarizationOutput:
    def __init__(self, *, with_regular: bool) -> None:
        self.exclusive_speaker_diarization = _FakeTurns(
            [(0.0, 2.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01")]
        )
        if with_regular:
            self.speaker_diarization = _FakeTurns(
                [(0.0, 3.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01")]
            )


class _FakePipeline:
    def __init__(self, output: object) -> None:
        self._output = output

    def to(self, *args: object, **kwargs: object) -> _FakePipeline:
        return self

    def __call__(self, audio: object, *, num_speakers=None, hook=None):
        if hook is not None:
            hook("segmentation", None, file={"uri": "test"}, total=1, completed=1)
        return self._output


def _run_diarizer(output: object, monkeypatch: pytest.MonkeyPatch) -> PyannoteSpeakerDiarizer:
    diarizer = PyannoteSpeakerDiarizer(Device.CPU)
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: _FakePipeline(output))
    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        lambda _path, **_kwargs: np.zeros(16000, dtype=np.float32),
    )
    diarizer.diarize(Path("audio.wav"))
    return diarizer


def test_overlap_regions_from_regular_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    diarizer = _run_diarizer(_FakeDiarizationOutput(with_regular=True), monkeypatch)

    assert diarizer.overlap_regions() == [SpeakerOverlap(start=2.0, end=3.0)]


def test_overlap_regions_empty_without_regular_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diarizer = _run_diarizer(_FakeDiarizationOutput(with_regular=False), monkeypatch)

    assert diarizer.overlap_regions() == []


def test_overlap_regions_empty_before_diarize() -> None:
    assert PyannoteSpeakerDiarizer(Device.CPU).overlap_regions() == []


# --- интеграция с конвейером -------------------------------------------------


class _FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return ([TranscriptionSegment(start=0.0, end=1.0, text="привет")], "ru", 1.0)


class _OverlapDiarizer:
    def diarize(self, audio_path: Path, *, num_speakers: int | None = None):
        return [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    def overlap_regions(self):
        return [SpeakerOverlap(start=0.5, end=1.5)]


class _SingleEntryMerger:
    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        return [TranscriptEntry(start=0.0, end=2.0, text="спор")], []


def test_pipeline_marks_overlap_entries(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_OverlapDiarizer(),
        merger=_SingleEntryMerger(),
    )

    assert result.entries[0].overlap is True
    content = (output_dir / "sample.txt").read_text(encoding="utf-8")
    assert "[наложение речи]" in content


def test_pipeline_skips_overlap_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        mark_overlap=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_OverlapDiarizer(),
        merger=_SingleEntryMerger(),
    )

    assert result.entries[0].overlap is False


def test_pipeline_degrades_without_overlap_method(audio_file: Path, tmp_path: Path) -> None:
    class _PlainDiarizer:
        def diarize(self, audio_path: Path, *, num_speakers: int | None = None):
            return [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_PlainDiarizer(),
        merger=_SingleEntryMerger(),
    )

    assert result.entries[0].overlap is False
