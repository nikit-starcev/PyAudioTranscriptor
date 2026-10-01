"""Тесты таймлайна «кто когда говорил» (текстовая сводка и HTML)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_transcriber.export.timeline import (
    build_speaker_tracks,
    render_timeline_html,
    render_timeline_text,
    write_timeline,
)
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.pipeline import run_pipeline


def _timeline_result(audio_file: Path) -> TranscriptionResult:
    ivan = Speaker(id="SPEAKER_00", display_name="Иван")
    maria = Speaker(id="SPEAKER_01", display_name="Мария")
    return TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=60.0,
        entries=[
            TranscriptEntry(start=0.0, end=29.0, text="раз", speaker=ivan),
            TranscriptEntry(start=29.0, end=40.0, text="два", speaker=ivan),
            TranscriptEntry(start=40.0, end=53.0, text="три", speaker=maria),
            TranscriptEntry(start=53.0, end=59.0, text="четыре", speaker=ivan),
        ],
        speakers=[ivan, maria],
    )


def test_build_tracks_groups_consecutive_speaker_entries(audio_file: Path) -> None:
    tracks = build_speaker_tracks(_timeline_result(audio_file))

    assert [(track.speaker_id, track.intervals) for track in tracks] == [
        ("SPEAKER_00", ((0.0, 40.0), (53.0, 59.0))),
        ("SPEAKER_01", ((40.0, 53.0),)),
    ]


def test_render_timeline_text_is_compact_and_labelled(audio_file: Path) -> None:
    text = render_timeline_text(_timeline_result(audio_file))

    assert text == (
        "Спикер 1 (Иван): 00:00–00:40, 00:53–00:59\n"
        "Спикер 2 (Мария): 00:40–00:53"
    )


def test_write_timeline_creates_self_contained_html(audio_file: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "call.timeline.html"

    assert write_timeline(_timeline_result(audio_file), output_path) is True

    content = output_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Иван" in content
    assert "Мария" in content
    assert "00:00–00:40" in content
    assert "00:40–00:53" in content
    # Самодостаточность: никаких внешних ссылок/скриптов/CDN.
    assert "http://" not in content
    assert "https://" not in content
    assert "<script" not in content


def test_render_timeline_html_without_speakers_is_empty_chart(audio_file: Path) -> None:
    result = TranscriptionResult(
        source_path=audio_file,
        language=None,
        duration=5.0,
        entries=[TranscriptEntry(start=0.0, end=1.0, text="без говорящего")],
        speakers=[],
    )

    assert build_speaker_tracks(result) == []
    assert render_timeline_text(result) == ""
    assert "Кто когда говорил" in render_timeline_html(result)


def test_write_timeline_skips_without_diarization(audio_file: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "call.timeline.html"
    result = TranscriptionResult(
        source_path=audio_file,
        language=None,
        duration=5.0,
        entries=[TranscriptEntry(start=0.0, end=1.0, text="без говорящего")],
        speakers=[],
    )

    assert write_timeline(result, output_path) is False
    assert not output_path.exists()


def test_build_tracks_places_extra_speakers_on_their_own_tracks(audio_file: Path) -> None:
    ivan = Speaker(id="SPEAKER_00", display_name="Иван")
    maria = Speaker(id="SPEAKER_01", display_name="Мария")
    result = TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=10.0,
        entries=[
            TranscriptEntry(
                start=0.0,
                end=5.0,
                text="спор",
                speaker=ivan,
                extra_speakers=[maria],
                overlap=True,
                speaker_confidence=0.3,
            ),
        ],
        speakers=[ivan, maria],
    )

    tracks = build_speaker_tracks(result)

    assert [(track.speaker_id, track.intervals) for track in tracks] == [
        ("SPEAKER_00", ((0.0, 5.0),)),
        ("SPEAKER_01", ((0.0, 5.0),)),
    ]
    text = render_timeline_text(result)
    assert "Спикер 1 (Иван): 00:00–00:05" in text
    assert "Спикер 2 (Мария): 00:00–00:05" in text


def test_render_timeline_text_uses_long_form_for_hours(audio_file: Path) -> None:
    speaker = Speaker(id="SPEAKER_00", display_name="Спикер 1")
    result = TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=4000.0,
        entries=[TranscriptEntry(start=0.0, end=3661.0, text="долго", speaker=speaker)],
        speakers=[speaker],
    )

    # Имя не дублируется, если совпадает с базовой меткой говорящего.
    assert render_timeline_text(result) == "Спикер 1: 00:00–1:01:01"


def test_render_timeline_text_does_not_duplicate_speaker_name(audio_file: Path) -> None:
    numbered = Speaker(id="SPEAKER_00", display_name="Спикер 1")
    empty_name = Speaker(id="SPEAKER_01", display_name="")
    result = TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=60.0,
        entries=[
            TranscriptEntry(start=0.0, end=10.0, text="раз", speaker=numbered),
            TranscriptEntry(start=10.0, end=20.0, text="два", speaker=empty_name),
        ],
        speakers=[numbered, empty_name],
    )

    assert render_timeline_text(result) == "Спикер 1: 00:00–00:10\nСпикер 2: 00:10–00:20"


class _Recognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return (
            [
                TranscriptionSegment(start=0.0, end=1.0, text="привет"),
                TranscriptionSegment(start=1.0, end=2.0, text="пока"),
            ],
            "ru",
            2.0,
        )


class _Diarizer:
    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: object = None,
    ):
        return [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]


def _run(config: AppConfig) -> TranscriptionResult:
    return run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_Recognizer(),
        diarizer=_Diarizer(),
        merger=OverlapSegmentMerger(),
    )


def test_pipeline_writes_timeline_when_diarization_present(
    audio_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )

    with caplog.at_level(logging.INFO, logger="audio_transcriber.pipeline"):
        _run(config)

    timeline_path = output_dir / f"{audio_file.stem}.timeline.html"
    assert timeline_path.exists()
    assert "Таймлайн говорящих" in caplog.text
    assert "Спикер 1" in caplog.text


def test_pipeline_skips_timeline_without_diarization(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        denoise=False,
    )

    _run(config)

    assert not (output_dir / f"{audio_file.stem}.timeline.html").exists()


def test_pipeline_skips_timeline_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        timeline=False,
    )

    _run(config)

    assert not (output_dir / f"{audio_file.stem}.timeline.html").exists()
