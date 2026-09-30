"""Тесты сериализации результата для веб-API (контракт ``/result``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.web.results import serialize_result


def _result(*, include_extra_in_speakers: bool) -> TranscriptionResult:
    anya = Speaker(id="SPEAKER_00", display_name="Аня")
    boris = Speaker(id="SPEAKER_01", display_name="Боря")
    entry = TranscriptEntry(
        start=0.0,
        end=1.0,
        text="хором",
        speaker=anya,
        overlap=True,
        extra_speakers=[boris],
        speaker_confidence=0.3,
    )
    speakers = [anya, boris] if include_extra_in_speakers else [anya]
    return TranscriptionResult(
        source_path=Path("call.mp3"),
        language="ru",
        duration=1.0,
        entries=[entry],
        speakers=speakers,
        low_confidence_threshold=-1.0,
    )


def test_serialize_result_exposes_extra_speakers_and_confidence() -> None:
    payload = serialize_result(_result(include_extra_in_speakers=True))
    entry = payload["entries"][0]

    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["extra_speaker_ids"] == ["SPEAKER_01"]
    assert entry["speaker_confidence"] == pytest.approx(0.3)
    assert entry["low_speaker_confidence"] is True
    assert entry["overlap"] is True
    assert {mark["key"] for mark in payload["marks"]} == {
        "low_confidence",
        "speaker_uncertain",
        "overlap",
    }


def test_serialize_result_adds_missing_extra_to_speakers() -> None:
    # Даже если доп. говорящий не попал в result.speakers, его id должен
    # разрешаться фронтендом — сериализатор добавляет его в список.
    payload = serialize_result(_result(include_extra_in_speakers=False))

    ids = [speaker["id"] for speaker in payload["speakers"]]
    assert ids == ["SPEAKER_00", "SPEAKER_01"]
    assert payload["entries"][0]["extra_speaker_ids"] == ["SPEAKER_01"]
