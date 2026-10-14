"""Тесты сериализации результата для веб-API (контракт ``/result``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)
from audio_transcriber.export.annotations import is_low_speaker_confidence
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.web.results import (
    entry_to_dict,
    numpy_json_default,
    serialize_result,
)


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_aligner_coerces_numpy_speaker_times_to_python_float(dtype: type) -> None:
    speaker = SpeakerSegment(start=dtype(0.0), end=dtype(0.4), speaker_id="SPEAKER_00")
    segment = TranscriptionSegment(start=0.0, end=1.0, text="привет")

    entries, _ = OverlapSegmentMerger(max_gap=2.0).merge([segment], [speaker])

    assert type(entries[0].speaker_confidence) is float
    assert entries[0].speaker_confidence == pytest.approx(0.4)


def test_is_low_speaker_confidence_returns_python_bool_for_numpy_scalar() -> None:
    entry = TranscriptEntry(
        start=0.0, end=1.0, text="привет", speaker_confidence=np.float64(0.4)
    )

    result = is_low_speaker_confidence(entry)

    assert type(result) is bool
    assert result is True


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_serialize_result_with_numpy_speaker_confidence_is_json_serializable(
    dtype: type,
) -> None:
    speaker = SpeakerSegment(start=dtype(0.0), end=dtype(0.4), speaker_id="SPEAKER_00")
    segment = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="привет",
        words=[
            WordTimestamp(
                text="привет",
                start=np.float32(0.0),
                end=np.float32(0.4),
                probability=np.float32(0.9),
            )
        ],
    )
    entries, speakers = OverlapSegmentMerger(max_gap=2.0).merge([segment], [speaker])
    result = TranscriptionResult(
        source_path=Path("call.mp3"),
        language="ru",
        duration=1.0,
        entries=entries,
        speakers=speakers,
        low_confidence_threshold=-1.0,
    )

    payload = serialize_result(result)

    json.dumps(payload, default=numpy_json_default)
    entry = payload["entries"][0]
    assert type(entry["speaker_confidence"]) is float
    assert type(entry["low_speaker_confidence"]) is bool
    assert type(entry["words"][0]["probability"]) is float


def test_entry_to_dict_yields_pure_python_types() -> None:
    entry = TranscriptEntry(
        start=np.float64(0.0),
        end=np.float64(1.0),
        text="привет",
        speaker_confidence=np.float32(0.4),
        avg_logprob=np.float64(-2.0),
    )

    payload = entry_to_dict(entry, threshold=np.float64(-1.0))

    assert type(payload["speaker_confidence"]) is float
    assert type(payload["low_speaker_confidence"]) is bool
    assert type(payload["low_confidence"]) is bool


def test_numpy_json_default_converts_numpy_scalars_and_arrays() -> None:
    assert numpy_json_default(np.bool_(True)) is True
    assert numpy_json_default(np.float32(0.5)) == pytest.approx(0.5)
    assert type(numpy_json_default(np.float32(0.5))) is float
    assert numpy_json_default(np.int64(7)) == 7
    assert numpy_json_default(np.array([1, 2, 3])) == [1, 2, 3]

    with pytest.raises(TypeError):
        numpy_json_default(object())


def test_result_save_does_not_fail_with_numpy_scalar_fields() -> None:
    payload = {
        "speaker_confidence": np.float32(0.3),
        "low_speaker_confidence": np.bool_(True),
        "count": np.int64(3),
    }

    dumped = json.dumps(payload, default=numpy_json_default)
    decoded = json.loads(dumped)

    assert decoded["speaker_confidence"] == pytest.approx(0.3)
    assert decoded["low_speaker_confidence"] is True
    assert decoded["count"] == 3

