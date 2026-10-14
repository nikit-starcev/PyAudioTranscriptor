"""Тесты протокола изоляции движков (#143)."""

from __future__ import annotations

import pytest

from audio_transcriber.cache.serialization import asr_from_payload, diarization_from_payload
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment, TranscriptionSegment
from audio_transcriber.isolation.protocol import (
    KIND_ASR,
    KIND_DIARIZATION,
    PROGRESS_PREFIX,
    WorkerRequest,
    asr_result,
    diarization_result,
    parse_progress_line,
    progress_line,
    request_from_json,
    request_to_json,
)
from audio_transcriber.progress import ProgressEvent


def test_request_roundtrip_asr() -> None:
    request = WorkerRequest(
        kind=KIND_ASR,
        audio_path="/tmp/a.wav",
        language="ru",
        options={"model_name": "small", "device": "cpu"},
    )

    restored = request_from_json(request_to_json(request))

    assert restored == request
    assert restored.options == {"model_name": "small", "device": "cpu"}


def test_request_roundtrip_diarization_with_waveform() -> None:
    request = WorkerRequest(
        kind=KIND_DIARIZATION,
        audio_path="/tmp/a.wav",
        waveform_path="/tmp/w.npy",
        options={"num_speakers": 3},
    )

    assert request_from_json(request_to_json(request)) == request


def test_request_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        WorkerRequest(kind="nope", audio_path="/tmp/a.wav")

    with pytest.raises(ValueError):
        request_from_json('{"kind": "nope", "audio_path": "/tmp/a.wav", "options": {}}')


def test_request_rejects_missing_audio_path() -> None:
    with pytest.raises(ValueError):
        request_from_json('{"kind": "asr", "options": {}}')


def test_request_rejects_non_object_options() -> None:
    with pytest.raises(ValueError):
        request_from_json('{"kind": "asr", "audio_path": "/x", "options": []}')


def test_progress_line_roundtrip() -> None:
    event = ProgressEvent(stage="asr", message="Речь", fraction=0.5, detail="шаг")

    line = progress_line(event)

    assert line.startswith(PROGRESS_PREFIX)
    assert parse_progress_line(line) == event


def test_progress_line_ignores_foreign_and_malformed() -> None:
    assert parse_progress_line("обычный лог движка") is None
    assert parse_progress_line(PROGRESS_PREFIX + "{не json") is None
    assert parse_progress_line(PROGRESS_PREFIX + '{"fraction": 1}') is None


def test_asr_result_is_readable_by_cache_parser() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="привет", avg_logprob=-0.2)]

    payload = asr_result(segments, "ru", 1.0)
    parsed, language, duration = asr_from_payload(payload)

    assert parsed == segments
    assert language == "ru"
    assert duration == 1.0


def test_diarization_result_is_readable_by_cache_parser() -> None:
    segments = [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]
    overlaps = [SpeakerOverlap(start=0.2, end=0.4, speaker_ids=("SPEAKER_00", "SPEAKER_01"))]

    payload = diarization_result(segments, overlaps)
    parsed_segments, parsed_overlaps = diarization_from_payload(payload)

    assert parsed_segments == segments
    assert parsed_overlaps == overlaps
