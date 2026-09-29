"""Тесты разметки уверенности ASR (``avg_logprob``) и её экспорта."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from docx import Document

from audio_transcriber.domain.models import (
    TranscriptEntry,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_transcriber.export.docx_exporter import DocxExporter
from audio_transcriber.export.json_exporter import JsonExporter
from audio_transcriber.export.txt_exporter import TxtExporter
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.merging.sentence_merger import SentenceMerger
from audio_transcriber.transcription.whisper_cpp_engine import _segment_avg_logprob


def _entry(
    start: float, end: float, text: str, *, logprob: float | None = None, overlap: bool = False
) -> TranscriptEntry:
    return TranscriptEntry(start=start, end=end, text=text, avg_logprob=logprob, overlap=overlap)


def _result(entries: list[TranscriptEntry], threshold: float | None = -1.0) -> TranscriptionResult:
    return TranscriptionResult(
        source_path=Path("call.mp3"),
        language="ru",
        duration=5.0,
        entries=entries,
        speakers=[],
        low_confidence_threshold=threshold,
    )


# --- проброс и склейка -------------------------------------------------------


def test_aligner_propagates_avg_logprob() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="привет", avg_logprob=-0.4)]

    entries, _ = OverlapSegmentMerger().merge(segments, [])

    assert entries[0].avg_logprob == -0.4


def test_aligner_keeps_none_when_absent() -> None:
    segments = [TranscriptionSegment(start=0.0, end=1.0, text="привет")]

    entries, _ = OverlapSegmentMerger().merge(segments, [])

    assert entries[0].avg_logprob is None


def test_sentence_merger_takes_worst_logprob() -> None:
    entries = [
        _entry(0.0, 1.0, "раз", logprob=-0.2),
        _entry(1.0, 2.0, "два", logprob=-1.5),
        _entry(2.0, 3.0, "три", logprob=-0.9),
    ]

    merged = SentenceMerger().merge(entries)

    assert len(merged) == 1
    assert merged[0].avg_logprob == -1.5


def test_sentence_merger_ignores_none_logprobs() -> None:
    entries = [_entry(0.0, 1.0, "раз", logprob=None), _entry(1.0, 2.0, "два", logprob=-1.2)]

    merged = SentenceMerger().merge(entries)

    assert merged[0].avg_logprob == -1.2


def test_sentence_merger_all_none_stays_none() -> None:
    entries = [_entry(0.0, 1.0, "раз"), _entry(1.0, 2.0, "два")]

    merged = SentenceMerger().merge(entries)

    assert merged[0].avg_logprob is None


def test_sentence_merger_ors_overlap_flag() -> None:
    entries = [
        _entry(0.0, 1.0, "раз", overlap=True),
        _entry(1.0, 2.0, "два", overlap=False),
    ]

    merged = SentenceMerger().merge(entries)

    assert merged[0].overlap is True


# --- извлечение из whisper.cpp ----------------------------------------------


def test_whisper_cpp_avg_logprob_skips_special_tokens() -> None:
    item = {
        "tokens": [
            {"text": "[_BEG_]", "p": 1.0},
            {"text": " привет", "p": 0.5},
            {"text": " мир", "p": 0.25},
            {"text": "[_TT_1]", "p": 0.01},
        ]
    }

    expected = (math.log(0.5) + math.log(0.25)) / 2

    assert _segment_avg_logprob(item) == pytest.approx(expected)


def test_whisper_cpp_avg_logprob_none_without_tokens() -> None:
    assert _segment_avg_logprob({}) is None
    assert _segment_avg_logprob({"tokens": []}) is None
    assert _segment_avg_logprob({"tokens": [{"text": "[_BEG_]", "p": 1.0}]}) is None


def test_whisper_cpp_avg_logprob_ignores_non_positive_probability() -> None:
    item = {"tokens": [{"text": " слово", "p": 0.0}, {"text": " ещё", "p": 0.5}]}

    assert _segment_avg_logprob(item) == pytest.approx(math.log(0.5))


# --- экспорт -----------------------------------------------------------------


def test_txt_marks_low_confidence_entry(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    TxtExporter().export(_result([_entry(0.0, 1.0, "плохо", logprob=-1.5)]), output)

    content = output.read_text(encoding="utf-8")
    assert "плохо ⚠ [низкая уверенность]" in content


def test_txt_does_not_mark_confident_entry(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    TxtExporter().export(_result([_entry(0.0, 1.0, "хорошо", logprob=-0.1)]), output)

    assert "низкая уверенность" not in output.read_text(encoding="utf-8")


def test_txt_does_not_mark_when_threshold_disabled(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    TxtExporter().export(_result([_entry(0.0, 1.0, "плохо", logprob=-9.0)], threshold=None), output)

    assert "низкая уверенность" not in output.read_text(encoding="utf-8")


def test_txt_does_not_mark_when_logprob_unknown(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    TxtExporter().export(_result([_entry(0.0, 1.0, "неизвестно")]), output)

    assert "низкая уверенность" not in output.read_text(encoding="utf-8")


def test_docx_marks_low_confidence_entry(tmp_path: Path) -> None:
    output = tmp_path / "out.docx"

    DocxExporter().export(_result([_entry(0.0, 1.0, "плохо", logprob=-2.0)]), output)

    document = Document(str(output))
    full_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "низкая уверенность" in full_text


def test_json_exports_confidence_and_overlap_fields(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    entries = [
        _entry(0.0, 1.0, "плохо", logprob=-2.0),
        _entry(1.0, 2.0, "хорошо", logprob=-0.1, overlap=True),
        _entry(2.0, 3.0, "неизвестно"),
    ]

    JsonExporter().export(_result(entries), output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["entries"][0] == {
        "start": 0.0,
        "end": 1.0,
        "text": "плохо",
        "speaker": None,
        "avg_logprob": -2.0,
        "low_confidence": True,
        "overlap": False,
    }
    assert payload["entries"][1]["avg_logprob"] == -0.1
    assert payload["entries"][1]["low_confidence"] is False
    assert payload["entries"][1]["overlap"] is True
    assert payload["entries"][2]["avg_logprob"] is None
    assert payload["entries"][2]["low_confidence"] is False
    assert payload["entries"][2]["overlap"] is False


def test_json_low_confidence_false_when_threshold_disabled(tmp_path: Path) -> None:
    output = tmp_path / "out.json"

    JsonExporter().export(
        _result([_entry(0.0, 1.0, "плохо", logprob=-9.0)], threshold=None), output
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["entries"][0]["avg_logprob"] == -9.0
    assert payload["entries"][0]["low_confidence"] is False
