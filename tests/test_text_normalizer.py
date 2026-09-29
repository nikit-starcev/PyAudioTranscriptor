"""Тесты безопасной нормализации текста (``TextNormalizer``)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.cleaning.text_normalizer import TextNormalizer, normalize_text
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import TranscriptEntry, TranscriptionSegment
from audio_transcriber.pipeline import run_pipeline


def test_plain_text_is_unchanged() -> None:
    entry = TranscriptEntry(start=0.0, end=1.0, text="Обычная нормальная речь.")

    result = TextNormalizer().normalize([entry])

    assert result[0] is entry


def test_collapses_spaces_and_punctuation() -> None:
    assert normalize_text("Привет   мир !!!  Как   дела ??") == "Привет мир! Как дела?"


def test_collapses_excess_dots() -> None:
    assert normalize_text("Ну......") == "Ну..."
    assert normalize_text("Ну..") == "Ну."


def test_keeps_standard_ellipsis() -> None:
    assert normalize_text("Ну...") == "Ну..."


def test_normalizes_spaces_around_brackets() -> None:
    assert normalize_text("Он сказал ( очень важно )") == "Он сказал (очень важно)"


def test_elementary_number_normalization() -> None:
    assert normalize_text("Итого 3 , 14 и 50 %") == "Итого 3,14 и 50%"
    assert normalize_text("Время 12 : 30") == "Время 12:30"
    assert normalize_text("Дробь 1 / 2") == "Дробь 1/2"


def test_does_not_touch_words() -> None:
    assert normalize_text("взаимоприимания") == "взаимоприимания"


def test_is_idempotent() -> None:
    text = "Ну   это...   очень !!! странно , да"

    once = normalize_text(text)

    assert normalize_text(once) == once


def test_blank_text_is_unchanged() -> None:
    assert normalize_text("   ") == "   "


# --- интеграция с конвейером -------------------------------------------------


class _FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return ([TranscriptionSegment(start=0.0, end=1.0, text="привет")], "ru", 1.0)


class _TextMerger:
    def __init__(self, text: str) -> None:
        self._text = text

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        return [TranscriptEntry(start=0.0, end=1.0, text=self._text)], []


def test_pipeline_normalizes_text_by_default(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        denoise=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        merger=_TextMerger("Привет   мир !!!"),
    )

    assert result.entries[0].text == "Привет мир!"


def test_pipeline_keeps_text_when_normalization_disabled(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        denoise=False,
        normalize_text=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        merger=_TextMerger("Привет   мир !!!"),
    )

    assert result.entries[0].text == "Привет   мир !!!"
