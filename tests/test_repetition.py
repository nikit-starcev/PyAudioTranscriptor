"""Тесты схлопывания подряд повторяющихся реплик (``RepetitionCleaner``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.cleaning.repetition_filter import RepetitionCleaner
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import TranscriptEntry, TranscriptionSegment
from audio_transcriber.pipeline import run_pipeline


def _entry(text: str, start: float = 0.0, end: float = 1.0) -> TranscriptEntry:
    return TranscriptEntry(start=start, end=end, text=text)


def test_collapses_consecutive_identical_entries() -> None:
    entries = [
        _entry("Продолжение следует", 0.0, 1.0),
        _entry("Продолжение следует", 1.0, 2.0),
        _entry("Продолжение следует", 2.0, 3.0),
    ]

    result = RepetitionCleaner().clean(entries)

    assert len(result) == 1
    assert result[0].text == "Продолжение следует"
    assert result[0].start == 0.0
    assert result[0].end == 3.0


def test_collapses_case_and_punctuation_variants() -> None:
    entries = [
        _entry("Спасибо за просмотр!"),
        _entry("спасибо за просмотр"),
        _entry("СПАСИБО, ЗА ПРОСМОТР..."),
    ]

    assert len(RepetitionCleaner().clean(entries)) == 1


def test_collapses_near_identical_above_threshold() -> None:
    entries = [_entry("обсуждение вопроса"), _entry("обсуждение вопросов")]

    assert len(RepetitionCleaner().clean(entries)) == 1


def test_keeps_dissimilar_entries() -> None:
    entries = [_entry("совершенно разные фразы"), _entry("абсолютно другой текст")]

    assert len(RepetitionCleaner().clean(entries)) == 2


def test_keeps_short_repeated_speech() -> None:
    # «да, да» — осмысленная короткая речь, схлопывать нельзя.
    assert len(RepetitionCleaner().clean([_entry("да"), _entry("да")])) == 2


def test_collapses_known_hallucination_even_if_short() -> None:
    entries = [_entry("Конец"), _entry("Конец")]

    assert len(RepetitionCleaner().clean(entries)) == 1


def test_does_not_merge_distinct_numbered_entries() -> None:
    # Цифры сохраняются в нормализации — «Пункт 1» и «Пункт 2» не повтор.
    entries = [_entry("Пункт 1"), _entry("Пункт 2")]

    assert len(RepetitionCleaner().clean(entries)) == 2


def test_similarity_threshold_is_configurable() -> None:
    entries = [_entry("привет мир друзья"), _entry("привет мир враги")]

    assert len(RepetitionCleaner(similarity=0.3).clean(entries)) == 1
    assert len(RepetitionCleaner(similarity=0.99).clean(entries)) == 2


def test_min_words_is_configurable() -> None:
    entries = [_entry("да"), _entry("да")]

    assert len(RepetitionCleaner(min_words=1).clean(entries)) == 1


def test_metadata_preserved_when_collapsing() -> None:
    first = _entry("Продолжение следует", 0.0, 2.5)
    second = _entry("Продолжение следует", 3.0, 4.0)

    result = RepetitionCleaner().clean([first, second])

    assert result[0].start == 0.0
    assert result[0].end == 4.0


@pytest.mark.parametrize("min_words", [0, -1])
def test_invalid_min_words_rejected(min_words: int) -> None:
    with pytest.raises(ValueError):
        RepetitionCleaner(min_words=min_words)


@pytest.mark.parametrize("similarity", [0.0, -0.1, 1.5])
def test_invalid_similarity_rejected(similarity: float) -> None:
    with pytest.raises(ValueError):
        RepetitionCleaner(similarity=similarity)


# --- интеграция с конвейером -------------------------------------------------


class _FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return ([TranscriptionSegment(start=0.0, end=1.0, text="привет")], "ru", 1.0)


#: Нейтральная повторяющаяся фраза (не шаблонная галлюцинация): ArtifactCleaner
#: вырезает известные фразы-заглушки (в т.ч. «Продолжение следует»), поэтому для
#: проверки именно схлопывания повторов берём обычную реплику.
_REPEATED_PHRASE = "Повторяющаяся реплика"


class _DuplicateMerger:
    """Возвращает две подряд одинаковые реплики одного говорящего."""

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        return [
            _entry(_REPEATED_PHRASE, 0.0, 1.0),
            _entry(_REPEATED_PHRASE, 1.0, 2.0),
        ], []


def test_pipeline_collapses_repeats_by_default(audio_file: Path, tmp_path: Path) -> None:
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
        merger=_DuplicateMerger(),
    )

    assert [entry.text for entry in result.entries] == [_REPEATED_PHRASE]


def test_pipeline_keeps_repeats_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        denoise=False,
        collapse_repeats=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        merger=_DuplicateMerger(),
    )

    # Реплики не схлопнуты и затем склеены SentenceMerger'ом в один текст.
    assert result.entries[0].text.count(_REPEATED_PHRASE) == 2
