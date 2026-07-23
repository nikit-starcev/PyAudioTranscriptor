"""Тесты постобработки текста через SymSpell."""

from __future__ import annotations

from audio_transcriber.correction.symspell_corrector import SymSpellTextCorrector
from audio_transcriber.domain.models import Speaker, TranscriptEntry


def test_symspell_corrects_similar_word() -> None:
    corrector = SymSpellTextCorrector(["взаимопонимание", "взаимопонимания"])
    speaker = Speaker(id="SPEAKER_00", display_name="Иван")
    entries = [
        TranscriptEntry(
            start=0.0,
            end=1.0,
            text="Взаимоприимания мы не нашли.",
            speaker=speaker,
        )
    ]

    corrected = corrector.correct(entries)

    assert corrected[0].text == "Взаимопонимания мы не нашли."
    assert corrected[0].speaker is speaker


def test_symspell_keeps_unknown_words() -> None:
    corrector = SymSpellTextCorrector(["взаимопонимание"])
    entries = [TranscriptEntry(start=0.0, end=1.0, text="Совершенно другое слово.")]

    corrected = corrector.correct(entries)

    assert corrected[0].text == "Совершенно другое слово."


def test_symspell_preserves_short_words_and_punctuation() -> None:
    corrector = SymSpellTextCorrector(["договор"])
    entries = [TranscriptEntry(start=0.0, end=1.0, text="И да, мы.")]

    corrected = corrector.correct(entries)

    assert corrected[0].text == "И да, мы."


def test_symspell_empty_dictionary_is_noop() -> None:
    corrector = SymSpellTextCorrector([])
    entries = [TranscriptEntry(start=0.0, end=1.0, text="Взаимоприимания")]

    assert corrector.correct(entries)[0].text == "Взаимоприимания"
