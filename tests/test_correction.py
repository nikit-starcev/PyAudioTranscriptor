"""Тесты автоисправления опечаток (морфология русского языка)."""

from __future__ import annotations

from audio_transcriber.correction.morph_corrector import (
    MorphTextCorrector,
    is_spelling_like_form,
)
from audio_transcriber.domain.models import Speaker, TranscriptEntry


def test_spelling_like_accepts_asr_typo() -> None:
    assert is_spelling_like_form("взаимоприимания", "взаимопонимания")


def test_spelling_like_rejects_unrelated_and_short_names() -> None:
    assert not is_spelling_like_form("есть", "часть")
    # Сходство ниже порога автоисправления (~0.85).
    assert not is_spelling_like_form("тестовик", "тестоват")
    assert not is_spelling_like_form("вопросы", "допроса")


def test_morph_corrector_fixes_unknown_asr_typo() -> None:
    corrector = MorphTextCorrector()
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


def test_morph_corrector_does_not_touch_known_words() -> None:
    corrector = MorphTextCorrector()
    original = "Есть вопросы к представителем суда."
    entries = [TranscriptEntry(start=0.0, end=1.0, text=original)]

    assert corrector.correct(entries)[0].text == original


def test_morph_corrector_does_not_invent_from_short_surname() -> None:
    corrector = MorphTextCorrector()
    # Короткая неизвестная фамилия не должна «исправляться» в похожее слово.
    original = "Сивков передал документы."
    entries = [TranscriptEntry(start=0.0, end=1.0, text=original)]

    assert corrector.correct(entries)[0].text == original


def test_morph_corrector_respects_custom_min_word_length() -> None:
    # При высоком пороге длины длинная опечатка всё ещё правится.
    corrector = MorphTextCorrector(min_word_length=10)
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="Взаимоприимания мы не нашли.")
    ]

    assert corrector.correct(entries)[0].text == "Взаимопонимания мы не нашли."
