"""Тесты автоисправления опечаток (морфология русского языка)."""

from __future__ import annotations

from audio_transcriber.correction.morph_corrector import (
    MorphTextCorrector,
    _is_spelling_like_fix,
)
from audio_transcriber.domain.models import Speaker, TranscriptEntry


def test_spelling_like_accepts_asr_typo() -> None:
    assert _is_spelling_like_fix("взаимоприимания", "взаимопонимания")


def test_spelling_like_rejects_unrelated_and_short_names() -> None:
    assert not _is_spelling_like_fix("есть", "часть")
    # Сходство ниже порога автоисправления (~0.85).
    assert not _is_spelling_like_fix("тестовик", "тестоват")
    assert not _is_spelling_like_fix("вопросы", "допроса")


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
