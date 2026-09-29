"""Тесты автоисправления опечаток (морфология русского языка)."""

from __future__ import annotations

from typing import Any

from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_EDIT_DISTANCE,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
)
from audio_transcriber.correction.morph_corrector import (
    MorphTextCorrector,
    contains_latin,
    is_spelling_like_form,
)
from audio_transcriber.domain.models import Speaker, TranscriptEntry


def _correct(text: str, **kwargs: Any) -> str:
    corrector = MorphTextCorrector(**kwargs)
    return corrector.correct([TranscriptEntry(start=0.0, end=1.0, text=text)])[0].text


def test_default_similarity_and_edit_distance_are_conservative() -> None:
    assert DEFAULT_CORRECTION_MIN_SIMILARITY == 0.93
    assert DEFAULT_CORRECTION_MAX_EDIT_DISTANCE == 1


def test_spelling_like_accepts_single_edit_asr_typo() -> None:
    assert is_spelling_like_form("конфиденциальнасть", "конфиденциальность")


def test_spelling_like_rejects_multi_edit() -> None:
    # «взаимоприимания» → «взаимопонимания» — две правки символа.
    assert not is_spelling_like_form("взаимоприимания", "взаимопонимания")


def test_spelling_like_rejects_latin_words() -> None:
    assert not is_spelling_like_form("login", "логин")
    assert not is_spelling_like_form("залогинился", "залоснилсяx")


def test_spelling_like_rejects_unrelated_and_short_names() -> None:
    assert not is_spelling_like_form("есть", "часть")
    # Сходство ниже порога автоисправления.
    assert not is_spelling_like_form("тестовик", "тестоват")
    assert not is_spelling_like_form("вопросы", "допроса")


def test_morph_corrector_fixes_unknown_single_edit_typo() -> None:
    corrector = MorphTextCorrector()
    speaker = Speaker(id="SPEAKER_00", display_name="Иван")
    entries = [
        TranscriptEntry(
            start=0.0,
            end=1.0,
            text="Конфиденциальнасть мы не нашли.",
            speaker=speaker,
        )
    ]

    corrected = corrector.correct(entries)

    assert corrected[0].text == "Конфиденциальность мы не нашли."
    assert corrected[0].speaker is speaker


def test_morph_corrector_does_not_touch_technical_words() -> None:
    """Тех-заимствования не должны «исправляться» на похожие русские слова."""
    corrector = MorphTextCorrector()
    original = "Он залогинился, залочился экран, пошло перелагирование."
    entries = [TranscriptEntry(start=0.0, end=1.0, text=original)]

    assert corrector.correct(entries)[0].text == original


def test_morph_corrector_does_not_touch_words_with_latin() -> None:
    corrector = MorphTextCorrector()
    original = "Сделал login и reloadConfig в API_KEY."
    entries = [TranscriptEntry(start=0.0, end=1.0, text=original)]

    assert corrector.correct(entries)[0].text == original


def test_morph_corrector_respects_similarity_threshold() -> None:
    # Одна правка, но сходство ~0.917 < 0.93 — по умолчанию не меняем.
    assert _correct("Безопаснасть важна.") == "Безопаснасть важна."
    # При более мягком пороге та же опечатка правится.
    assert _correct("Безопаснасть важна.", min_similarity=0.85) == "Безопасность важна."


def test_morph_corrector_respects_max_edit_distance() -> None:
    # Две правки: по умолчанию слово не трогаем.
    assert _correct("Взаимоприимания мы не нашли.") == "Взаимоприимания мы не нашли."
    # С явным допуском двух правок и мягким порогом — правится.
    assert (
        _correct("Взаимоприимания мы не нашли.", min_similarity=0.8, max_edit_distance=2)
        == "Взаимопонимания мы не нашли."
    )


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
    # При высоком пороге длины длинная опечатка (одна правка) всё ещё правится.
    assert (
        _correct("Конфиденциальнасть мы не нашли.", min_word_length=10)
        == "Конфиденциальность мы не нашли."
    )


def test_contains_latin() -> None:
    assert contains_latin("login")
    assert contains_latin("залогинился2x")
    assert not contains_latin("залогинился")
    assert not contains_latin("ёж")
