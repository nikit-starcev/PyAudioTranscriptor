"""Тесты очистки стенограммы от неречевых артефактов Whisper."""

from __future__ import annotations

from audio_transcriber.cleaning import ArtifactCleaner
from audio_transcriber.domain.models import Speaker, TranscriptEntry


def _entry(text: str, start: float = 0.0, end: float = 1.0) -> TranscriptEntry:
    return TranscriptEntry(start=start, end=end, text=text)


def test_bracket_only_noise_entry_is_removed() -> None:
    cleaner = ArtifactCleaner()

    assert cleaner.clean([_entry("[АПЛОДИСМЕНТЫ]")]) == []


def test_noise_marker_inside_text_is_cut_out() -> None:
    cleaner = ArtifactCleaner()

    result = cleaner.clean([_entry("Текст [СМЕХ] ещё текст")])

    assert len(result) == 1
    assert result[0].text == "Текст ещё текст"
    assert "СМЕХ" not in result[0].text


def test_music_symbols_blank_audio_and_parentheses_are_removed() -> None:
    cleaner = ArtifactCleaner()

    entries = [
        _entry("♪♪ музыка ♪"),
        _entry("[BLANK_AUDIO]"),
        _entry("(аплодисменты)"),
    ]

    assert cleaner.clean(entries) == []


def test_meaningful_parenthetical_text_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("Он сказал (это важно)")

    result = cleaner.clean([entry])

    assert result == [entry]
    assert result[0].text == "Он сказал (это важно)"


def test_meaningful_phrase_with_noise_word_inside_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("Он подал (важный сигнал)")

    result = cleaner.clean([entry])

    assert result[0].text == "Он подал (важный сигнал)"


def test_case_and_inflection_are_ignored() -> None:
    cleaner = ArtifactCleaner()

    assert cleaner.clean([_entry("[Аплодисменты]")]) == []
    assert cleaner.clean([_entry("[аплодисменты]")]) == []
    assert cleaner.clean([_entry("[АПЛОДИСМЕНТАМИ]")]) == []


def test_combined_noise_annotation_is_removed() -> None:
    cleaner = ArtifactCleaner()

    assert cleaner.clean([_entry("[СМЕХ И АПЛОДИСМЕНТЫ]")]) == []
    assert cleaner.clean([_entry("[ПАУЗА 5 СЕК]")]) == []


def test_music_symbols_around_text_are_stripped() -> None:
    cleaner = ArtifactCleaner()

    result = cleaner.clean([_entry("♪ привет ♪")])

    assert len(result) == 1
    assert result[0].text == "привет"


def test_plain_text_is_returned_untouched() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("Обычная речь без пометок.")

    result = cleaner.clean([entry])

    assert result[0] is entry
    assert result[0].text == "Обычная речь без пометок."


def test_punctuation_only_text_is_not_dropped() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("...")

    result = cleaner.clean([entry])

    assert result == [entry]


def test_metadata_is_preserved_when_marker_removed() -> None:
    cleaner = ArtifactCleaner()
    speaker = Speaker(id="SPEAKER_00", display_name="Иван")
    entry = TranscriptEntry(start=1.5, end=3.0, text="привет [шум]", speaker=speaker)

    result = cleaner.clean([entry])

    assert len(result) == 1
    assert result[0].text == "привет"
    assert result[0].start == 1.5
    assert result[0].end == 3.0
    assert result[0].speaker is speaker
