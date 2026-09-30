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


# --- «голые» шумовые слова без скобок --------------------------------------


def test_bare_repeated_noise_entry_is_removed() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("АПЛОДИСМЕНТЫ " * 15)

    assert cleaner.clean([entry]) == []


def test_bare_trailing_noise_is_trimmed_from_speech() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("112 без изменений. АИ тоже. АПЛОДИСМЕНТЫ")

    result = cleaner.clean([entry])

    assert len(result) == 1
    assert result[0].text == "112 без изменений. АИ тоже."
    assert result[0].start == entry.start
    assert result[0].end == entry.end


def test_bare_repeated_noise_tail_is_trimmed() -> None:
    cleaner = ArtifactCleaner()

    result = cleaner.clean([_entry("Спасибо. СМЕХ СМЕХ")])

    assert [entry.text for entry in result] == ["Спасибо."]


def test_bare_leading_noise_is_trimmed_from_speech() -> None:
    cleaner = ArtifactCleaner()

    result = cleaner.clean([_entry("Шум. Мы начали работу")])

    assert [entry.text for entry in result] == ["Мы начали работу"]


def test_noise_word_inside_speech_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entries = [_entry("Мы обсудили шум в записи"), _entry("Проверим сигнал")]

    result = cleaner.clean(entries)

    assert result == entries


def test_single_bare_noise_word_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entries = [_entry("Тишина."), _entry("Звонок."), _entry("Шум."), _entry("Стук.")]

    result = cleaner.clean(entries)

    assert result == entries


def test_consecutive_bare_noise_entries_are_removed() -> None:
    cleaner = ArtifactCleaner()
    entries = [_entry("АПЛОДИСМЕНТЫ") for _ in range(5)]

    assert cleaner.clean(entries) == []


def test_single_noise_entry_between_speech_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entries = [_entry("Первый тезис."), _entry("Тишина."), _entry("Второй тезис.")]

    result = cleaner.clean(entries)

    assert [entry.text for entry in result] == [
        "Первый тезис.",
        "Тишина.",
        "Второй тезис.",
    ]


def test_bare_noise_cleaning_is_idempotent() -> None:
    cleaner = ArtifactCleaner()
    samples = [
        "112 без изменений. АИ тоже. АПЛОДИСМЕНТЫ",
        "Спасибо. СМЕХ СМЕХ",
        "АПЛОДИСМЕНТЫ " * 10,
        "Мы обсудили шум в записи",
    ]

    for text in samples:
        cleaned_once = cleaner.clean([_entry(text)])
        cleaned_twice = cleaner.clean(cleaned_once)
        assert [entry.text for entry in cleaned_once] == [entry.text for entry in cleaned_twice]


def test_bare_noise_entry_with_different_stems_is_removed_when_repeated() -> None:
    cleaner = ArtifactCleaner()

    # Разные шумовые основы, но серия из двух слов — тоже артефакт.
    assert cleaner.clean([_entry("СМЕХ АПЛОДИСМЕНТЫ")]) == []


# --- шаблонные галлюцинации --------------------------------------------------


def test_template_hallucinations_are_removed() -> None:
    cleaner = ArtifactCleaner()
    entries = [
        _entry("Продолжение следует"),
        _entry("Продолжение в следующей части"),
        _entry("Добро пожаловать в Казахстан!"),
        _entry("Спасибо за просмотр"),
        _entry("Редактор субтитров А. Иванов"),
        _entry("Субтитры подготовил Иван"),
    ]

    assert cleaner.clean(entries) == []


def test_hallucination_matching_is_case_and_punctuation_insensitive() -> None:
    cleaner = ArtifactCleaner()

    assert cleaner.clean([_entry("ПРОДОЛЖЕНИЕ, СЛЕДУЕТ...")]) == []


def test_real_phrase_mentioning_subtitles_not_at_start_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("Обсудили, как делаются субтитры")

    assert cleaner.clean([entry]) == [entry]


# --- эвристика «мало текста на длинном интервале» ----------------------------


def test_extreme_sparse_long_entry_is_removed() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("Спасибо.", start=0.0, end=30.0)

    assert cleaner.clean([entry]) == []


def test_sparse_long_entry_with_enough_chars_is_kept() -> None:
    cleaner = ArtifactCleaner()
    # 26 с, 3 слова, но 16 символов — не «совсем пусто», сохраняем.
    entry = _entry("Пока еще ошибки.", start=300.0, end=326.0)

    assert cleaner.clean([entry]) == [entry]


def test_sparse_short_interval_is_kept() -> None:
    cleaner = ArtifactCleaner()
    entry = _entry("Спасибо.", start=0.0, end=1.0)

    assert cleaner.clean([entry]) == [entry]


def test_sparse_long_entry_is_kept_when_dropping_disabled() -> None:
    cleaner = ArtifactCleaner(drop_sparse_long=False)
    entry = _entry("Спасибо.", start=0.0, end=30.0)

    assert cleaner.clean([entry]) == [entry]
