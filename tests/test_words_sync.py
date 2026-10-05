"""Тесты синхронизации текста и пословных меток при правках (#84)."""

from __future__ import annotations

from audio_transcriber.cleaning import ArtifactCleaner
from audio_transcriber.cleaning.text_normalizer import TextNormalizer
from audio_transcriber.domain.models import TranscriptEntry, WordTimestamp
from audio_transcriber.utils.text import sync_words_to_text


def _words(*texts: str) -> list[WordTimestamp]:
    return [
        WordTimestamp(text=text, start=float(index), end=float(index) + 0.5)
        for index, text in enumerate(texts)
    ]


def test_no_words_stays_empty() -> None:
    assert sync_words_to_text("привет мир", [], "привет") == []


def test_unchanged_text_keeps_words() -> None:
    words = _words("привет", "мир")

    result = sync_words_to_text("привет мир", words, "привет мир")

    assert result == words
    assert result is not words


def test_removed_marker_drops_its_word() -> None:
    words = _words("привет", "[СМЕХ]", "мир")

    result = sync_words_to_text("привет [СМЕХ] мир", words, "привет мир")

    assert [word.text for word in result] == ["привет", "мир"]
    assert all("СМЕХ" not in word.text for word in result)


def test_punctuation_normalization_keeps_words() -> None:
    words = _words("привет!!!")

    result = sync_words_to_text("привет!!!", words, "привет!")

    assert [word.text for word in result] == ["привет!"]
    assert result[0].start == words[0].start


def test_rewritten_word_invalidates_words() -> None:
    words = _words("превет", "мир")

    # Слово заменено (корректор), точное соответствие невозможно → инвалидация.
    assert sync_words_to_text("превет мир", words, "привет мир") == []


def test_token_count_mismatch_invalidates_words() -> None:
    words = _words("привет")

    assert sync_words_to_text("привет мир", words, "привет") == []


def test_artifact_cleaner_syncs_words() -> None:
    entry = TranscriptEntry(
        start=0.0,
        end=3.0,
        text="привет [СМЕХ] мир",
        words=_words("привет", "[СМЕХ]", "мир"),
    )

    result = ArtifactCleaner().clean([entry])

    assert result[0].text == "привет мир"
    assert [word.text for word in result[0].words] == ["привет", "мир"]
    assert all("СМЕХ" not in word.text for word in result[0].words)


def test_text_normalizer_syncs_words() -> None:
    entry = TranscriptEntry(
        start=0.0,
        end=1.0,
        text="привет!!!",
        words=_words("привет!!!"),
    )

    result = TextNormalizer().normalize([entry])

    assert result[0].text == "привет!"
    assert [word.text for word in result[0].words] == ["привет!"]
