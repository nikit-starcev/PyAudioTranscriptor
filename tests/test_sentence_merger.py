"""Тесты склейки подряд идущих реплик одного говорящего (``SentenceMerger``)."""

from __future__ import annotations

import pytest

from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.merging.sentence_merger import SentenceMerger

_SPEAKER_1 = Speaker(id="SPEAKER_00", display_name="Спикер 1")
_SPEAKER_2 = Speaker(id="SPEAKER_01", display_name="Спикер 2")


def _entry(start: float, end: float, text: str, speaker: Speaker | None) -> TranscriptEntry:
    return TranscriptEntry(start=start, end=end, text=text, speaker=speaker)


def test_merges_consecutive_same_speaker() -> None:
    entries = [
        _entry(0.0, 1.0, "привет", _SPEAKER_1),
        _entry(1.1, 2.0, "как дела", _SPEAKER_1),
        _entry(2.1, 3.0, "что нового", _SPEAKER_1),
    ]

    merged = SentenceMerger().merge(entries)

    assert len(merged) == 1
    assert merged[0].text == "привет как дела что нового"
    assert merged[0].start == 0.0
    assert merged[0].end == 3.0
    assert merged[0].speaker is _SPEAKER_1


def test_does_not_merge_different_speakers() -> None:
    entries = [
        _entry(0.0, 1.0, "привет", _SPEAKER_1),
        _entry(1.0, 2.0, "здравствуйте", _SPEAKER_2),
    ]

    merged = SentenceMerger().merge(entries)

    assert [entry.text for entry in merged] == ["привет", "здравствуйте"]
    assert [entry.speaker for entry in merged] == [_SPEAKER_1, _SPEAKER_2]


def test_speaker_none_entries_are_merged_together() -> None:
    entries = [
        _entry(0.0, 1.0, "раз", None),
        _entry(1.0, 2.0, "два", None),
        _entry(2.0, 3.0, "три", None),
    ]

    merged = SentenceMerger().merge(entries)

    assert len(merged) == 1
    assert merged[0].text == "раз два три"
    assert merged[0].speaker is None
    assert merged[0].start == 0.0
    assert merged[0].end == 3.0


def test_large_gap_breaks_none_series() -> None:
    entries = [
        _entry(0.0, 1.0, "раз", None),
        _entry(10.0, 11.0, "два", None),
    ]

    merged = SentenceMerger(max_gap=2.0).merge(entries)

    assert [entry.text for entry in merged] == ["раз", "два"]


def test_none_is_not_attached_to_named_speaker() -> None:
    entries = [
        _entry(0.0, 1.0, "привет", _SPEAKER_1),
        _entry(1.0, 2.0, "шум", None),
        _entry(2.0, 3.0, "продолжаю", _SPEAKER_1),
    ]

    merged = SentenceMerger().merge(entries)

    # None-реплика разрывает серию и не присоединяется ни к кому
    assert [entry.text for entry in merged] == ["привет", "шум", "продолжаю"]


def test_merges_same_speaker_across_natural_pause() -> None:
    # Пауза 3 с внутри речи одного говорящего — естественная, не разрывает
    # реплику (порог по умолчанию 5 с, #113).
    entries = [
        _entry(0.0, 1.0, "первая", _SPEAKER_1),
        _entry(4.0, 5.0, "вторая", _SPEAKER_1),
    ]

    merged = SentenceMerger().merge(entries)

    assert len(merged) == 1
    assert merged[0].text == "первая вторая"


def test_default_gap_breaks_long_silence() -> None:
    entries = [
        _entry(0.0, 1.0, "первая", _SPEAKER_1),
        _entry(20.0, 21.0, "вторая", _SPEAKER_1),
    ]

    merged = SentenceMerger().merge(entries)

    assert [entry.text for entry in merged] == ["первая", "вторая"]


def test_large_gap_breaks_utterance() -> None:
    entries = [
        _entry(0.0, 1.0, "первая", _SPEAKER_1),
        _entry(10.0, 11.0, "вторая", _SPEAKER_1),
    ]

    merged = SentenceMerger(max_gap=2.0).merge(entries)

    assert [entry.text for entry in merged] == ["первая", "вторая"]


def test_gap_at_boundary_still_merges() -> None:
    entries = [
        _entry(0.0, 1.0, "первая", _SPEAKER_1),
        _entry(3.0, 4.0, "вторая", _SPEAKER_1),
    ]

    merged = SentenceMerger(max_gap=2.0).merge(entries)

    assert len(merged) == 1
    assert merged[0].text == "первая вторая"


def test_collapses_repeated_spaces() -> None:
    entries = [
        _entry(0.0, 1.0, "привет  ", _SPEAKER_1),
        _entry(1.0, 2.0, "   как   дела", _SPEAKER_1),
    ]

    merged = SentenceMerger().merge(entries)

    assert merged[0].text == "привет как дела"


def test_empty_list_returns_empty() -> None:
    assert SentenceMerger().merge([]) == []


def test_negative_max_gap_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceMerger(max_gap=-1.0)


_SPEAKER_3 = Speaker(id="SPEAKER_02", display_name="Спикер 3")


def test_merge_unions_extra_speakers_without_duplicates() -> None:
    entries = [
        TranscriptEntry(
            start=0.0,
            end=1.0,
            text="раз",
            speaker=_SPEAKER_1,
            extra_speakers=[_SPEAKER_2],
            overlap=True,
            speaker_confidence=0.9,
        ),
        TranscriptEntry(
            start=1.1,
            end=2.0,
            text="два",
            speaker=_SPEAKER_1,
            extra_speakers=[_SPEAKER_2, _SPEAKER_3],
            overlap=True,
            speaker_confidence=0.7,
        ),
    ]

    merged = SentenceMerger().merge(entries)

    assert len(merged) == 1
    assert [speaker.id for speaker in merged[0].extra_speakers] == [
        "SPEAKER_01",
        "SPEAKER_02",
    ]
    assert merged[0].overlap is True


def test_merge_keeps_worst_speaker_confidence() -> None:
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="раз", speaker=_SPEAKER_1, speaker_confidence=0.9),
        TranscriptEntry(start=1.1, end=2.0, text="два", speaker=_SPEAKER_1, speaker_confidence=0.4),
    ]

    merged = SentenceMerger().merge(entries)

    # Сомнение не теряется при склейке: берётся минимум уверенности.
    assert merged[0].speaker_confidence == 0.4


def test_merge_confidence_none_is_ignored() -> None:
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="раз", speaker=_SPEAKER_1),
        TranscriptEntry(
            start=1.1, end=2.0, text="два", speaker=_SPEAKER_1, speaker_confidence=0.6
        ),
    ]

    merged = SentenceMerger().merge(entries)

    assert merged[0].speaker_confidence == 0.6
