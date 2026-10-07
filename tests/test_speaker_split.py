"""Тесты авторазделения реплики по смене говорящего (#93)."""

from __future__ import annotations

from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    WordTimestamp,
)
from audio_transcriber.merging.split import SpeakerChangeSplitter

_SPK0 = Speaker(id="SPEAKER_00", display_name="Спикер 1")
_SPK1 = Speaker(id="SPEAKER_01", display_name="Спикер 2")


def _word(text: str, start: float, end: float) -> WordTimestamp:
    return WordTimestamp(text=text, start=start, end=end, probability=0.9)


def _entry(words: list[WordTimestamp], **kwargs: object) -> TranscriptEntry:
    start = words[0].start
    end = max(word.end for word in words)
    text = " ".join(word.text for word in words)
    return TranscriptEntry(start=start, end=end, text=text, words=words, **kwargs)  # type: ignore[arg-type]


def test_splits_entry_when_speaker_changes_across_words() -> None:
    entry = _entry(
        [
            _word("Алексей,", 0.0, 0.5),
            _word("ты", 0.5, 0.7),
            _word("с", 0.7, 0.8),
            _word("нами?", 0.8, 1.5),
            _word("Да,", 2.0, 2.1),
            _word("я", 2.1, 2.2),
            _word("здесь.", 2.2, 3.1),
            _word("Продолжаю", 4.0, 5.0),
        ],
        speaker=_SPK0,
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.5, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=3.1, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=4.0, end=5.0, speaker_id="SPEAKER_00"),
    ]
    speakers = [_SPK0]

    result, result_speakers = SpeakerChangeSplitter().split(
        [entry], speaker_segments, speakers
    )

    assert len(result) == 3
    assert [part.speaker.id for part in result] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert [part.text for part in result] == [
        "Алексей, ты с нами?",
        "Да, я здесь.",
        "Продолжаю",
    ]
    assert result[1].start == 2.0 and result[1].end == 3.1
    assert [word.text for word in result[1].words] == ["Да,", "я", "здесь."]
    # Появившийся участник попал в список говорящих.
    assert {speaker.id for speaker in result_speakers} == {"SPEAKER_00", "SPEAKER_01"}


def test_short_insertion_is_split_not_swallowed() -> None:
    """Короткая (~1 с) вставка другого говорящего — отдельная реплика."""
    entry = _entry(
        [
            _word("Вопрос", 0.0, 0.8),
            _word("Да,", 1.0, 1.0),
            _word("я", 1.0, 1.0),
            _word("здесь.", 1.0, 1.9),
            _word("Продолжение", 2.0, 4.0),
        ],
        speaker=_SPK0,
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=0.8, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=1.9, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=2.0, end=4.0, speaker_id="SPEAKER_00"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert [part.text for part in result] == ["Вопрос", "Да, я здесь.", "Продолжение"]
    assert result[1].speaker is not None
    assert result[1].speaker.id == "SPEAKER_01"


def test_tiny_flicker_is_not_split() -> None:
    """Шумовое колебание разметки (< порога) не дробит реплику."""
    entry = _entry(
        [
            _word("раз", 0.0, 1.0),
            _word("два", 1.0, 1.1),
            _word("три", 1.1, 2.0),
        ],
        speaker=_SPK0,
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=1.1, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=1.1, end=2.0, speaker_id="SPEAKER_00"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert len(result) == 1
    assert result[0] is entry


def test_single_speaker_entry_untouched() -> None:
    entry = _entry(
        [_word("раз", 0.0, 1.0), _word("два", 1.0, 2.0)],
        speaker=_SPK0,
    )
    speaker_segments = [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert result == [entry]


def test_edited_entry_is_not_split() -> None:
    """Ручная правка текста (#26) сохраняется — реплика не делится."""
    entry = _entry(
        [_word("раз", 0.0, 1.0), _word("два", 1.0, 2.0)],
        speaker=_SPK0,
        edited=True,
        original_text="раз два",
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert result == [entry]


def test_no_words_entry_untouched() -> None:
    entry = TranscriptEntry(start=0.0, end=2.0, text="текст", speaker=_SPK0)
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert result == [entry]


def test_no_speaker_segments_returns_input() -> None:
    entry = _entry([_word("раз", 0.0, 1.0), _word("два", 1.0, 2.0)], speaker=_SPK0)

    result, speakers = SpeakerChangeSplitter().split([entry], [], [_SPK0])

    assert result == [entry]
    assert speakers == [_SPK0]


def test_split_keeps_words_and_recomputes_confidence() -> None:
    entry = _entry(
        [_word("первый", 0.0, 2.0), _word("второй", 3.0, 5.0)],
        speaker=_SPK0,
        speaker_confidence=0.5,
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=3.0, end=5.0, speaker_id="SPEAKER_01"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert len(result) == 2
    # Каждая часть полностью покрыта своим говорящим → уверенность 1.0.
    assert [part.speaker_confidence for part in result] == [1.0, 1.0]
    assert [word.text for word in result[0].words] == ["первый"]
    assert [word.text for word in result[1].words] == ["второй"]


def test_words_not_matching_text_are_not_split() -> None:
    """Если текст не восстанавливается из слов, реплику не трогаем."""
    entry = TranscriptEntry(
        start=0.0,
        end=2.0,
        text="совершенно другой текст",
        speaker=_SPK0,
        words=[_word("раз", 0.0, 1.0), _word("два", 1.0, 2.0)],
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert result == [entry]


def test_non_monotonic_words_are_not_split() -> None:
    """Неотсортированные пословные метки (стык чанков) не режутся."""
    entry = _entry(
        [
            _word("первый", 3.0, 4.0),
            _word("второй", 0.0, 1.0),
        ],
        speaker=_SPK0,
    )
    speaker_segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=3.0, end=4.0, speaker_id="SPEAKER_00"),
    ]

    result, _ = SpeakerChangeSplitter().split([entry], speaker_segments, [_SPK0])

    assert result == [entry]
