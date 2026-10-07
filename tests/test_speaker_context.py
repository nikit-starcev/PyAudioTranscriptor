"""Тесты добора говорящего коротким репликам в «дырках» разметки (#93)."""

from __future__ import annotations

from audio_transcriber.domain.models import Speaker, TranscriptEntry, WordTimestamp
from audio_transcriber.merging.context import assign_context_speakers

_SPK0 = Speaker(id="SPEAKER_00", display_name="Спикер 1")
_SPK1 = Speaker(id="SPEAKER_01", display_name="Спикер 2")


def _entry(
    start: float,
    end: float,
    text: str,
    speaker: Speaker | None,
    words: int = 1,
) -> TranscriptEntry:
    word_list = [
        WordTimestamp(text=f"w{i}", start=start + i * 0.1, end=start + i * 0.1 + 0.05)
        for i in range(words)
    ]
    return TranscriptEntry(start=start, end=end, text=text, speaker=speaker, words=word_list)


def test_assigns_speaker_when_both_sides_match() -> None:
    entries = [
        _entry(0.0, 1.0, "слева", _SPK0),
        _entry(5.0, 6.0, "Спасибо.", None),
        _entry(10.0, 11.0, "справа", _SPK0),
    ]

    result = assign_context_speakers(entries)

    assert result[1].speaker is not None
    assert result[1].speaker.id == "SPEAKER_00"


def test_assigns_nearest_side_when_sides_differ() -> None:
    """Соседи разные, но правый ближе — берём его говорящего."""
    entries = [
        _entry(0.0, 1.0, "слева", _SPK0),
        _entry(5.0, 6.0, "непонятно", None),
        _entry(6.5, 7.5, "справа", _SPK1),
    ]

    result = assign_context_speakers(entries)

    assert result[1].speaker is not None
    assert result[1].speaker.id == "SPEAKER_01"


def test_tie_prefers_left_neighbour() -> None:
    entries = [
        _entry(0.0, 2.0, "слева", _SPK0),
        _entry(4.0, 5.0, "непонятно", None),
        _entry(7.0, 9.0, "справа", _SPK1),
    ]

    result = assign_context_speakers(entries)

    assert result[1].speaker is not None
    assert result[1].speaker.id == "SPEAKER_00"


def test_leaves_long_entry_without_speaker() -> None:
    entries = [
        _entry(0.0, 1.0, "слева", _SPK0),
        _entry(5.0, 9.0, "длинная реплика без говорящего", None, words=6),
        _entry(10.0, 11.0, "справа", _SPK0),
    ]

    result = assign_context_speakers(entries)

    assert result[1].speaker is None


def test_leaves_when_neighbours_too_far() -> None:
    entries = [
        _entry(0.0, 1.0, "слева", _SPK0),
        _entry(100.0, 101.0, "далеко", None),
        _entry(200.0, 201.0, "справа", _SPK1),
    ]

    result = assign_context_speakers(entries, max_gap=15.0)

    assert result[1].speaker is None


def test_assigns_when_only_one_side_within_gap() -> None:
    entries = [
        _entry(0.0, 1.0, "шум", None),
        _entry(5.0, 6.0, "речь", _SPK0),
    ]

    result = assign_context_speakers(entries)

    assert result[0].speaker is not None
    assert result[0].speaker.id == "SPEAKER_00"


def test_entries_with_speaker_unchanged() -> None:
    entries = [_entry(0.0, 1.0, "раз", _SPK0), _entry(1.0, 2.0, "два", _SPK1)]

    result = assign_context_speakers(entries)

    assert result == entries
