"""Тесты сборки контекста/промпта и разбора цитат для чата по стенограмме (#54/#96)."""

from __future__ import annotations

from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm.chat import (
    CHAT_SYSTEM_PROMPT,
    build_chat_messages,
    build_transcript_context,
    format_timecode,
    parse_citations,
)


def _entries() -> list[TranscriptEntry]:
    ivan = Speaker(id="SPEAKER_00", display_name="Спикер 1 — Иван")
    maria = Speaker(id="SPEAKER_01", display_name="Спикер 2 — Мария")
    return [
        TranscriptEntry(start=0.0, end=4.0, text="Привет, начнём встречу", speaker=ivan),
        TranscriptEntry(start=65.0, end=70.0, text="Срок — до пятницы", speaker=maria),
        TranscriptEntry(start=3661.0, end=3670.0, text="Хорошо, зафиксировали", speaker=ivan),
    ]


def test_format_timecode_variants() -> None:
    assert format_timecode(0) == "0:00"
    assert format_timecode(65) == "1:05"
    assert format_timecode(3661) == "1:01:01"
    assert format_timecode(-5) == "0:00"


def test_build_transcript_context_numbers_and_labels() -> None:
    context = build_transcript_context(_entries())
    lines = context.splitlines()
    assert lines[0] == "[0] [0:00] Иван: Привет, начнём встречу"
    assert lines[1] == "[1] [1:05] Мария: Срок — до пятницы"
    assert lines[2] == "[2] [1:01:01] Иван: Хорошо, зафиксировали"


def test_build_transcript_context_skips_empty_and_truncates() -> None:
    empty = TranscriptEntry(start=0.0, end=1.0, text="   ", speaker=None)
    entries = [*_entries(), empty]
    tiny = build_transcript_context(entries, max_chars=40)
    # Первая реплика влезает, остальные опущены с явной пометкой.
    assert tiny.splitlines()[0].startswith("[0]")
    assert any("опущено" in line for line in tiny.splitlines())
    assert "[3]" not in build_transcript_context(entries)


def test_empty_transcript_context() -> None:
    assert build_transcript_context([]) == "(стенограмма пуста)"


def test_parse_citations_dedup_and_range() -> None:
    entries = _entries()
    citations = parse_citations("Решили [1], а срок [1, 2] и мусор [99].", entries)
    assert [item.index for item in citations] == [1, 2]
    assert citations[0].speaker == "Мария"
    assert citations[0].start == 65.0
    assert citations[1].text == "Хорошо, зафиксировали"


def test_parse_citations_none() -> None:
    assert parse_citations("Не увидел ссылок", _entries()) == []


def test_build_chat_messages_structure_and_history_trim() -> None:
    history = [
        {"role": "user", "content": f"вопрос {index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"ответ {index}"}
        for index in range(20)
    ]
    messages = build_chat_messages(
        _entries(), None, history, "о чём договорились?", max_history=4
    )
    assert messages[0]["role"] == "system"
    assert CHAT_SYSTEM_PROMPT in messages[0]["content"]
    assert "[0] [0:00]" in messages[0]["content"]
    # system + 4 последних истории + вопрос
    assert len(messages) == 6
    assert messages[1]["content"] == "вопрос 16"
    assert messages[-1] == {"role": "user", "content": "о чём договорились?"}


def test_build_chat_messages_drops_invalid_history() -> None:
    history = [
        {"role": "system", "content": "нельзя"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "нормально"},
    ]
    messages = build_chat_messages(_entries(), None, history, "вопрос")
    roles = [item["role"] for item in messages]
    assert roles == ["system", "user", "user"]
    assert messages[1]["content"] == "нормально"
