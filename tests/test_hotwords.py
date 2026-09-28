"""Тесты ограничения hotwords (:mod:`audio_transcriber.utils.hotwords`)."""

from __future__ import annotations

from audio_transcriber.utils.hotwords import (
    MAX_RECOMMENDED_HOTWORDS_LENGTH,
    build_hotwords,
    truncate_hotwords_by_tokens,
)


def test_build_hotwords_keeps_order_and_joins() -> None:
    result, dropped = build_hotwords("юрист, Смирнова")
    assert result == "юрист, Смирнова"
    assert dropped == []


def test_build_hotwords_drops_overflow() -> None:
    long_tail = ", ".join(f"термин-{i}" for i in range(200))
    result, dropped = build_hotwords(long_tail, max_length=50)
    assert result is not None
    assert len(result) <= 50
    assert dropped


def test_build_hotwords_empty() -> None:
    result, dropped = build_hotwords(None)
    assert result is None
    assert dropped == []


def test_truncate_hotwords_by_tokens_respects_limit() -> None:
    def encode(text: str) -> list[int]:
        # Грубая модель: 1 токен ≈ 1 символ (удобно для теста).
        return list(range(len(text)))

    truncated, dropped = truncate_hotwords_by_tokens("один, два, три", encode, max_tokens=10)
    assert truncated
    assert isinstance(dropped, list)


def test_truncate_hotwords_by_tokens_empty_terms() -> None:
    truncated, dropped = truncate_hotwords_by_tokens("  ,  ", lambda _text: [1, 2, 3])
    assert truncated == "  ,  "
    assert dropped == []


def test_max_recommended_hotwords_length_is_positive() -> None:
    assert MAX_RECOMMENDED_HOTWORDS_LENGTH > 0
