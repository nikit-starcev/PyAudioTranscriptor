"""Тесты редакторских правок (#51): правила частых ошибок и орфография.

Проверяем чистую логику без веб-слоя: сбор предложений, приоритет при
пересечении, выборочное применение и морфологическую подсказку.
"""

from __future__ import annotations

from audio_transcriber.correction.editorial import (
    KIND_COMMON,
    KIND_SPELLING,
    apply_suggestions,
    build_suggestions,
    suggest_common,
)
from audio_transcriber.correction.morph_corrector import MorphTextCorrector


def _apply(text: str, **kwargs) -> tuple[str, list]:
    suggestions = build_suggestions(text, entry_index=0, corrector=MorphTextCorrector(), **kwargs)
    return apply_suggestions(text, suggestions)


def test_common_rules_fix_spaces_and_punctuation() -> None:
    corrected, applied = _apply("привет ,мир  как дела", check_spelling=False)

    assert corrected == "привет, мир как дела"
    assert applied
    assert all(item.kind == KIND_COMMON for item in applied)


def test_common_rules_do_not_touch_correct_text() -> None:
    assert suggest_common("Привет, мир. Всё хорошо!", entry_index=0) == []


def test_common_rules_fix_dash_and_percent() -> None:
    corrected, _ = _apply("слово - слово и 50 % готово", check_spelling=False)

    assert corrected == "слово — слово и 50% готово"


def test_common_rules_strip_edges_and_collapse_spaces() -> None:
    corrected, _ = _apply("  двойной   пробел  ", check_spelling=False)

    assert corrected == "двойной пробел"


def test_sentence_space_rule_is_position_based() -> None:
    suggestions = suggest_common("Привет.Пока", entry_index=0)

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert (suggestion.before, suggestion.after) == (".", ". ")
    assert suggestion.id == f"0:{suggestion.start}:{suggestion.end}:{KIND_COMMON}"


def test_apply_suggestions_respects_selection() -> None:
    text = "привет ,мир  как дела"
    suggestions = suggest_common(text, entry_index=0)
    # Оставляем только правку двойного пробела.
    chosen = [item for item in suggestions if item.reason == "двойной пробел"]

    corrected, applied = apply_suggestions(text, chosen)

    assert corrected == "привет ,мир как дела"
    assert [item.reason for item in applied] == ["двойной пробел"]


def test_apply_suggestions_skips_overlaps() -> None:
    text = "а ,б"
    suggestions = suggest_common(text, entry_index=0)
    corrected, _ = apply_suggestions(text, suggestions)

    assert corrected == "а, б"


def test_spelling_suggestion_from_morph_corrector() -> None:
    corrected, applied = _apply("Конфиденциальнасть мы не нашли.", fix_common=False)

    assert corrected == "Конфиденциальность мы не нашли."
    assert len(applied) == 1
    assert applied[0].kind == KIND_SPELLING
    assert applied[0].before == "Конфиденциальнасть"


def test_spelling_can_be_disabled() -> None:
    suggestions = build_suggestions(
        "Конфиденциальнасть мы не нашли.",
        entry_index=0,
        check_spelling=False,
        corrector=None,
    )

    assert suggestions == []
