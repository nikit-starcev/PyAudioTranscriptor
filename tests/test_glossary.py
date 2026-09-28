"""Тесты детерминированного матчера глоссария (без LLM)."""

from __future__ import annotations

import os
from pathlib import Path

from audio_transcriber.llm.glossary import (
    Glossary,
    collect_term_suggestions,
    load_glossary,
    normalize_glossary_paths,
    split_glossary_paths,
    write_suggested_terms,
)


def test_glossary_loads_terms_skipping_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "glossary.txt"
    path.write_text("# комментарий\n\nОИБ\nКИСУСС\n\n  транскрибер  \n", encoding="utf-8")

    glossary = Glossary(path=path)

    assert "ОИБ" in glossary
    assert "КИСУСС" in glossary
    assert "транскрибер" in glossary
    assert len(glossary) == 3


def test_correct_text_leaves_exact_term_untouched() -> None:
    glossary = Glossary(terms=["ОИБ"])
    text, replacements = glossary.correct_text("Этим занимается ОИБ.")

    assert text == "Этим занимается ОИБ."
    assert replacements == []


def test_correct_text_fixes_asr_typo_of_acronym() -> None:
    glossary = Glossary(terms=["ОИБ"])
    text, replacements = glossary.correct_text("Этим занимается АИБ.")

    assert text == "Этим занимается ОИБ."
    assert replacements == [("АИБ", "ОИБ")]


def test_correct_text_preserves_case_of_replacement() -> None:
    glossary = Glossary(terms=["ОИБ"])
    text, replacements = glossary.correct_text("этим занимается аиб.")

    assert text == "этим занимается оиб."
    assert replacements == [("аиб", "оиб")]


def test_correct_text_does_not_touch_declined_form() -> None:
    glossary = Glossary(terms=["транскрибер"])
    text, replacements = glossary.correct_text("Мы настроили транскрибера.")

    assert text == "Мы настроили транскрибера."
    assert replacements == []


def test_correct_text_leaves_unrelated_words_untouched() -> None:
    glossary = Glossary(terms=["ОИБ", "транскрибер"])
    text, replacements = glossary.correct_text("Собрание по проекту прошло штатно.")

    assert text == "Собрание по проекту прошло штатно."
    assert replacements == []


def test_correct_text_fixes_longer_term_typo() -> None:
    glossary = Glossary(terms=["КИСУСС"])
    text, replacements = glossary.correct_text("В системе КИСУС сбой.")

    assert text == "В системе КИСУСС сбой."
    assert replacements == [("КИСУС", "КИСУСС")]


def test_suggest_new_terms_finds_acronym_and_unknown_noun() -> None:
    glossary = Glossary(terms=["ОИБ"])

    suggestions = glossary.suggest_new_terms("Обсуждали ХТТП и платформу Осыка. ОИБ уже в списке.")

    assert "ХТТП" in suggestions
    assert "Осыка" in suggestions
    assert "ОИБ" not in suggestions


def test_suggest_new_terms_skips_known_common_words() -> None:
    glossary = Glossary()

    suggestions = glossary.suggest_new_terms("Безопасность и Система важны.")

    assert "Безопасность" not in suggestions
    assert "Система" not in suggestions


# ---------------------------------------------------------------------------
# Регистр аббревиатур: падежные формы не должны превращаться в «Арм»/«Субд».
# ---------------------------------------------------------------------------


def test_correct_text_keeps_acronym_uppercase_in_declined_form_arm() -> None:
    glossary = Glossary(terms=["АРМ"])
    text, replacements = glossary.correct_text("данные в АРМе оператора")

    assert text == "данные в АРМ оператора"
    assert replacements == [("АРМе", "АРМ")]


def test_correct_text_keeps_acronym_uppercase_in_declined_form_subd() -> None:
    glossary = Glossary(terms=["СУБД"])
    text, replacements = glossary.correct_text("данные в СУБДе")

    assert text == "данные в СУБД"
    assert replacements == [("СУБДе", "СУБД")]


def test_correct_text_all_lower_acronym_stays_lower() -> None:
    glossary = Glossary(terms=["АРМ"])
    text, replacements = glossary.correct_text("данные в арме оператора")

    assert text == "данные в арм оператора"
    assert replacements == [("арме", "арм")]


# ---------------------------------------------------------------------------
# Авто-предложение новых терминов: файл предложений.
# ---------------------------------------------------------------------------


def test_suggested_path_is_next_to_glossary(tmp_path: Path) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("ОИБ\n", encoding="utf-8")

    glossary = Glossary(path=glossary_path)

    assert glossary.suggested_path == tmp_path / "glossary.suggested.txt"


def test_collect_term_suggestions_counts_and_sorts(tmp_path: Path) -> None:
    glossary = Glossary(terms=["ОИБ"])
    text = "ХТТП и ещё ХТТП. Тут АБВ. ОИБ в списке."

    assert collect_term_suggestions(glossary, text) == [("ХТТП", 2), ("АБВ", 1)]


def test_write_suggested_terms_creates_file_with_count_and_source(
    tmp_path: Path,
) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("ОИБ\n", encoding="utf-8")
    glossary = Glossary(path=glossary_path)
    text = "ХТТП и ХТТП, снова ХТТП. ОИБ уже в списке."

    out = write_suggested_terms(glossary, text, source="call_001.txt")

    assert out == glossary_path.with_name("glossary.suggested.txt")
    assert out is not None
    content = out.read_text(encoding="utf-8")
    assert "ХТТП\t3\tcall_001.txt" in content
    assert "ОИБ" not in content


def test_write_suggested_terms_does_not_duplicate_previous(tmp_path: Path) -> None:
    glossary_path = tmp_path / "glossary.txt"
    glossary_path.write_text("ОИБ\n", encoding="utf-8")
    glossary = Glossary(path=glossary_path)
    text = "Снова про ХТТП."

    write_suggested_terms(glossary, text, source="call_001.txt")
    out = write_suggested_terms(glossary, text, source="call_002.txt")

    assert out is not None
    content = out.read_text(encoding="utf-8")
    assert content.count("ХТТП") == 1
    assert "ХТТП\t1\tcall_001.txt" in content
    assert "call_002.txt" not in content


def test_write_suggested_terms_without_glossary_path_returns_none() -> None:
    glossary = Glossary(terms=["ОИБ"])

    assert write_suggested_terms(glossary, "ХТТП", source="call.txt") is None


# ---------------------------------------------------------------------------
# Задача 1: несколько глоссариев в одном пути.
# ---------------------------------------------------------------------------


def test_split_glossary_paths_handles_comma_and_pathsep() -> None:
    assert split_glossary_paths("a.txt,b.txt") == ["a.txt", "b.txt"]
    assert split_glossary_paths(f"a.txt{os.pathsep}b.txt") == ["a.txt", "b.txt"]
    assert split_glossary_paths("  ") == []


def test_normalize_glossary_paths_accepts_single_path(tmp_path: Path) -> None:
    single = tmp_path / "glossary.txt"

    assert normalize_glossary_paths(single) == [single]
    assert normalize_glossary_paths(str(single)) == [single]
    assert normalize_glossary_paths(None) == []


def test_load_glossary_merges_multiple_paths(tmp_path: Path) -> None:
    first = tmp_path / "tz.txt"
    second = tmp_path / "user.txt"
    first.write_text("ОИБ\n", encoding="utf-8")
    second.write_text("АИП\n", encoding="utf-8")

    glossary = load_glossary([first, second])

    assert glossary is not None
    assert "ОИБ" in glossary
    assert "АИП" in glossary


def test_load_glossary_accepts_comma_separated_string(tmp_path: Path) -> None:
    first = tmp_path / "tz.txt"
    second = tmp_path / "user.txt"
    first.write_text("ОИБ\n", encoding="utf-8")
    second.write_text("АИП\n", encoding="utf-8")

    glossary = load_glossary(f"{first},{second}")

    assert glossary is not None
    assert "ОИБ" in glossary
    assert "АИП" in glossary


def test_load_glossary_single_path_still_works(tmp_path: Path) -> None:
    path = tmp_path / "glossary.txt"
    path.write_text("ОИБ\n", encoding="utf-8")

    glossary = load_glossary(path)

    assert glossary is not None
    assert "ОИБ" in glossary
    assert glossary.source_path == path


# ---------------------------------------------------------------------------
# Задача 3: защита от коллизий коротких аббревиатур.
# ---------------------------------------------------------------------------


def test_correct_text_keeps_exact_acronym_untouched() -> None:
    # «ОИБ» есть в глоссарии — не тянется к похожему «ОИВ».
    glossary = Glossary(terms=["ОИБ", "ОИВ", "АИС"])
    text, replacements = glossary.correct_text("вопросы ОИБ обсудили")

    assert text == "вопросы ОИБ обсудили"
    assert replacements == []


def test_correct_text_skips_ambiguous_short_acronym() -> None:
    # «АИБ» на расстоянии 1 и от «ОИБ», и от «АИС» — ничья, слово не трогаем.
    glossary = Glossary(terms=["ОИБ", "АИС"])
    text, replacements = glossary.correct_text("АИБ система")

    assert text == "АИБ система"
    assert replacements == []


def test_correct_text_fixes_unambiguous_short_acronym() -> None:
    glossary = Glossary(terms=["ОИБ"])
    text, replacements = glossary.correct_text("АИБ система")

    assert text == "ОИБ система"
    assert replacements == [("АИБ", "ОИБ")]


def test_correct_text_does_not_corrupt_common_words_into_acronyms() -> None:
    # Обычные слова не должны тянуться к аббревиатурам на одну правку.
    glossary = Glossary(terms=["ЧТЗ", "ППО", "АОН", "ЦОД", "АБД"])
    text, replacements = glossary.correct_text("что про он год ад")

    assert text == "что про он год ад"
    assert replacements == []


def test_correct_text_keeps_long_term_fuzzy_behaviour() -> None:
    # Длинный термин правится по-прежнему (лучшее сходство, без ничьей).
    glossary = Glossary(terms=["КИСУСС"])
    text, replacements = glossary.correct_text("В системе КИСУС сбой.")

    assert text == "В системе КИСУСС сбой."
    assert replacements == [("КИСУС", "КИСУСС")]


# ---------------------------------------------------------------------------
# Задача 4: явные пары «как слышит ASR = канон».
# ---------------------------------------------------------------------------


def test_explicit_pair_overrides_ambiguous_fuzzy_match() -> None:
    # Без пары fuzzy не тронул бы «АИБ» (ничья ОИБ/АИС), а с парой — заменит.
    glossary = Glossary(terms=["АИБ = ОИБ", "АИС"])
    text, replacements = glossary.correct_text("АИБ система")

    assert text == "ОИБ система"
    assert replacements == [("АИБ", "ОИБ")]


def test_explicit_pair_adds_canonical_as_term() -> None:
    glossary = Glossary(terms=["АИБ = ОИБ"])

    assert "ОИБ" in glossary
    text, replacements = glossary.correct_text("АИБ и ОИБ")

    assert text == "ОИБ и ОИБ"
    assert replacements == [("АИБ", "ОИБ")]


def test_explicit_pair_left_side_is_case_insensitive() -> None:
    glossary = Glossary(terms=["АИБ = ОИБ"])
    text, replacements = glossary.correct_text("аиб система")

    assert text == "ОИБ система"
    assert replacements == [("аиб", "ОИБ")]


def test_glossary_file_parses_explicit_pairs(tmp_path: Path) -> None:
    path = tmp_path / "glossary.user.txt"
    path.write_text("# комментарий\nАИБ = ОИБ\nОИБ\n", encoding="utf-8")

    glossary = Glossary(path=path)
    text, replacements = glossary.correct_text("АИБ система")

    assert text == "ОИБ система"
    assert replacements == [("АИБ", "ОИБ")]


def test_malformed_pair_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "glossary.txt"
    path.write_text("АИБ =\n= ОИБ\nОИБ\n", encoding="utf-8")

    glossary = Glossary(path=path)

    assert glossary.terms == ["оиб"]


# ---------------------------------------------------------------------------
# Задача 2: фразовые термины (из нескольких слов).
# ---------------------------------------------------------------------------


def test_correct_text_phrase_only_glossary_keeps_exact_match() -> None:
    # Глоссарий только из фразы: ранний выход не должен «съедать» обработку.
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text("Тема: информационная безопасность проекта.")

    assert text == "Тема: информационная безопасность проекта."
    assert replacements == []


def test_correct_text_phrase_only_glossary_fixes_typo() -> None:
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text("Тема: информационная безопастность проекта.")

    assert text == "Тема: информационная безопасность проекта."
    assert replacements == [("информационная безопастность", "информационная безопасность")]


def test_correct_text_phrase_fixes_typo_in_any_word() -> None:
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text("Обсуждали информационая безопасность проекта.")

    assert text == "Обсуждали информационная безопасность проекта."
    assert replacements == [("информационая безопасность", "информационная безопасность")]


def test_correct_text_phrase_preserves_case() -> None:
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text("Информационная безопастность важна.")

    assert text == "Информационная безопасность важна."
    assert replacements == [("Информационная безопастность", "Информационная безопасность")]


def test_correct_text_phrase_does_not_touch_declined_form() -> None:
    # Склонённая фраза — это форма того же термина, а не опечатка.
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text(
        "О вопросах информационной безопасности речи не шло."
    )

    assert text == "О вопросах информационной безопасности речи не шло."
    assert replacements == []


def test_correct_text_phrase_coexists_with_single_terms() -> None:
    glossary = Glossary(terms=["ОИБ", "информационная безопасность"])

    text, replacements = glossary.correct_text("В системе АИБ и информационная безопастность.")

    assert text == "В системе ОИБ и информационная безопасность."
    assert ("АИБ", "ОИБ") in replacements
    assert (
        "информационная безопастность",
        "информационная безопасность",
    ) in replacements


def test_explicit_phrase_pair_applied_always() -> None:
    glossary = Glossary(terms=["информационная безопастность = информационная безопасность"])

    assert "информационная безопасность" in glossary

    text, replacements = glossary.correct_text("Про информационная безопастность.")

    assert text == "Про информационная безопасность."
    assert replacements == [("информационная безопастность", "информационная безопасность")]


def test_correct_text_phrase_shorter_word_count_not_matched() -> None:
    # Фраза не должна «затягивать» текст с другим числом слов.
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text("Информационная и безопасность.")

    assert text == "Информационная и безопасность."
    assert replacements == []


def test_correct_text_phrase_does_not_swallow_punctuation() -> None:
    # Слова, разделённые знаками препинания, не считаются слитной фразой —
    # иначе замена выкинула бы запятую или оставила лишнюю скобку.
    glossary = Glossary(terms=["информационная безопасность"])

    text, replacements = glossary.correct_text("Информационная, безопасность — важна.")

    assert text == "Информационная, безопасность — важна."
    assert replacements == []
