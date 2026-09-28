"""Общие текстовые хелперы для правки терминов и опечаток ASR.

Здесь собраны примитивы, которые нужны сразу нескольким подсистемам
(морфологический корректор и глоссарий): выделение слов, перенос регистра,
метрики сходства строк. Единое место исключает расхождение поведения между
ними.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Выделяет «слова» без цифр и знаков препинания (Unicode-буквы).
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def match_case(original: str, replacement: str, *, acronym: bool = False) -> str:
    """Сохраняет регистр исходного слова у замены.

    Для канонических аббревиатур (``acronym=True``) замена всегда пишется
    ПРОПИСНЫМИ, если исходное слово начиналось с заглавной буквы: падежные
    формы вроде «АРМе» или «СУБДе» — это всё та же аббревиатура, а не
    отдельное слово, поэтому «Арм»/«Субд» были бы ошибкой. Полностью
    строчные формы («арме») сохраняют нижний регистр, как и обычные слова.
    """
    if acronym and original[:1].isupper():
        return replacement.upper()
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def ratio(left: str, right: str) -> float:
    """Доля сходства двух строк без учёта регистра (0..1).

    Основано на :class:`difflib.SequenceMatcher` и не зависит от порядка
    общих фрагментов.
    """
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def levenshtein(left: str, right: str) -> int:
    """Расстояние Левенштейна между двумя строками."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def levenshtein_ratio(left: str, right: str) -> float:
    """Сходство двух строк по расстоянию Левенштейна (0..1).

    Регистр не меняется: вызывающий сам решает, нормализовать ли строки
    (в глоссарии обе строки уже приведены к нижнему регистру).
    """
    if not left and not right:
        return 1.0
    max_len = max(len(left), len(right))
    return 1.0 - levenshtein(left, right) / max_len
