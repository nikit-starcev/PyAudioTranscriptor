"""Строки таблицы результатов TUI и их фильтрация поиском."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class ResultRow(NamedTuple):
    """Строка таблицы результатов: время, говорящий, метки, текст реплики."""

    time: str
    speaker: str
    marks: str
    text: str


def _search_terms(query: str) -> list[str]:
    """Разбивает запрос поиска на нормализованные (casefold) слова."""
    return [term.casefold() for term in query.split() if term]


def filter_result_rows(rows: Sequence[ResultRow], query: str) -> list[ResultRow]:
    """Фильтрует строки стенограммы по запросу без учёта регистра.

    Запрос разбивается по пробелам, и строка попадает в выборку, только если
    **все** слова запроса встречаются как подстроки в тексте реплики или в
    имени говорящего (логика «И»: ``"иван привет"`` найдёт реплики Ивана, в
    которых есть «привет»). Пустой запрос возвращает все строки.
    """
    terms = _search_terms(query)
    if not terms:
        return list(rows)

    matching: list[ResultRow] = []
    for row in rows:
        haystack = f"{row.speaker}\n{row.text}".casefold()
        if all(term in haystack for term in terms):
            matching.append(row)
    return matching
