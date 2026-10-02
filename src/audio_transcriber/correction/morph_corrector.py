"""Автоисправление опечаток ASR по морфологии русского языка.

Без пользовательских словарей и без нейросетей: неизвестные словоформы
подбираются к ближайшей известной форме из OpenCorpora (через pymorphy3).
Известные слова и склонения не изменяются.

Правки намеренно консервативны, чтобы не «исправлять» тех-заимствования
(«залогинился» → «залоснился») на похожие русские слова:

* допускается не более одной правки символа (``max_edit_distance``);
* порог сходства высокий (по умолчанию 0.93);
* слова с латиницей не трогаются вовсе.

Пороги длины и сходства задаются через ``AppConfig`` / ``config.env``
(см. ``CORRECTION_MIN_WORD_LENGTH``, ``CORRECTION_MIN_SIMILARITY``,
``CORRECTION_MAX_CANDIDATES``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any

from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MAX_EDIT_DISTANCE,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.models import TranscriptEntry
from audio_transcriber.utils.text import WORD_PATTERN, levenshtein, match_case, ratio

logger = logging.getLogger(__name__)

# Латиница в слове: тех-заимствование (login, reloadConfig, API_KEY) — не трогаем.
_LATIN_RE = re.compile(r"[A-Za-z]")


def contains_latin(word: str) -> bool:
    """Есть ли в слове латинские буквы (a–z, A–Z)."""
    return _LATIN_RE.search(word) is not None


def is_spelling_like_form(
    original: str,
    corrected: str,
    *,
    min_word_length: int = DEFAULT_CORRECTION_MIN_WORD_LENGTH,
    min_similarity: float = DEFAULT_CORRECTION_MIN_SIMILARITY,
    max_edit_distance: int = DEFAULT_CORRECTION_MAX_EDIT_DISTANCE,
) -> bool:
    """Похожа ли замена на исправление опечатки, а не на другое слово.

    Отклоняет слова с латиницей (тех-заимствования) и замены, требующие более
    ``max_edit_distance`` правок символов.
    """
    if contains_latin(original) or contains_latin(corrected):
        return False

    if len(original) < min_word_length or len(corrected) < min_word_length:
        return False

    if levenshtein(original.casefold(), corrected.casefold()) > max_edit_distance:
        return False

    max_len_delta = max(2, len(original) // 5)
    if abs(len(original) - len(corrected)) > max_len_delta:
        return False

    return ratio(original, corrected) >= min_similarity


class MorphTextCorrector:
    """Исправляет только неизвестные словоформы ближайшим словом русского языка.

    Реализует протокол ``TextCorrector``.
    """

    def __init__(
        self,
        *,
        min_word_length: int = DEFAULT_CORRECTION_MIN_WORD_LENGTH,
        min_similarity: float = DEFAULT_CORRECTION_MIN_SIMILARITY,
        max_candidates: int = DEFAULT_CORRECTION_MAX_CANDIDATES,
        max_edit_distance: int = DEFAULT_CORRECTION_MAX_EDIT_DISTANCE,
    ) -> None:
        self._min_word_length = min_word_length
        self._min_similarity = min_similarity
        self._max_candidates = max_candidates
        self._max_edit_distance = max_edit_distance

        self._morph: Any = None
        self._words_dawg: Any = None
        self._suggestion_cache: dict[str, str | None] = {}
        self._known_cache: dict[str, bool] = {}

    def _ensure_loaded(self) -> None:
        if self._morph is not None:
            return

        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()
        self._words_dawg = self._morph.dictionary.words

    def _is_known_russian_word(self, word: str) -> bool:
        cached = self._known_cache.get(word)
        if cached is not None:
            return cached

        assert self._morph is not None
        known = any(parse.is_known for parse in self._morph.parse(word))
        self._known_cache[word] = known
        return known

    def _is_spelling_like(self, original: str, corrected: str) -> bool:
        return is_spelling_like_form(
            original,
            corrected,
            min_word_length=self._min_word_length,
            min_similarity=self._min_similarity,
            max_edit_distance=self._max_edit_distance,
        )

    def _best_known_form(self, word: str) -> str | None:
        """Ищет в OpenCorpora ближайшую известную словоформу к опечатке."""
        assert self._words_dawg is not None

        folded = word.casefold()
        if folded in self._suggestion_cache:
            return self._suggestion_cache[folded]

        best: str | None = None
        best_score = 0.0

        # Сначала более длинный префикс (уже набор кандидатов), затем короче.
        max_pref = min(8, len(folded) - 1)
        for pref_len in range(max_pref, 3, -1):
            prefix = folded[:pref_len]
            candidates = self._words_dawg.keys(prefix)
            if not candidates:
                continue

            # На коротком префиксе keys() может вернуть огромный список —
            # ограничиваем перебор и при необходимости пробуем другой префикс.
            for counted, candidate in enumerate(candidates, start=1):
                if counted > self._max_candidates:
                    break

                if not self._is_spelling_like(folded, candidate):
                    continue

                score = ratio(folded, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score

            if best is not None and best_score >= self._min_similarity:
                break

        result = best if best_score >= self._min_similarity else None
        self._suggestion_cache[folded] = result
        return result

    def _suggestions(self, text: str) -> list[tuple[int, int, str, str]]:
        """Находит замены-опечатки в тексте с позициями, не меняя сам текст.

        Возвращает список ``(start, end, before, after)`` в порядке появления.
        Общая основа и для :meth:`_correct_text`, и для :meth:`suggest_spans`.
        """
        found: list[tuple[int, int, str, str]] = []
        for match in WORD_PATTERN.finditer(text):
            original_word = match.group(0)

            if len(original_word) < self._min_word_length:
                continue
            # Тех-заимствования с латиницей (login, reloadConfig) не трогаем.
            if contains_latin(original_word):
                continue
            if self._is_known_russian_word(original_word):
                continue

            candidate = self._best_known_form(original_word)
            if candidate is None:
                continue
            if candidate.casefold() == original_word.casefold():
                continue

            replacement = match_case(original_word, candidate)
            found.append((match.start(), match.end(), original_word, replacement))
        return found

    def suggest_spans(self, text: str) -> list[tuple[int, int, str, str]]:
        """Предложения правок с позициями ``(start, end, before, after)``.

        Публичная точка входа для редакторской проверки (#51): в отличие от
        :meth:`correct`, ничего не применяет и отдаёт позиции для «принять/
        отклонить» на стороне UI.
        """
        self._ensure_loaded()
        return self._suggestions(text)

    def _correct_text(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Правит опечатки в одном тексте, возвращая его и список замен.

        Отдельный метод (а не замыкание в цикле) — так ``applied`` не является
        переменной цикла, и поведение не зависит от отложенного вызова.
        """
        spans = self._suggestions(text)
        if not spans:
            return text, []

        applied: list[tuple[str, str]] = []
        parts: list[str] = []
        cursor = 0
        for start, end, original_word, replacement in spans:
            parts.append(text[cursor:start])
            parts.append(replacement)
            cursor = end
            applied.append((original_word, replacement))
        parts.append(text[cursor:])
        return "".join(parts), applied

    def correct(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        self._ensure_loaded()

        corrected_entries: list[TranscriptEntry] = []
        total_replacements = 0

        for entry in entries:
            if not entry.text.strip():
                corrected_entries.append(entry)
                continue

            new_text, applied = self._correct_text(entry.text)
            total_replacements += len(applied)

            for old, new in applied:
                logger.info("Автоисправление: «%s» → «%s»", old, new)

            if applied:
                corrected_entries.append(replace(entry, text=new_text))
            else:
                corrected_entries.append(entry)

        if total_replacements:
            logger.info("Автоисправление: исправлено слов — %d", total_replacements)
        else:
            logger.info("Автоисправление: замен не потребовалось")

        return corrected_entries
