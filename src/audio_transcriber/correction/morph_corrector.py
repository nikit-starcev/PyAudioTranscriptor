"""Автоисправление опечаток ASR по морфологии русского языка.

Без пользовательских словарей и без нейросетей: неизвестные словоформы
подбираются к ближайшей известной форме из OpenCorpora (через pymorphy3).
Известные слова и склонения не изменяются.

Пороги длины и сходства задаются через ``AppConfig`` / ``config.env``
(см. ``CORRECTION_MIN_WORD_LENGTH``, ``CORRECTION_MIN_SIMILARITY``,
``CORRECTION_MAX_CANDIDATES``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from difflib import SequenceMatcher

from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.models import TranscriptEntry

logger = logging.getLogger(__name__)

# Выделяет «слова» без цифр и знаков препинания (Unicode-буквы).
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _match_case(original: str, replacement: str) -> str:
    """Сохраняет регистр исходного слова у замены."""
    if original.isupper():
        return replacement.upper()

    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]

    return replacement


def _similarity(left: str, right: str) -> float:
    """Доля сходства двух строк без учёта регистра (0..1)."""
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def is_spelling_like_form(
    original: str,
    corrected: str,
    *,
    min_word_length: int = DEFAULT_CORRECTION_MIN_WORD_LENGTH,
    min_similarity: float = DEFAULT_CORRECTION_MIN_SIMILARITY,
) -> bool:
    """Проверяет, похожа ли замена на исправление опечатки, а не на другое слово."""
    if len(original) < min_word_length or len(corrected) < min_word_length:
        return False

    max_len_delta = max(2, len(original) // 5)
    if abs(len(original) - len(corrected)) > max_len_delta:
        return False

    return _similarity(original, corrected) >= min_similarity


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
    ) -> None:
        self._min_word_length = min_word_length
        self._min_similarity = min_similarity
        self._max_candidates = max_candidates

        self._morph = None
        self._words_dawg = None
        self._suggestion_cache: dict[str, str | None] = {}

    def _ensure_loaded(self) -> None:
        if self._morph is not None:
            return

        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()
        self._words_dawg = self._morph.dictionary.words

    def _is_known_russian_word(self, word: str) -> bool:
        assert self._morph is not None
        return any(parse.is_known for parse in self._morph.parse(word))

    def _is_spelling_like(self, original: str, corrected: str) -> bool:
        return is_spelling_like_form(
            original,
            corrected,
            min_word_length=self._min_word_length,
            min_similarity=self._min_similarity,
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

                score = _similarity(folded, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score

            if best is not None and best_score >= self._min_similarity:
                break

        result = best if best_score >= self._min_similarity else None
        self._suggestion_cache[folded] = result
        return result

    def correct(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        self._ensure_loaded()

        corrected_entries: list[TranscriptEntry] = []
        total_replacements = 0

        for entry in entries:
            if not entry.text.strip():
                corrected_entries.append(entry)
                continue

            applied: list[tuple[str, str]] = []

            def replace_match(match: re.Match[str]) -> str:
                original_word = match.group(0)

                if len(original_word) < self._min_word_length:
                    return original_word

                if self._is_known_russian_word(original_word):
                    return original_word

                candidate = self._best_known_form(original_word)
                if candidate is None:
                    return original_word

                if candidate.casefold() == original_word.casefold():
                    return original_word

                replacement = _match_case(original_word, candidate)
                applied.append((original_word, replacement))
                return replacement

            new_text = _WORD_PATTERN.sub(replace_match, entry.text)
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
