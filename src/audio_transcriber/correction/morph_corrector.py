"""Автоисправление опечаток ASR по морфологии русского языка.

Без пользовательских словарей и без нейросетей: неизвестные словоформы
подбираются к ближайшей известной форме из OpenCorpora (через pymorphy3).
Известные слова и склонения не изменяются.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from difflib import SequenceMatcher

from audio_transcriber.domain.models import TranscriptEntry

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
_MIN_WORD_LENGTH = 6
# Порог подогнан так, чтобы «взаимоприимания»→«взаимопонимания» (~0.87)
# проходило, а короткие неизвестные фамилии (~0.80 сходства) — нет.
_MIN_SIMILARITY = 0.85
_MAX_CANDIDATES_PER_PREFIX = 8000


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def _is_spelling_like_fix(original: str, corrected: str) -> bool:
    if len(original) < _MIN_WORD_LENGTH or len(corrected) < _MIN_WORD_LENGTH:
        return False
    if abs(len(original) - len(corrected)) > max(2, len(original) // 5):
        return False
    return _similarity(original, corrected) >= _MIN_SIMILARITY


class MorphTextCorrector:
    """Исправляет только неизвестные словоформы ближайшим словом русского языка.

    Реализует протокол ``TextCorrector``.
    """

    def __init__(self) -> None:
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

    def _best_known_form(self, word: str) -> str | None:
        """Ищет в OpenCorpora ближайшую известную словоформу к опечатке."""
        assert self._words_dawg is not None

        folded = word.casefold()
        if folded in self._suggestion_cache:
            return self._suggestion_cache[folded]

        best: str | None = None
        best_score = 0.0
        # Сначала более длинный префикс (уже кандидаты), затем короче.
        max_pref = min(8, len(folded) - 1)
        for pref_len in range(max_pref, 3, -1):
            prefix = folded[:pref_len]
            candidates = self._words_dawg.keys(prefix)
            # keys() может вернуть огромный список на коротком префиксе —
            # в таком случае пропускаем и пробуем другой префикс.
            if not candidates:
                continue

            counted = 0
            for candidate in candidates:
                counted += 1
                if counted > _MAX_CANDIDATES_PER_PREFIX:
                    break
                if not _is_spelling_like_fix(folded, candidate):
                    continue
                score = _similarity(folded, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score

            if best is not None and best_score >= _MIN_SIMILARITY:
                break

        result = best if best_score >= _MIN_SIMILARITY else None
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
                if len(original_word) < _MIN_WORD_LENGTH:
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
            corrected_entries.append(
                replace(entry, text=new_text) if applied else entry
            )

        if total_replacements:
            logger.info("Автоисправление: исправлено слов — %d", total_replacements)
        else:
            logger.info("Автоисправление: замен не потребовалось")

        return corrected_entries
