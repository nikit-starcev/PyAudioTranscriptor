"""Постобработка текста через SymSpell по пользовательскому словарю."""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from symspellpy import SymSpell, Verbosity

from audio_transcriber.domain.models import TranscriptEntry

logger = logging.getLogger(__name__)

# Буквенные токены (включая кириллицу). Цифры и пунктуация не трогаем.
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Короткие слова (союзы, предлоги) не корректируем — слишком много ложных срабатываний.
_MIN_WORD_LENGTH = 4


def _match_case(original: str, replacement: str) -> str:
    """Переносит регистр исходного слова на замену из словаря."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class SymSpellTextCorrector:
    """Исправляет слова в репликах по словарю через SymSpell.

    Реализует протокол ``TextCorrector``. Алгоритм язык-агностичен: работает
    для русского, английского и других языков, если термины есть в словаре.
    """

    def __init__(
        self,
        terms: list[str],
        *,
        max_edit_distance: int = 2,
    ) -> None:
        self._max_edit_distance = max_edit_distance
        self._sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )
        self._known_forms: set[str] = set()

        added = 0
        for term in terms:
            normalized = term.strip()
            if not normalized:
                continue
            # Многословные термины регистрируем целиком и по отдельным словам —
            # так ловятся и «взаимопонимания», и отдельные части фраз.
            parts = [normalized, *normalized.split()]
            for part in parts:
                key = part.casefold()
                if len(key) < _MIN_WORD_LENGTH:
                    continue
                if self._sym_spell.create_dictionary_entry(key, 10):
                    added += 1
                self._known_forms.add(key)

        logger.debug("SymSpell: загружено %d форм из %d терминов словаря", added, len(terms))

    def correct(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        if not self._known_forms:
            return entries

        corrected_entries: list[TranscriptEntry] = []
        total_replacements = 0

        for entry in entries:
            new_text, replacements = self._correct_text(entry.text)
            total_replacements += len(replacements)
            for old, new in replacements:
                logger.info("Коррекция словаря: «%s» → «%s»", old, new)
            corrected_entries.append(replace(entry, text=new_text) if replacements else entry)

        if total_replacements:
            logger.info("Постобработка словаря: исправлено слов — %d", total_replacements)
        else:
            logger.info("Постобработка словаря: замен не потребовалось")

        return corrected_entries

    def _correct_text(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        replacements: list[tuple[str, str]] = []

        def replace_match(match: re.Match[str]) -> str:
            original = match.group(0)
            if len(original) < _MIN_WORD_LENGTH:
                return original

            key = original.casefold()
            if key in self._known_forms:
                return original

            suggestions = self._sym_spell.lookup(
                key,
                Verbosity.CLOSEST,
                max_edit_distance=self._max_edit_distance,
                include_unknown=False,
            )
            if not suggestions:
                return original

            best = suggestions[0]
            if best.distance == 0 or best.term == key:
                return original

            corrected = _match_case(original, best.term)
            if corrected != original:
                replacements.append((original, corrected))
            return corrected

        return _WORD_PATTERN.sub(replace_match, text), replacements
