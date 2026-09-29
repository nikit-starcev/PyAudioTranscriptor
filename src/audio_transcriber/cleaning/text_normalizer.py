"""Аккуратная нормализация текста реплик.

ASR иногда оставляет в тексте «мусорную» типографику: повторную пунктуацию
(``!!!``, ``???``), лишние многоточия (``..``, ``......``), двойные пробелы,
пробелы перед знаками препинания. Модуль приводит такое к единому виду.

Принципы (сознательно консервативные):

* слова **не переписываются** — нормализуется только «оболочка» текста;
* числа и даты трогаются лишь в элементарных безопасных случаях (пробелы
  внутри десятичной дроби, вокруг ``:`` в времени, перед ``%``, вокруг ``/``
  между цифрами); текстовые числительные не разворачиваются;
* текст, не отличающийся от нормы, возвращается без изменений;
* операция идемпотентна.

Реализует протокол ``TextNormalizerProtocol``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from audio_transcriber.domain.models import TranscriptEntry

logger = logging.getLogger(__name__)

_NBSP_RE = re.compile(r"\u00a0")
_WHITESPACE_RE = re.compile(r"\s+")
# Повтор одной и той же пунктуационной марки: «!!!» -> «!», «,,» -> «,».
_REPEATED_PUNCT_RE = re.compile(r"([!?;:,])\1+")
_DOTS_RE = re.compile(r"\.{2,}")
# Элементарные числовые случаи: пробелы внутри числа/времени/процентов.
_NUMBER_DECIMAL_RE = re.compile(r"(\d)\s*([.,])\s*(\d)")
_NUMBER_COLON_RE = re.compile(r"(\d)\s*:\s*(\d)")
_NUMBER_PERCENT_RE = re.compile(r"(\d)\s+%")
_NUMBER_SLASH_RE = re.compile(r"(\d)\s*/\s*(\d)")
# Пробелы перед закрывающими/одиночными знаками и после открывающих.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,!?;:…)\]}».])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{«])\s+")


def _collapse_dots(match: re.Match[str]) -> str:
    """Лишние точки: ``..`` -> ``.``, ``...`` и больше -> ``...``."""
    return "..." if len(match.group(0)) >= 3 else "."


def normalize_text(text: str) -> str:
    """Нормализует типографику одного текста, не трогая слова.

    Идемпотентна: повторный вызов не меняет уже нормализованный текст.
    """
    if not text.strip():
        return text

    result = text.replace("\u00a0", " ")
    result = _WHITESPACE_RE.sub(" ", result)
    result = _NUMBER_DECIMAL_RE.sub(r"\1\2\3", result)
    result = _NUMBER_COLON_RE.sub(r"\1:\2", result)
    result = _NUMBER_PERCENT_RE.sub(r"\1%", result)
    result = _NUMBER_SLASH_RE.sub(r"\1/\2", result)
    result = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
    result = _SPACE_AFTER_OPEN_RE.sub(r"\1", result)
    result = _REPEATED_PUNCT_RE.sub(r"\1", result)
    result = _DOTS_RE.sub(_collapse_dots, result)
    result = _WHITESPACE_RE.sub(" ", result)
    return result.strip()


class TextNormalizer:
    """Нормализует пробелы и пунктуацию в тексте реплик.

    Реализует протокол ``TextNormalizerProtocol``.
    """

    def normalize(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        result: list[TranscriptEntry] = []
        changed = 0

        for entry in entries:
            new_text = normalize_text(entry.text)
            if new_text != entry.text:
                changed += 1
                logger.debug("Нормализация: «%s» → «%s»", entry.text, new_text)
                result.append(replace(entry, text=new_text))
            else:
                result.append(entry)

        if changed:
            logger.info("Нормализация текста: изменено реплик — %d", changed)
        else:
            logger.info("Нормализация текста: изменений не потребовалось")

        return result
