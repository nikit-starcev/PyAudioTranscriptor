"""Схлопывание подряд повторяющихся реплик (зацикливания ASR).

Whisper иногда «залипает» на одной фразе и выдаёт её подряд много раз
(«Продолжение следует», «Спасибо за просмотр», «Подписывайтесь на канал»,
титры редактора/корректора субтитров и т.п.). После разбиения на сегменты это
превращается в стенограмму из десятков одинаковых строк.

Модуль удаляет **серию** подряд идущих одинаковых/почти одинаковых реплик,
оставляя одну (первую). Чтобы не вырезать осмысленную короткую повторяющуюся
речь («да, да», «ну да»), схлопываются только реплики не короче
``min_words`` слов (исключение — известные фразы-галлюцинации). Сравнение
идёт по нормализованному тексту (регистр и пунктуация не важны).

Реализует протокол ``RepetitionCleanerProtocol``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from audio_transcriber.domain.models import TranscriptEntry
from audio_transcriber.utils.text import ratio

logger = logging.getLogger(__name__)

DEFAULT_REPEAT_SIMILARITY = 0.9
DEFAULT_REPEAT_MIN_WORDS = 2

# Известные «залипания» Whisper — субтитровые заглушки/концовки, которые
# повторяются подряд. Значения уже нормализованы (нижний регистр, без
# пунктуации). Список расширяем; это лишь подстраховка поверх общего правила.
KNOWN_REPETITIONS: frozenset[str] = frozenset(
    {
        "продолжение следует",
        "спасибо за просмотр",
        "спасибо за внимание",
        "подписывайтесь на канал",
        "подпишитесь на канал",
        "до новых встреч",
        "всем спасибо",
        "всем пока",
        "конец",
        "конец фильма",
        "редактор субтитров",
        "корректор субтитров",
        "субтитры сделал",
        "субтитры подогнал",
        "translated by",
        "subs by",
    }
)

_PUNCTUATION_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    """Нормализует текст для сравнения: регистр и пунктуация не важны.

    Цифры сохраняются, поэтому «Пункт 1» и «Пункт 2» не считаются повтором.
    """
    stripped = _PUNCTUATION_RE.sub(" ", text.casefold())
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _word_count(normalized: str) -> int:
    return len(_TOKEN_RE.findall(normalized))


class RepetitionCleaner:
    """Схлопывает идущие подряд одинаковые/почти одинаковые реплики.

    Реализует протокол ``RepetitionCleanerProtocol``.
    """

    def __init__(
        self,
        *,
        min_words: int = DEFAULT_REPEAT_MIN_WORDS,
        similarity: float = DEFAULT_REPEAT_SIMILARITY,
    ) -> None:
        if min_words < 1:
            raise ValueError("min_words не может быть меньше 1")
        if not (0.0 < similarity <= 1.0):
            raise ValueError("similarity должен быть в диапазоне (0; 1]")
        self._min_words = min_words
        self._similarity = similarity

    def _should_collapse(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left != right and ratio(left, right) < self._similarity:
            return False
        # Текст совпал (точно или достаточно похоже). Известной фразе-заглушке
        # ограничение на длину не помеха; остальное должно быть не короче
        # ``min_words`` — так сохраняется короткая осмысленная речь («да, да»).
        if left in KNOWN_REPETITIONS:
            return True
        return _word_count(left) >= self._min_words and _word_count(right) >= self._min_words

    def clean(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        result: list[TranscriptEntry] = []
        collapsed = 0

        for entry in entries:
            if result:
                previous = result[-1]
                if self._should_collapse(_normalize(previous.text), _normalize(entry.text)):
                    result[-1] = replace(previous, end=max(previous.end, entry.end))
                    collapsed += 1
                    logger.debug(
                        "Схлопнут повтор: «%s» (оставлена одна реплика)",
                        entry.text,
                    )
                    continue
            result.append(entry)

        if collapsed:
            logger.info("Схлопывание повторов: удалено повторных реплик — %d", collapsed)
        else:
            logger.info("Схлопывание повторов: повторяющихся реплик не найдено")

        return result
