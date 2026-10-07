"""Схлопывание зацикливаний ASR (повторы реплик и n-грамм внутри реплики).

Whisper иногда «залипает» на одной фразе и выдаёт её подряд много раз
(«Продолжение следует», «Спасибо за просмотр», «Подписывайтесь на канал»,
титры редактора/корректора субтитров и т.п.). После разбиения на сегменты это
превращается в стенограмму из десятков одинаковых строк. Кроме того, внутри
**одной** реплики whisper.cpp может продублировать фразу («это все прекрасно.
все прекрасно…») — это уже не серия одинаковых реплик, и прежний фильтр её не
видел.

Модуль делает две вещи:

1. Удаляет **серию** подряд идущих одинаковых/почти одинаковых реплик,
   оставляя одну (первую). Чтобы не вырезать осмысленную короткую
   повторяющуюся речь («да, да», «ну да»), схлопываются только реплики не
   короче ``min_words`` слов (исключение — известные фразы-галлюцинации).
2. Схлопывает **соседние повторяющиеся n-граммы внутри одной реплики**
   (whisper-loop). Повтор ищется только среди **соседних** одинаковых
   последовательностей слов (регистр/пунктуация не важны), поэтому легитимные
   повторы в разных местах фразы не трогаются: «перед тем как … перед ним»,
   «двадцать один … двадцать два». Схлопывается n-грамма длиной не меньше
   ``min_words``; дополнительно одиночный повтор буквенного слова, не
   входящего в :data:`LEGIT_SINGLE_REPEATS`, — типичная whisper-заика («перед
   перед запуском», «ещё ещё»), тогда как удвоения коротких междометий
   («да», «ну», «вот») и повторы чисел сохраняются.

После правки текста пословные метки согласуются через
:func:`~audio_transcriber.utils.text.sync_words_to_text` (#84).

Реализует протокол ``RepetitionCleanerProtocol``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from audio_transcriber.domain.models import TranscriptEntry
from audio_transcriber.utils.text import ratio, sync_words_to_text

logger = logging.getLogger(__name__)

DEFAULT_REPEAT_SIMILARITY = 0.9
DEFAULT_REPEAT_MIN_WORDS = 2

#: Короткие слова-междометия/частицы, удвоение которых — нормальная речь
#: («да да», «ну ну», «вот вот», «так так», «нет нет»), а не заикание ASR.
LEGIT_SINGLE_REPEATS: frozenset[str] = frozenset(
    {
        "да",
        "ну",
        "вот",
        "так",
        "нет",
        "ага",
        "угу",
        "ой",
        "ах",
        "эх",
        "мм",
        "эм",
        "ок",
        "ай",
        "а",
        "и",
        "у",
        "о",
        "э",
    }
)

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
#: Ключ слова для сравнения n-грамм: буквы/цифры без пунктуации, нижний регистр.
_WORD_KEY_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    """Нормализует текст для сравнения: регистр и пунктуация не важны.

    Цифры сохраняются, поэтому «Пункт 1» и «Пункт 2» не считаются повтором.
    """
    stripped = _PUNCTUATION_RE.sub(" ", text.casefold())
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _word_count(normalized: str) -> int:
    return len(_TOKEN_RE.findall(normalized))


def _token_key(token: str) -> str:
    """Ключ токена без пунктуации и регистра (для сравнения n-грамм).

    Чистая пунктуация («.–») сопоставляется буквально, иначе все знаки
    препинания считались бы равными друг другу.
    """
    key = "".join(_WORD_KEY_RE.findall(token)).casefold()
    return key or token.casefold()


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

    def _unit_collapsible(self, unit: list[str]) -> bool:
        """Достойна ли n-грамма схлопывания как whisper-loop.

        n-грамма не короче ``min_words`` схлопывается. Одиночное слово —
        только если оно буквенное и не входит в :data:`LEGIT_SINGLE_REPEATS`:
        так убирается заикание на содержательном слове («перед перед»,
        «ещё ещё»), но сохраняются естественные удвоения междометий («да да»,
        «ну ну», «вот вот») и повторяющиеся числа («9 9 9»).
        """
        if len(unit) >= self._min_words:
            return True
        if len(unit) == 1:
            key = _token_key(unit[0])
            if not any(character.isalpha() for character in key):
                return False
            return key not in LEGIT_SINGLE_REPEATS
        return False

    def _collapse_intra(self, text: str) -> tuple[str, int]:
        """Схлопывает соседние повторяющиеся n-граммы внутри текста реплики.

        Возвращает ``(новый_текст, сколько_копий_удалено)``. Ищем только
        **соседние** одинаковые последовательности слов (регистр/пунктуация не
        важны), поэтому повторы в разных местах фразы («перед тем как … перед
        ним») не задеваются. На каждой позиции берётся самая длинная
        подходящая n-грамма; удаляется вторая копия, и позиция проверяется
        снова — так схлопываются и тройные повторы.

        Чисто пунктуационные токены («–», «.») в сравнении не участвуют, иначе
        вставленный знак разрывал бы визуально одинаковые повторы
        («все хосты это сервер» / «Все хосты – это сервер»).
        """
        tokens = text.split()
        if len(tokens) < 2:
            return text, 0

        removed = 0
        while True:
            # Позиции «сигнальных» токенов (не чистая пунктуация) и их ключи.
            positions = [
                pos for pos, token in enumerate(tokens) if _WORD_KEY_RE.search(token)
            ]
            if len(positions) < 2:
                break
            keys = [_token_key(tokens[pos]) for pos in positions]

            index = 0
            while index < len(keys):
                max_unit = (len(keys) - index) // 2
                best_unit = 0
                for unit in range(1, max_unit + 1):
                    if keys[index : index + unit] == keys[index + unit : index + 2 * unit] and (
                        self._unit_collapsible(
                            [tokens[pos] for pos in positions[index : index + unit]]
                        )
                    ):
                        best_unit = unit
                if best_unit:
                    # Удаляем вторую копию из исходных токенов вместе со знаками
                    # препинания внутри неё.
                    start = positions[index + best_unit]
                    end = positions[index + 2 * best_unit - 1] + 1
                    del tokens[start:end]
                    removed += 1
                    break
                index += 1
            if index >= len(keys):
                # Больше повторов нет.
                break

        if not removed:
            return text, 0
        return " ".join(tokens), removed

    @staticmethod
    def _with_text(entry: TranscriptEntry, new_text: str) -> TranscriptEntry:
        """Заменяет текст реплики, согласуя с ним пословные метки (#84)."""
        return replace(
            entry,
            text=new_text,
            words=sync_words_to_text(entry.text, list(entry.words), new_text),
        )

    def collapse_intra(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        """Схлопывает соседние повторяющиеся n-граммы внутри каждой реплики.

        Отдельный проход поверх готовых реплик: вызывается после склейки
        коротких сегментов в предложения, которая могла создать повтор на стыке
        двух сегментов. Повторы, уже схлопнутые в :meth:`clean`, повторно не
        меняются.
        """
        result: list[TranscriptEntry] = []
        removed = 0
        for entry in entries:
            new_text, count = self._collapse_intra(entry.text)
            if count:
                removed += count
                logger.debug(
                    "Схлопнут повтор внутри реплики: «%s» → «%s»",
                    entry.text,
                    new_text,
                )
                result.append(self._with_text(entry, new_text))
            else:
                result.append(entry)
        if removed:
            logger.info(
                "Схлопывание повторов внутри реплик (после склейки): удалено — %d",
                removed,
            )
        return result

    def clean(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        result: list[TranscriptEntry] = []
        collapsed = 0
        intra = 0

        for entry in entries:
            # 1) Повторы-заикания внутри одной реплики (whisper-loop).
            new_text, removed = self._collapse_intra(entry.text)
            if removed:
                intra += removed
                logger.debug(
                    "Схлопнут повтор внутри реплики: «%s» → «%s»",
                    entry.text,
                    new_text,
                )
                entry = self._with_text(entry, new_text)

            # 2) Серия подряд идущих одинаковых/почти одинаковых реплик.
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

        if collapsed or intra:
            logger.info(
                "Схлопывание повторов: повторных реплик — %d, "
                "повторов n-грамм внутри реплик — %d",
                collapsed,
                intra,
            )
        else:
            logger.info("Схлопывание повторов: повторяющихся реплик не найдено")

        return result

