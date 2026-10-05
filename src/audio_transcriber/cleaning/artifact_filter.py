"""Очистка стенограммы от неречевых артефактов ASR.

Whisper (и faster-whisper, и whisper.cpp) на музыке, тишине или посторонних
звуках (например, щелчках клавиатуры рядом с микрофоном) вставляет в текст
пометки, которых не было в речи: ``[АПЛОДИСМЕНТЫ]``, ``(смех)``,
``[BLANK_AUDIO]``, музыкальные символы ``♪ ♫ ♬ ♩`` и т.п. Такие пометки не
несут смысла и засоряют стенограмму.

Отдельно встречается случай, когда тот же шум распознан **без скобок** —
whisper.cpp после фикса потери кусков речи выдаёт «голые» шумовые слова,
часто целыми сериями: ``АПЛОДИСМЕНТЫ АПЛОДИСМЕНТЫ АПЛОДИСМЕНТЫ …`` или
хвостом у осмысленной фразы: ``112 без изменений. АИ тоже. АПЛОДИСМЕНТЫ``.

Модуль удаляет:

* скобочные пометки ``[...]``, ``(...)``, ``{...}``, если внутри **только**
  «шумовые» слова (список :data:`NOISE_WORD_STEMS` расширяем);
* музыкальные символы;
* «голые» серии шумовых слов: реплику, состоящую только из шумовых слов
  **с повтором** (одна основа ≥ 2 раз или ≥ 2 шумовых слова), — целиком;
* одиночные шумовые слова-хвосты/заголовки у осмысленной фразы, если они
  отделены знаком конца предложения (``.``/``!``/``?``/``…``) или идут
  серией из двух и более слов;
* шаблонные галлюцинации Whisper (субтитровые заглушки/концовки вроде
  «Продолжение следует», «Добро пожаловать в Казахстан», «Редактор субтитров
  …», а также подписи сервисов расшифровки — «Transcription by CastingWords»,
  «Subtitles by …») — по спискам :data:`HALLUCINATION_PHRASES` и
  :data:`HALLUCINATION_PREFIXES`;
* «совсем пустые» длинные реплики по эвристике :data:`SPARSE_LONG_SECONDS` /
  :data:`SPARSE_MAX_WORDS` / :data:`SPARSE_MAX_CHARS` (спорные случаи
  сохраняются и только помечаются в логе);
* нормализует пробелы и пунктуацию, оставшиеся после удаления;
* реплики, ставшие пустыми, исключаются целиком.

Одиночное «голое» шумовое слово сохраняется: оно может быть настоящей
короткой репликой («Тишина.», «Звонок.», «Сигнал.», «Шум.», «Стук.»).
Поэтому срабатывают только серии/повторы. Обычный текст не изменяется: если
в реплике не было артефактов, она возвращается как есть.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from audio_transcriber.domain.models import TranscriptEntry
from audio_transcriber.utils.text import WORD_PATTERN

logger = logging.getLogger(__name__)

# Основы «шумовых» слов (без учёта регистра). Сравнение — по началу слова,
# поэтому одна основа покрывает склонения/формы: «аплоди» — аплодисменты,
# аплодисментов, аплодировать; «applaus» — applause и т.д.
NOISE_WORD_STEMS: tuple[str, ...] = (
    # --- русский ---
    "аплоди",
    "смех",
    "смеш",
    "смея",
    "музык",
    "шум",
    "пауз",
    "тишин",
    "кашел",
    "кашл",
    "вздох",
    "вздых",
    "неразборчив",
    "звон",
    "сигнал",
    "щелч",
    "стук",
    "хлоп",
    "фонов",
    "помех",
    # --- английский ---
    "applaus",
    "laugh",
    "music",
    "noise",
    "pause",
    "silence",
    "silent",
    "cough",
    "sigh",
    "inaudible",
    "unintelligible",
    "indistinct",
    "ring",
    "signal",
    "beep",
    "blank",
    "audio",
    "click",
    "knock",
    "background",
)

# Слова-связки и единицы длительности, допустимые внутри шумовой пометки:
# «[СМЕХ И АПЛОДИСМЕНТЫ]», «[ПАУЗА 5 СЕК]». Сравнение — точное.
NOISE_EXACT_WORDS: frozenset[str] = frozenset(
    {
        "и",
        "and",
        "with",
        "с",
        "сек",
        "sec",
        "s",
        "second",
        "seconds",
        "мин",
        "min",
        "minute",
        "minutes",
        "минут",
        "минута",
        "минуты",
        "секунд",
        "секунда",
        "секунды",
    }
)

# Шаблонные галлюцинации Whisper: субтитровые заглушки/концовки, которые модель
# выдаёт на тишине, музыке или неразборчивом звуке. Значения нормализованы
# (нижний регистр, без пунктуации); совпадение — по целой реплике.
HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        "продолжение следует",
        "продолжение в следующей части",
        "добро пожаловать в казахстан",
        "спасибо за просмотр",
        "спасибо за внимание",
        "подписывайтесь на канал",
        "подпишитесь на канал",
        "ставьте лайки",
        "до новых встреч",
        "всем спасибо",
        "всем пока",
        "конец фильма",
        "thanks for watching",
        "thank you for watching",
        "subscribe to my channel",
        "please subscribe",
        # Подписи сервисов расшифровки, встречающиеся в обучающих данных Whisper.
        # Значения нормализованы (пунктуация выброшена: «rev.com» → «rev com»),
        # совпадение — по целой реплике.
        "transcription by castingwords",
        "castingwords",
        "rev com",
        "otter ai",
        "sonix",
        "temi",
        "veed io",
        "descript",
    }
)

# Заглушки-«титры» с изменяемым хвостом (автор, сайт): совпадение по началу
# фразы. Ограничение на длину не позволяет вырезать осмысленное предложение,
# случайно начинающееся с этих слов.
HALLUCINATION_PREFIXES: tuple[str, ...] = (
    "субтитры",
    "редактор субтитров",
    "корректор субтитров",
    "перевод субтитров",
    "субтитры сделал",
    "субтитры создал",
    "субтитры подогнал",
    "subtitles by",
    "subtitles created by",
    "translated by",
    "transcribed by",
    "transcription by",
    "amara.org",
)
HALLUCINATION_PREFIX_MAX_WORDS = 12

# Эвристика «аномально мало текста на длинном интервале». Пороги — константы.
# Срабатывание по «или» помечает реплику как подозрительную, по «и» (совсем
# мало и слов, и символов) — удаляет: спорные случаи сохраняем, чтобы не терять
# реальную короткую реплику, растянутую на длинный интервал.
SPARSE_LONG_SECONDS = 20.0
SPARSE_MAX_WORDS = 4
SPARSE_MAX_CHARS = 14

# Скобочные пометки: (...) [] {...}. Группы с внутренним содержимым, парные
# скобки не смешиваются (открывающая определяет закрывающую).
_BRACKET_PATTERN = re.compile(r"\[([^\[\]]*)\]|\(([^()]*)\)|\{([^{}]*)\}")

# Музыкальные символы и подобные им графические пометки.
_MUSIC_PATTERN = re.compile(r"[♪♫♬♩♭♮♯🎵🎶🎼]+")

# Признак осмысленного текста: есть хотя бы одна буква или цифра.
_MEANINGFUL_PATTERN = re.compile(r"[^\W_]", re.UNICODE)

# Знаки конца предложения: «голый» шум отделяется ими от настоящей речи.
_SENTENCE_ENDINGS: frozenset[str] = frozenset(".!?…")

_WHITESPACE_PATTERN = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…])")
_DUPLICATE_PUNCT = re.compile(r"([,;:!?])\1+")
_DUPLICATE_DOT = re.compile(r"\.{2,}")
_LEADING_SEPARATORS = re.compile(r"^[,.;:!?…\s]+")


def _is_noise_word(word: str) -> bool:
    """Является ли слово частью шумовой пометки."""
    folded = word.casefold()
    if folded in NOISE_EXACT_WORDS:
        return True
    return any(folded.startswith(stem) for stem in NOISE_WORD_STEMS)


def _is_noise_stem_word(word: str) -> bool:
    """Является ли слово именно «шумовым» (не связкой/единицей длительности)."""
    folded = word.casefold()
    return any(folded.startswith(stem) for stem in NOISE_WORD_STEMS)


def _noise_stem(word: str) -> str | None:
    """Возвращает основу шумового слова (или ``None``)."""
    folded = word.casefold()
    for stem in NOISE_WORD_STEMS:
        if folded.startswith(stem):
            return stem
    return None


def _is_noise_only(text: str) -> bool:
    """Состоит ли текст только из шумовых слов (и связок).

    Пустой текст и текст без слов — не «только шум» (``False``).
    """
    words = WORD_PATTERN.findall(text)
    return bool(words) and all(_is_noise_word(word) for word in words)


def _noise_word_count(text: str) -> int:
    """Сколько в тексте собственно шумовых слов (связки не считаются)."""
    return sum(1 for word in WORD_PATTERN.findall(text) if _is_noise_stem_word(word))


def _is_repeated_noise(text: str) -> bool:
    """«Голый» шум с повтором: только шумовые слова и их ≥ 2.

    Учитываются только настоящие шумовые слова: текст «И И» (одни связки)
    повтором не считается — иначе легко удалить короткую осмысленную реплику.
    """
    if not _is_noise_only(text):
        return False
    return _noise_word_count(text) >= 2


def _has_repeated_stem(text: str) -> bool:
    """Встречается ли одна и та же шумовая основа ≥ 2 раз (для серии реплик).

    Осторожный критерий для подряд идущих «голых» шумовых реплик: удаляем
    серию только при явном повторе одной основы (например, десятки
    «АПЛОДИСМЕНТЫ»), а разные одиночные шумовые слова («Тишина.», «Звонок.»)
    сохраняем.
    """
    if not _is_noise_only(text):
        return False
    seen: set[str] = set()
    for word in WORD_PATTERN.findall(text):
        stem = _noise_stem(word)
        if stem is None:
            continue
        if stem in seen:
            return True
        seen.add(stem)
    return False


def _has_meaningful(text: str) -> bool:
    """Есть ли в тексте буквы или цифры (т.е. он не пустой по смыслу)."""
    return _MEANINGFUL_PATTERN.search(text) is not None


# Нормализация текста для сверки с шаблонными галлюцинациями: нижний регистр,
# пунктуация и лишние пробелы не важны.
_MATCH_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_MATCH_WHITESPACE = re.compile(r"\s+", re.UNICODE)


def _normalize_for_match(text: str) -> str:
    """Нижний регистр без пунктуации — ключ для сверки с фразами-шаблонами."""
    folded = _MATCH_NON_WORD.sub(" ", text.casefold())
    return _MATCH_WHITESPACE.sub(" ", folded).strip()


def _word_count(text: str) -> int:
    """Число слов (буквенных токенов) в тексте."""
    return len(WORD_PATTERN.findall(text))


def _is_hallucination(text: str) -> bool:
    """Шаблонная галлюцинация Whisper (субтитровая заглушка/концовка)?

    Совпадение — по целой реплике: точное (регистр и пунктуация не важны) для
    :data:`HALLUCINATION_PHRASES` либо по началу фразы (с ограничением длины)
    для «титров» из :data:`HALLUCINATION_PREFIXES`.
    """
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    if normalized in HALLUCINATION_PHRASES:
        return True
    return (
        _word_count(normalized) <= HALLUCINATION_PREFIX_MAX_WORDS
        and normalized.startswith(HALLUCINATION_PREFIXES)
    )


def _sparse_metrics(entry: TranscriptEntry) -> tuple[float, int, int]:
    """(длительность, число слов, число символов) для эвристики разреженности."""
    text = entry.text.strip()
    return entry.end - entry.start, _word_count(text), len(text)


def _is_sparse_long(entry: TranscriptEntry) -> bool:
    """Подозрительно мало текста на длинном интервале (широкое условие — «или»)."""
    duration, words, chars = _sparse_metrics(entry)
    if duration <= SPARSE_LONG_SECONDS:
        return False
    return words <= SPARSE_MAX_WORDS or chars <= SPARSE_MAX_CHARS


def _is_extreme_sparse_long(entry: TranscriptEntry) -> bool:
    """Совсем мало и слов, и символов на длинном интервале (узкое — «и»)."""
    duration, words, chars = _sparse_metrics(entry)
    if duration <= SPARSE_LONG_SECONDS:
        return False
    return words <= SPARSE_MAX_WORDS and chars <= SPARSE_MAX_CHARS


def _normalize(text: str) -> str:
    """Нормализует пробелы и пунктуацию после удаления пометок."""
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _DUPLICATE_PUNCT.sub(r"\1", text)
    text = _DUPLICATE_DOT.sub(".", text)
    text = _LEADING_SEPARATORS.sub("", text)
    return text.strip()


def _has_sentence_ending(text: str, start: int, end: int) -> bool:
    """Есть ли между позициями ``start`` и ``end`` знак конца предложения."""
    return any(char in _SENTENCE_ENDINGS for char in text[start:end])


def _trim_noise_edges(text: str) -> tuple[str, list[str]]:
    """Обрезает серии «голых» шумовых слов в начале/конце реплики.

    Возвращает (текст, список удалённых слов). Реплика с осмысленным словом
    внутри сохраняется: удаляется только шумовая «рамка». Одиночное шумовое
    слово в начале/конце без знака конца предложения не трогается — оно может
    быть частью настоящей фразы («Проверим сигнал»).
    """
    matches = list(WORD_PATTERN.finditer(text))
    if len(matches) < 2:
        return text, []
    words = [match.group(0) for match in matches]
    noise = [_is_noise_word(word) for word in words]
    if all(noise):
        # Полностью шумовая реплика — её судьбу решает `_is_repeated_noise`.
        return text, []

    leading = 0
    while leading < len(words) and noise[leading]:
        leading += 1

    trailing = 0
    while trailing < len(words) - leading and noise[len(words) - 1 - trailing]:
        trailing += 1

    trim_start = 0
    if leading and (
        leading >= 2
        or _has_sentence_ending(text, matches[leading - 1].end(), matches[leading].start())
    ):
        trim_start = leading

    trim_end = 0
    if trailing and (
        trailing >= 2
        or _has_sentence_ending(
            text,
            matches[len(words) - trailing - 1].end(),
            matches[len(words) - trailing].start(),
        )
    ):
        trim_end = trailing

    if not trim_start and not trim_end:
        return text, []

    removed = (
        words[:trim_start] + words[len(words) - trim_end :] if trim_end else words[:trim_start]
    )

    start = matches[trim_start].start() if trim_start else 0
    end = matches[len(words) - trim_end].start() if trim_end else len(text)
    return text[start:end], removed


class ArtifactCleaner:
    """Удаляет из реплик неречевые пометки Whisper.

    Реализует протокол ``ArtifactCleanerProtocol``.
    """

    def __init__(self, *, drop_sparse_long: bool = True) -> None:
        # Удалять ли «совсем пустые» длинные реплики (см. _is_extreme_sparse_long).
        # При False такие реплики только помечаются в логе, но сохраняются.
        self._drop_sparse_long = drop_sparse_long

    def _clean_text(self, text: str) -> tuple[str, list[str]]:
        """Возвращает очищенный текст и список удалённых артефактов.

        Если артефактов не найдено, текст возвращается без изменений — обычная
        речь не нормализуется и не трогается.
        """
        removed: list[str] = []

        def replace_bracket(match: re.Match[str]) -> str:
            inner = next((group for group in match.groups() if group is not None), "")
            if _is_noise_only(inner):
                removed.append(match.group(0))
                return " "
            return match.group(0)

        cleaned = _BRACKET_PATTERN.sub(replace_bracket, text)

        def replace_music(match: re.Match[str]) -> str:
            removed.append(match.group(0))
            return " "

        cleaned = _MUSIC_PATTERN.sub(replace_music, cleaned)
        if removed:
            cleaned = _normalize(cleaned)

        # «Голые» шумовые слова-рамка: 'Текст. СМЕХ СМЕХ' → 'Текст.'.
        cleaned, trimmed = _trim_noise_edges(cleaned)
        if trimmed:
            cleaned = _normalize(cleaned)
            removed.extend(trimmed)

        if not removed:
            return text, []

        return cleaned, removed

    def clean(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        cleaned_entries: list[TranscriptEntry] = []
        removed_markers = 0
        dropped_entries = 0
        sparse_marked = 0
        # Буфер подряд идущих «голых» шумовых реплик: одиночную сохраняем
        # (может быть реальной короткой фразой), серию из двух и более — нет.
        pending_noise: list[TranscriptEntry] = []

        def flush_pending() -> None:
            nonlocal dropped_entries
            if not pending_noise:
                return
            combined = " ".join(entry.text for entry in pending_noise)
            single_repeat = len(pending_noise) == 1 and _is_repeated_noise(combined)
            series_repeat = len(pending_noise) >= 2 and _has_repeated_stem(combined)
            if single_repeat or series_repeat:
                dropped_entries += len(pending_noise)
                logger.debug("Очистка артефактов: шумовая серия удалена — «%s»", combined)
            else:
                cleaned_entries.extend(pending_noise)
            pending_noise.clear()

        for entry in entries:
            # Шаблонная галлюцинация Whisper — удаляем целиком до прочей очистки.
            if _is_hallucination(entry.text):
                flush_pending()
                dropped_entries += 1
                logger.debug(
                    "Очистка артефактов: шаблонная галлюцинация удалена — «%s»",
                    entry.text,
                )
                continue

            # «Совсем пустая» длинная реплика — почти наверняка галлюцинация.
            if _is_extreme_sparse_long(entry):
                flush_pending()
                if self._drop_sparse_long:
                    dropped_entries += 1
                    logger.warning(
                        "Очистка артефактов: удалена реплика с аномально малым "
                        "текстом на длинном интервале (%.1f с) — «%s»",
                        entry.end - entry.start,
                        entry.text,
                    )
                    continue
                logger.warning(
                    "Очистка артефактов: подозрительно мало текста на длинном "
                    "интервале (%.1f с) — «%s» (оставлена)",
                    entry.end - entry.start,
                    entry.text,
                )
            elif _is_sparse_long(entry):
                # Спорный случай: текста мало, но полностью «пустым» он не выглядит.
                sparse_marked += 1
                logger.warning(
                    "Очистка артефактов: подозрительно мало текста на длинном "
                    "интервале (%.1f с) — «%s» (оставлена)",
                    entry.end - entry.start,
                    entry.text,
                )

            new_text, removed = self._clean_text(entry.text)
            removed_markers += len(removed)

            # Реплика исчезает, если после удаления пометок не осталось
            # осмысленного текста, либо если она и была только шумовой пометкой
            # (например, «♪♪ музыка ♪» без скобок).
            if removed and not _has_meaningful(new_text):
                flush_pending()
                dropped_entries += 1
                logger.debug("Очистка артефактов: реплика удалена — «%s»", entry.text)
                continue
            if removed and _is_noise_only(new_text):
                flush_pending()
                dropped_entries += 1
                logger.debug(
                    "Очистка артефактов: шумовая реплика удалена — «%s» → «%s»",
                    entry.text,
                    new_text,
                )
                continue

            # «Голое» шумовое слово без пометок — копим серию для проверки.
            if _is_noise_only(new_text):
                pending_noise.append(
                    replace(entry, text=new_text) if new_text != entry.text else entry
                )
                continue

            flush_pending()
            if new_text != entry.text:
                logger.debug(
                    "Очистка артефактов: «%s» → «%s» (удалено: %s)",
                    entry.text,
                    new_text,
                    ", ".join(removed),
                )
                cleaned_entries.append(replace(entry, text=new_text))
            else:
                cleaned_entries.append(entry)

        flush_pending()

        if removed_markers or dropped_entries or sparse_marked:
            logger.info(
                "Очистка артефактов: удалено пометок — %d, реплик — %d, "
                "подозрительных длинных реплик — %d",
                removed_markers,
                dropped_entries,
                sparse_marked,
            )
        else:
            logger.info("Очистка артефактов: пометок не найдено")

        return cleaned_entries
