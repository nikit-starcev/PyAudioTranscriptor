"""Очистка стенограммы от неречевых артефактов ASR.

Whisper (и faster-whisper, и whisper.cpp) на музыке, тишине или посторонних
звуках (например, щелчках клавиатуры рядом с микрофоном) вставляет в текст
пометки, которых не было в речи: ``[АПЛОДИСМЕНТЫ]``, ``(смех)``,
``[BLANK_AUDIO]``, музыкальные символы ``♪ ♫ ♬ ♩`` и т.п. Такие пометки не
несут смысла и засоряют стенограмму.

Модуль удаляет:

* скобочные пометки ``[...]``, ``(...)``, ``{...}``, если внутри **только**
  «шумовые» слова (список :data:`NOISE_WORD_STEMS` расширяем);
* музыкальные символы;
* нормализует пробелы и пунктуацию, оставшиеся после удаления;
* реплики, ставшие пустыми, исключаются целиком.

Обычный текст не изменяется: если в реплике не было артефактов, она
возвращается как есть.
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

# Скобочные пометки: (...) [] {...}. Группы с внутренним содержимым, парные
# скобки не смешиваются (открывающая определяет закрывающую).
_BRACKET_PATTERN = re.compile(r"\[([^\[\]]*)\]|\(([^()]*)\)|\{([^{}]*)\}")

# Музыкальные символы и подобные им графические пометки.
_MUSIC_PATTERN = re.compile(r"[♪♫♬♩♭♮♯🎵🎶🎼]+")

# Признак осмысленного текста: есть хотя бы одна буква или цифра.
_MEANINGFUL_PATTERN = re.compile(r"[^\W_]", re.UNICODE)

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


def _is_noise_only(text: str) -> bool:
    """Состоит ли текст только из шумовых слов (и связок).

    Пустой текст и текст без слов — не «только шум» (``False``).
    """
    words = WORD_PATTERN.findall(text)
    return bool(words) and all(_is_noise_word(word) for word in words)


def _has_meaningful(text: str) -> bool:
    """Есть ли в тексте буквы или цифры (т.е. он не пустой по смыслу)."""
    return _MEANINGFUL_PATTERN.search(text) is not None


def _normalize(text: str) -> str:
    """Нормализует пробелы и пунктуацию после удаления пометок."""
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _DUPLICATE_PUNCT.sub(r"\1", text)
    text = _DUPLICATE_DOT.sub(".", text)
    text = _LEADING_SEPARATORS.sub("", text)
    return text.strip()


class ArtifactCleaner:
    """Удаляет из реплик неречевые пометки Whisper.

    Реализует протокол ``ArtifactCleanerProtocol``.
    """

    def _clean_text(self, text: str) -> tuple[str, list[str]]:
        """Возвращает очищенный текст и список удалённых пометок.

        Если пометок не найдено, текст возвращается без изменений — обычная
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

        if not removed:
            return text, removed

        return _normalize(cleaned), removed

    def clean(self, entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
        cleaned_entries: list[TranscriptEntry] = []
        removed_markers = 0
        dropped_entries = 0

        for entry in entries:
            new_text, removed = self._clean_text(entry.text)
            removed_markers += len(removed)

            # Реплика исчезает, если после удаления пометок не осталось
            # осмысленного текста, либо если она и была только шумовой пометкой
            # (например, «♪♪ музыка ♪» без скобок). Реплики, в которых пометок
            # не было, не трогаются вовсе — даже состоящие из пунктуации.
            if removed and not _has_meaningful(new_text):
                dropped_entries += 1
                logger.debug("Очистка артефактов: реплика удалена — «%s»", entry.text)
                continue
            if removed and _is_noise_only(new_text):
                dropped_entries += 1
                logger.debug(
                    "Очистка артефактов: шумовая реплика удалена — «%s» → «%s»",
                    entry.text,
                    new_text,
                )
                continue

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

        if removed_markers or dropped_entries:
            logger.info(
                "Очистка артефактов: удалено пометок — %d, пустых реплик — %d",
                removed_markers,
                dropped_entries,
            )
        else:
            logger.info("Очистка артефактов: пометок не найдено")

        return cleaned_entries
