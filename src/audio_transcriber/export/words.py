"""Пословная разметка реплик для субтитровых экспортёров.

Подсветка слов включается только тогда, когда у реплики есть пословные
таймстемпы (#45): без них формат выводится как раньше. WebVTT умеет
караоке-подсветку нативно — inline-таймкод ``<ЧЧ:ММ:СС.ммм>`` перед словом
заставляет плеер подсвечивать слово во время его произнесения. SRT такой
возможности не имеет, поэтому там применяется статическая подсветка каждого
слова цветным тегом ``<font>``.
"""

from __future__ import annotations

from audio_transcriber.domain.models import TranscriptEntry, WordTimestamp
from audio_transcriber.export.timestamps import format_vtt_timestamp

#: Цвет статической подсветки слов в SRT (SRT не умеет синхронную подсветку).
SRT_HIGHLIGHT_COLOR = "#00BFFF"


def escape_vtt_text(text: str) -> str:
    """Экранирует служебные символы в тексте cue по правилам WebVTT."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_srt_text(text: str) -> str:
    """Экранирует ``<``/``>``/``&`` в тексте SRT, не трогая теги экспортёра."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def has_words(entry: TranscriptEntry) -> bool:
    """Есть ли у реплики пословные таймстемпы."""

    return bool(entry.words)


def vtt_cue_text(words: list[WordTimestamp], *, highlight: bool) -> str:
    """Текст cue WebVTT с караоке-подсветкой слов (inline-таймкод + ``<c>``)."""

    parts: list[str] = []
    for word in words:
        text = escape_vtt_text(word.text)
        if highlight:
            parts.append(f"<{format_vtt_timestamp(word.start)}><c>{text}</c>")
        else:
            parts.append(text)
    return " ".join(parts)


def srt_cue_text(words: list[WordTimestamp], *, highlight: bool) -> str:
    """Текст SRT с пословными тегами (SRT не поддерживает синхронную подсветку)."""

    parts: list[str] = []
    for word in words:
        text = escape_srt_text(word.text)
        if highlight:
            parts.append(f'<font color="{SRT_HIGHLIGHT_COLOR}">{text}</font>')
        else:
            parts.append(text)
    return " ".join(parts)
