"""Форматирование временных меток для текстовых и субтитровых экспортёров."""

from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды как ``ЧЧ:ММ:СС``."""

    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Форматирует секунды как ``ЧЧ:ММ:СС,ммм`` (формат SubRip)."""

    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
