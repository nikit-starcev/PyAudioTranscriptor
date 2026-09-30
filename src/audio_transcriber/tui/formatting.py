"""Форматирование значений для TUI и разбор значений из ``config.env``."""

from __future__ import annotations

from collections.abc import Sequence

#: Символы для строки амплитуды (от тишины к пику).
_AMP_CHARS = "▁▂▃▄▅▆▇█"


def _amplitude_line(envelope: Sequence[float], width: int | None = None) -> str:
    """Строка амплитуды из блочных символов (по одному на окно)."""
    values = list(envelope)
    if width is not None:
        values = values[:width]
    last = len(_AMP_CHARS) - 1
    chars = []
    for value in values:
        clamped = min(1.0, max(0.0, value))
        chars.append(_AMP_CHARS[round(clamped * last)])
    return "".join(chars)


def _progress_bar(fraction: float, width: int) -> str:
    """Полоса прогресса с курсором ``●`` на позиции ``fraction`` (0..1)."""
    if width <= 0:
        return ""
    position = min(width - 1, max(0, round(fraction * (width - 1))))
    return "─" * position + "●" + "─" * (width - 1 - position)


def _format_size(size: int) -> str:
    """Человекочитаемый размер файла (Б/КБ/МБ/ГБ)."""
    value = float(size)
    for unit in ("Б", "КБ", "МБ"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


def _fmt_duration(seconds: float) -> str:
    """Форматирует длительность: «<1 с», «12 с» или «1:23» для минут."""
    if seconds < 1:
        return "<1 с"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs} с"


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str | None, default: bool = False) -> bool:
    """Разбирает булево значение из ``config.env`` («true», «1», «да», …)."""
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "да")
