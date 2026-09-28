"""Значения по умолчанию, не зависящие от подсистем.

Вынесены в отдельный модуль, чтобы :mod:`audio_transcriber.config` не
импортировал из :mod:`audio_transcriber.llm` (инверсия зависимостей): и
конфигурация, и клиент LLM берут значение отсюда.
"""

from __future__ import annotations

# 4096 токенов: 7B Q4_K_M влезает в 8 ГБ VRAM вместе с рабочим столом.
DEFAULT_CONTEXT_SIZE = 4096
