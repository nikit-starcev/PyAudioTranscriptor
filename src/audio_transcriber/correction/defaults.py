"""Значения по умолчанию для автоисправления опечаток ASR.

Дублируются в ``config.example.env`` как
``CORRECTION_MIN_WORD_LENGTH``, ``CORRECTION_MIN_SIMILARITY``,
``CORRECTION_MAX_CANDIDATES``.
"""

DEFAULT_CORRECTION_MIN_WORD_LENGTH = 6
# Порог сходства поднят до 0.93: правки стали консервативными, чтобы
# тех-заимствования («залогинился», «залочился») не тянулись к похожим русским
# словам. Дополнительно ограничена дистанция правки (см. ниже).
DEFAULT_CORRECTION_MIN_SIMILARITY = 0.93
# Максимум правок символов в слове: выйти за одну правку — не менять слово.
DEFAULT_CORRECTION_MAX_EDIT_DISTANCE = 1
DEFAULT_CORRECTION_MAX_CANDIDATES = 8000
