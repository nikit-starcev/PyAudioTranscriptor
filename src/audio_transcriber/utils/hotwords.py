"""Ограничение длины hotwords для подсказок ASR (faster-whisper)."""

from __future__ import annotations

from collections.abc import Callable

# Whisper допускает максимум 448 токенов на окно. faster-whisper кладёт в
# prompt и hotwords, и предыдущий текст — до ~223 токенов каждый. Если оба
# близки к лимиту, места на генерацию не остаётся (ошибка
# "The maximum decoding length must be > 0"). Держим hotwords заметно короче,
# чтобы оставался запас под previous_tokens и новые токены.
#
# Для кириллицы ~2.5 символа на токен, поэтому символьный лимит ниже, чем
# кажется: ~280 символов ≈ 100–110 токенов. Это приближение «на входе»;
# после загрузки модели движок дополнительно обрезает по реальным токенам.
MAX_RECOMMENDED_HOTWORDS_LENGTH = 280

# Жёсткий потолок в токенах Whisper (после токенизации модели).
MAX_HOTWORDS_TOKENS = 100


def build_hotwords(
    extra: str | None = None,
    *,
    max_length: int = MAX_RECOMMENDED_HOTWORDS_LENGTH,
) -> tuple[str | None, list[str]]:
    """Собирает строку ``--hotwords`` с ограничением по длине в символах.

    Термины добавляются по порядку, пока итоговая строка не превысит
    ``max_length``. Не поместившиеся термины возвращаются отдельно.

    :param extra: значение, переданное напрямую через ``--hotwords``.
    :param max_length: ограничение на длину итоговой строки в символах.
    :return: пара (итоговая строка hotwords или ``None``, список отброшенных терминов).
    """
    all_terms: list[str] = []
    if extra and extra.strip():
        all_terms.extend(part.strip() for part in extra.split(",") if part.strip())

    included: list[str] = []
    dropped: list[str] = []
    length = 0

    for term in all_terms:
        added_length = len(term) + (2 if included else 0)  # ", " между терминами
        if length + added_length > max_length:
            dropped.append(term)
            continue
        included.append(term)
        length += added_length

    return (", ".join(included) if included else None), dropped


def truncate_hotwords_by_tokens(
    hotwords: str,
    encode: Callable[[str], list[int]],
    *,
    max_tokens: int = MAX_HOTWORDS_TOKENS,
) -> tuple[str, list[str]]:
    """Обрезает hotwords по реальным токенам модели, сохраняя целые термины.

    :param hotwords: строка терминов через ``", "``.
    :param encode: функция токенизации (например, ``tokenizer.encode(...).ids``).
    :param max_tokens: максимальное число токенов Whisper, допустимое для hotwords.
    :return: пара (укороченная строка, список отброшенных терминов).
    """
    terms = [part.strip() for part in hotwords.split(",") if part.strip()]
    if not terms:
        return hotwords, []

    included: list[str] = []
    dropped: list[str] = []

    for term in terms:
        candidate = ", ".join([*included, term])
        token_count = len(encode(" " + candidate))
        if token_count > max_tokens:
            dropped.append(term)
            continue
        included.append(term)

    if not included:
        return "", terms

    return ", ".join(included), dropped
