"""Загрузка пользовательского словаря терминов для распознавания речи."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from audio_transcriber.utils.exceptions import ConfigurationError

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


def load_vocabulary_terms(path: Path) -> list[str]:
    """Читает термины из текстового файла (по одному на строке).

    Пустые строки и строки, начинающиеся с ``#``, считаются комментариями
    и игнорируются. Порядок строк в файле — это порядок приоритета: если
    термины не влезут в лимит модели, будут отброшены самые последние.
    """
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"Не удалось прочитать файл словаря {path}: {exc}") from exc

    return [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]


def build_hotwords(
    terms: list[str],
    extra: str | None = None,
    *,
    max_length: int = MAX_RECOMMENDED_HOTWORDS_LENGTH,
) -> tuple[str | None, list[str]]:
    """Объединяет термины словаря и строку ``--hotwords`` в одну строку.

    Термины добавляются по порядку (сначала словарь, затем ``extra``), пока
    итоговая строка не превысит ``max_length`` символов — приблизительный
    запас под токенный лимит модели (для кириллицы символов нужно меньше,
    чем для латиницы). Термины, не поместившиеся в лимит, не включаются в
    результат и возвращаются отдельно, чтобы вызывающий код мог явно
    предупредить о них пользователя.

    :param terms: термины, загруженные из файла словаря.
    :param extra: значение, переданное напрямую через ``--hotwords``.
    :param max_length: ограничение на длину итоговой строки в символах.
    :return: пара (итоговая строка hotwords или ``None``, список отброшенных терминов).
    """
    all_terms = list(terms)
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
