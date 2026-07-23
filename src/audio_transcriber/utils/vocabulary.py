"""Загрузка пользовательского словаря терминов для распознавания речи."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.utils.exceptions import ConfigurationError

# faster-whisper учитывает лишь ограниченный объём hotwords на каждый отрезок
# распознавания и обрезает всё, что не влезло, с конца строки. Ограничиваем
# объём заранее и по терминам целиком, чтобы явно сообщать, какие термины не
# поместились, а не молча резать строку посередине слова.
MAX_RECOMMENDED_HOTWORDS_LENGTH = 900


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
    итоговая строка не превысит ``max_length`` символов — это приблизительный,
    но безопасный запас под реальный лимит контекста модели. Термины, не
    поместившиеся в лимит, не включаются в результат и возвращаются отдельно,
    чтобы вызывающий код мог явно предупредить о них пользователя.

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
