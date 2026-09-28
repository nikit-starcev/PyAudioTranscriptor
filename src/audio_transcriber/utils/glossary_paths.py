"""Нормализация путей к файлам глоссария.

Хелперы не зависят от подсистемы LLM, поэтому используются и конфигурацией
(:class:`~audio_transcriber.config.settings.AppConfig`), и загрузчиком
глоссария.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

# Путь(и) к глоссарию на входе: одиночный путь, строка со списком через
# запятую/``os.pathsep``, либо любая последовательность путей/строк.
GlossaryPathArg = Path | str | Iterable[Path | str] | None


def split_glossary_paths(value: str) -> list[str]:
    """Разбирает строку со списком путей глоссариев.

    Поддерживаются разделители — запятая и ``os.pathsep``. Пустые элементы
    отбрасываются. Буква диска Windows (``C:``) не считается разделителем и
    склеивается со следующей частью.
    """
    if not value:
        return []

    raw_parts = value.split(",")
    if os.pathsep != ",":
        expanded: list[str] = []
        for part in raw_parts:
            expanded.extend(part.split(os.pathsep))
        raw_parts = expanded

    result: list[str] = []
    for index, part in enumerate(raw_parts):
        item = part.strip()
        if not item:
            continue
        if len(item) == 1 and item.isalpha() and index + 1 < len(raw_parts):
            # «C» + «\path» — буква диска Windows, склеиваем обратно.
            raw_parts[index + 1] = item + raw_parts[index + 1]
            continue
        result.append(item)
    return result


def normalize_glossary_paths(value: GlossaryPathArg) -> list[Path]:
    """Приводит любой вход к списку путей глоссариев.

    Принимает ``None``, одиночный путь (:class:`~pathlib.Path` или ``str``),
    строку со списком путей (через запятую или ``os.pathsep``) либо
    последовательность путей. Возвращает список :class:`~pathlib.Path`.
    """
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(item) for item in split_glossary_paths(str(value))]

    result: list[Path] = []
    for item in value:
        if isinstance(item, str):
            result.extend(Path(part) for part in split_glossary_paths(item))
        elif isinstance(item, Path):
            result.append(item)
    return result


def normalize_glossary_paths_tuple(value: GlossaryPathArg) -> tuple[Path, ...]:
    """Как :func:`normalize_glossary_paths`, но сразу возвращает кортеж.

    Используется конвертером поля :attr:`AppConfig.glossary_path`, чтобы
    инвариант «после нормализации — всегда ``tuple[Path, ...]``» был выражен
    в типе.
    """
    return tuple(normalize_glossary_paths(value))
