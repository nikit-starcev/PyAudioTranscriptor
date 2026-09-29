"""Библиотека образцов голоса (``voices/``) и её слияние с явными образцами.

В каталоге-библиотеке (``voices_dir``, по умолчанию ``./voices``) файлы
``Иван.wav``, ``Мария.wav`` и т.п. трактуются как образцы голоса: имя участника
— это имя файла без расширения (stem). Такие образцы **добавляются** к явно
заданным ``--speaker-reference`` и вместе с ними участвуют в enrollment.

Приоритет и дедупликация: явные образцы идут первыми; файлы библиотеки
дописываются к одноимённому имени; полностью одинаковые пути не дублируются.
Одноимённые образцы (явные + из библиотеки) усредняются при сопоставлении —
это повышает устойчивость эмбеддинга. Отсутствие каталога — не ошибка:
библиотека просто пуста.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from audio_transcriber.utils.text import sanitize_filename

logger = logging.getLogger(__name__)


def collect_voice_library(directory: Path | None) -> dict[str, tuple[Path, ...]]:
    """Собирает образцы из каталога: ``stem файла -> кортеж путей``.

    Учитываются только ``*.wav`` верхнего уровня (регистр расширения не важен).
    Отсутствующий или недоступный каталог даёт пустой словарь (мягкая деградация).
    """
    if directory is None:
        return {}
    try:
        if not directory.is_dir():
            return {}
        files = sorted(path for path in directory.iterdir() if path.is_file())
    except OSError as exc:
        logger.warning("Библиотека голосов недоступна (%s): %s", directory, exc)
        return {}

    library: dict[str, list[Path]] = {}
    for path in files:
        if path.suffix.lower() != ".wav":
            continue
        name = path.stem.strip()
        if not name:
            continue
        library.setdefault(name, []).append(path)
    return {name: tuple(paths) for name, paths in library.items()}


def merge_references(
    explicit: Mapping[str, Sequence[Path]],
    library: Mapping[str, Sequence[Path]],
) -> dict[str, tuple[Path, ...]]:
    """Добавляет образцы библиотеки к явным, не дублируя одинаковые пути.

    Возвращает новый словарь ``имя -> кортеж путей``. Имена, которых нет среди
    явных, появляются только из библиотеки; для одноимённых образцы объединяются.
    """
    merged: dict[str, list[Path]] = {}
    for raw_name, paths in explicit.items():
        name = raw_name.strip()
        if not name:
            continue
        merged.setdefault(name, []).extend(Path(path) for path in paths)

    for raw_name, paths in library.items():
        name = raw_name.strip()
        if not name:
            continue
        bucket = merged.setdefault(name, [])
        for path in paths:
            candidate = Path(path)
            if candidate not in bucket:
                bucket.append(candidate)

    return {name: tuple(paths) for name, paths in merged.items() if paths}


def save_speaker_sample(source: Path, directory: Path, name: str) -> Path:
    """Копирует образец говорящего в библиотеку как ``<имя>.wav``.

    Создаёт каталог при необходимости и возвращает путь сохранённого файла.
    Ошибки ввода-вывода (нет прав, диск) пробрасываются вызывающему.
    """
    target = Path(directory) / f"{sanitize_filename(name)}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target
