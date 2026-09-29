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


def delete_voice_sample(path: Path, directory: Path | None = None) -> bool:
    """Безопасно удаляет образец ``.wav`` из библиотеки голосов.

    Удаление разрешено, только если ``path`` — это ``.wav`` **внутри** каталога
    библиотеки ``directory`` (символические ссылки разрешаются перед проверкой).
    Иначе — отказ (``False``): нельзя случайно стереть файл вне библиотеки.
    Если ``directory`` не задан, библиотекой считается родительский каталог
    ``path`` (тогда проверка вырождается в «это файл ``.wav``»).

    Возвращает ``True`` при успешном удалении и ``False`` при отказе/ошибке.
    """
    candidate = Path(path)
    root = Path(directory) if directory is not None else candidate.parent
    if candidate.suffix.lower() != ".wav":
        return False
    try:
        target = candidate.resolve()
        root_resolved = root.resolve()
    except OSError as exc:
        logger.warning("Библиотека голосов: не удалось разрешить путь %s: %s", candidate, exc)
        return False

    if target != root_resolved and root_resolved not in target.parents:
        logger.warning("Отказ удаления: %s вне библиотеки %s", candidate, root)
        return False
    try:
        if not target.is_file():
            return False
        target.unlink()
    except OSError as exc:
        logger.warning("Библиотека голосов: не удалось удалить %s: %s", target, exc)
        return False
    logger.info("Библиотека голосов: удалён образец %s", target)
    return True
