"""Библиотека образцов голоса (``voices/``) и её слияние с явными образцами.

В каталоге-библиотеке (``voices_dir``, по умолчанию ``./voices``) файлы
``Иван.wav``, ``Мария.wav`` и т.п. трактуются как образцы голоса: имя участника
— это имя файла без расширения (stem). У одного человека может быть несколько
образцов: ``Иван.wav`` — первый, ``Иван (2).wav``, ``Иван (3).wav`` и т.д. —
дополнительные; они группируются под общим именем. Такие образцы
**добавляются** к явно заданным ``--speaker-reference`` и вместе с ними
участвуют в enrollment.

Приоритет и дедупликация: явные образцы идут первыми; файлы библиотеки
дописываются к одноимённому имени; полностью одинаковые пути не дублируются.
Все образцы человека (явные + из библиотеки) усредняются при сопоставлении —
это повышает устойчивость эмбеддинга. Отсутствие каталога — не ошибка:
библиотека просто пуста.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from audio_transcriber.diarization.reference import (
    ReferencePrepareOptions,
    ReferenceQuality,
    prepare_reference,
)
from audio_transcriber.utils.audio import load_waveform, write_wav
from audio_transcriber.utils.text import sanitize_filename

logger = logging.getLogger(__name__)

#: Суффикс-дубликат в имени файла: ``Иван (2).wav`` — второй образец «Иван».
_DUPLICATE_SUFFIX = re.compile(r"^(?P<base>.+?)\s*\((?P<index>\d+)\)$")


def base_sample_name(stem: str) -> str:
    """Базовое (человеческое) имя образца: ``Иван (2)`` → ``Иван``.

    Соглашение о нескольких образцах: ``Иван.wav`` — первый, ``Иван (2).wav``,
    ``Иван (3).wav`` и т.д. — дополнительные. Имя без суффикса ``(N)``
    возвращается как есть, поэтому старые файлы ``Иван.wav`` читаются как раньше.
    """
    name = stem.strip()
    match = _DUPLICATE_SUFFIX.match(name)
    return match.group("base").strip() if match else name


def sample_index(stem: str) -> int:
    """Порядковый номер образца в группе: без суффикса — 1, ``(N)`` → ``N``."""
    match = _DUPLICATE_SUFFIX.match(stem.strip())
    return int(match.group("index")) if match else 1


def collect_voice_library(directory: Path | None) -> dict[str, tuple[Path, ...]]:
    """Собирает образцы из каталога: ``базовое имя -> кортеж путей``.

    Несколько файлов одного человека (``Иван.wav``, ``Иван (2).wav``, …)
    группируются под общим именем; внутри группы порядок — основной образец,
    затем дубликаты по возрастанию номера. Учитываются только ``*.wav`` верхнего
    уровня (регистр расширения не важен). Отсутствующий или недоступный каталог
    даёт пустой словарь (мягкая деградация).
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
        name = base_sample_name(path.stem)
        if not name:
            continue
        library.setdefault(name, []).append(path)
    return {
        name: tuple(sorted(paths, key=lambda path: (sample_index(path.stem), path.name.casefold())))
        for name, paths in library.items()
    }


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


def unique_sample_path(directory: Path, name: str) -> Path:
    """Свободный путь для нового образца: ``<имя>.wav`` или ``<имя> (N).wav``.

    Существующие файлы не перезаписываются: второй образец «Иван» сохраняется как
    ``Иван (2).wav``, третий — ``Иван (3).wav`` и т.д.
    """
    stem = sanitize_filename(name)
    candidate = Path(directory) / f"{stem}.wav"
    counter = 2
    while candidate.exists():
        candidate = Path(directory) / f"{stem} ({counter}).wav"
        counter += 1
    return candidate


def save_speaker_sample(source: Path, directory: Path, name: str) -> Path:
    """Копирует образец говорящего в библиотеку, не перезаписывая существующие.

    Первый образец имени сохраняется как ``<имя>.wav``, последующие — как
    ``<имя> (2).wav``, ``<имя> (3).wav`` и т.д. Создаёт каталог при
    необходимости и возвращает путь сохранённого файла. Ошибки ввода-вывода (нет
    прав, диск) пробрасываются вызывающему.
    """
    target = unique_sample_path(Path(directory), name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def save_reference_sample(
    source: Path,
    directory: Path,
    name: str,
    *,
    options: ReferencePrepareOptions | None = None,
) -> tuple[Path, ReferenceQuality | None]:
    """Сохраняет образец в библиотеку, подготавливая его (VAD + нормализация).

    Если подготовка включена (``options.enabled``), исходный аудиофайл
    декодируется в 16 кГц моно, обрезается до речи (3–10 с) и нормализуется по
    RMS, а качество возвращается вызывающему для предупреждений. Имя файла —
    свободный ``<имя>.wav``/``<имя> (N).wav``. При выключенной подготовке файл
    копируется как есть (``quality`` — ``None``). Ошибки ввода-вывода
    пробрасываются вызывающему.
    """
    target = unique_sample_path(Path(directory), name)
    target.parent.mkdir(parents=True, exist_ok=True)
    options = options if options is not None else ReferencePrepareOptions()
    if not options.enabled:
        shutil.copyfile(source, target)
        return target, None

    waveform = load_waveform(Path(source))
    prepared = prepare_reference(waveform, options=options)
    if prepared.waveform.size == 0:
        raise ValueError("пустой образец после подготовки")
    write_wav(target, prepared.waveform)
    return target, prepared.quality


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


def delete_voice_samples(directory: Path | None, name: str) -> int:
    """Удаляет **все** образцы человека (все файлы группы ``name``).

    Возвращает число успешно удалённых файлов. Имя нормализуется
    (:func:`base_sample_name`), поэтому ``Иван (2)`` и ``Иван`` указывают на
    одну группу. Отсутствие каталога/имени — не ошибка (возвращается 0).
    """
    library = collect_voice_library(directory)
    paths = library.get(base_sample_name(name), ())
    if not paths:
        return 0
    deleted = 0
    for path in paths:
        if delete_voice_sample(path, directory):
            deleted += 1
    return deleted


def merge_voice_people(
    directory: Path | None, source: str, target: str
) -> list[tuple[Path, Path]]:
    """Переносит все образцы человека ``source`` в группу имени ``target``.

    Файлы переименовываются в свободные ``<target>.wav``/``<target> (N).wav``
    (существующие образцы цели не перезаписываются), поэтому после операции
    библиотека содержит одного человека под одним именем. Имена нормализуются
    (:func:`base_sample_name`). Возвращает пары ``(старый, новый)`` только для
    успешно перенесённых файлов; при совпадении/отсутствии имён — пустой список.
    Ошибки ввода-вывода по отдельному файлу пропускаются (остальные переносятся).
    """
    source_name = base_sample_name(source)
    target_name = base_sample_name(target)
    if not source_name or not target_name or source_name == target_name:
        return []
    if directory is None:
        return []
    root = Path(directory)
    paths = collect_voice_library(directory).get(source_name, ())
    if not paths:
        return []
    moved: list[tuple[Path, Path]] = []
    for path in paths:
        new_path = unique_sample_path(root, target_name)
        try:
            path.rename(new_path)
        except OSError as exc:
            logger.warning("Библиотека голосов: не удалось перенести %s: %s", path, exc)
            continue
        moved.append((path, new_path))
    if moved:
        logger.info(
            "Библиотека голосов: образцы «%s» объединены с «%s» (%d файл(ов))",
            source_name,
            target_name,
            len(moved),
        )
    return moved
