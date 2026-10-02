"""Хелперы для сборки окружения внешних процессов.

Движки на базе C++ (llama.cpp, whisper.cpp, nemo-speech) загружают разделяемые
библиотеки по ``LD_LIBRARY_PATH``; логика подмешивания каталога библиотек к
текущему окружению вынесена сюда, чтобы клиент LLM, движок whisper.cpp и движок
диаризации nemo-speech не дублировали её.

Дополнительно здесь проверка доступности бинарника и защита от затенения
системных библиотек: бандл-libstdc++/libgcc_s из каталога библиотек движка
ломают загрузку Vulkan-ICD (Radeon), поэтому такой каталог не подмешивается.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def with_library_path(
    env: Mapping[str, str], library_path: str | None
) -> dict[str, str]:
    """Возвращает копию ``env`` с добавленным ``LD_LIBRARY_PATH``.

    Если ``library_path`` не задан, окружение возвращается без изменений
    (копией). Иначе каталог библиотек ставится первым, сохраняя прежнее
    значение ``LD_LIBRARY_PATH`` после него.
    """
    result = dict(env)
    if library_path:
        existing = result.get("LD_LIBRARY_PATH")
        result["LD_LIBRARY_PATH"] = library_path + (":" + existing if existing else "")
    return result


def binary_available(binary: str) -> bool:
    """Доступен ли бинарник: путь существует либо имя найдено в ``PATH``."""
    if not binary:
        return False
    candidate = Path(binary).expanduser()
    if candidate.is_file():
        return True
    return shutil.which(binary) is not None


def effective_library_path(
    lib_path: str | None,
    *,
    forbidden_libs: Sequence[str] = (),
) -> str | None:
    """Каталог библиотек для ``LD_LIBRARY_PATH`` с защитой от затенения stdlib.

    Возвращает путь каталога, если он существует и не содержит запрещённых
    библиотек (``forbidden_libs``). Иначе (или если каталога нет) возвращает
    ``None``: бандл-версии libstdc++/libgcc_s затеняют системные и ломают
    загрузку Vulkan-ICD, поэтому лучше положиться на RUNPATH бинарника
    (``$ORIGIN/../lib``), а в лог выдать предупреждение.
    """
    if not lib_path:
        return None
    directory = Path(lib_path).expanduser()
    if not directory.is_dir():
        logger.warning(
            "Каталог библиотек не найден: %s — запуск без LD_LIBRARY_PATH",
            directory,
        )
        return None
    shadows = [name for name in forbidden_libs if (directory / name).exists()]
    if shadows:
        logger.warning(
            "В каталоге библиотек %s найдены %s — они затеняют системные и ломают "
            "загрузку Vulkan-ICD; LD_LIBRARY_PATH не задаётся, используется "
            "RUNPATH бинарника. Удалите эти файлы из каталога (например, "
            "перенесите их в lib.bundlebak/).",
            directory,
            ", ".join(shadows),
        )
        return None
    return str(directory)
