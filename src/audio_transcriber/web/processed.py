"""Пометка обработанных загруженных файлов (issue #16).

После успешного прогона задачи исходный файл считается «обработанным»: он
исключается из основного списка ``GET /api/files``, но **физически остаётся на
месте**. Так сохраняются доступ к аудио (``GET /api/jobs/{id}/audio``),
повторный запуск и стадийный кэш, ключ которого завязан на путь к файлу.

Признак хранится рядом с файлом — sidecar-маркер ``<файл>.processed``. Он не
является медиафайлом, поэтому сам не попадает в выдачу ``/api/files``. Когда
исходный файл удаляют (``DELETE /api/files/{name}``), маркер снимается вместе
с ним, а ``POST /api/files/{name}/restore`` возвращает файл в основной список.
"""

from __future__ import annotations

from pathlib import Path

#: Суффикс sidecar-маркера обработанного файла.
PROCESSED_SUFFIX = ".processed"


def processed_marker(path: Path) -> Path:
    """Путь sidecar-маркера для исходного файла (``<файл>.processed``)."""
    return path.with_name(path.name + PROCESSED_SUFFIX)


def is_processed(path: Path) -> bool:
    """Помечен ли файл как обработанный; ``False`` — при ошибке доступа."""
    try:
        return processed_marker(path).is_file()
    except OSError:
        return False


def mark_processed(path: Path) -> bool:
    """Помечает файл обработанным; ``False``, если маркер создать не удалось."""
    try:
        processed_marker(path).touch(exist_ok=True)
    except OSError:
        return False
    return True


def clear_processed(path: Path) -> bool:
    """Снимает метку обработанного файла; ``False`` — при ошибке доступа."""
    try:
        processed_marker(path).unlink(missing_ok=True)
    except OSError:
        return False
    return True
