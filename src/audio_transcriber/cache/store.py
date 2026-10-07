"""Файловое хранилище постадийного кэша.

Ключ стадии — SHA-256 от стабильно сериализованного описания входа: версия
формата кэша, имя стадии, идентичность исходного аудиофайла (путь + размер +
mtime) и параметры стадии. Одинаковые входные данные дают один и тот же ключ,
изменение любого параметра или самого файла — новый.

Хранилище намеренно устроено с мягкой деградацией: любая ошибка чтения/записи
кэша не роняет конвейер, а лишь приводит к промаху и повторному расчёту.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Версия формата кэша. Меняется при несовместимом изменении структуры файлов;
#: старые файлы после этого игнорируются и пересчитываются.
CACHE_FORMAT_VERSION = 1


def _source_identity(source: Path) -> dict[str, Any]:
    """Стабильное описание исходного аудиофайла для ключа кэша."""
    try:
        stat = source.stat()
    except OSError:
        return {"path": str(source), "size": None, "mtime_ns": None}
    return {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def compute_cache_key(stage: str, source: Path, params: Mapping[str, Any]) -> str:
    """Считает ключ кэша стадии: хеш от файла и параметров.

    Параметры сериализуются с ``sort_keys=True`` и компактными разделителями,
    поэтому словари с одинаковыми значениями дают одинаковый ключ независимо
    от порядка ключей.
    """
    payload = {
        "cache_version": CACHE_FORMAT_VERSION,
        "stage": stage,
        "source": _source_identity(source),
        "params": dict(params),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class StageCache:
    """Постадийный кэш на диске: JSON-файл на ключ плюс аудиофайлы денойза."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        #: Сериализация записей в рамках процесса. От межпроцессных гонок
        #: защищает не лок, а уникальное имя временного файла + атомарный
        #: ``os.replace`` (см. :meth:`_write_atomic`).
        self._write_lock = threading.Lock()

    def key(self, stage: str, source: Path, params: Mapping[str, Any]) -> str:
        """Возвращает ключ кэша для стадии."""
        return compute_cache_key(stage, source, params)

    def _json_path(self, stage: str, key: str) -> Path:
        return self.directory / f"{stage}-{key}.json"

    def _audio_path(self, stage: str, key: str) -> Path:
        return self.directory / f"{stage}-{key}.wav"

    def load(self, stage: str, key: str) -> dict[str, Any] | None:
        """Читает данные стадии из кэша.

        Возвращает ``None`` при промахе, а также при повреждённом или
        несовместимом файле (с предупреждением в лог) — вызывающий код в этом
        случае должен пересчитать стадию.
        """
        if not self.enabled:
            return None
        path = self._json_path(stage, key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Кэш %s повреждён, будет пересчитан (%s): %s", stage, path.name, exc)
            return None

        if (
            not isinstance(raw, dict)
            or raw.get("cache_version") != CACHE_FORMAT_VERSION
            or raw.get("stage") != stage
        ):
            logger.warning("Кэш %s несовместим, будет пересчитан: %s", stage, path.name)
            return None

        data = raw.get("data")
        if not isinstance(data, dict):
            logger.warning("Кэш %s повреждён (нет данных), будет пересчитан: %s", stage, path.name)
            return None
        return data

    def save(self, stage: str, key: str, data: Mapping[str, Any]) -> None:
        """Сохраняет данные стадии. Ошибки записи не критичны."""
        if not self.enabled:
            return
        payload = {"cache_version": CACHE_FORMAT_VERSION, "stage": stage, "data": dict(data)}
        path = self._json_path(stage, key)
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            with self._write_lock:
                self._write_atomic(
                    path,
                    lambda tmp: tmp.write_text(text, encoding="utf-8"),
                )
        except OSError as exc:
            logger.warning("Не удалось сохранить кэш %s: %s", stage, exc)

    def load_audio(self, stage: str, key: str) -> Path | None:
        """Возвращает путь к закэшированному аудиофайлу стадии (или ``None``)."""
        if not self.enabled:
            return None
        path = self._audio_path(stage, key)
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            return None
        return None

    def save_audio(self, stage: str, key: str, source: Path) -> Path | None:
        """Копирует аудиофайл стадии в кэш. При ошибке возвращает ``None``."""
        if not self.enabled:
            return None
        path = self._audio_path(stage, key)
        try:
            with self._write_lock:
                self._write_atomic(
                    path,
                    lambda tmp: shutil.copyfile(source, tmp),
                )
        except OSError as exc:
            logger.warning("Не удалось сохранить аудио в кэш %s: %s", stage, exc)
            return None
        return path

    def _temp_path(self, target: Path) -> Path:
        """Уникальный путь временного файла рядом с целевым.

        В имя входят PID и UUID: параллельные записи (в том числе из разных
        процессов) не делят один и тот же ``.tmp`` и не портят файл друг друга.
        """
        return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")

    def _write_atomic(self, target: Path, producer: Callable[[Path], object]) -> None:
        """Пишет во временный файл уникального имени и публикует его ``os.replace``.

        ``os.replace`` в пределах одной ФС атомарен: читатель видит либо старое
        содержимое, либо уже полностью записанное новое, но никогда — обрывок.
        Временный файл удаляется при любой ошибке (в том числе на ``os.replace``).
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp_path = self._temp_path(target)
        try:
            producer(tmp_path)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Не удалось удалить временный файл кэша %s", tmp_path)
            raise

    def clear(self) -> int:
        """Удаляет все файлы кэша и возвращает их количество."""
        if not self.directory.is_dir():
            return 0
        removed = 0
        # ``*.tmp`` захватывает и уникальные скрытые ``.<name>.<pid>.<uuid>.tmp``
        # (pathlib не выделяет файлы с ведущей точкой), поэтому отдельных
        # шаблонов не нужно.
        for pattern in ("*.json", "*.wav", "*.tmp"):
            for entry in self.directory.glob(pattern):
                try:
                    entry.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("Не удалось удалить файл кэша %s: %s", entry, exc)
        return removed
