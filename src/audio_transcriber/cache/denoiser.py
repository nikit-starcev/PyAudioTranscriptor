"""Кэширующая обёртка вокруг денойзера.

Шумоподавление — нейросетевой инференс, и при повторном запуске на том же
файле его результат не нужно считать заново. Обёртка кладёт очищенный WAV в
каталог кэша и при совпадении ключа возвращает его напрямую, минуя внутренний
движок.

Если денойзер мягко деградировал (вернул исходный путь — движок недоступен),
результат не кэшируется, чтобы следующая попытка не «залипла» на неудаче.
Внутренний ``close()`` вызывается как обычно и удаляет временные файлы, не
затрагивая файл в кэше.
"""

from __future__ import annotations

import logging
from pathlib import Path

from audio_transcriber.cache.store import StageCache
from audio_transcriber.denoising.base import DenoiserProtocol

logger = logging.getLogger(__name__)


class CachingDenoiser:
    """Обёртка ``DenoiserProtocol``, кэширующая очищенный аудиофайл."""

    def __init__(self, inner: DenoiserProtocol, cache: StageCache, *, source: Path) -> None:
        self._inner = inner
        self._cache = cache
        self._source = source

    def _key(self) -> str:
        return self._cache.key(
            "denoise",
            self._source,
            {"denoise": True, "engine": type(self._inner).__name__},
        )

    def denoise(self, input_path: Path) -> Path:
        key = self._key()
        cached = self._cache.load_audio("denoise", key)
        if cached is not None:
            logger.info("Кэш шумоподавления: попадание (%s)", key[:12])
            return cached

        logger.info("Кэш шумоподавления: промах — выполняю денойз")
        result = self._inner.denoise(input_path)
        # Копируем результат в кэш для следующих запусков, но текущему потребителю
        # возвращаем исходный путь (поведение без кэша не меняется).
        if result != input_path and result.is_file():
            self._cache.save_audio("denoise", key, result)
        return result

    def close(self) -> None:
        self._inner.close()
