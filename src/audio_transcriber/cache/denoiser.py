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

import numpy as np

from audio_transcriber.cache.store import StageCache
from audio_transcriber.denoising.base import DenoiserProtocol
from audio_transcriber.progress import ProgressCallback

logger = logging.getLogger(__name__)


class CachingDenoiser:
    """Обёртка ``DenoiserProtocol``, кэширующая очищенный аудиофайл.

    :param on_progress: приёмник событий прогресса. Если задан, прокидывается
        во внутренний денойзер (атрибут ``on_progress``), чтобы длинная стадия
        шумоподавления отдавала реальный процент. Обёртка сама событий не
        генерирует: на попадании в кэш инференса нет, а на промахе весь
        прогресс приходит от ``inner``.
    """

    def __init__(
        self,
        inner: DenoiserProtocol,
        cache: StageCache,
        *,
        source: Path,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._source = source
        self._last_hit = False
        self._last_waveform: np.ndarray | None = None
        #: Ключ денойза последнего вызова ``denoise``. Участвует в ключах ASR и
        #: диаризации: он детерминированно описывает фактически применённый
        #: денойз (движок + параметры + исходный файл), в отличие от одной лишь
        #: настройки ``denoise`` (#83).
        self._last_key: str | None = None
        self._on_progress = on_progress
        if on_progress is not None and hasattr(inner, "on_progress"):
            inner.on_progress = on_progress

    @property
    def last_hit(self) -> bool:
        """Был ли последний :meth:`denoise` обслужен из кэша."""
        return self._last_hit

    @property
    def last_waveform(self) -> np.ndarray | None:
        """Пробрасывает waveform внутреннего денойзера (``None`` при попадании в кэш).

        Кэшированный WAV читается с диска, поэтому декодированного массива в
        памяти нет — при попадании возвращаем ``None``, и диаризация декодирует
        файл сама (как раньше). При промахе переиспользуем waveform, который уже
        посчитал внутренний денойзер.
        """
        return self._last_waveform

    @property
    def cache_key(self) -> str | None:
        """Ключ денойза последнего :meth:`denoise` (``None``, пока не вызывался).

        Потребители (ключи ASR/диаризации) используют его как стабильную
        «подпись» фактически применённого денойза (#83): он уже включает
        исходный файл, движок и его параметры.
        """
        return self._last_key

    def _params(self) -> dict[str, object]:
        """Параметры денойзера, влияющие на результат (движок + чанкинг/частота).

        Без них смена ``chunk_seconds``/``overlap_seconds``/
        ``output_sample_rate``/бинарника/версии реализации переиспользовала бы
        старый очищенный WAV (#83, #50). Параметры читаются через ``getattr``:
        произвольный ``DenoiserProtocol`` может их не иметь — тогда в ключ
        входит только имя класса движка.
        """
        params: dict[str, object] = {"engine": type(self._inner).__name__}
        for name in (
            "chunk_seconds",
            "overlap_seconds",
            "output_sample_rate",
            # Внешний бинарник и версия реализации: переход с Python-пакета
            # ``deepfilternet`` на CLI ``deep-filter`` (или смена бинарника)
            # меняет результат при тех же параметрах (#50).
            "binary",
            "impl_version",
        ):
            value = getattr(self._inner, name, None)
            if value is not None:
                params[name] = value
        return params

    def _key(self) -> str:
        return self._cache.key("denoise", self._source, self._params())

    def denoise(self, input_path: Path) -> Path:
        key = self._key()
        self._last_key = key
        cached = self._cache.load_audio("denoise", key)
        if cached is not None:
            logger.info("Кэш шумоподавления: попадание (%s)", key[:12])
            self._last_hit = True
            self._last_waveform = None
            return cached

        logger.info("Кэш шумоподавления: промах — выполняю денойз")
        self._last_hit = False
        result = self._inner.denoise(input_path)
        self._last_waveform = getattr(self._inner, "last_waveform", None)
        # Копируем результат в кэш для следующих запусков, но текущему потребителю
        # возвращаем исходный путь (поведение без кэша не меняется).
        if result != input_path and result.is_file():
            self._cache.save_audio("denoise", key, result)
        return result

    def close(self) -> None:
        self._inner.close()
        self._last_waveform = None
