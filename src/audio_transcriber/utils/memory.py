"""Освобождение оперативной памяти после тяжёлых стадий конвейера (#141).

Модели в процессах (pyannote/faster-whisper/GigaAM/sherpa) держат RSS до конца
прогона, если их не освободить явно. Здесь собрана единая точка освобождения:

1. сборка мусора Python (``gc.collect``);
2. возврат страниц аллокатора ОС (``malloc_trim`` на Linux/glibc) — без него
   освобождённые ONNX/torch-арены часто остаются в RSS процесса;
3. очистка кэша CUDA — **только** для torch-движков.

Импорт ``torch`` здесь **не** выполняется: на слабом железе и на CPU-путях его
загрузка стоит ~сотни МБ и секунды, а вызывать :func:`release_memory` можно и
после движков без torch. Если ``torch`` уже импортирован (pyannote его грузит),
берём модуль из ``sys.modules`` и чистим CUDA-кэш.
"""

from __future__ import annotations

import gc
import logging
import sys

logger = logging.getLogger(__name__)

__all__ = ["release_memory"]

#: Кэш libc (``None`` — ещё не пытались загрузить, ``False`` — недоступна).
_libc: object | None = None


def _malloc_trim() -> None:
    """Пытается вернуть освобождённые страницы аллокатора ОС (Linux/glibc).

    glibc удерживает освобождённые арены в RSS процесса; ``malloc_trim(0)``
    отдаёт их ядру. Для ONNX Runtime это заметно: после выгрузки CAM++ крупная
    часть памяти освобождается именно здесь. На не-Linux/не-glibc — тихо ничего
    не делает (мягкая деградация).
    """
    global _libc
    if not sys.platform.startswith("linux"):
        return
    if _libc is None:
        try:
            import ctypes

            _libc = ctypes.CDLL("libc.so.6")
        except Exception as exc:  # noqa: BLE001 — libc может отсутствовать (musl)
            logger.debug("malloc_trim недоступен: %s", exc)
            _libc = False
    if _libc is False:
        return
    try:
        _libc.malloc_trim(0)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 — чистка памяти не должна ронять прогон
        logger.debug("malloc_trim завершился ошибкой: %s", exc)


def _empty_cuda_cache() -> None:
    """Чистит кэш CUDA, если ``torch`` уже импортирован (иначе — ничего)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception as exc:  # noqa: BLE001 — очистка кэша не должна ронять прогон
        logger.debug("Не удалось очистить кэш CUDA: %s", exc)


def release_memory(*, cuda: bool = False) -> None:
    """Освобождает память: сборка мусора + возврат страниц ОС (+ кэш CUDA).

    :param cuda: чистить ли кэш CUDA. Передаётся ``True`` только torch-движками
        (pyannote, эмбеддер pyannote). При ``False`` ``torch`` не трогается
        вовсе, чтобы не провоцировать его импорт там, где он не нужен.
    """
    gc.collect()
    if cuda:
        _empty_cuda_cache()
    _malloc_trim()
