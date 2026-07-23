"""Настройка логирования приложения через стандартный модуль ``logging``
с красивым выводом в консоль через ``rich`` и (опционально) сохранением
подробных логов в файл.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

_CONSOLE_LOG_FORMAT = "%(message)s"
_FILE_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def _silence_third_party_noise() -> None:
    """Отключает предупреждения сторонних библиотек, не относящиеся к делу.

    Эти сообщения выводятся напрямую в stderr мимо нашего логгера (через
    ``warnings.warn`` или собственный логгер библиотеки) и описывают
    ожидаемое, не влияющее на результат поведение:

    - ``pyannote.audio`` при каждом запуске предупреждает об отсутствии
      ``torchcodec`` — декодирование аудио в этом проекте выполняется через
      PyAV (см. ``utils/audio.py``), ``torchcodec`` не используется вовсе;
    - ``pyannote.audio`` осознанно отключает TensorFloat-32 для
      воспроизводимости результатов — это не ошибка;
    - PyTorch предупреждает об отсутствии ``triton`` (используется только
      для профилирования FLOPs, не для самого распознавания/диаризации) —
      на Windows ``triton`` не поддерживается.
    """

    # Важно: фильтры настраиваются по имени модуля-источника, без импорта
    # pyannote.audio/torch — сама эта функция вызывается до того, как
    # диаризация лениво импортирует их (см. pyannote_engine.py), и не должна
    # провоцировать их загрузку раньше времени.
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*torchcodec is not installed correctly.*",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pyannote\.audio\.utils\.reproducibility",
    )


def setup_logging(*, verbose: bool = False, log_dir: Path | None = None) -> Path | None:
    """Настраивает корневой логгер приложения.

    В консоль всегда выводятся логи (уровень DEBUG при ``verbose=True``,
    иначе INFO). Если передан ``log_dir``, дополнительно создаётся файл с
    полными DEBUG-логами — независимо от уровня, выбранного для консоли —
    чтобы после сбоя можно было изучить полную историю выполнения.

    :param verbose: подробный (DEBUG) уровень логов в консоли.
    :param log_dir: директория для файла логов. Если ``None``, логи
        сохраняются только в консоль.
    :return: путь к созданному файлу логов, либо ``None``, если файл не
        создавался.
    """

    _silence_third_party_noise()

    console_handler = RichHandler(
        show_time=verbose,
        show_path=verbose,
        rich_tracebacks=True,
        markup=False,
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    handlers: list[logging.Handler] = [console_handler]
    log_file: Path | None = None

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"audio-transcriber_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(_FILE_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
            )
            handlers.append(file_handler)
        except OSError:
            log_file = None

    logging.basicConfig(
        level=logging.DEBUG,
        format=_CONSOLE_LOG_FORMAT,
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    if log_dir is not None and log_file is None:
        logger.warning(
            "Не удалось создать файл логов в %s, логи сохраняются только в консоль",
            log_dir,
        )

    return log_file
