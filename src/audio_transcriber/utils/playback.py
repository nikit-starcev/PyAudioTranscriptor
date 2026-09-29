"""Локальное проигрывание аудио-образцов без блокировки интерфейса.

Используется TUI, чтобы прослушать образец голоса выбранного говорящего.
Плеер выбирается из установленных в системе (ffplay / paplay / aplay / mpv /
afplay) и запускается отдельным процессом — интерфейс не блокируется. Если
плечера нет или запуск не удался, функция возвращает ``False``, а вызывающий
код показывает уведомление. Исключения наружу не пробрасываются.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: Поддерживаемые плееры в порядке предпочтения: (бинарник, аргументы).
_PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
    ("paplay", ()),
    ("aplay", ("-q",)),
    ("mpv", ("--no-video", "--really-quiet")),
    ("afplay", ()),
)


def find_player() -> tuple[str, tuple[str, ...]] | None:
    """Возвращает первый доступный плеер (бинарник и аргументы) или ``None``."""
    for binary, options in _PLAYERS:
        if shutil.which(binary):
            return binary, options
    return None


def play_audio_file(path: Path) -> bool:
    """Запускает неблокирующее проигрывание файла; ``False`` — не получилось.

    Ничего не ждёт: процесс плеера отвязывается от TUI (``start_new_session``),
    а его вывод глушится, чтобы не портить полноэкранный интерфейс.
    """
    player = find_player()
    if player is None:
        logger.warning("Аудио-плеер не найден — образец %s не проигран", path.name)
        return False
    binary, options = player
    try:
        subprocess.Popen(
            [binary, *options, str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        logger.warning("Не удалось запустить плеер %s: %s", binary, exc)
        return False
    return True
