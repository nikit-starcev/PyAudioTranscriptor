"""Десктоп-уведомления о завершении обработки (Linux: ``notify-send``).

Модуль не обращается к сети и не требует внешних сервисов: уведомление
передаётся локальной утилите ``notify-send`` (libnotify), если она есть в
``PATH``. Если утилиты нет или запуск не удался, функция молча возвращает
``False`` и пишет причину в DEBUG-лог — конвейер никогда не падает из-за
уведомлений.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

_APP_NAME = "AudioTranscriptor"
_DEFAULT_TIMEOUT_MS = 5000
_RUN_TIMEOUT_S = 5


def notify(title: str, message: str, *, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> bool:
    """Отправляет десктоп-уведомление через ``notify-send``.

    Args:
        title: Заголовок уведомления.
        message: Текст уведомления.
        timeout_ms: Время показа уведомления в миллисекундах.

    Returns:
        ``True``, если уведомление передано утилите, и ``False``, если
        ``notify-send`` недоступен или запуск завершился ошибкой.
        Исключения наружу не пробрасываются.
    """
    binary = shutil.which("notify-send")
    if binary is None:
        logger.debug("notify-send не найден — уведомление пропущено: %s — %s", title, message)
        return False

    try:
        subprocess.run(
            [binary, "--app-name", _APP_NAME, "--expire-time", str(timeout_ms), title, message],
            check=False,
            capture_output=True,
            timeout=_RUN_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Не удалось отправить уведомление: %s", exc)
        return False
    return True
