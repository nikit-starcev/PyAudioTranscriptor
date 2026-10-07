"""Подбор свободного TCP-порта для локального веб-сервера.

Модуль не зависит от веб-зависимостей (``fastapi``/``uvicorn``): только
стандартный ``socket``. Это позволяет команде ``audio-transcriber web``
решить, на каком порту запускаться, ещё до ленивого импорта сервера, а
лаунчерам — не печатать неверный URL при автоподборе (issue #27).
"""

from __future__ import annotations

import socket

#: Порт веб-интерфейса по умолчанию.
DEFAULT_WEB_PORT = 8790
#: Сколько портов после дефолтного просматривать при автоподборе.
WEB_PORT_SCAN_LIMIT = 50
#: Верхняя граница допустимых портов (IANA dynamic/private range).
MAX_PORT = 65535


def _address_families(host: str) -> list[int]:
    """Семейства адресов, в которых имеет смысл пробовать ``bind`` для ``host``.

    ``0.0.0.0``/``127.0.0.1`` дают только ``AF_INET``, ``::`` — ``AF_INET6``,
    ``localhost`` — оба. Если имя не разрешается, возвращается ``AF_INET``:
    последующий ``bind`` сообщит о проблеме с хостом, а не с портом.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return [socket.AF_INET]
    families: list[int] = []
    for family, _type, _proto, _canon, _sockaddr in infos:
        if family not in families:
            families.append(family)
    return families or [socket.AF_INET]


def is_port_available(host: str, port: int) -> bool:
    """Возвращает ``True``, если ``host:port`` можно занять прямо сейчас.

    Доступность проверяется попыткой ``bind`` для каждого семейства адресов
    хоста: если порт уже слушает другой процесс, ядро вернёт ``EADDRINUSE``.
    ``SO_REUSEADDR`` выставлен, чтобы порт в состоянии ``TIME_WAIT`` (без
    активного слушателя) считался свободным — повторный запуск сразу после
    остановки сервера должен работать.
    """
    for family in _address_families(host):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                return False
    return True


def find_available_port(
    host: str,
    start: int = DEFAULT_WEB_PORT,
    limit: int = WEB_PORT_SCAN_LIMIT,
) -> int | None:
    """Ищет первый свободный порт в диапазоне ``[start, start + limit]``.

    Возвращает ``None``, если свободных портов в диапазоне нет (или превышена
    граница :data:`MAX_PORT`). Граница включается: при ``limit=50`` проверяется
    ``start + 50``.
    """
    end = min(start + limit, MAX_PORT)
    for port in range(start, end + 1):
        if is_port_available(host, port):
            return port
    return None
