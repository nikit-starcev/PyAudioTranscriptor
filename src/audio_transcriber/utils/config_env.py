"""Поиск и разбор файла ``config.env``.

Файл ``config.env`` лежит рядом с проектом (не попадает в git) и содержит
настройки запуска в формате ``КЛЮЧ=значение``. Логика поиска и разбора
вынесена в отдельный модуль, чтобы её переиспользовали TUI и команда
``doctor`` (единый формат и приоритет каталогов поиска).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

#: Каталог проекта: ``src/audio_transcriber/utils/config_env.py`` -> parents[3].
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    """Возвращает корневой каталог проекта (для поиска ``config.env``)."""
    return _PROJECT_ROOT


def parse_config_env(text: str) -> dict[str, str]:
    """Разбирает содержимое ``config.env`` в словарь настроек.

    Пустые строки и комментарии (``#``) пропускаются. Значения снимаются без
    окружающих кавычек.
    """
    defaults: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        defaults[key.strip()] = value.strip().strip('"').strip("'")
    return defaults


def find_config_env(candidates: Sequence[Path] | None = None) -> Path | None:
    """Ищет ``config.env`` в каталоге запуска, затем в корне проекта."""
    roots = candidates if candidates is not None else (Path.cwd(), _PROJECT_ROOT)
    for root in roots:
        config_path = Path(root) / "config.env"
        if config_path.is_file():
            return config_path
    return None


def load_config_env(candidates: Sequence[Path] | None = None) -> tuple[Path | None, dict[str, str]]:
    """Возвращает путь к ``config.env`` (или ``None``) и разобранные настройки."""
    path = find_config_env(candidates)
    if path is None:
        return None, {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return path, {}
    return path, parse_config_env(text)
