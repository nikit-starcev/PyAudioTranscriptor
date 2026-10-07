"""Поиск и разбор файла ``config.env``.

Файл ``config.env`` лежит рядом с проектом (не попадает в git) и содержит
настройки запуска в формате ``КЛЮЧ=значение``. Логика поиска и разбора
вынесена в отдельный модуль, чтобы её переиспользовали TUI и команда
``doctor`` (единый формат и приоритет каталогов поиска).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

#: Каталог пакета: ``.../audio_transcriber/utils/config_env.py`` -> parents[1].
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: Строковые значения, считающиеся истиной в ``config.env`` (регистр не важен).
TRUE_VALUES = frozenset({"1", "true", "yes", "on", "да"})
#: Строковые значения, считающиеся ложью в ``config.env`` (регистр не важен).
FALSE_VALUES = frozenset({"0", "false", "no", "off", "нет"})


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Единый разбор булева значения ``config.env``.

    ``None``, пустая строка и неизвестное значение означают «не задано» —
    возвращается ``default`` (мягкий разбор, как в CLI и вебе). Иначе значение
    сверяется с :data:`TRUE_VALUES`/:data:`FALSE_VALUES` без учёта регистра.
    """
    if value is None or not value.strip():
        return default
    stripped = value.strip().casefold()
    if stripped in TRUE_VALUES:
        return True
    if stripped in FALSE_VALUES:
        return False
    return default


def _source_checkout_root(package_dir: Path) -> Path | None:
    """Корень репозитория, если пакет запущен из исходников (``src/``).

    Для layout ``<repo>/src/audio_transcriber`` каталог пакета лежит в ``src``,
    и корень репозитория — двумя уровнями выше. Для wheel-установки
    (``site-packages/audio_transcriber``) вернуть нечего: жёсткого «корня репо»
    там нет, и подставлять ``site-packages``/``lib/pythonX.Y`` нельзя.
    """
    if package_dir.parent.name == "src":
        return package_dir.parent.parent
    return None


def detect_project_root(package_dir: Path, *, cwd: Path | None = None) -> Path:
    """Определяет каталог для поиска ``config.env`` (тестируемая логика).

    Исходники (``src``-layout) → корень репозитория. Wheel-установка → рабочий
    каталог: данные (``config.env``) лежат рядом с запуском или рядом с пакетом,
    но не в жёстко зашитом каталоге репозитория (issue #90).
    """
    source_root = _source_checkout_root(Path(package_dir))
    if source_root is not None:
        return source_root
    return Path(cwd) if cwd is not None else Path.cwd()


def project_root() -> Path:
    """Возвращает корневой каталог проекта (для поиска ``config.env``)."""
    return detect_project_root(_PACKAGE_DIR)


def _default_roots() -> tuple[Path, ...]:
    """Каталоги поиска ``config.env``: рабочий каталог, затем корень проекта.

    Порядок: сначала ``cwd`` (там обычно запускают и кладут файл), затем
    вычисленный :func:`project_root`. Для wheel-установки это ``cwd`` и
    каталог рядом с пакетом — без несуществующих путей репозитория.
    """
    roots: list[Path] = [Path.cwd()]
    for candidate in (_source_checkout_root(_PACKAGE_DIR), _PACKAGE_DIR.parent):
        if candidate is not None and candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


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
    """Ищет ``config.env`` в каталоге запуска, затем рядом с проектом/пакетом."""
    roots = candidates if candidates is not None else _default_roots()
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
