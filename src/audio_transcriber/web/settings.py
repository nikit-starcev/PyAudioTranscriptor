"""Редактируемые настройки веб-интерфейса (``web-data/settings.json``).

Настройки — тонкий слой поверх ``config.env``: значения по умолчанию берутся из
окружения, а сохранённые в JSON поля их переопределяют. При создании задачи
этот срез накладывается на ``config.env`` (см.
:func:`audio_transcriber.web.config.build_job_config`), а путь к БД глоссария
используется эндпоинтами ``/api/glossary``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from audio_transcriber.config import defaults as config_defaults
from audio_transcriber.config.defaults import DEFAULT_VOICES_DIR
from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.web.config import _as_bool, _env_export_formats, env_defaults

logger = logging.getLogger(__name__)

#: Допустимые форматы экспорта (нижний регистр).
VALID_FORMATS: tuple[str, ...] = tuple(export_format.value for export_format in ExportFormat)


class SettingsError(ValueError):
    """Некорректные настройки веб-интерфейса (для ответа 400)."""


@dataclass(slots=True)
class WebSettings:
    """Редактируемый срез настроек веб-интерфейса."""

    glossary_enabled: bool = True
    glossary_db: str = ""
    voices_dir: str = ""
    export_formats: list[str] = field(default_factory=lambda: ["txt"])
    llm_enabled: bool = False
    llm_summary: bool = True
    denoise: bool = True
    mark_overlap: bool = True
    normalize_text: bool = True
    clean_artifacts: bool = True
    protocol_auto: bool = False

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответа API."""
        return asdict(self)

    def env_overrides(self) -> dict[str, str]:
        """Срез в виде переменных ``config.env`` для :func:`build_job_config`."""
        return {
            "GLOSSARY_ENABLED": _format_bool(self.glossary_enabled),
            "GLOSSARY_DB": self.glossary_db,
            "VOICES_DIR": self.voices_dir,
            "EXPORT_FORMATS": ",".join(self.export_formats),
            "LLM_ENABLED": _format_bool(self.llm_enabled),
            "LLM_SUMMARY": _format_bool(self.llm_summary),
            "DENOISE": _format_bool(self.denoise),
            "MARK_OVERLAP": _format_bool(self.mark_overlap),
            "NORMALIZE_TEXT": _format_bool(self.normalize_text),
            "CLEAN_ARTIFACTS": _format_bool(self.clean_artifacts),
            "PROTOCOL_AUTO": _format_bool(self.protocol_auto),
        }

    def resolved_glossary_db(self) -> Path:
        """Путь к SQLite-БД глоссария: настройка или значение по умолчанию."""
        raw = self.glossary_db.strip()
        return Path(raw).expanduser() if raw else Path(config_defaults.DEFAULT_GLOSSARY_DB)

    def resolved_voices_dir(self) -> Path:
        """Каталог-библиотека образцов голоса: настройка или ``./voices``."""
        raw = self.voices_dir.strip()
        return Path(raw).expanduser() if raw else Path(DEFAULT_VOICES_DIR)


def default_settings(defaults: Mapping[str, str] | None = None) -> WebSettings:
    """Настройки по умолчанию из ``config.env`` (пусто — значения класса)."""
    source = defaults if defaults is not None else env_defaults()
    return WebSettings(
        glossary_enabled=_as_bool(source.get("GLOSSARY_ENABLED"), default=True),
        glossary_db=source.get("GLOSSARY_DB", "").strip(),
        voices_dir=source.get("VOICES_DIR", "").strip(),
        export_formats=[fmt.value for fmt in _env_export_formats(dict(source))],
        llm_enabled=_as_bool(source.get("LLM_ENABLED")),
        llm_summary=_as_bool(source.get("LLM_SUMMARY"), default=True),
        denoise=_as_bool(source.get("DENOISE"), default=True),
        mark_overlap=_as_bool(source.get("MARK_OVERLAP"), default=True),
        normalize_text=_as_bool(source.get("NORMALIZE_TEXT"), default=True),
        clean_artifacts=_as_bool(source.get("CLEAN_ARTIFACTS"), default=True),
        protocol_auto=_as_bool(source.get("PROTOCOL_AUTO")),
    )


def settings_from_mapping(
    raw: Mapping[str, object], *, base: WebSettings | None = None
) -> WebSettings:
    """Собирает :class:`WebSettings` из частичного словаря поверх ``base``."""
    current = base or default_settings()

    def pick_bool(key: str, fallback: bool) -> bool:
        value = raw.get(key, fallback)
        return _coerce_bool(value, fallback)

    def pick_str(key: str, fallback: str) -> str:
        value = raw.get(key, fallback)
        return value.strip() if isinstance(value, str) else fallback

    return replace(
        current,
        glossary_enabled=pick_bool("glossary_enabled", current.glossary_enabled),
        glossary_db=pick_str("glossary_db", current.glossary_db),
        voices_dir=pick_str("voices_dir", current.voices_dir),
        export_formats=_coerce_formats(raw.get("export_formats"), current.export_formats),
        llm_enabled=pick_bool("llm_enabled", current.llm_enabled),
        llm_summary=pick_bool("llm_summary", current.llm_summary),
        denoise=pick_bool("denoise", current.denoise),
        mark_overlap=pick_bool("mark_overlap", current.mark_overlap),
        normalize_text=pick_bool("normalize_text", current.normalize_text),
        clean_artifacts=pick_bool("clean_artifacts", current.clean_artifacts),
        protocol_auto=pick_bool("protocol_auto", current.protocol_auto),
    )


def validate_settings(settings: WebSettings) -> None:
    """Проверяет форматы экспорта и пути; неизменяемо, только чтение.

    :raises SettingsError: если форматов нет/есть неизвестные или каталог
        родителя для ``glossary_db``/``voices_dir`` не существует.
    """
    formats = [fmt.strip().casefold() for fmt in settings.export_formats if fmt.strip()]
    if not formats:
        raise SettingsError("Выберите хотя бы один формат экспорта")
    unknown = [fmt for fmt in formats if fmt not in VALID_FORMATS]
    if unknown:
        raise SettingsError(f"Неизвестные форматы экспорта: {', '.join(sorted(set(unknown)))}")
    settings.export_formats = list(dict.fromkeys(formats))

    for label, value in (("glossary_db", settings.glossary_db), ("voices_dir", settings.voices_dir)):
        if not value.strip():
            continue
        _validate_path(label, value)


def _validate_path(label: str, value: str) -> None:
    try:
        path = Path(value).expanduser()
    except (OSError, ValueError) as exc:
        raise SettingsError(f"Некорректный путь {label}: {exc}") from exc
    if path.exists():
        return
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.is_dir():
        raise SettingsError(f"Каталог не существует: {parent}")


class SettingsStore:
    """Чтение/запись ``web-data/settings.json`` (отсутствие — значения по умолчанию)."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Путь к JSON-файлу настроек."""
        return self._path

    def load(self) -> WebSettings:
        """Эффективные настройки: ``config.env`` плюс сохранённые переопределения."""
        base = default_settings()
        saved = self._read()
        if not saved:
            return base
        return settings_from_mapping(saved, base=base)

    def save(self, settings: WebSettings) -> WebSettings:
        """Сохраняет настройки, возвращая то, что записано."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(settings.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise SettingsError(f"Не удалось сохранить настройки: {exc}") from exc
        return settings

    def _read(self) -> dict[str, object]:
        try:
            if not self._path.is_file():
                return {}
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Не удалось прочитать настройки %s — берутся значения по умолчанию", self._path)
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items()}


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _coerce_bool(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().casefold()
        if stripped in {"1", "true", "yes", "on", "да"}:
            return True
        if stripped in {"0", "false", "no", "off", "нет"}:
            return False
    return fallback


def _coerce_formats(value: object, fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return list(fallback)
    return [str(item).strip().casefold() for item in items if str(item).strip()]
