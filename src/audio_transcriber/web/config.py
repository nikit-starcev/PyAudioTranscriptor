"""Сборка конфигурации конвейера для веб-задач из ``config.env``/окружения.

Единый источник настроек — ``config.env`` (как у TUI). Веб-специфичные
отличия: результат не экспортируется автоматически (``protocol_auto=False``),
десктоп-уведомления и таймлайн выключены, а образцы голоса сохраняются, чтобы
их можно было прослушать в браузере.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_MIN_DURATION_OFF,
    DEFAULT_GLOSSARY_DB,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.utils.config_env import load_config_env
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple


def env_defaults() -> dict[str, str]:
    """Настройки из ``config.env`` рядом с проектом (пустой dict, если нет)."""
    _, defaults = load_config_env()
    return defaults


def _as_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on", "да"}


def _as_int(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip().isdigit():
        return default
    return int(raw.strip())


def _as_float(raw: str | None, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _as_optional_float(raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class WebConfig:
    """Безопасный срез настроек для ``GET /api/config``."""

    input_dir: str
    output_dir: str
    export_formats: list[str]
    llm_enabled: bool
    glossary_enabled: bool
    voices_dir: str

    def as_dict(self) -> dict[str, object]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "export_formats": self.export_formats,
            "llm_enabled": self.llm_enabled,
            "glossary_enabled": self.glossary_enabled,
            "voices_dir": self.voices_dir,
        }


def public_config(*, input_dir: Path, output_dir: Path) -> WebConfig:
    """Безопасный срез настроек (без токенов и путей к моделям)."""
    defaults = env_defaults()
    export_formats = _env_export_formats(defaults)
    voices_dir = defaults.get("VOICES_DIR", "").strip() or "voices"
    return WebConfig(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        export_formats=[fmt.value for fmt in export_formats],
        llm_enabled=_as_bool(defaults.get("LLM_ENABLED")),
        glossary_enabled=_as_bool(defaults.get("GLOSSARY_ENABLED"), default=True),
        voices_dir=voices_dir,
    )


def _env_export_formats(defaults: dict[str, str]) -> tuple[ExportFormat, ...]:
    raw = defaults.get("EXPORT_FORMATS", "").strip()
    if not raw:
        return (ExportFormat.TXT,)
    formats: list[ExportFormat] = []
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            formats.append(ExportFormat(value))
        except ValueError:
            continue
    return tuple(formats) or (ExportFormat.TXT,)


def _env_backend(defaults: dict[str, str]) -> AsrBackend:
    raw = defaults.get("ASR_BACKEND", "").strip().casefold()
    try:
        return AsrBackend(raw)
    except ValueError:
        return AsrBackend.FASTER_WHISPER


def _env_device(defaults: dict[str, str]) -> Device:
    raw = defaults.get("DEVICE", "").strip().casefold()
    try:
        return Device(raw)
    except ValueError:
        return Device.AUTO


def build_job_config(
    source_path: Path,
    *,
    output_dir: Path,
    data_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> AppConfig:
    """Собирает :class:`AppConfig` для одной веб-задачи из настроек окружения.

    ``overrides`` — срез настроек веб-интерфейса (``web-data/settings.json``),
    накладываемый поверх ``config.env``: ключи совпадают с переменными
    ``config.env`` (``GLOSSARY_ENABLED``, ``EXPORT_FORMATS`` и т.д.). Пустые
    значения пропускаются — тогда действует значение из окружения.
    """
    defaults = dict(env_defaults())
    if overrides:
        defaults.update(
            {
                key: value
                for key, value in overrides.items()
                if isinstance(value, str) and value != ""
            }
        )
    glossary_db_raw = defaults.get("GLOSSARY_DB", "").strip()
    glossary_db = Path(glossary_db_raw) if glossary_db_raw else Path(DEFAULT_GLOSSARY_DB)
    voices_raw = defaults.get("VOICES_DIR", "").strip()
    pyannote_raw = defaults.get("PYANNOTE_LOCAL_MODEL", "").strip()
    wcp_model_raw = defaults.get("WHISPER_CPP_MODEL", "").strip()
    llm_model_raw = defaults.get("LLM_MODEL", "").strip()
    glossary_path_raw = defaults.get("GLOSSARY_PATH", "").strip()
    min_duration_off_raw = defaults.get("DIARIZATION_MIN_DURATION_OFF", "").strip()
    clustering_threshold_raw = defaults.get("DIARIZATION_CLUSTERING_THRESHOLD", "").strip()
    clustering_fb_raw = defaults.get("DIARIZATION_CLUSTERING_FB", "").strip()

    return AppConfig(
        input_file=source_path,
        output_dir=output_dir,
        model_name=defaults.get("MODEL", "").strip() or "large-v3-turbo",
        language=None,
        device=_env_device(defaults),
        export_formats=_env_export_formats(defaults),
        num_speakers=None,
        diarization_enabled=_as_bool(defaults.get("DIARIZATION_ENABLED"), default=True),
        diarization_min_duration_off=_as_float(
            min_duration_off_raw, DEFAULT_DIARIZATION_MIN_DURATION_OFF
        ),
        diarization_clustering_threshold=_as_optional_float(clustering_threshold_raw),
        diarization_clustering_fb=_as_optional_float(clustering_fb_raw),
        export_speaker_samples=True,
        hf_token=defaults.get("HF_TOKEN") or None,
        pyannote_local_model=Path(pyannote_raw) if pyannote_raw else None,
        initial_prompt=defaults.get("INITIAL_PROMPT") or None,
        hotwords=defaults.get("HOTWORDS") or None,
        clean_artifacts=_as_bool(defaults.get("CLEAN_ARTIFACTS"), default=True),
        collapse_repeats=_as_bool(defaults.get("COLLAPSE_REPEATS"), default=True),
        repeat_min_words=_as_int(defaults.get("REPEAT_MIN_WORDS"), 4),
        repeat_similarity=_as_float(defaults.get("REPEAT_SIMILARITY"), 0.9),
        normalize_text=_as_bool(defaults.get("NORMALIZE_TEXT"), default=True),
        denoise=_as_bool(defaults.get("DENOISE"), default=True),
        mark_overlap=_as_bool(defaults.get("MARK_OVERLAP"), default=True),
        use_cache=_as_bool(defaults.get("USE_CACHE"), default=True),
        cache_dir=data_dir / "cache",
        notifications=False,
        timeline=False,
        low_confidence_threshold=_as_float(
            defaults.get("LOW_CONFIDENCE_THRESHOLD"), DEFAULT_LOW_CONFIDENCE_THRESHOLD
        ),
        enable_correction=_as_bool(defaults.get("ENABLE_CORRECTION")),
        correction_min_word_length=_as_int(defaults.get("CORRECTION_MIN_WORD_LENGTH"), 5),
        correction_min_similarity=_as_float(defaults.get("CORRECTION_MIN_SIMILARITY"), 0.85),
        correction_max_candidates=_as_int(defaults.get("CORRECTION_MAX_CANDIDATES"), 3),
        asr_backend=_env_backend(defaults),
        whisper_cpp_model=Path(wcp_model_raw) if wcp_model_raw else None,
        whisper_cpp_binary=defaults.get("WHISPER_CPP_BINARY", "").strip() or "whisper-cli",
        whisper_cpp_lib_path=defaults.get("WHISPER_CPP_LIB_PATH") or None,
        whisper_cpp_threads=(
            int(defaults["WHISPER_CPP_THREADS"])
            if defaults.get("WHISPER_CPP_THREADS", "").strip().isdigit()
            else None
        ),
        llm_enabled=_as_bool(defaults.get("LLM_ENABLED")),
        llm_model=Path(llm_model_raw) if llm_model_raw else None,
        llm_binary=defaults.get("LLM_BINARY", "").strip() or "llama-server",
        llm_lib_path=defaults.get("LLM_LIB_PATH") or None,
        llm_gpu=_as_bool(defaults.get("LLM_GPU"), default=True),
        llm_context_size=_as_int(defaults.get("LLM_CONTEXT"), 4096),
        llm_summary=_as_bool(defaults.get("LLM_SUMMARY"), default=True),
        llm_extract_names=_as_bool(defaults.get("LLM_EXTRACT_NAMES")),
        glossary_path=normalize_glossary_paths_tuple(glossary_path_raw or None),
        glossary_db=glossary_db,
        glossary_enabled=_as_bool(defaults.get("GLOSSARY_ENABLED"), default=True),
        voices_dir=Path(voices_raw) if voices_raw else None,
        # Экспорт по кнопке (протокол) — по умолчанию; может быть включён
        # настройкой веб-интерфейса (PROTOCOL_AUTO).
        protocol_auto=_as_bool(defaults.get("PROTOCOL_AUTO"), default=False),
    )
