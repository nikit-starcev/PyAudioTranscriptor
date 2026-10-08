"""Редактируемые настройки веб-интерфейса (``web-data/settings.json``).

Настройки — тонкий слой поверх ``config.env``: значения по умолчанию берутся из
окружения, а сохранённые в JSON поля их переопределяют. При создании задачи
этот срез накладывается на ``config.env`` (см.
:func:`audio_transcriber.web.config.build_job_config`), а путь к БД глоссария
используется эндпоинтами ``/api/glossary``.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from audio_transcriber.config import defaults as config_defaults
from audio_transcriber.config.defaults import (
    DEFAULT_DEEP_FILTER_BINARY,
    DEFAULT_DIARIZATION_ENGINE,
    DEFAULT_DIARIZATION_ESTIMATE_ENABLED,
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
    DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_ENABLED,
    DEFAULT_DIARIZATION_HYBRID_LINKAGE,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS,
    DEFAULT_NEMO_SPEECH_BINARY,
    DEFAULT_NEMO_SPEECH_DEVICE,
    DEFAULT_NEMO_SPEECH_MODEL,
    DEFAULT_SENTENCE_MERGE_MAX_GAP,
    DEFAULT_VOICES_DIR,
    VALID_DIARIZATION_ENGINES,
    VALID_DIARIZATION_LINKAGES,
    VALID_LLM_PROVIDERS,
    VALID_NEMO_SPEECH_DEVICES,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.utils.config_env import FALSE_VALUES, TRUE_VALUES
from audio_transcriber.web.config import (
    _as_bool,
    _as_float,
    _as_int,
    _env_export_formats,
    env_defaults,
)

logger = logging.getLogger(__name__)

#: Допустимые форматы экспорта (нижний регистр).
VALID_FORMATS: tuple[str, ...] = tuple(export_format.value for export_format in ExportFormat)

#: Допустимые бэкенды распознавания.
VALID_ASR_BACKENDS: tuple[str, ...] = tuple(backend.value for backend in AsrBackend)

#: Допустимые устройства вычислений.
VALID_DEVICES: tuple[str, ...] = tuple(device.value for device in Device)


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
    #: Путь/имя внешнего Rust-CLI ``deep-filter`` (DeepFilterNet) для денойза.
    deep_filter_binary: str = DEFAULT_DEEP_FILTER_BINARY
    mark_overlap: bool = True
    #: Сводить кластеры с одинаковым уверенным именем в одного говорящего
    #: (enrollment many-to-one). Безымянные «Спикер N» не сливаются.
    merge_same_name_speakers: bool = True
    sentence_merge_max_gap: float = DEFAULT_SENTENCE_MERGE_MAX_GAP
    normalize_text: bool = True
    clean_artifacts: bool = True
    #: Автоисправление опечаток (стадия ``correction``, ``pymorphy3``): правит
    #: только неизвестные словоформы. Осторожный режим, по умолчанию выключен.
    enable_correction: bool = False
    protocol_auto: bool = False
    #: Пословные таймстемпы (#45): собирать слова с временами из токенов ASR.
    #: По умолчанию включено; доступны в результате API.
    word_timestamps: bool = True
    #: Системные уведомления о завершении/ошибке/отмене веб-задачи (#35).
    #: Отдельная настройка веб-интерфейса; переопределяет ``NOTIFICATIONS``
    #: из ``config.env`` для веб-задач.
    notifications: bool = True
    #: Бэкенд распознавания (``faster-whisper``, ``whisper-cpp`` или ``gigaam``).
    asr_backend: str = "faster-whisper"
    #: Устройство вычислений (``auto`` / ``cpu`` / ``cuda``).
    device: str = "auto"
    #: Пути к локальным моделям и бинарникам (совпадают с ключами ``config.env``).
    whisper_cpp_model: str = ""
    whisper_cpp_binary: str = "whisper-cli"
    #: Каталог библиотек whisper.cpp (Linux — ``LD_LIBRARY_PATH``). Заполняется
    #: кнопкой автоустановки бинарника (#98) или вручную.
    whisper_cpp_lib_path: str = ""
    llm_model: str = ""
    llm_binary: str = "llama-server"
    #: Каталог библиотек llama.cpp (нужен для ``llama-server``).
    llm_lib_path: str = ""
    #: Провайдер LLM (``llama`` — локальный, ``openai`` — внешний API).
    llm_provider: str = "llama"
    #: Базовый URL и имя модели внешнего OpenAI-совместимого API.
    llm_base_url: str = ""
    llm_model_name: str = ""
    #: API-ключ внешней LLM — секрет: хранится в ``secrets.json``, не в
    #: ``settings.json``. В этом срезе намеренно отсутствует.
    pyannote_local_model: str = ""
    #: Движок диаризации и параметры NeMo-Speech.cpp (#62):
    #: ``auto``/``pyannote``/``nemo-speech``, бинарник, каталог библиотек,
    #: модель и устройство (``auto``/``vulkan``/``cpu``).
    diarization_engine: str = DEFAULT_DIARIZATION_ENGINE
    nemo_speech_binary: str = DEFAULT_NEMO_SPEECH_BINARY
    nemo_speech_lib_path: str = ""
    nemo_speech_model: str = DEFAULT_NEMO_SPEECH_MODEL
    nemo_speech_device: str = DEFAULT_NEMO_SPEECH_DEVICE
    #: Оценщик числа говорящих и маршрутизация ``auto`` (#64): включение,
    #: длительность анализа речи (сек), порог кластеризации, модель и cap
    #: маршрутизации (до него ``auto`` выбирает nemo-speech).
    diarization_estimate_enabled: bool = DEFAULT_DIARIZATION_ESTIMATE_ENABLED
    diarization_estimate_seconds: float = DEFAULT_DIARIZATION_ESTIMATE_SECONDS
    diarization_estimate_threshold: float = DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD
    diarization_estimate_model: str = DEFAULT_DIARIZATION_ESTIMATE_MODEL
    diarization_route_max_speakers: int = DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS
    #: Гибридная диаризация (#64, часть 2): оконный nemo-speech + глобальная
    #: склейка говорящих по эмбеддингам (обход лимита 4). Окно/перекрытие/порог.
    diarization_hybrid_enabled: bool = DEFAULT_DIARIZATION_HYBRID_ENABLED
    diarization_hybrid_window_seconds: float = DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS
    diarization_hybrid_overlap_seconds: float = DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS
    diarization_hybrid_min_speaker_seconds: float = DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS
    #: Linkage и порог пороговой ветки кластеризации гибрида (порог — в единицах
    #: евклидова расстояния при linkage=ward, не путать с порогом оценщика).
    diarization_hybrid_linkage: str = DEFAULT_DIARIZATION_HYBRID_LINKAGE
    diarization_hybrid_threshold: float = DEFAULT_DIARIZATION_HYBRID_THRESHOLD
    #: GigaAM v3 (RU) через onnx-asr (бэкенд ``gigaam``): имя модели, локальный
    #: каталог снимка, квантизация (``int8``/пусто) и встроенный VAD. Пустое имя
    #: означает значение по умолчанию из ``config.defaults``.
    gigaam_model: str = ""
    gigaam_model_path: str = ""
    gigaam_quantization: str = ""
    gigaam_vad: bool = True

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответа API (без секретов)."""
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
            "DEEP_FILTER_BINARY": self.deep_filter_binary,
            "MARK_OVERLAP": _format_bool(self.mark_overlap),
            "MERGE_SAME_NAME_SPEAKERS": _format_bool(self.merge_same_name_speakers),
            "SENTENCE_MERGE_MAX_GAP": str(self.sentence_merge_max_gap),
            "NORMALIZE_TEXT": _format_bool(self.normalize_text),
            "CLEAN_ARTIFACTS": _format_bool(self.clean_artifacts),
            "ENABLE_CORRECTION": _format_bool(self.enable_correction),
            "PROTOCOL_AUTO": _format_bool(self.protocol_auto),
            "WORD_TIMESTAMPS": _format_bool(self.word_timestamps),
            "NOTIFICATIONS": _format_bool(self.notifications),
            "ASR_BACKEND": self.asr_backend,
            "DEVICE": self.device,
            "WHISPER_CPP_MODEL": self.whisper_cpp_model,
            "WHISPER_CPP_BINARY": self.whisper_cpp_binary,
            "WHISPER_CPP_LIB_PATH": self.whisper_cpp_lib_path,
            "LLM_MODEL": self.llm_model,
            "LLM_BINARY": self.llm_binary,
            "LLM_LIB_PATH": self.llm_lib_path,
            "LLM_PROVIDER": self.llm_provider,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL_NAME": self.llm_model_name,
            "PYANNOTE_LOCAL_MODEL": self.pyannote_local_model,
            "DIARIZATION_ENGINE": self.diarization_engine,
            "NEMO_SPEECH_BINARY": self.nemo_speech_binary,
            "NEMO_SPEECH_LIB_PATH": self.nemo_speech_lib_path,
            "NEMO_SPEECH_MODEL": self.nemo_speech_model,
            "NEMO_SPEECH_DEVICE": self.nemo_speech_device,
            "DIARIZATION_ESTIMATE_ENABLED": _format_bool(self.diarization_estimate_enabled),
            "DIARIZATION_ESTIMATE_SECONDS": str(self.diarization_estimate_seconds),
            "DIARIZATION_ESTIMATE_THRESHOLD": str(self.diarization_estimate_threshold),
            "DIARIZATION_ESTIMATE_MODEL": self.diarization_estimate_model,
            "DIARIZATION_ROUTE_MAX_SPEAKERS": str(self.diarization_route_max_speakers),
            "DIARIZATION_HYBRID_ENABLED": _format_bool(self.diarization_hybrid_enabled),
            "DIARIZATION_HYBRID_WINDOW_SECONDS": str(
                self.diarization_hybrid_window_seconds
            ),
            "DIARIZATION_HYBRID_OVERLAP_SECONDS": str(
                self.diarization_hybrid_overlap_seconds
            ),
            "DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS": str(
                self.diarization_hybrid_min_speaker_seconds
            ),
            "DIARIZATION_HYBRID_LINKAGE": self.diarization_hybrid_linkage,
            "DIARIZATION_HYBRID_THRESHOLD": str(self.diarization_hybrid_threshold),
            "GIGAAM_MODEL": self.gigaam_model,
            "GIGAAM_MODEL_PATH": self.gigaam_model_path,
            "GIGAAM_QUANTIZATION": self.gigaam_quantization,
            "GIGAAM_VAD": _format_bool(self.gigaam_vad),
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
        deep_filter_binary=source.get("DEEP_FILTER_BINARY", "").strip()
        or DEFAULT_DEEP_FILTER_BINARY,
        mark_overlap=_as_bool(source.get("MARK_OVERLAP"), default=True),
        merge_same_name_speakers=_as_bool(
            source.get("MERGE_SAME_NAME_SPEAKERS"), default=True
        ),
        sentence_merge_max_gap=_as_float(
            source.get("SENTENCE_MERGE_MAX_GAP"), DEFAULT_SENTENCE_MERGE_MAX_GAP
        ),
        normalize_text=_as_bool(source.get("NORMALIZE_TEXT"), default=True),
        clean_artifacts=_as_bool(source.get("CLEAN_ARTIFACTS"), default=True),
        enable_correction=_as_bool(source.get("ENABLE_CORRECTION")),
        protocol_auto=_as_bool(source.get("PROTOCOL_AUTO")),
        word_timestamps=_as_bool(source.get("WORD_TIMESTAMPS"), default=True),
        notifications=_as_bool(source.get("NOTIFICATIONS"), default=True),
        asr_backend=source.get("ASR_BACKEND", "").strip() or "faster-whisper",
        device=source.get("DEVICE", "").strip() or "auto",
        whisper_cpp_model=source.get("WHISPER_CPP_MODEL", "").strip(),
        whisper_cpp_binary=source.get("WHISPER_CPP_BINARY", "").strip() or "whisper-cli",
        whisper_cpp_lib_path=source.get("WHISPER_CPP_LIB_PATH", "").strip(),
        llm_model=source.get("LLM_MODEL", "").strip(),
        llm_binary=source.get("LLM_BINARY", "").strip() or "llama-server",
        llm_lib_path=source.get("LLM_LIB_PATH", "").strip(),
        llm_provider=source.get("LLM_PROVIDER", "").strip().casefold() or "llama",
        llm_base_url=source.get("LLM_BASE_URL", "").strip(),
        llm_model_name=source.get("LLM_MODEL_NAME", "").strip(),
        pyannote_local_model=source.get("PYANNOTE_LOCAL_MODEL", "").strip(),
        diarization_engine=source.get("DIARIZATION_ENGINE", "").strip().casefold()
        or DEFAULT_DIARIZATION_ENGINE,
        nemo_speech_binary=source.get("NEMO_SPEECH_BINARY", "").strip()
        or DEFAULT_NEMO_SPEECH_BINARY,
        nemo_speech_lib_path=source.get("NEMO_SPEECH_LIB_PATH", "").strip(),
        nemo_speech_model=source.get("NEMO_SPEECH_MODEL", "").strip()
        or DEFAULT_NEMO_SPEECH_MODEL,
        nemo_speech_device=source.get("NEMO_SPEECH_DEVICE", "").strip().casefold()
        or DEFAULT_NEMO_SPEECH_DEVICE,
        diarization_estimate_enabled=_as_bool(
            source.get("DIARIZATION_ESTIMATE_ENABLED"),
            default=DEFAULT_DIARIZATION_ESTIMATE_ENABLED,
        ),
        diarization_estimate_seconds=_as_float(
            source.get("DIARIZATION_ESTIMATE_SECONDS"),
            DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
        ),
        diarization_estimate_threshold=_as_float(
            source.get("DIARIZATION_ESTIMATE_THRESHOLD"),
            DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
        ),
        diarization_estimate_model=source.get("DIARIZATION_ESTIMATE_MODEL", "").strip()
        or DEFAULT_DIARIZATION_ESTIMATE_MODEL,
        diarization_route_max_speakers=_as_int(
            source.get("DIARIZATION_ROUTE_MAX_SPEAKERS"),
            DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS,
        ),
        diarization_hybrid_enabled=_as_bool(
            source.get("DIARIZATION_HYBRID_ENABLED"),
            default=DEFAULT_DIARIZATION_HYBRID_ENABLED,
        ),
        diarization_hybrid_window_seconds=_as_float(
            source.get("DIARIZATION_HYBRID_WINDOW_SECONDS"),
            DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
        ),
        diarization_hybrid_overlap_seconds=_as_float(
            source.get("DIARIZATION_HYBRID_OVERLAP_SECONDS"),
            DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
        ),
        diarization_hybrid_min_speaker_seconds=_as_float(
            source.get("DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS"),
            DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
        ),
        diarization_hybrid_linkage=source.get("DIARIZATION_HYBRID_LINKAGE", "").strip().casefold()
        or DEFAULT_DIARIZATION_HYBRID_LINKAGE,
        diarization_hybrid_threshold=_as_float(
            source.get("DIARIZATION_HYBRID_THRESHOLD"),
            DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
        ),
        gigaam_model=source.get("GIGAAM_MODEL", "").strip()
        or config_defaults.DEFAULT_GIGAAM_MODEL,
        gigaam_model_path=source.get("GIGAAM_MODEL_PATH", "").strip(),
        gigaam_quantization=source.get("GIGAAM_QUANTIZATION", "").strip(),
        gigaam_vad=_as_bool(source.get("GIGAAM_VAD"), default=True),
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

    def pick_nonempty(key: str, fallback: str) -> str:
        value = raw.get(key, fallback)
        if isinstance(value, str):
            return value.strip() or fallback
        return fallback

    def pick_float(key: str, fallback: float) -> float:
        value = raw.get(key, fallback)
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip())
            except ValueError:
                return fallback
        return fallback

    def pick_int(key: str, fallback: int) -> int:
        value = raw.get(key, fallback)
        if isinstance(value, bool):
            return fallback
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            return int(value.strip())
        return fallback

    return replace(
        current,
        glossary_enabled=pick_bool("glossary_enabled", current.glossary_enabled),
        glossary_db=pick_str("glossary_db", current.glossary_db),
        voices_dir=pick_str("voices_dir", current.voices_dir),
        export_formats=_coerce_formats(raw.get("export_formats"), current.export_formats),
        llm_enabled=pick_bool("llm_enabled", current.llm_enabled),
        llm_summary=pick_bool("llm_summary", current.llm_summary),
        denoise=pick_bool("denoise", current.denoise),
        deep_filter_binary=pick_nonempty(
            "deep_filter_binary", current.deep_filter_binary
        ),
        mark_overlap=pick_bool("mark_overlap", current.mark_overlap),
        merge_same_name_speakers=pick_bool(
            "merge_same_name_speakers", current.merge_same_name_speakers
        ),
        sentence_merge_max_gap=pick_float(
            "sentence_merge_max_gap", current.sentence_merge_max_gap
        ),
        normalize_text=pick_bool("normalize_text", current.normalize_text),
        clean_artifacts=pick_bool("clean_artifacts", current.clean_artifacts),
        enable_correction=pick_bool("enable_correction", current.enable_correction),
        protocol_auto=pick_bool("protocol_auto", current.protocol_auto),
        word_timestamps=pick_bool("word_timestamps", current.word_timestamps),
        notifications=pick_bool("notifications", current.notifications),
        asr_backend=pick_nonempty("asr_backend", current.asr_backend),
        device=pick_nonempty("device", current.device),
        whisper_cpp_model=pick_str("whisper_cpp_model", current.whisper_cpp_model),
        whisper_cpp_binary=pick_nonempty("whisper_cpp_binary", current.whisper_cpp_binary),
        whisper_cpp_lib_path=pick_str("whisper_cpp_lib_path", current.whisper_cpp_lib_path),
        llm_model=pick_str("llm_model", current.llm_model),
        llm_binary=pick_nonempty("llm_binary", current.llm_binary),
        llm_lib_path=pick_str("llm_lib_path", current.llm_lib_path),
        llm_provider=pick_nonempty("llm_provider", current.llm_provider),
        llm_base_url=pick_str("llm_base_url", current.llm_base_url),
        llm_model_name=pick_str("llm_model_name", current.llm_model_name),
        pyannote_local_model=pick_str("pyannote_local_model", current.pyannote_local_model),
        diarization_engine=pick_nonempty("diarization_engine", current.diarization_engine).casefold(),
        nemo_speech_binary=pick_nonempty("nemo_speech_binary", current.nemo_speech_binary),
        nemo_speech_lib_path=pick_str("nemo_speech_lib_path", current.nemo_speech_lib_path),
        nemo_speech_model=pick_nonempty("nemo_speech_model", current.nemo_speech_model),
        nemo_speech_device=pick_nonempty("nemo_speech_device", current.nemo_speech_device).casefold(),
        diarization_estimate_enabled=pick_bool(
            "diarization_estimate_enabled", current.diarization_estimate_enabled
        ),
        diarization_estimate_seconds=pick_float(
            "diarization_estimate_seconds", current.diarization_estimate_seconds
        ),
        diarization_estimate_threshold=pick_float(
            "diarization_estimate_threshold", current.diarization_estimate_threshold
        ),
        diarization_estimate_model=pick_nonempty(
            "diarization_estimate_model", current.diarization_estimate_model
        ),
        diarization_route_max_speakers=pick_int(
            "diarization_route_max_speakers", current.diarization_route_max_speakers
        ),
        diarization_hybrid_enabled=pick_bool(
            "diarization_hybrid_enabled", current.diarization_hybrid_enabled
        ),
        diarization_hybrid_window_seconds=pick_float(
            "diarization_hybrid_window_seconds", current.diarization_hybrid_window_seconds
        ),
        diarization_hybrid_overlap_seconds=pick_float(
            "diarization_hybrid_overlap_seconds", current.diarization_hybrid_overlap_seconds
        ),
        diarization_hybrid_min_speaker_seconds=pick_float(
            "diarization_hybrid_min_speaker_seconds",
            current.diarization_hybrid_min_speaker_seconds,
        ),
        diarization_hybrid_linkage=pick_nonempty(
            "diarization_hybrid_linkage", current.diarization_hybrid_linkage
        ).casefold(),
        diarization_hybrid_threshold=pick_float(
            "diarization_hybrid_threshold", current.diarization_hybrid_threshold
        ),
        gigaam_model=pick_str("gigaam_model", current.gigaam_model),
        gigaam_model_path=pick_str("gigaam_model_path", current.gigaam_model_path),
        gigaam_quantization=pick_str("gigaam_quantization", current.gigaam_quantization),
        gigaam_vad=pick_bool("gigaam_vad", current.gigaam_vad),
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

    if settings.asr_backend not in VALID_ASR_BACKENDS:
        raise SettingsError(
            f"Неизвестный бэкенд распознавания: {settings.asr_backend!r} "
            f"(допустимо: {', '.join(VALID_ASR_BACKENDS)})"
        )
    if settings.device not in VALID_DEVICES:
        raise SettingsError(
            f"Неизвестное устройство: {settings.device!r} (допустимо: {', '.join(VALID_DEVICES)})"
        )

    settings.llm_provider = settings.llm_provider.strip().casefold()
    settings.llm_base_url = settings.llm_base_url.strip()
    settings.llm_model_name = settings.llm_model_name.strip()
    if settings.llm_provider not in VALID_LLM_PROVIDERS:
        raise SettingsError(
            f"Неизвестный провайдер LLM: {settings.llm_provider!r} "
            f"(допустимо: {', '.join(VALID_LLM_PROVIDERS)})"
        )

    settings.diarization_engine = settings.diarization_engine.strip().casefold()
    if settings.diarization_engine not in VALID_DIARIZATION_ENGINES:
        raise SettingsError(
            f"Неизвестный движок диаризации: {settings.diarization_engine!r} "
            f"(допустимо: {', '.join(VALID_DIARIZATION_ENGINES)})"
        )
    settings.nemo_speech_device = settings.nemo_speech_device.strip().casefold()
    if settings.nemo_speech_device not in VALID_NEMO_SPEECH_DEVICES:
        raise SettingsError(
            f"Неизвестное устройство NeMo-Speech.cpp: {settings.nemo_speech_device!r} "
            f"(допустимо: {', '.join(VALID_NEMO_SPEECH_DEVICES)})"
        )
    settings.nemo_speech_binary = settings.nemo_speech_binary.strip() or DEFAULT_NEMO_SPEECH_BINARY
    settings.nemo_speech_model = settings.nemo_speech_model.strip() or DEFAULT_NEMO_SPEECH_MODEL
    settings.nemo_speech_lib_path = settings.nemo_speech_lib_path.strip()

    settings.deep_filter_binary = (
        settings.deep_filter_binary.strip() or DEFAULT_DEEP_FILTER_BINARY
    )

    if settings.sentence_merge_max_gap <= 0.0:
        raise SettingsError("SENTENCE_MERGE_MAX_GAP должно быть положительным числом")

    if not isinstance(settings.word_timestamps, bool):
        raise SettingsError("WORD_TIMESTAMPS должно быть true или false")

    if not isinstance(settings.diarization_estimate_enabled, bool):
        raise SettingsError("DIARIZATION_ESTIMATE_ENABLED должно быть true или false")
    if settings.diarization_estimate_seconds <= 0.0:
        raise SettingsError("DIARIZATION_ESTIMATE_SECONDS должно быть положительным числом")
    if not (0.0 < settings.diarization_estimate_threshold < 2.0):
        raise SettingsError(
            "DIARIZATION_ESTIMATE_THRESHOLD должно быть числом в диапазоне (0; 2)"
        )
    settings.diarization_estimate_model = (
        settings.diarization_estimate_model.strip() or DEFAULT_DIARIZATION_ESTIMATE_MODEL
    )
    if settings.diarization_route_max_speakers < 1:
        raise SettingsError("DIARIZATION_ROUTE_MAX_SPEAKERS должно быть целым числом >= 1")

    if not isinstance(settings.diarization_hybrid_enabled, bool):
        raise SettingsError("DIARIZATION_HYBRID_ENABLED должно быть true или false")
    if settings.diarization_hybrid_window_seconds <= 0.0:
        raise SettingsError(
            "DIARIZATION_HYBRID_WINDOW_SECONDS должно быть положительным числом"
        )
    if settings.diarization_hybrid_overlap_seconds < 0.0:
        raise SettingsError(
            "DIARIZATION_HYBRID_OVERLAP_SECONDS не может быть отрицательным"
        )
    if (
        settings.diarization_hybrid_overlap_seconds
        >= settings.diarization_hybrid_window_seconds
    ):
        raise SettingsError(
            "DIARIZATION_HYBRID_OVERLAP_SECONDS должно быть меньше "
            "DIARIZATION_HYBRID_WINDOW_SECONDS"
        )
    if settings.diarization_hybrid_min_speaker_seconds <= 0.0:
        raise SettingsError(
            "DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS должно быть положительным числом"
        )
    if settings.diarization_hybrid_linkage not in VALID_DIARIZATION_LINKAGES:
        raise SettingsError(
            "DIARIZATION_HYBRID_LINKAGE должно быть одним из "
            f"{', '.join(VALID_DIARIZATION_LINKAGES)}"
        )
    if not (0.0 < settings.diarization_hybrid_threshold < 2.0):
        raise SettingsError(
            "DIARIZATION_HYBRID_THRESHOLD должно быть числом в диапазоне (0; 2)"
        )

    if settings.llm_base_url:
        validate_llm_base_url(settings.llm_base_url)

    paths = (
        ("glossary_db", settings.glossary_db),
        ("voices_dir", settings.voices_dir),
        ("whisper_cpp_model", settings.whisper_cpp_model),
        ("whisper_cpp_lib_path", settings.whisper_cpp_lib_path),
        ("llm_model", settings.llm_model),
        ("llm_lib_path", settings.llm_lib_path),
        ("pyannote_local_model", settings.pyannote_local_model),
        ("gigaam_model_path", settings.gigaam_model_path),
        ("nemo_speech_binary", settings.nemo_speech_binary),
        ("nemo_speech_lib_path", settings.nemo_speech_lib_path),
    )
    for label, value in paths:
        if not value.strip():
            continue
        _validate_path(label, value)


def validate_llm_base_url(value: str) -> None:
    """Проверяет base_url внешней LLM (SSRF, #87).

    Разрешён только абсолютный http(s)-URL с хостом. Дополнительно
    отклоняются link-local адреса (``169.254.0.0/16``, ``fe80::/10``), которые
    в локальном приложении не бывают легитимным LLM-сервером, но часто служат
    целью SSRF (метаданные облака). Loopback/приватные адреса разрешены —
    Ollama/llama.cpp/LM Studio обычно живут там.

    :raises SettingsError: если значение не является допустимым URL.
    """
    if not value:
        return
    if not _is_http_url(value):
        raise SettingsError(
            f"Некорректный LLM_BASE_URL: {value!r} "
            "(ожидается http(s)://host[:port][/path])"
        )
    host = urlparse(value).hostname or ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if address.is_link_local:
        raise SettingsError(
            f"Недопустимый адрес LLM_BASE_URL: {host!r} "
            "(link-local/метаданные облака запрещены)"
        )


def _is_http_url(value: str) -> bool:
    """Похоже ли значение на абсолютный http(s)-URL с хостом."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
        if stripped in TRUE_VALUES:
            return True
        if stripped in FALSE_VALUES:
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
