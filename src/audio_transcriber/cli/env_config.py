"""Значения :class:`AppConfig` из ``config.env`` для CLI (issue #76).

CLI без явных флагов должен вести себя ровно так же, как ``config.env``
(единый источник настроек, как у веба через
:func:`audio_transcriber.web.config.build_job_config`), а флаги — только
переопределять отдельные настройки. Модуль собирает kwargs для
:class:`AppConfig` из словаря ``config.env``: отсутствующая или пустая
настройка означает «значение по умолчанию :class:`AppConfig`» (для
bool-настроек это ``true`` — как в ``config.example.env``).

Разбор намеренно мягкий (как в вебе): некорректное значение не роняет CLI, а
трактуется как «настройка не задана» — дальше действует дефолт или флаг.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
    DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_DIARIZATION_MIN_DURATION_OFF,
    DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS,
    DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS,
    DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS,
    DEFAULT_LLM_REQUEST_TIMEOUT,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_REFERENCE_PREPARE,
    DEFAULT_REFERENCE_TARGET_DBFS,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.llm.client import DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "да"})


def as_bool(raw: str | None, default: bool = False) -> bool:
    """Разбирает булево значение ``config.env`` (``true/1/yes/on/да``)."""
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in _TRUE_VALUES


def as_int(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip().isdigit():
        return default
    return int(raw.strip())


def as_float(raw: str | None, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def as_optional_float(raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


def as_optional_int(raw: str | None) -> int | None:
    if raw is None or not raw.strip().isdigit():
        return None
    return int(raw.strip())


def env_export_formats(env: Mapping[str, str]) -> tuple[ExportFormat, ...] | None:
    """Форматы экспорта из ``EXPORT_FORMATS`` (веб) или ``FORMATS`` (CLI/TUI)."""
    raw = (env.get("EXPORT_FORMATS") or env.get("FORMATS") or "").strip()
    if not raw:
        return None
    formats: list[ExportFormat] = []
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            formats.append(ExportFormat(value))
        except ValueError:
            continue
    deduplicated = tuple(dict.fromkeys(formats))
    return deduplicated or None


def env_backend(env: Mapping[str, str]) -> AsrBackend:
    raw = (env.get("ASR_BACKEND") or "").strip().casefold()
    try:
        return AsrBackend(raw)
    except ValueError:
        return AsrBackend.FASTER_WHISPER


def env_device(env: Mapping[str, str]) -> Device:
    raw = (env.get("DEVICE") or "").strip().casefold()
    try:
        return Device(raw)
    except ValueError:
        return Device.AUTO


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def collect_env_kwargs(
    env: Mapping[str, str],
    *,
    input_file: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Собирает kwargs :class:`AppConfig` из настроек ``config.env``.

    Возвращает только те поля, для которых в ``config.env`` задано непустое
    значение. Остальные поля остаются на дефолтах :class:`AppConfig`, которые
    совпадают с дефолтами CLI-опций, — поэтому «пустой» ``config.env`` даёт
    прежнее поведение CLI.
    """
    kwargs: dict[str, object] = {"input_file": input_file, "output_dir": output_dir}

    def val(key: str) -> str:
        return (env.get(key) or "").strip()

    def has(key: str) -> bool:
        return bool(val(key))

    def put(field: str, value: object) -> None:
        kwargs[field] = value

    # --- ASR/модель ---------------------------------------------------------
    if has("MODEL"):
        put("model_name", val("MODEL"))
    if has("ASR_BACKEND"):
        put("asr_backend", env_backend(env))
    if has("WHISPER_CPP_MODEL"):
        put("whisper_cpp_model", Path(val("WHISPER_CPP_MODEL")))
    if has("WHISPER_CPP_BINARY"):
        put("whisper_cpp_binary", val("WHISPER_CPP_BINARY"))
    if has("WHISPER_CPP_LIB_PATH"):
        put("whisper_cpp_lib_path", val("WHISPER_CPP_LIB_PATH"))
    if has("WHISPER_CPP_THREADS"):
        put("whisper_cpp_threads", as_optional_int(val("WHISPER_CPP_THREADS")))

    # --- GigaAM (#46) -------------------------------------------------------
    if has("GIGAAM_MODEL"):
        put("gigaam_model", val("GIGAAM_MODEL"))
    if has("GIGAAM_MODEL_PATH"):
        put("gigaam_model_path", Path(val("GIGAAM_MODEL_PATH")))
    if has("GIGAAM_QUANTIZATION"):
        put("gigaam_quantization", val("GIGAAM_QUANTIZATION"))
    if has("GIGAAM_VAD"):
        put("gigaam_vad", as_bool(val("GIGAAM_VAD")))

    # --- Гибридный ASR (#57) ------------------------------------------------
    if has("HYBRID_ASR"):
        put("hybrid_asr", as_bool(val("HYBRID_ASR")))
    if has("HYBRID_FALLBACK_BACKEND"):
        raw = val("HYBRID_FALLBACK_BACKEND").casefold()
        if raw in {item.value for item in AsrBackend}:
            put("hybrid_fallback_backend", AsrBackend(raw))
    if has("HYBRID_LOW_LOGPROB_THRESHOLD"):
        put("hybrid_low_logprob_threshold", as_float(
            val("HYBRID_LOW_LOGPROB_THRESHOLD"), -1.0
        ))
    if has("HYBRID_NO_SPEECH_THRESHOLD"):
        put("hybrid_no_speech_threshold", as_float(val("HYBRID_NO_SPEECH_THRESHOLD"), 0.6))
    if has("HYBRID_SILENCE_RMS_THRESHOLD"):
        put("hybrid_silence_rms_threshold", as_float(
            val("HYBRID_SILENCE_RMS_THRESHOLD"), 0.003
        ))
    if has("HYBRID_MIN_SEGMENT_SECONDS"):
        put("hybrid_min_segment_seconds", as_float(
            val("HYBRID_MIN_SEGMENT_SECONDS"), 0.5
        ))
    if has("HYBRID_CONTEXT_SECONDS"):
        put("hybrid_context_seconds", as_float(val("HYBRID_CONTEXT_SECONDS"), 0.4))

    # --- Язык/устройство/форматы -------------------------------------------
    if has("LANGUAGE"):
        put("language", val("LANGUAGE"))
    if has("DEVICE"):
        put("device", env_device(env))
    formats = env_export_formats(env)
    if formats is not None:
        put("export_formats", formats)

    # --- Диаризация: число говорящих и гиперпараметры -----------------------
    if has("NUM_SPEAKERS"):
        num_speakers = as_optional_int(val("NUM_SPEAKERS"))
        if num_speakers is not None:
            put("num_speakers", num_speakers)
    if has("MIN_SPEAKERS"):
        min_speakers = as_optional_int(val("MIN_SPEAKERS"))
        if min_speakers is not None:
            put("min_speakers", min_speakers)
    if has("MAX_SPEAKERS"):
        max_speakers = as_optional_int(val("MAX_SPEAKERS"))
        if max_speakers is not None:
            put("max_speakers", max_speakers)
    if has("DIARIZATION_ENABLED"):
        put("diarization_enabled", as_bool(val("DIARIZATION_ENABLED"), default=True))
    if has("DIARIZATION_MIN_DURATION_OFF"):
        put("diarization_min_duration_off", as_float(
            val("DIARIZATION_MIN_DURATION_OFF"), DEFAULT_DIARIZATION_MIN_DURATION_OFF
        ))
    if has("DIARIZATION_CLUSTERING_THRESHOLD"):
        put("diarization_clustering_threshold", as_optional_float(
            val("DIARIZATION_CLUSTERING_THRESHOLD")
        ))
    if has("DIARIZATION_CLUSTERING_FB"):
        put("diarization_clustering_fb", as_optional_float(val("DIARIZATION_CLUSTERING_FB")))

    # --- Движок диаризации и NeMo-Speech.cpp (#62) -------------------------
    if has("DIARIZATION_ENGINE"):
        put("diarization_engine", val("DIARIZATION_ENGINE").casefold())
    if has("NEMO_SPEECH_BINARY"):
        put("nemo_speech_binary", val("NEMO_SPEECH_BINARY"))
    if has("NEMO_SPEECH_LIB_PATH"):
        put("nemo_speech_lib_path", val("NEMO_SPEECH_LIB_PATH"))
    if has("NEMO_SPEECH_MODEL"):
        put("nemo_speech_model", val("NEMO_SPEECH_MODEL"))
    if has("NEMO_SPEECH_DEVICE"):
        put("nemo_speech_device", val("NEMO_SPEECH_DEVICE").casefold())

    # --- Оценщик числа говорящих и маршрутизация auto (#64) ----------------
    if has("DIARIZATION_ESTIMATE_ENABLED"):
        put("diarization_estimate_enabled", as_bool(
            val("DIARIZATION_ESTIMATE_ENABLED"), default=True
        ))
    if has("DIARIZATION_ESTIMATE_SECONDS"):
        put("diarization_estimate_seconds", as_float(
            val("DIARIZATION_ESTIMATE_SECONDS"), DEFAULT_DIARIZATION_ESTIMATE_SECONDS
        ))
    if has("DIARIZATION_ESTIMATE_THRESHOLD"):
        put("diarization_estimate_threshold", as_float(
            val("DIARIZATION_ESTIMATE_THRESHOLD"), DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD
        ))
    if has("DIARIZATION_ESTIMATE_MODEL"):
        put("diarization_estimate_model", val("DIARIZATION_ESTIMATE_MODEL"))
    if has("DIARIZATION_ROUTE_MAX_SPEAKERS"):
        put("diarization_route_max_speakers", as_int(
            val("DIARIZATION_ROUTE_MAX_SPEAKERS"), DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS
        ))

    # --- Гибридная диаризация (#64, часть 2) --------------------------------
    if has("DIARIZATION_HYBRID_ENABLED"):
        put("diarization_hybrid_enabled", as_bool(
            val("DIARIZATION_HYBRID_ENABLED"), default=True
        ))
    if has("DIARIZATION_HYBRID_WINDOW_SECONDS"):
        put("diarization_hybrid_window_seconds", as_float(
            val("DIARIZATION_HYBRID_WINDOW_SECONDS"), DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS
        ))
    if has("DIARIZATION_HYBRID_OVERLAP_SECONDS"):
        put("diarization_hybrid_overlap_seconds", as_float(
            val("DIARIZATION_HYBRID_OVERLAP_SECONDS"),
            DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
        ))
    if has("DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS"):
        put("diarization_hybrid_min_speaker_seconds", as_float(
            val("DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS"),
            DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
        ))
    if has("DIARIZATION_HYBRID_LINKAGE"):
        put("diarization_hybrid_linkage", val("DIARIZATION_HYBRID_LINKAGE").casefold())
    if has("DIARIZATION_HYBRID_THRESHOLD"):
        put("diarization_hybrid_threshold", as_float(
            val("DIARIZATION_HYBRID_THRESHOLD"), DEFAULT_DIARIZATION_HYBRID_THRESHOLD
        ))
    if has("DIARIZATION_HYBRID_OVERLOAD_SPLIT"):
        put("diarization_hybrid_overload_split", as_bool(
            val("DIARIZATION_HYBRID_OVERLOAD_SPLIT"), default=True
        ))
    if has("DIARIZATION_HYBRID_SUBWINDOW_SECONDS"):
        put("diarization_hybrid_subwindow_seconds", as_float(
            val("DIARIZATION_HYBRID_SUBWINDOW_SECONDS"),
            DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
        ))
    if has("DIARIZATION_HYBRID_MAX_SPLIT_DEPTH"):
        put("diarization_hybrid_max_split_depth", as_int(
            val("DIARIZATION_HYBRID_MAX_SPLIT_DEPTH"), DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH
        ))

    # --- HF-токен и локальная модель диаризации -----------------------------
    if has("HF_TOKEN"):
        put("hf_token", val("HF_TOKEN"))
    if has("PYANNOTE_LOCAL_MODEL"):
        put("pyannote_local_model", Path(val("PYANNOTE_LOCAL_MODEL")))

    # --- Имена/образцы говорящих -------------------------------------------
    if has("SPEAKER_NAMES"):
        put("speaker_names", AppConfig.parse_speaker_names(_split_list(val("SPEAKER_NAMES"))))
    if has("SPEAKER_REFERENCES"):
        put(
            "speaker_references",
            AppConfig.parse_speaker_references(_split_list(val("SPEAKER_REFERENCES"))),
        )
    if has("ENROLLMENT_MIN_SIMILARITY"):
        put("enrollment_min_similarity", as_float(val("ENROLLMENT_MIN_SIMILARITY"), 0.6))
    if has("VOICES_DIR"):
        put("voices_dir", Path(val("VOICES_DIR")))
    if has("EXPORT_SPEAKER_SAMPLES"):
        put("export_speaker_samples", as_bool(val("EXPORT_SPEAKER_SAMPLES"), default=True))

    # --- Подготовка эталона голоса (#29) ------------------------------------
    if has("REFERENCE_PREPARE"):
        put("reference_prepare", as_bool(val("REFERENCE_PREPARE"), default=DEFAULT_REFERENCE_PREPARE))
    if has("ENROLLMENT_MIN_SAMPLE_SECONDS"):
        put("enrollment_min_sample_seconds", as_float(
            val("ENROLLMENT_MIN_SAMPLE_SECONDS"), DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS
        ))
    if has("ENROLLMENT_MAX_SAMPLE_SECONDS"):
        put("enrollment_max_sample_seconds", as_float(
            val("ENROLLMENT_MAX_SAMPLE_SECONDS"), DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS
        ))
    if has("REFERENCE_TARGET_DBFS"):
        put("reference_target_dbfs", as_float(
            val("REFERENCE_TARGET_DBFS"), DEFAULT_REFERENCE_TARGET_DBFS
        ))

    # --- LLM-постобработка --------------------------------------------------
    if has("LLM_ENABLED"):
        put("llm_enabled", as_bool(val("LLM_ENABLED")))
    if has("LLM_PROVIDER"):
        put("llm_provider", val("LLM_PROVIDER").casefold())
    if has("LLM_BASE_URL"):
        put("llm_base_url", val("LLM_BASE_URL"))
    if has("LLM_MODEL_NAME"):
        put("llm_model_name", val("LLM_MODEL_NAME"))
    if has("LLM_API_KEY"):
        put("llm_api_key", val("LLM_API_KEY"))
    if has("LLM_MODEL"):
        put("llm_model", Path(val("LLM_MODEL")))
    if has("LLM_BINARY"):
        put("llm_binary", val("LLM_BINARY"))
    if has("LLM_LIB_PATH"):
        put("llm_lib_path", val("LLM_LIB_PATH"))
    if has("LLM_GPU"):
        put("llm_gpu", as_bool(val("LLM_GPU"), default=True))
    if has("LLM_CONTEXT"):
        put("llm_context_size", as_int(val("LLM_CONTEXT"), DEFAULT_LLM_CONTEXT_SIZE))
    if has("LLM_REQUEST_TIMEOUT"):
        put("llm_request_timeout", as_float(
            val("LLM_REQUEST_TIMEOUT"), DEFAULT_LLM_REQUEST_TIMEOUT
        ))
    if has("LLM_EXTRACT_NAMES"):
        put("llm_extract_names", as_bool(val("LLM_EXTRACT_NAMES")))
    if has("LLM_SUGGEST_TERMS"):
        put("llm_suggest_terms", as_bool(val("LLM_SUGGEST_TERMS")))
    if has("LLM_SUMMARY"):
        put("llm_summary", as_bool(val("LLM_SUMMARY"), default=True))
    if has("LLM_PROMPT_EXTRA"):
        put("llm_prompt_extra", val("LLM_PROMPT_EXTRA"))
    if has("LLM_PROMPT_FILE"):
        put("llm_prompt_file", Path(val("LLM_PROMPT_FILE")))

    # --- Глоссарий ----------------------------------------------------------
    if has("GLOSSARY_PATH"):
        put("glossary_path", normalize_glossary_paths_tuple(val("GLOSSARY_PATH")))
    if has("GLOSSARY_DB"):
        put("glossary_db", Path(val("GLOSSARY_DB")))
    if has("GLOSSARY_ENABLED"):
        put("glossary_enabled", as_bool(val("GLOSSARY_ENABLED"), default=True))

    # --- Артефакты/очистка/качество -----------------------------------------
    if has("CLEAN_ARTIFACTS"):
        put("clean_artifacts", as_bool(val("CLEAN_ARTIFACTS"), default=True))
    if has("COLLAPSE_REPEATS"):
        put("collapse_repeats", as_bool(val("COLLAPSE_REPEATS"), default=True))
    if has("REPEAT_MIN_WORDS"):
        put("repeat_min_words", as_int(val("REPEAT_MIN_WORDS"), 4))
    if has("REPEAT_SIMILARITY"):
        put("repeat_similarity", as_float(val("REPEAT_SIMILARITY"), 0.9))
    if has("NORMALIZE_TEXT"):
        put("normalize_text", as_bool(val("NORMALIZE_TEXT"), default=True))
    if has("MARK_OVERLAP"):
        put("mark_overlap", as_bool(val("MARK_OVERLAP"), default=True))
    if has("MERGE_SAME_NAME_SPEAKERS"):
        put("merge_same_name_speakers", as_bool(
            val("MERGE_SAME_NAME_SPEAKERS"), default=True
        ))
    if has("LOW_CONFIDENCE_THRESHOLD"):
        put("low_confidence_threshold", as_float(
            val("LOW_CONFIDENCE_THRESHOLD"), DEFAULT_LOW_CONFIDENCE_THRESHOLD
        ))
    if has("DENOISE"):
        put("denoise", as_bool(val("DENOISE"), default=True))
    if has("USE_CACHE"):
        put("use_cache", as_bool(val("USE_CACHE"), default=True))
    if has("CACHE_DIR"):
        put("cache_dir", Path(val("CACHE_DIR")))
    if has("NOTIFICATIONS"):
        put("notifications", as_bool(val("NOTIFICATIONS"), default=True))
    if has("TIMELINE"):
        put("timeline", as_bool(val("TIMELINE"), default=True))
    if has("PROTOCOL_AUTO"):
        put("protocol_auto", as_bool(val("PROTOCOL_AUTO"), default=True))
    if has("ENABLE_CORRECTION"):
        put("enable_correction", as_bool(val("ENABLE_CORRECTION")))
    if has("CORRECTION_MIN_WORD_LENGTH"):
        put("correction_min_word_length", as_int(
            val("CORRECTION_MIN_WORD_LENGTH"), DEFAULT_CORRECTION_MIN_WORD_LENGTH
        ))
    if has("CORRECTION_MIN_SIMILARITY"):
        put("correction_min_similarity", as_float(
            val("CORRECTION_MIN_SIMILARITY"), DEFAULT_CORRECTION_MIN_SIMILARITY
        ))
    if has("CORRECTION_MAX_CANDIDATES"):
        put("correction_max_candidates", as_int(
            val("CORRECTION_MAX_CANDIDATES"), DEFAULT_CORRECTION_MAX_CANDIDATES
        ))
    if has("VERBOSE"):
        put("verbose", as_bool(val("VERBOSE")))

    return kwargs
