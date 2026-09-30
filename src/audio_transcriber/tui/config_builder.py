"""Сборка :class:`AppConfig` из значений виджетов TUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Checkbox, Input, Select, Switch

from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.llm.client import (
    DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE,
)
from audio_transcriber.tui.formatting import _to_bool, _to_float, _to_int
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple

if TYPE_CHECKING:
    from audio_transcriber.tui.app import TranscriberApp


def build_config_from_widgets(app: TranscriberApp, input_file: Path) -> AppConfig:
    """Собирает :class:`AppConfig` из значений виджетов TUI.

    Вынесено из :meth:`TranscriberApp._build_config`, чтобы сборка конфигурации
    не разрасталась внутри приложения.
    """
    num_speakers_raw = app.query_one("#num_speakers", Input).value.strip()
    num_speakers = int(num_speakers_raw) if num_speakers_raw else None

    formats_value = app.query_one("#formats", Select).value
    formats_raw = formats_value if isinstance(formats_value, str) else "txt"
    export_formats = tuple(
        dict.fromkeys(ExportFormat(f.strip()) for f in formats_raw.split(",") if f.strip())
    )

    enable_correction = app.query_one("#correction", Switch).value
    clean_artifacts = app.query_one("#clean", Switch).value
    collapse_repeats = app.query_one("#collapse_repeats", Switch).value
    normalize_text = app.query_one("#normalize", Switch).value
    mark_overlap = app.query_one("#overlap", Switch).value
    diarization_enabled = app.query_one("#diarization", Switch).value
    denoise_enabled = app.query_one("#denoise", Switch).value
    use_cache = app.query_one("#cache", Switch).value
    notifications_enabled = app.query_one("#notifications", Switch).value
    timeline_enabled = app.query_one("#timeline", Switch).value

    backend_raw = str(app.query_one("#backend", Select).value or "")
    backend = (
        AsrBackend(backend_raw)
        if backend_raw in {item.value for item in AsrBackend}
        else AsrBackend.FASTER_WHISPER
    )
    device_raw = str(app.query_one("#device", Select).value or "")
    device = Device(device_raw) if device_raw in {item.value for item in Device} else Device.AUTO

    model = app.query_one("#model", Input).value.strip() or "large-v3-turbo"
    wcp_model = app.query_one("#wcp_model", Input).value.strip()
    wcp_binary = app.query_one("#wcp_binary", Input).value.strip() or "whisper-cli"
    wcp_lib = app.query_one("#wcp_lib", Input).value.strip()

    speaker_names_raw = app.query_one("#speaker_names", Input).value.strip()
    speaker_names = (
        AppConfig.parse_speaker_names(
            [p.strip() for p in speaker_names_raw.split(",") if p.strip()]
        )
        if speaker_names_raw
        else {}
    )

    speaker_references_raw = app.query_one("#speaker_references", Input).value.strip()
    speaker_references = (
        AppConfig.parse_speaker_references(
            [p.strip() for p in speaker_references_raw.split(",") if p.strip()]
        )
        if speaker_references_raw
        else {}
    )
    enrollment_min_raw = app.query_one("#enrollment_min_similarity", Input).value.strip()
    enrollment_min_similarity = _to_float(
        enrollment_min_raw or None, DEFAULT_ENROLLMENT_MIN_SIMILARITY
    )
    export_speaker_samples = app.query_one("#speaker_samples", Switch).value
    voices_dir_raw = app.query_one("#voices_dir", Input).value.strip()
    voices_dir = Path(voices_dir_raw) if voices_dir_raw else None

    hotwords = app.query_one("#hotwords", Input).value.strip() or None
    output_dir = app.query_one("#output_dir", Input).value.strip() or "output"

    llm_enabled = app.query_one("#llm", Switch).value
    llm_extract_names = app.query_one("#llm_names", Switch).value
    llm_summary = app.query_one("#llm_summary", Switch).value
    llm_model = app.query_one("#llm_model", Input).value.strip()
    llm_binary = app.query_one("#llm_binary", Input).value.strip() or "llama-server"
    llm_lib = app.query_one("#llm_lib", Input).value.strip()
    llm_prompt_extra = app.query_one("#llm_prompt_extra", Input).value.strip() or None
    llm_prompt_file_raw = app.query_one("#llm_prompt_file", Input).value.strip()
    llm_prompt_file = Path(llm_prompt_file_raw) if llm_prompt_file_raw else None
    glossary_path = app.query_one("#glossary_path", Input).value.strip()
    glossary_enabled = app.query_one("#glossary_enabled", Checkbox).value
    glossary_db_raw = app.query_one("#glossary_db", Input).value.strip()
    glossary_db = Path(glossary_db_raw) if glossary_db_raw else None

    language_value = app.query_one("#language", Select).value
    language = language_value if isinstance(language_value, str) and language_value else None

    defaults = app._defaults
    pyannote_local_raw = defaults.get("PYANNOTE_LOCAL_MODEL", "").strip()
    pyannote_local_model = Path(pyannote_local_raw) if pyannote_local_raw else None

    wcp_threads_raw = defaults.get("WHISPER_CPP_THREADS", "").strip()
    whisper_cpp_threads = int(wcp_threads_raw) if wcp_threads_raw.isdigit() else None

    low_conf_raw = app.query_one("#low_conf", Input).value.strip()
    low_confidence_threshold = _to_float(low_conf_raw or None, DEFAULT_LOW_CONFIDENCE_THRESHOLD)

    cache_dir_raw = defaults.get("CACHE_DIR", "").strip()
    cache_dir = Path(cache_dir_raw) if cache_dir_raw else None

    return AppConfig(
        input_file=input_file,
        output_dir=Path(output_dir),
        model_name=model,
        language=language,
        device=device,
        export_formats=export_formats,
        num_speakers=num_speakers,
        diarization_enabled=diarization_enabled,
        speaker_names=speaker_names,
        speaker_references=speaker_references,
        enrollment_min_similarity=enrollment_min_similarity,
        voices_dir=voices_dir,
        export_speaker_samples=export_speaker_samples,
        hf_token=defaults.get("HF_TOKEN") or None,
        pyannote_local_model=pyannote_local_model,
        initial_prompt=defaults.get("INITIAL_PROMPT") or None,
        hotwords=hotwords,
        clean_artifacts=clean_artifacts,
        collapse_repeats=collapse_repeats,
        repeat_min_words=_to_int(defaults.get("REPEAT_MIN_WORDS"), DEFAULT_REPEAT_MIN_WORDS),
        repeat_similarity=_to_float(defaults.get("REPEAT_SIMILARITY"), DEFAULT_REPEAT_SIMILARITY),
        normalize_text=normalize_text,
        denoise=denoise_enabled,
        mark_overlap=mark_overlap,
        use_cache=use_cache,
        cache_dir=cache_dir,
        notifications=notifications_enabled,
        timeline=timeline_enabled,
        low_confidence_threshold=low_confidence_threshold,
        enable_correction=enable_correction,
        correction_min_word_length=_to_int(
            defaults.get("CORRECTION_MIN_WORD_LENGTH"),
            DEFAULT_CORRECTION_MIN_WORD_LENGTH,
        ),
        correction_min_similarity=_to_float(
            defaults.get("CORRECTION_MIN_SIMILARITY"),
            DEFAULT_CORRECTION_MIN_SIMILARITY,
        ),
        correction_max_candidates=_to_int(
            defaults.get("CORRECTION_MAX_CANDIDATES"),
            DEFAULT_CORRECTION_MAX_CANDIDATES,
        ),
        verbose=_to_bool(defaults.get("VERBOSE")),
        asr_backend=backend,
        whisper_cpp_model=Path(wcp_model) if wcp_model else None,
        whisper_cpp_binary=wcp_binary,
        whisper_cpp_lib_path=wcp_lib or None,
        whisper_cpp_threads=whisper_cpp_threads,
        llm_enabled=llm_enabled,
        llm_model=Path(llm_model) if llm_model else None,
        llm_binary=llm_binary,
        llm_lib_path=llm_lib or None,
        llm_gpu=_to_bool(defaults.get("LLM_GPU"), default=True),
        llm_context_size=_to_int(defaults.get("LLM_CONTEXT"), DEFAULT_LLM_CONTEXT_SIZE),
        llm_suggest_terms=_to_bool(defaults.get("LLM_SUGGEST_TERMS")),
        llm_extract_names=llm_extract_names,
        llm_summary=llm_summary,
        llm_prompt_extra=llm_prompt_extra,
        llm_prompt_file=llm_prompt_file,
        glossary_path=normalize_glossary_paths_tuple(glossary_path or None),
        glossary_db=glossary_db,
        glossary_enabled=glossary_enabled,
        # TUI не пишет протокол автоматически: пользователь сначала проверяет
        # стенограмму и правит имена, а протокол собирает кнопкой.
        protocol_auto=False,
    )
