"""Сборка :class:`AppConfig` из значений виджетов TUI.

Базовый слой настроек берётся из того же единого источника, что и у CLI, —
:func:`audio_transcriber.cli.env_config.collect_env_kwargs` (``config.env``),
поверх него накладываются значения виджетов. Благодаря этому TUI автоматически
переносит все настройки, для которых нет отдельных виджетов (``NEMO_SPEECH_*``,
``DIARIZATION_*``/HYBRID/ESTIMATE, ``WORD_TIMESTAMPS``,
``MERGE_SAME_NAME_SPEAKERS``, ``LLM_*``, ``INITIAL_PROMPT`` и т.д.), вместо
того чтобы дублировать список полей (issue #90).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Checkbox, Input, Select, Switch

from audio_transcriber.cli.env_config import collect_env_kwargs
from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_SENTENCE_MERGE_MAX_GAP,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.tui.formatting import _to_float
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple

if TYPE_CHECKING:
    from audio_transcriber.tui.app import TranscriberApp


def build_config_from_widgets(app: TranscriberApp, input_file: Path) -> AppConfig:
    """Собирает :class:`AppConfig` из значений виджетов TUI.

    Значения ``config.env`` (единый источник, как у CLI/веба) служат базой;
    виджеты задают явные override. Вынесено из
    :meth:`TranscriberApp._build_config`, чтобы сборка конфигурации не
    разрасталась внутри приложения.
    """
    defaults = app._defaults
    output_dir = app.query_one("#output_dir", Input).value.strip() or "output"

    # База — настройки config.env, разобранные общим с CLI кодом. Отсутствующие
    # ключи остаются на дефолтах AppConfig.
    kwargs = collect_env_kwargs(
        defaults,
        input_file=input_file,
        output_dir=Path(output_dir),
    )

    num_speakers_raw = app.query_one("#num_speakers", Input).value.strip()
    # Пустой виджет не должен затирать значение из config.env (NUM_SPEAKERS).
    if num_speakers_raw:
        kwargs["num_speakers"] = int(num_speakers_raw)

    formats_value = app.query_one("#formats", Select).value
    formats_raw = formats_value if isinstance(formats_value, str) else "txt"
    kwargs["export_formats"] = tuple(
        dict.fromkeys(ExportFormat(f.strip()) for f in formats_raw.split(",") if f.strip())
    )

    backend_raw = str(app.query_one("#backend", Select).value or "")
    kwargs["asr_backend"] = (
        AsrBackend(backend_raw)
        if backend_raw in {item.value for item in AsrBackend}
        else AsrBackend.FASTER_WHISPER
    )
    device_raw = str(app.query_one("#device", Select).value or "")
    kwargs["device"] = (
        Device(device_raw) if device_raw in {item.value for item in Device} else Device.AUTO
    )

    kwargs["model_name"] = app.query_one("#model", Input).value.strip() or "large-v3-turbo"
    wcp_model = app.query_one("#wcp_model", Input).value.strip()
    kwargs["whisper_cpp_model"] = Path(wcp_model) if wcp_model else None
    kwargs["whisper_cpp_binary"] = (
        app.query_one("#wcp_binary", Input).value.strip() or "whisper-cli"
    )
    kwargs["whisper_cpp_lib_path"] = app.query_one("#wcp_lib", Input).value.strip() or None

    speaker_names_raw = app.query_one("#speaker_names", Input).value.strip()
    kwargs["speaker_names"] = (
        AppConfig.parse_speaker_names(
            [p.strip() for p in speaker_names_raw.split(",") if p.strip()]
        )
        if speaker_names_raw
        else {}
    )

    speaker_references_raw = app.query_one("#speaker_references", Input).value.strip()
    kwargs["speaker_references"] = (
        AppConfig.parse_speaker_references(
            [p.strip() for p in speaker_references_raw.split(",") if p.strip()]
        )
        if speaker_references_raw
        else {}
    )
    enrollment_min_raw = app.query_one("#enrollment_min_similarity", Input).value.strip()
    kwargs["enrollment_min_similarity"] = _to_float(
        enrollment_min_raw or None, DEFAULT_ENROLLMENT_MIN_SIMILARITY
    )
    kwargs["export_speaker_samples"] = app.query_one("#speaker_samples", Switch).value
    voices_dir_raw = app.query_one("#voices_dir", Input).value.strip()
    kwargs["voices_dir"] = Path(voices_dir_raw) if voices_dir_raw else None

    language_value = app.query_one("#language", Select).value
    kwargs["language"] = (
        language_value if isinstance(language_value, str) and language_value else None
    )
    kwargs["hotwords"] = app.query_one("#hotwords", Input).value.strip() or None

    kwargs["diarization_enabled"] = app.query_one("#diarization", Switch).value
    kwargs["denoise"] = app.query_one("#denoise", Switch).value
    kwargs["use_cache"] = app.query_one("#cache", Switch).value
    kwargs["notifications"] = app.query_one("#notifications", Switch).value
    kwargs["timeline"] = app.query_one("#timeline", Switch).value
    kwargs["clean_artifacts"] = app.query_one("#clean", Switch).value
    kwargs["collapse_repeats"] = app.query_one("#collapse_repeats", Switch).value
    kwargs["normalize_text"] = app.query_one("#normalize", Switch).value
    kwargs["mark_overlap"] = app.query_one("#overlap", Switch).value
    kwargs["enable_correction"] = app.query_one("#correction", Switch).value

    low_conf_raw = app.query_one("#low_conf", Input).value.strip()
    kwargs["low_confidence_threshold"] = _to_float(
        low_conf_raw or None, DEFAULT_LOW_CONFIDENCE_THRESHOLD
    )

    sentence_merge_gap_raw = app.query_one("#sentence_merge_max_gap", Input).value.strip()
    kwargs["sentence_merge_max_gap"] = _to_float(
        sentence_merge_gap_raw or None, DEFAULT_SENTENCE_MERGE_MAX_GAP
    )

    kwargs["llm_enabled"] = app.query_one("#llm", Switch).value
    kwargs["llm_extract_names"] = app.query_one("#llm_names", Switch).value
    kwargs["llm_summary"] = app.query_one("#llm_summary", Switch).value
    llm_model = app.query_one("#llm_model", Input).value.strip()
    kwargs["llm_model"] = Path(llm_model) if llm_model else None
    kwargs["llm_binary"] = app.query_one("#llm_binary", Input).value.strip() or "llama-server"
    kwargs["llm_lib_path"] = app.query_one("#llm_lib", Input).value.strip() or None
    kwargs["llm_prompt_extra"] = app.query_one("#llm_prompt_extra", Input).value.strip() or None
    llm_prompt_file_raw = app.query_one("#llm_prompt_file", Input).value.strip()
    kwargs["llm_prompt_file"] = Path(llm_prompt_file_raw) if llm_prompt_file_raw else None

    glossary_path = app.query_one("#glossary_path", Input).value.strip()
    kwargs["glossary_path"] = normalize_glossary_paths_tuple(glossary_path or None)
    glossary_db_raw = app.query_one("#glossary_db", Input).value.strip()
    kwargs["glossary_db"] = Path(glossary_db_raw) if glossary_db_raw else None
    kwargs["glossary_enabled"] = app.query_one("#glossary_enabled", Checkbox).value

    # TUI не пишет протокол автоматически: пользователь сначала проверяет
    # стенограмму и правит имена, а протокол собирает кнопкой.
    kwargs["protocol_auto"] = False

    return AppConfig(**kwargs)  # type: ignore[arg-type]
