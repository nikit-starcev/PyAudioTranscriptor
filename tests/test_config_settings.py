"""Тесты сборки и валидации конфигурации приложения (:class:`AppConfig`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.utils.exceptions import ConfigurationError


def test_valid_config_does_not_create_output_dir(tmp_path: Path, audio_file: Path) -> None:
    output_dir = tmp_path / "output" / "nested"

    config = AppConfig(input_file=audio_file, output_dir=output_dir)

    # Конструктор не выполняет I/O — каталог создаётся отдельно.
    assert not output_dir.exists()
    assert config.device is Device.AUTO
    assert config.export_formats == (ExportFormat.TXT,)


def test_ensure_output_dir_creates_directory(tmp_path: Path, audio_file: Path) -> None:
    output_dir = tmp_path / "output" / "nested"

    config = AppConfig(input_file=audio_file, output_dir=output_dir)
    config.ensure_output_dir()

    assert output_dir.is_dir()


def test_llm_enabled_without_model_warns(
    audio_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        AppConfig(input_file=audio_file, llm_enabled=True)

    assert any("модель не задана" in record.message for record in caplog.records)


def test_missing_input_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.mp3"

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=missing)


def test_input_file_must_not_be_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "a-directory"
    directory.mkdir()

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=directory)


def test_num_speakers_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, num_speakers=0)


def test_diarization_enabled_defaults_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).diarization_enabled is True


def test_diarization_can_be_disabled(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, diarization_enabled=False).diarization_enabled is False


def test_diarization_enabled_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_enabled="yes")  # type: ignore[arg-type]


def test_denoise_defaults_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).denoise is True


def test_denoise_can_be_disabled(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, denoise=False).denoise is False


def test_denoise_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, denoise="yes")  # type: ignore[arg-type]


def test_quality_features_defaults(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.collapse_repeats is True
    assert config.normalize_text is True
    assert config.mark_overlap is True
    assert config.repeat_min_words == 2
    assert config.repeat_similarity == pytest.approx(0.9)
    assert config.low_confidence_threshold == pytest.approx(-1.0)


def test_collapse_repeats_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, collapse_repeats="yes")  # type: ignore[arg-type]


def test_normalize_text_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, normalize_text="yes")  # type: ignore[arg-type]


def test_mark_overlap_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, mark_overlap="yes")  # type: ignore[arg-type]


def test_repeat_min_words_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, repeat_min_words=0)


def test_repeat_similarity_must_be_in_unit_interval(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, repeat_similarity=0.0)

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, repeat_similarity=1.5)


def test_low_confidence_threshold_must_be_non_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, low_confidence_threshold=0.5)

    assert (
        AppConfig(input_file=audio_file, low_confidence_threshold=-3.0).low_confidence_threshold
        == -3.0
    )


def test_at_least_one_export_format_required(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, export_formats=())


def test_correction_min_word_length_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, correction_min_word_length=0)


def test_correction_min_similarity_must_be_in_unit_interval(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, correction_min_similarity=0.0)

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, correction_min_similarity=1.5)


def test_correction_max_candidates_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, correction_max_candidates=0)


def test_llm_context_defaults_to_4096(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.llm_context_size == 4096
    assert config.llm_suggest_terms is False


def test_llm_context_must_be_at_least_128(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_context_size=64)


def test_llm_request_timeout_defaults_to_600(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.llm_request_timeout == 600.0


def test_llm_request_timeout_accepts_positive_value(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, llm_request_timeout=30.0)

    assert config.llm_request_timeout == 30.0


@pytest.mark.parametrize("value", [0.0, -5.0])
def test_llm_request_timeout_must_be_positive(audio_file: Path, value: float) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_request_timeout=value)


def test_llm_request_timeout_rejects_non_number(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_request_timeout="быстро")  # type: ignore[arg-type]


def test_llm_extract_names_defaults_to_false(audio_file: Path) -> None:
    # определение имён — экспериментальная функция, по умолчанию выключена
    assert AppConfig(input_file=audio_file).llm_extract_names is False


def test_llm_extract_names_accepts_explicit_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, llm_extract_names=True).llm_extract_names is True


def test_llm_extract_names_accepts_explicit_false(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, llm_extract_names=False).llm_extract_names is False


def test_llm_extract_names_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_extract_names="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["0=Иван"], {"SPEAKER_00": "Иван"}),
        (["0=Иван", "1=Мария"], {"SPEAKER_00": "Иван", "SPEAKER_01": "Мария"}),
        ([], {}),
    ],
)
def test_parse_speaker_names_valid(raw: list[str], expected: dict[str, str]) -> None:
    assert AppConfig.parse_speaker_names(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        ["без-разделителя"],
        ["x=Иван"],
        ["0="],
        ["0=   "],
    ],
)
def test_parse_speaker_names_invalid(raw: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig.parse_speaker_names(raw)


def test_glossary_path_single_path_normalized_to_tuple(tmp_path: Path, audio_file: Path) -> None:
    glossary = tmp_path / "glossary.txt"
    glossary.write_text("ОИБ\n", encoding="utf-8")

    config = AppConfig(input_file=audio_file, glossary_path=glossary)

    assert config.glossary_path == (glossary,)


def test_glossary_path_accepts_list_of_paths(tmp_path: Path, audio_file: Path) -> None:
    first = tmp_path / "tz.txt"
    second = tmp_path / "user.txt"
    first.write_text("ОИБ\n", encoding="utf-8")
    second.write_text("АИП\n", encoding="utf-8")

    config = AppConfig(input_file=audio_file, glossary_path=[first, second])

    assert config.glossary_path == (first, second)


def test_glossary_path_accepts_comma_separated_string(tmp_path: Path, audio_file: Path) -> None:
    first = tmp_path / "tz.txt"
    second = tmp_path / "user.txt"
    first.write_text("ОИБ\n", encoding="utf-8")
    second.write_text("АИП\n", encoding="utf-8")

    config = AppConfig(input_file=audio_file, glossary_path=f"{first},{second}")

    assert config.glossary_path == (first, second)


def test_missing_glossary_path_raises(tmp_path: Path, audio_file: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, glossary_path=missing)


# --- Пакет 2 «LLM»: резюме и доп. инструкции ------------------------------


def test_llm_summary_defaults_to_true(audio_file: Path) -> None:
    # резюме включено по умолчанию и применяется вместе с LLM
    assert AppConfig(input_file=audio_file).llm_summary is True


def test_llm_summary_accepts_explicit_false(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, llm_summary=False).llm_summary is False


def test_llm_summary_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_summary="yes")  # type: ignore[arg-type]


def test_llm_prompt_extra_defaults_to_none(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.llm_prompt_extra is None
    assert config.llm_prompt_file is None


def test_llm_prompt_extra_empty_normalized_to_none(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, llm_prompt_extra="   ").llm_prompt_extra is None


def test_llm_prompt_extra_kept_as_text(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, llm_prompt_extra="Отвечай кратко")

    assert config.llm_prompt_extra == "Отвечай кратко"


def test_llm_prompt_file_missing_raises(tmp_path: Path, audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_prompt_file=tmp_path / "missing.txt")


def test_llm_prompt_file_existing_accepted(tmp_path: Path, audio_file: Path) -> None:
    prompt_file = tmp_path / "extra.txt"
    prompt_file.write_text("Инструкция", encoding="utf-8")

    config = AppConfig(input_file=audio_file, llm_prompt_file=prompt_file)

    assert config.llm_prompt_file == prompt_file


# --- Пакет 3 «надёжность»: кэш результатов ---------------------------------


def test_use_cache_defaults_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).use_cache is True


def test_use_cache_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, use_cache="yes")  # type: ignore[arg-type]


def test_resolved_cache_dir_defaults_to_output_subdir(
    tmp_path: Path, audio_file: Path
) -> None:
    config = AppConfig(input_file=audio_file, output_dir=tmp_path / "out")

    assert config.resolved_cache_dir() == tmp_path / "out" / ".cache"


def test_resolved_cache_dir_honors_explicit_value(tmp_path: Path, audio_file: Path) -> None:
    custom = tmp_path / "custom-cache"
    config = AppConfig(input_file=audio_file, output_dir=tmp_path / "out", cache_dir=custom)

    assert config.resolved_cache_dir() == custom


# --- Пакет 4 «интерфейс»: уведомления о завершении --------------------------


def test_notifications_default_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).notifications is True


def test_notifications_accepts_explicit_false(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, notifications=False).notifications is False


def test_notifications_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, notifications="yes")  # type: ignore[arg-type]


# --- Пакет 5 «enrollment-диаризация»: образцы голоса ------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["Иван=ivan.wav"], {"Иван": (Path("ivan.wav"),)}),
        (
            ["Иван=a.wav", "Иван=b.wav", "Мария=c.wav"],
            {"Иван": (Path("a.wav"), Path("b.wav")), "Мария": (Path("c.wav"),)},
        ),
        ([], {}),
    ],
)
def test_parse_speaker_references_valid(raw: list[str], expected: dict[str, tuple[Path, ...]]) -> None:
    assert AppConfig.parse_speaker_references(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        ["без-разделителя"],
        ["=ivan.wav"],
        ["Иван="],
        ["Иван=   "],
        ["   =x.wav"],
    ],
)
def test_parse_speaker_references_invalid(raw: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig.parse_speaker_references(raw)


def test_speaker_references_accept_single_path_value(tmp_path: Path, audio_file: Path) -> None:
    reference = tmp_path / "ivan.wav"
    reference.write_bytes(b"")

    config = AppConfig(input_file=audio_file, speaker_references={"Иван": reference})

    assert config.speaker_references == {"Иван": (reference,)}


def test_speaker_references_missing_file_rejected(tmp_path: Path, audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            speaker_references={"Иван": tmp_path / "missing.wav"},
        )


def test_enrollment_min_similarity_default(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).enrollment_min_similarity == pytest.approx(0.6)


@pytest.mark.parametrize("value", [-1.5, 1.01])
def test_enrollment_min_similarity_out_of_range(audio_file: Path, value: float) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, enrollment_min_similarity=value)


def test_timeline_defaults_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).timeline is True


def test_timeline_accepts_explicit_false(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, timeline=False).timeline is False


def test_timeline_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, timeline="yes")  # type: ignore[arg-type]


# --- Пакет 8 «образцы голоса и библиотека» ---------------------------------


def test_export_speaker_samples_defaults_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).export_speaker_samples is True


def test_export_speaker_samples_can_be_disabled(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, export_speaker_samples=False)

    assert config.export_speaker_samples is False


def test_export_speaker_samples_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, export_speaker_samples="yes")  # type: ignore[arg-type]


def test_voices_dir_defaults_to_none(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).voices_dir is None


def test_resolved_voices_dir_defaults_to_voices(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.resolved_voices_dir() == Path("voices")


# --- Подготовка эталона голоса (#29) ----------------------------------------


def test_reference_prepare_defaults(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.reference_prepare is True
    assert config.enrollment_min_sample_seconds == pytest.approx(3.0)
    assert config.enrollment_max_sample_seconds == pytest.approx(10.0)
    assert config.reference_target_dbfs == pytest.approx(-30.0)


def test_reference_prepare_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, reference_prepare="yes")  # type: ignore[arg-type]


def test_reference_sample_duration_range_validated(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            enrollment_min_sample_seconds=5.0,
            enrollment_max_sample_seconds=3.0,
        )
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, enrollment_min_sample_seconds=0.0)


def test_reference_target_dbfs_must_be_negative(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, reference_target_dbfs=0.0)


def test_resolved_voices_dir_honors_explicit_value(tmp_path: Path, audio_file: Path) -> None:
    custom = tmp_path / "my-voices"
    config = AppConfig(input_file=audio_file, voices_dir=custom)

    assert config.resolved_voices_dir() == custom


def test_voices_dir_missing_warns(
    tmp_path: Path, audio_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        AppConfig(input_file=audio_file, voices_dir=tmp_path / "missing-voices")

    assert any("библиотеки голосов" in record.message for record in caplog.records)


def test_resolved_speaker_references_merges_library(tmp_path: Path, audio_file: Path) -> None:
    explicit = tmp_path / "explicit.wav"
    explicit.write_bytes(b"")
    voices = tmp_path / "voices"
    voices.mkdir()
    library_file = voices / "Мария.wav"
    library_file.write_bytes(b"")

    config = AppConfig(
        input_file=audio_file,
        speaker_references={"Иван": explicit},
        voices_dir=voices,
    )

    assert config.resolved_speaker_references() == {
        "Иван": (explicit,),
        "Мария": (library_file,),
    }


def test_resolved_speaker_references_without_library_is_explicit(
    tmp_path: Path, audio_file: Path
) -> None:
    explicit = tmp_path / "explicit.wav"
    explicit.write_bytes(b"")
    config = AppConfig(
        input_file=audio_file,
        speaker_references={"Иван": explicit},
        voices_dir=tmp_path / "absent",
    )

    assert config.resolved_speaker_references() == {"Иван": (explicit,)}


# --- Пакет «протокол по кнопке»: protocol_auto ------------------------------


def test_protocol_auto_defaults_to_true(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file).protocol_auto is True


def test_protocol_auto_can_be_disabled(audio_file: Path) -> None:
    assert AppConfig(input_file=audio_file, protocol_auto=False).protocol_auto is False


def test_protocol_auto_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, protocol_auto="yes")  # type: ignore[arg-type]


# --- Диаризация: диапазон говорящих и гиперпараметры pyannote ----------------

def test_diarization_hyperparameters_defaults(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.diarization_min_duration_off == pytest.approx(0.5)
    assert config.diarization_clustering_threshold is None
    assert config.diarization_clustering_fb is None
    assert config.min_speakers is None
    assert config.max_speakers is None


def test_min_duration_off_must_be_non_negative(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_min_duration_off=-0.1)

    config = AppConfig(input_file=audio_file, diarization_min_duration_off=0.0)
    assert config.diarization_min_duration_off == pytest.approx(0.0)


def test_min_duration_off_must_be_number(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_min_duration_off="0.5")  # type: ignore[arg-type]


def test_clustering_threshold_must_be_in_unit_interval(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_clustering_threshold=0.0)

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_clustering_threshold=1.5)

    assert AppConfig(input_file=audio_file, diarization_clustering_threshold=0.6).diarization_clustering_threshold == pytest.approx(0.6)


def test_clustering_fb_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_clustering_fb=0.0)

    assert AppConfig(input_file=audio_file, diarization_clustering_fb=1.5).diarization_clustering_fb == pytest.approx(1.5)


def test_min_speakers_greater_than_max_rejected(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, min_speakers=5, max_speakers=3)


def test_speaker_bounds_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, min_speakers=0)

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, max_speakers=0)


def test_speaker_range_preserved_when_valid(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, min_speakers=2, max_speakers=4)

    assert config.min_speakers == 2
    assert config.max_speakers == 4


def test_num_speakers_overrides_range(
    audio_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        config = AppConfig(
            input_file=audio_file, num_speakers=3, min_speakers=2, max_speakers=5
        )

    assert config.min_speakers is None
    assert config.max_speakers is None
    assert any("игнорируются" in record.message for record in caplog.records)


def test_llm_provider_defaults_to_llama(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.llm_provider == "llama"
    assert config.llm_base_url is None
    assert config.llm_model_name is None
    assert config.llm_api_key is None


def test_llm_provider_normalized_and_validated(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, llm_provider="OpenAI")

    assert config.llm_provider == "openai"

    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, llm_provider="anthropic")


def test_llm_provider_blank_fields_normalized_to_none(audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        llm_enabled=True,
        llm_provider="openai",
        llm_base_url="  ",
        llm_model_name="  ",
        llm_api_key="  ",
    )

    assert config.llm_base_url is None
    assert config.llm_model_name is None
    assert config.llm_api_key is None


# --- GigaAM (#46) и гибридный ASR (#57) -------------------------------------


def test_gigaam_backend_uses_defaults(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, asr_backend=AsrBackend.GIGAAM)

    assert config.asr_backend is AsrBackend.GIGAAM
    assert config.gigaam_model == "gigaam-v3-e2e-rnnt"
    assert config.gigaam_model_path is None
    assert config.gigaam_quantization is None
    assert config.gigaam_vad is True


def test_gigaam_model_path_string_is_normalized_to_path(audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        asr_backend=AsrBackend.GIGAAM,
        gigaam_model_path="models/gigaam",
    )
    assert config.gigaam_model_path == Path("models/gigaam")


def test_gigaam_quantization_blank_normalized_to_none(audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file, asr_backend=AsrBackend.GIGAAM, gigaam_quantization="  "
    )
    assert config.gigaam_quantization is None


def test_gigaam_vad_must_be_boolean(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, asr_backend=AsrBackend.GIGAAM, gigaam_vad="yes")  # type: ignore[arg-type]


def test_hybrid_disabled_by_default(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.hybrid_asr is False
    assert config.hybrid_fallback_backend is AsrBackend.FASTER_WHISPER
    assert config.hybrid_low_logprob_threshold == pytest.approx(-1.0)
    assert config.hybrid_no_speech_threshold == pytest.approx(0.6)
    assert config.hybrid_context_seconds > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hybrid_low_logprob_threshold", 0.5),
        ("hybrid_no_speech_threshold", 1.5),
        ("hybrid_silence_rms_threshold", -0.1),
        ("hybrid_min_segment_seconds", 0.0),
        ("hybrid_context_seconds", -0.1),
    ],
)
def test_hybrid_thresholds_reject_out_of_range(
    audio_file: Path, field: str, value: float
) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, **{field: value})


def test_hybrid_whisper_cpp_requires_model(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            hybrid_asr=True,
            hybrid_fallback_backend=AsrBackend.WHISPER_CPP,
        )


def test_hybrid_whisper_cpp_accepts_model(audio_file: Path, tmp_path: Path) -> None:
    model = tmp_path / "ggml.bin"
    model.write_bytes(b"x")

    config = AppConfig(
        input_file=audio_file,
        hybrid_asr=True,
        hybrid_fallback_backend=AsrBackend.WHISPER_CPP,
        whisper_cpp_model=model,
    )
    assert config.hybrid_fallback_backend is AsrBackend.WHISPER_CPP


def test_hybrid_same_engine_warns(
    audio_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        AppConfig(input_file=audio_file, hybrid_asr=True)

    assert any("совпадают" in record.message for record in caplog.records)


# --- Движок диаризации и NeMo-Speech.cpp (#62) ------------------------------


def test_diarization_engine_defaults_to_auto(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.diarization_engine == "auto"
    assert config.nemo_speech_binary == "nemo-speech"
    assert config.nemo_speech_lib_path is None
    assert config.nemo_speech_model == "nvidia/diar_streaming_sortformer_4spk-v2"
    assert config.nemo_speech_device == "auto"


def test_diarization_engine_normalizes_case(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, diarization_engine="NEMO-SPEECH")

    assert config.diarization_engine == "nemo-speech"


def test_unknown_diarization_engine_raises(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_engine="whisper")


def test_unknown_nemo_speech_device_raises(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, nemo_speech_device="cuda")


def test_nemo_speech_binary_must_be_nonempty(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, nemo_speech_binary="  ")


def test_nemo_speech_model_must_be_nonempty(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, nemo_speech_model="   ")


def test_nemo_speech_lib_path_blank_normalized_to_none(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, nemo_speech_lib_path="  ")

    assert config.nemo_speech_lib_path is None


def test_nemo_speech_speaker_limit_warns(
    audio_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        AppConfig(input_file=audio_file, diarization_engine="nemo-speech", num_speakers=6)

    assert any("не более 4" in record.message for record in caplog.records)


def test_nemo_speech_speaker_limit_not_warned_for_pyannote(
    audio_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        AppConfig(input_file=audio_file, diarization_engine="pyannote", num_speakers=6)

    assert not any("не более 4" in record.message for record in caplog.records)


# --- Оценщик числа говорящих и маршрутизация auto (#64) ---------------------


def test_diarization_estimate_defaults(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.diarization_estimate_enabled is True
    assert config.diarization_estimate_seconds == 30.0
    assert config.diarization_estimate_threshold == 0.7
    assert config.diarization_estimate_model.endswith(".onnx")
    assert config.diarization_route_max_speakers == 4


def test_diarization_estimate_seconds_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_estimate_seconds=0.0)


def test_diarization_estimate_threshold_range(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_estimate_threshold=2.5)
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_estimate_threshold=0.0)


def test_diarization_estimate_model_must_be_nonempty(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_estimate_model="   ")


def test_diarization_route_max_speakers_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_route_max_speakers=0)


def test_diarization_estimate_enabled_must_be_bool(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            diarization_estimate_enabled="yes",  # type: ignore[arg-type]
        )


# --- Гибридная диаризация (#64, часть 2) ------------------------------------


def test_diarization_hybrid_defaults(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file)

    assert config.diarization_hybrid_enabled is True
    assert config.diarization_hybrid_window_seconds == 90.0
    assert config.diarization_hybrid_overlap_seconds == 2.0
    assert config.diarization_hybrid_min_speaker_seconds == 1.5
    assert config.diarization_hybrid_overload_split is True
    assert config.diarization_hybrid_subwindow_seconds == 30.0
    assert config.diarization_hybrid_max_split_depth == 1


def test_diarization_hybrid_engine_is_valid(audio_file: Path) -> None:
    config = AppConfig(input_file=audio_file, diarization_engine="HYBRID")

    assert config.diarization_engine == "hybrid"


def test_diarization_hybrid_window_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_hybrid_window_seconds=0.0)


def test_diarization_hybrid_overlap_must_be_smaller_than_window(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            diarization_hybrid_window_seconds=10.0,
            diarization_hybrid_overlap_seconds=10.0,
        )


def test_diarization_hybrid_overlap_not_negative(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_hybrid_overlap_seconds=-1.0)


def test_diarization_hybrid_min_speaker_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_hybrid_min_speaker_seconds=0.0)


def test_diarization_hybrid_enabled_must_be_bool(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            diarization_hybrid_enabled="yes",  # type: ignore[arg-type]
        )


def test_diarization_hybrid_subwindow_must_be_positive(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_hybrid_subwindow_seconds=0.0)


def test_diarization_hybrid_subwindow_must_be_smaller_than_window(
    audio_file: Path,
) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            diarization_hybrid_window_seconds=30.0,
            diarization_hybrid_subwindow_seconds=30.0,
        )


def test_diarization_hybrid_max_split_depth_must_be_non_negative(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(input_file=audio_file, diarization_hybrid_max_split_depth=-1)


def test_diarization_hybrid_overload_split_must_be_bool(audio_file: Path) -> None:
    with pytest.raises(ConfigurationError):
        AppConfig(
            input_file=audio_file,
            diarization_hybrid_overload_split="yes",  # type: ignore[arg-type]
        )
