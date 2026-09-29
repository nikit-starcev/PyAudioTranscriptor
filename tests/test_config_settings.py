"""Тесты сборки и валидации конфигурации приложения (:class:`AppConfig`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
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
