"""Тесты сборки и валидации конфигурации приложения (:class:`AppConfig`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.utils.exceptions import ConfigurationError


def test_valid_config_creates_output_dir(tmp_path: Path, audio_file: Path) -> None:
    output_dir = tmp_path / "output" / "nested"

    config = AppConfig(input_file=audio_file, output_dir=output_dir)

    assert output_dir.is_dir()
    assert config.device is Device.AUTO
    assert config.export_formats == (ExportFormat.TXT,)


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
