"""Тесты CLI-команды ``transcribe`` через Typer ``CliRunner``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_transcriber import __version__
from audio_transcriber.cli import app as app_module
from audio_transcriber.cli.app import app
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult

runner = CliRunner()


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Подменяет ``run_pipeline`` в CLI фиктивной реализацией без реальных моделей."""

    def fake_run_pipeline(config, **kwargs):
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)],
            speakers=[speaker],
        )

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_missing_input_file_fails_before_reaching_config(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.mp3"

    result = runner.invoke(app, ["transcribe", str(missing)])

    assert result.exit_code != 0


def test_transcribe_with_defaults_resolves_cpu(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    output_dir = tmp_path / "out"
    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0
    assert "cpu" in result.stdout
    assert "Готово" in result.stdout
    assert output_dir.is_dir()


def test_transcribe_rejects_non_positive_num_speakers(audio_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "-n", "0"],
    )

    assert result.exit_code != 0


def test_transcribe_rejects_malformed_speaker_name(audio_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--speaker-name",
            "not-valid",
        ],
    )

    assert result.exit_code == 1


def test_transcribe_saves_debug_log_file(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    output_dir = tmp_path / "out"
    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0
    log_files = list((output_dir / "logs").glob("*.log"))
    assert len(log_files) == 1
    assert "Готово" in log_files[0].read_text(encoding="utf-8")


def test_transcribe_accepts_multiple_formats_and_speaker_names(
    audio_file: Path, tmp_path: Path, stub_pipeline
) -> None:
    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "-f",
            "txt",
            "-f",
            "docx",
            "--speaker-name",
            "0=Иван",
            "--speaker-name",
            "1=Мария",
            "-d",
            "cpu",
            "-v",
        ],
    )

    assert result.exit_code == 0
    assert "txt, docx" in result.stdout
    assert "Иван" in result.stdout
    assert "Мария" in result.stdout


def test_transcribe_passes_hotwords(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}

    def fake_run_pipeline(config: AppConfig, **kwargs):
        captured["config"] = config
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)],
            speakers=[speaker],
        )

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "--hotwords",
            "юрист Смирнова",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].hotwords == "юрист Смирнова"
    assert captured["config"].enable_correction is False


def test_transcribe_enable_correction_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}

    def fake_run_pipeline(config: AppConfig, **kwargs):
        captured["config"] = config
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)],
            speakers=[speaker],
        )

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--enable-correction",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].enable_correction is True


def _capturing_pipeline(captured: dict[str, AppConfig]):
    def fake_run_pipeline(config: AppConfig, **kwargs):
        captured["config"] = config
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)],
            speakers=[speaker],
        )

    return fake_run_pipeline


def test_transcribe_names_disabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    # экспериментальное определение имён выключено по умолчанию
    assert captured["config"].llm_extract_names is False


def test_transcribe_names_can_be_enabled(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--llm-names"],
    )

    assert result.exit_code == 0
    assert captured["config"].llm_extract_names is True


def test_transcribe_llm_no_names_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--llm-no-names",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].llm_extract_names is False


def test_transcribe_no_diarization_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-diarization"],
    )

    assert result.exit_code == 0
    assert captured["config"].diarization_enabled is False


def test_transcribe_diarization_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].diarization_enabled is True


def test_transcribe_cleaning_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].clean_artifacts is True


def test_transcribe_no_clean_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-clean"],
    )

    assert result.exit_code == 0
    assert captured["config"].clean_artifacts is False


def test_transcribe_hotwords_warns_when_over_limit(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    # Длинная строка --hotwords должна обрезаться с предупреждением.
    long_hotwords = ", ".join(f"термин-{i}" for i in range(80))

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "--hotwords",
            long_hotwords,
            "-v",
        ],
    )

    assert result.exit_code == 0
    log_files = list((output_dir / "logs").glob("*.log"))
    assert "Не поместилось в лимит hotwords ASR" in log_files[0].read_text(encoding="utf-8")
