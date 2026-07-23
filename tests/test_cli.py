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
    monkeypatch.setattr(app_module, "resolve_device", lambda device: Device.CPU)

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app, ["transcribe", str(audio_file), "-o", str(output_dir)]
    )

    assert result.exit_code == 0
    assert "cpu" in result.stdout
    assert "Готово" in result.stdout
    assert output_dir.is_dir()


def test_transcribe_rejects_non_positive_num_speakers(
    audio_file: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "-n", "0"],
    )

    assert result.exit_code != 0


def test_transcribe_rejects_malformed_speaker_name(
    audio_file: Path, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(app_module, "resolve_device", lambda device: Device.CPU)

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


def test_transcribe_loads_vocabulary_file_and_merges_with_hotwords(
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
    monkeypatch.setattr(app_module, "resolve_device", lambda device: Device.CPU)

    vocabulary_file = tmp_path / "vocabulary.txt"
    vocabulary_file.write_text("# участники\nИванов\nПетров\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "--vocabulary-file",
            str(vocabulary_file),
            "--hotwords",
            "юрист Смирнова",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].hotwords == "Иванов, Петров, юрист Смирнова"


def test_transcribe_warns_about_vocabulary_terms_over_the_limit(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda device: Device.CPU)

    # Первые термины укладываются в лимит модели (900 символов по умолчанию),
    # а последний — заведомо не влезает и должен быть отброшен с предупреждением.
    vocabulary_file = tmp_path / "vocabulary.txt"
    common_terms = "\n".join(f"термин-{i}" for i in range(80))
    vocabulary_file.write_text(f"{common_terms}\nочень-важный-но-не-влезающий-термин\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "--vocabulary-file",
            str(vocabulary_file),
            "-v",
        ],
    )

    assert result.exit_code == 0
    log_files = list((output_dir / "logs").glob("*.log"))
    assert "Не поместилось в лимит модели" in log_files[0].read_text(encoding="utf-8")


def test_transcribe_rejects_missing_vocabulary_file(audio_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--vocabulary-file",
            str(tmp_path / "missing-vocabulary.txt"),
        ],
    )

    assert result.exit_code != 0
