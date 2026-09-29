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
from audio_transcriber.utils.exceptions import AudioTranscriberError

runner = CliRunner()


@pytest.fixture(autouse=True)
def notify_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Перехватывает десктоп-уведомления, чтобы тесты не дёргали notify-send."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_module, "notify", lambda title, message: calls.append((title, message)) or True
    )
    return calls


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


def test_transcribe_denoise_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].denoise is True


def test_transcribe_no_denoise_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-denoise"],
    )

    assert result.exit_code == 0
    assert captured["config"].denoise is False


def test_transcribe_quality_defaults(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    config = captured["config"]
    assert config.collapse_repeats is True
    assert config.normalize_text is True
    assert config.mark_overlap is True
    assert config.low_confidence_threshold == pytest.approx(-1.0)


def test_transcribe_quality_flags(
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
            "--no-collapse-repeats",
            "--no-normalize",
            "--no-overlap",
            "--low-confidence-threshold=-2.5",
            "--repeat-min-words",
            "3",
            "--repeat-similarity",
            "0.8",
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert config.collapse_repeats is False
    assert config.normalize_text is False
    assert config.mark_overlap is False
    assert config.low_confidence_threshold == pytest.approx(-2.5)
    assert config.repeat_min_words == 3
    assert config.repeat_similarity == pytest.approx(0.8)


def test_transcribe_rejects_positive_confidence_threshold(
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
            "--low-confidence-threshold=0.5",
        ],
    )

    assert result.exit_code == 1


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


# --- Пакет 2 «LLM»: резюме и доп. инструкции -------------------------------


def test_transcribe_llm_summary_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].llm_summary is True


def test_transcribe_llm_no_summary_flag(
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
            "--no-llm-summary",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].llm_summary is False


def test_transcribe_llm_prompt_extra_and_file(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    prompt_file = tmp_path / "extra.txt"
    prompt_file.write_text("Инструкция", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--llm-prompt-extra",
            "Пиши кратко",
            "--llm-prompt-file",
            str(prompt_file),
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].llm_prompt_extra == "Пиши кратко"
    assert captured["config"].llm_prompt_file == prompt_file


# --- Пакет 3 «надёжность»: кэш результатов ---------------------------------


def test_transcribe_cache_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].use_cache is True


def test_transcribe_no_cache_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-cache"],
    )

    assert result.exit_code == 0
    assert captured["config"].use_cache is False


def test_transcribe_custom_cache_dir(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--cache-dir",
            str(cache_dir),
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].cache_dir == cache_dir
    assert captured["config"].resolved_cache_dir() == cache_dir


def test_transcribe_clear_cache_removes_files(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_pipeline
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    output_dir = tmp_path / "out"
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True)
    stale = cache_dir / "asr-deadbeef.json"
    stale.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(output_dir), "--clear-cache"],
    )

    assert result.exit_code == 0
    assert not stale.exists()


# --- Пакет 4 «интерфейс»: уведомления о завершении --------------------------


def test_transcribe_notifications_enabled_by_default(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline,
    notify_calls: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert len(notify_calls) == 1
    assert notify_calls[0][0] == "Транскрибация завершена"
    assert audio_file.name in notify_calls[0][1]


def test_transcribe_notifications_disabled_by_flag(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline,
    notify_calls: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-notify"],
    )

    assert result.exit_code == 0
    assert notify_calls == []


def test_transcribe_notifies_on_pipeline_error(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    notify_calls: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    def failing_pipeline(config, **kwargs):
        raise AudioTranscriberError("сбой распознавания")

    monkeypatch.setattr(app_module, "run_pipeline", failing_pipeline)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert len(notify_calls) == 1
    assert notify_calls[0][0] == "Транскрибация не удалась"
    assert "сбой распознавания" in notify_calls[0][1]


# --- Пакет 5 «enrollment-диаризация»: образцы голоса ------------------------


def test_transcribe_rejects_malformed_speaker_reference(
    audio_file: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--speaker-reference",
            "not-valid",
        ],
    )

    assert result.exit_code == 1


def test_transcribe_passes_speaker_references(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    first = tmp_path / "ivan.wav"
    second = tmp_path / "ivan2.wav"
    first.write_bytes(b"")
    second.write_bytes(b"")

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
            "--speaker-reference",
            f"Иван={first}",
            "--speaker-reference",
            f"Иван={second}",
            "--enrollment-min-similarity",
            "0.7",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].speaker_references == {"Иван": (first, second)}
    assert captured["config"].enrollment_min_similarity == pytest.approx(0.7)


def test_transcribe_rejects_missing_speaker_reference_file(
    audio_file: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--speaker-reference",
            f"Иван={tmp_path / 'missing.wav'}",
        ],
    )

    assert result.exit_code == 1
