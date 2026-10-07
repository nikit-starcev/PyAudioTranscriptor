"""Тесты CLI-команды ``transcribe`` через Typer ``CliRunner``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_transcriber import __version__
from audio_transcriber.cli import app as app_module
from audio_transcriber.cli.app import app
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import AsrBackend, Device
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.utils.exceptions import AudioTranscriberError

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolate_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует CLI-тесты от личного ``config.env`` разработчика.

    Боевой CLI по умолчанию читает ``config.env`` (issue #76). Чтобы результат
    тестов не зависел от личного файла, по умолчанию подсовываем пустое
    окружение; тесты, проверяющие слияние, задают его через ``config_env``.
    """
    monkeypatch.setattr(app_module, "load_config_env", lambda *_args, **_kwargs: (None, {}))


@pytest.fixture
def config_env(monkeypatch: pytest.MonkeyPatch):
    """Задаёт содержимое ``config.env``, которое увидит CLI в тесте."""

    def _install(defaults: dict[str, str]) -> None:
        monkeypatch.setattr(
            app_module, "load_config_env", lambda *_args, **_kwargs: (None, dict(defaults))
        )

    return _install


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


def test_transcribe_llm_request_timeout(
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
            "--llm-request-timeout",
            "45",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].llm_request_timeout == 45.0


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


def test_transcribe_diarization_hyperparameters_defaults(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    config = captured["config"]
    # Наш дефолт против дробления реплик — 0.5 (у pyannote 0.0).
    assert config.diarization_min_duration_off == pytest.approx(0.5)
    assert config.diarization_clustering_threshold is None
    assert config.diarization_clustering_fb is None
    assert config.min_speakers is None
    assert config.max_speakers is None


def test_transcribe_diarization_hyperparameter_flags(
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
            "--min-duration-off",
            "0.8",
            "--clustering-threshold",
            "0.6",
            "--clustering-fb",
            "1.5",
            "--min-speakers",
            "2",
            "--max-speakers",
            "5",
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert config.diarization_min_duration_off == pytest.approx(0.8)
    assert config.diarization_clustering_threshold == pytest.approx(0.6)
    assert config.diarization_clustering_fb == pytest.approx(1.5)
    assert config.min_speakers == 2
    assert config.max_speakers == 5


def test_transcribe_rejects_negative_min_duration_off(
    audio_file: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--min-duration-off",
            "-1",
        ],
    )

    assert result.exit_code != 0


def test_transcribe_rejects_min_greater_than_max_speakers(
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
            "--min-speakers",
            "5",
            "--max-speakers",
            "2",
        ],
    )

    assert result.exit_code == 1


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


# --- Пакет 6 «таймлайн говорящих» -------------------------------------------


def test_transcribe_timeline_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].timeline is True


def test_transcribe_no_timeline_flag(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-timeline"],
    )

    assert result.exit_code == 0
    assert captured["config"].timeline is False


def test_transcribe_word_timestamps_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].word_timestamps is True


def test_transcribe_no_word_timestamps_flag(
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
            "--no-word-timestamps",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].word_timestamps is False


def test_transcribe_word_timestamps_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    config_env({"WORD_TIMESTAMPS": "false"})

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].word_timestamps is False


# --- Пакет 8 «образцы голоса и библиотека» ---------------------------------


def test_transcribe_speaker_samples_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].export_speaker_samples is True


def test_transcribe_no_speaker_samples_flag(
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
            "--no-speaker-samples",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].export_speaker_samples is False


def test_transcribe_voices_dir_is_merged_into_references(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    voices = tmp_path / "voices"
    voices.mkdir()
    library_file = voices / "Мария.wav"
    library_file.write_bytes(b"")
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--voices-dir",
            str(voices),
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].voices_dir == voices
    assert captured["config"].resolved_speaker_references() == {"Мария": (library_file,)}


# --- Пакет «протокол по кнопке»: --no-protocol ------------------------------


def test_transcribe_protocol_enabled_by_default(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].protocol_auto is True


def test_transcribe_no_protocol_flag_disables_auto_protocol(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)

    result = runner.invoke(
        app,
        ["transcribe", str(audio_file), "-o", str(tmp_path / "out"), "--no-protocol"],
    )

    assert result.exit_code == 0
    assert captured["config"].protocol_auto is False


def _real_pipeline_with_stubs(config: AppConfig, **kwargs):
    """Прогон настоящего ``run_pipeline`` с фиктивными компонентами (без моделей)."""
    from audio_transcriber.domain.models import (
        Speaker,
        TranscriptEntry,
        TranscriptionSegment,
    )
    from audio_transcriber.pipeline import run_pipeline as real_run_pipeline

    class _Recognizer:
        def transcribe(self, audio_path: Path, *, language: str | None = None):
            return ([TranscriptionSegment(start=0.0, end=1.0, text="привет")], "ru", 1.0)

    class _Merger:
        def merge(self, transcription_segments, speaker_segments, known_speakers=None):
            speaker = Speaker(id="SPEAKER_00", display_name="Иван")
            entry = TranscriptEntry(start=0.0, end=1.0, text="привет", speaker=speaker)
            return [entry], [speaker]

    return real_run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_Recognizer(),
        merger=_Merger(),
    )


def test_transcribe_no_protocol_writes_no_files(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "run_pipeline", _real_pipeline_with_stubs)
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "--no-protocol",
            "--no-diarization",
            "--no-speaker-samples",
            "--no-timeline",
        ],
    )

    assert result.exit_code == 0
    assert not (output_dir / f"{audio_file.stem}.txt").exists()


def test_transcribe_default_protocol_writes_files(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "run_pipeline", _real_pipeline_with_stubs)
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(output_dir),
            "--no-diarization",
            "--no-speaker-samples",
            "--no-timeline",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / f"{audio_file.stem}.txt").is_file()


# --- Issue #76: CLI наследует config.env, флаги — явный override ------------


def test_cli_takes_asr_backend_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    wcp_model = tmp_path / "ggml-large-v3-turbo.bin"
    config_env({"ASR_BACKEND": "whisper-cpp", "WHISPER_CPP_MODEL": str(wcp_model)})

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    config = captured["config"]
    assert config.asr_backend is AsrBackend.WHISPER_CPP
    assert config.whisper_cpp_model == wcp_model
    assert "Бэкенд распознавания: whisper-cpp" in result.stdout


def test_cli_flag_overrides_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    config_env(
        {
            "ASR_BACKEND": "whisper-cpp",
            "WHISPER_CPP_MODEL": str(tmp_path / "ggml.bin"),
            "DENOISE": "false",
        }
    )

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(audio_file),
            "-o",
            str(tmp_path / "out"),
            "--asr-backend",
            "faster-whisper",
            "--denoise",
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    # Явные флаги перебивают config.env.
    assert config.asr_backend is AsrBackend.FASTER_WHISPER
    assert config.denoise is True


def test_cli_nemo_speech_and_pyannote_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    pyannote_local = tmp_path / "pyannote-models" / "speaker-diarization-community-1"
    config_env(
        {
            "DIARIZATION_ENGINE": "auto",
            "NEMO_SPEECH_BINARY": "/opt/nemo-speech/bin/nemo-speech",
            "NEMO_SPEECH_DEVICE": "vulkan",
            "PYANNOTE_LOCAL_MODEL": str(pyannote_local),
            "DEVICE": "cpu",
        }
    )

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    config = captured["config"]
    assert config.nemo_speech_binary == "/opt/nemo-speech/bin/nemo-speech"
    assert config.nemo_speech_device == "vulkan"
    assert config.pyannote_local_model == pyannote_local
    assert config.device is Device.CPU
    # В логе конфигурации видно GPU-движок, а не pyannote.
    assert "устройство vulkan" in result.stdout


def test_cli_bool_defaults_follow_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    config_env(
        {
            "DENOISE": "false",
            "MERGE_SAME_NAME_SPEAKERS": "false",
            "CLEAN_ARTIFACTS": "false",
            "MARK_OVERLAP": "false",
        }
    )

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    config = captured["config"]
    assert config.denoise is False
    assert config.merge_same_name_speakers is False
    assert config.clean_artifacts is False
    assert config.mark_overlap is False


def test_cli_llm_settings_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    config_env(
        {
            "LLM_ENABLED": "true",
            "LLM_PROVIDER": "llama",
            "LLM_MODEL": str(tmp_path / "qwen.gguf"),
            "LLM_CONTEXT": "2048",
            "LLM_REQUEST_TIMEOUT": "42",
            "LLM_EXTRACT_NAMES": "true",
        }
    )

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    config = captured["config"]
    assert config.llm_enabled is True
    assert config.llm_model == tmp_path / "qwen.gguf"
    assert config.llm_context_size == 2048
    assert config.llm_request_timeout == pytest.approx(42.0)
    assert config.llm_extract_names is True


def test_cli_hotwords_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    config_env({"HOTWORDS": "юрист Смирнова"})

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].hotwords == "юрист Смирнова"


# --- Issue #90: INITIAL_PROMPT и CLEAR_CACHE в едином источнике ------------


def test_cli_initial_prompt_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
) -> None:
    """INITIAL_PROMPT из config.env доходит до AppConfig без флага (issue #90)."""
    captured: dict[str, AppConfig] = {}
    monkeypatch.setattr(app_module, "run_pipeline", _capturing_pipeline(captured))
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    config_env({"INITIAL_PROMPT": "ОИБ, АРМ, КИСУСС"})

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(tmp_path / "out")])

    assert result.exit_code == 0
    assert captured["config"].initial_prompt == "ОИБ, АРМ, КИСУСС"


def test_cli_clear_cache_from_config_env(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_env,
    stub_pipeline,
) -> None:
    """CLEAR_CACHE=true в config.env очищает кэш так же, как флаг ``--clear-cache``.

    Раньше это делали обёртки run.sh/run.ps1 (дублирование маппинга, issue #90);
    теперь единый источник — CLI.
    """
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    output_dir = tmp_path / "out"
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True)
    stale = cache_dir / "asr-deadbeef.json"
    stale.write_text("{}", encoding="utf-8")
    config_env({"CLEAR_CACHE": "true"})

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0
    assert not stale.exists()


def test_cli_clear_cache_disabled_by_default(
    audio_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_pipeline,
) -> None:
    """Без CLEAR_CACHE/флага кэш не трогается."""
    monkeypatch.setattr(app_module, "resolve_device", lambda _device: Device.CPU)
    output_dir = tmp_path / "out"
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True)
    kept = cache_dir / "asr-deadbeef.json"
    kept.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["transcribe", str(audio_file), "-o", str(output_dir)])

    assert result.exit_code == 0
    assert kept.exists()
