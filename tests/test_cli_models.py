"""Тесты CLI-команды ``models`` и нейтрального пакета ``audio_transcriber.models``.

Сеть не используется: реальный загрузчик подменяется заглушкой, пишущей файлы
на диск. Проверяются список/скачивание/удаление, идемпотентность, gated без
токена, свободное место, коды выхода и переиспользование веб-механизма.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_transcriber import models as core_models
from audio_transcriber.cli import models_cmd
from audio_transcriber.cli.app import app
from audio_transcriber.web import models as web_models

runner = CliRunner()


class FakeDownloader:
    """Заглушка загрузчика: пишет файлы и уведомляет о прогрессе."""

    def __init__(self, *, fail_times: int = 0, block: threading.Event | None = None) -> None:
        self.fail_times = fail_times
        self.block = block
        self.calls: list[tuple[str, str]] = []

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress,
    ) -> None:
        self.calls.append((repo, filename))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("сеть недоступна")
        if self.block is not None:
            self.block.wait(timeout=5)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = b"x" * 4096
        destination.write_bytes(data)
        on_progress(len(data))

    def fetch_snapshot(
        self,
        *,
        repo: str,
        destination: Path,
        token: str | None,
        on_progress,
    ) -> None:
        self.calls.append((repo, "<snapshot>"))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.yaml").write_bytes(b"ok")
        on_progress(64)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Изолирует тесты от config.env, секретов и окружения разработчика."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("AUDIO_TRANSCRIBER_WEB_DATA", str(tmp_path / "web-data"))
    monkeypatch.setattr(models_cmd, "load_config_env", lambda: (None, {}))


def _combined_output(result) -> str:
    return (result.output or "") + (getattr(result, "stderr", "") or "")


def _install_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeDownloader) -> None:
    monkeypatch.setattr(models_cmd, "HfDownloader", lambda: fake)


def test_web_models_reexports_core() -> None:
    """Веб-фасад переэкспортирует тот же объект, что и нейтральный пакет."""
    assert web_models.MODEL_CATALOG is core_models.MODEL_CATALOG
    assert web_models.ModelDownloadManager is core_models.ModelDownloadManager
    assert web_models.HfDownloader is core_models.HfDownloader
    assert web_models.KIND_WHISPER == core_models.KIND_WHISPER


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["models", "--help"])

    assert result.exit_code == 0, _combined_output(result)
    assert "list" in result.output
    assert "download" in result.output
    assert "delete" in result.output


def test_models_list_shows_status_and_paths(tmp_path: Path) -> None:
    root = tmp_path / "models-root"
    target = root / "whisper-models"
    target.mkdir(parents=True)
    (target / "ggml-small.bin").write_bytes(b"z" * 10)

    result = runner.invoke(app, ["models", "list", "--models-root", str(root)])

    assert result.exit_code == 0, _combined_output(result)
    lines = result.output.splitlines()
    assert any(line.startswith("ID") for line in lines)
    small = next(line for line in lines if line.startswith("whisper-small"))
    assert "present" in small
    turbo = next(line for line in lines if line.startswith("whisper-large-v3-turbo"))
    assert "missing" in turbo
    assert str(target / "ggml-small.bin") in result.output
    assert "да" in next(line for line in lines if line.startswith("pyannote-community-1"))


def test_models_download_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDownloader()
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"

    result = runner.invoke(
        app,
        ["models", "download", "whisper-small", "--models-root", str(root)],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert (root / "whisper-models" / "ggml-small.bin").is_file()
    assert fake.calls == [("ggerganov/whisper.cpp", "ggml-small.bin")]
    assert "скачана" in result.output

    status = core_models.local_status(
        core_models.find_model("whisper-small"),  # type: ignore[arg-type]
        root / "whisper-models",
    )
    assert status.present is True


def test_models_download_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDownloader()
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"
    target = root / "whisper-models"
    target.mkdir(parents=True)
    (target / "ggml-small.bin").write_bytes(b"z" * 10)

    result = runner.invoke(
        app,
        ["models", "download", "whisper-small", "--models-root", str(root)],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert fake.calls == []
    assert "пропуск" in result.output


def test_models_download_retries_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDownloader(fail_times=1)
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"

    result = runner.invoke(
        app,
        [
            "models",
            "download",
            "whisper-small",
            "--models-root",
            str(root),
            "--retries",
            "2",
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    assert len(fake.calls) == 2
    assert (root / "whisper-models" / "ggml-small.bin").is_file()


def test_models_download_exhausts_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDownloader(fail_times=5)
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"

    result = runner.invoke(
        app,
        ["models", "download", "whisper-small", "--models-root", str(root), "--retries", "1"],
    )

    assert result.exit_code == 1
    assert "сеть недоступна" in _combined_output(result)


def test_models_download_gated_without_token_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeDownloader()
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"

    result = runner.invoke(
        app,
        ["models", "download", "pyannote-community-1", "--models-root", str(root)],
    )

    assert result.exit_code == 1
    assert "токен" in _combined_output(result)
    assert fake.calls == []


def test_models_download_all_skips_gated_and_downloads_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDownloader()
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"
    monkeypatch.setattr(models_cmd, "free_space", lambda _path: 10**12)

    result = runner.invoke(app, ["models", "download", "--all", "--models-root", str(root)])

    assert result.exit_code == 0, _combined_output(result)
    assert "pyannote-community-1" in result.output
    assert (root / "whisper-models" / "ggml-small.bin").is_file()
    assert (root / "sherpa-models").is_dir()
    assert (root / "gigaam-models" / "gigaam-v3-onnx" / "config.yaml").is_file()


def test_models_download_insufficient_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDownloader()
    _install_fake(monkeypatch, fake)
    root = tmp_path / "models-root"
    monkeypatch.setattr(models_cmd, "free_space", lambda _path: 1)

    result = runner.invoke(
        app, ["models", "download", "whisper-large-v3", "--models-root", str(root)]
    )

    assert result.exit_code == 1
    assert "места" in _combined_output(result)
    assert fake.calls == []


def test_models_download_unknown_id_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["models", "download", "nope", "--models-root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "не найдена" in _combined_output(result)


def test_models_download_requires_selection(tmp_path: Path) -> None:
    result = runner.invoke(app, ["models", "download", "--models-root", str(tmp_path)])

    assert result.exit_code == 2
    assert "укажите" in _combined_output(result).lower()


def test_models_download_all_and_ids_conflict(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["models", "download", "--all", "whisper-small", "--models-root", str(tmp_path)]
    )

    assert result.exit_code == 2


def test_models_delete(tmp_path: Path) -> None:
    root = tmp_path / "models-root"
    target = root / "whisper-models"
    target.mkdir(parents=True)
    (target / "ggml-small.bin").write_bytes(b"z" * 10)

    result = runner.invoke(
        app, ["models", "delete", "whisper-small", "--models-root", str(root)]
    )

    assert result.exit_code == 0, _combined_output(result)
    assert not (target / "ggml-small.bin").exists()

    again = runner.invoke(
        app, ["models", "delete", "whisper-small", "--models-root", str(root)]
    )
    assert again.exit_code == 0
    assert "нечего" in _combined_output(again)


def test_models_delete_unknown_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["models", "delete", "nope", "--models-root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "не найдена" in _combined_output(result)


def test_required_model_entries_reflects_config() -> None:
    default_ids = {entry.id for entry in models_cmd.required_model_entries({})}
    assert default_ids == {"sherpa-campplus-advanced", "pyannote-community-1"}

    whisper = models_cmd.required_model_entries({"ASR_BACKEND": "whisper-cpp"})
    assert "whisper-large-v3-turbo" in {entry.id for entry in whisper}
    assert "gigaam-v3-onnx" not in {entry.id for entry in whisper}

    configured = models_cmd.required_model_entries(
        {"ASR_BACKEND": "whisper-cpp", "WHISPER_CPP_MODEL": "/x/ggml-small.bin"}
    )
    assert "whisper-small" in {entry.id for entry in configured}
    assert "whisper-large-v3-turbo" not in {entry.id for entry in configured}

    full = models_cmd.required_model_entries(
        {"ASR_BACKEND": "gigaam", "LLM_ENABLED": "true", "DIARIZATION_ENGINE": "pyannote"}
    )
    assert {entry.id for entry in full} == {
        "gigaam-v3-onnx",
        "qwen2.5-7b-instruct-q4_k_m",
        "pyannote-community-1",
        "sherpa-campplus-advanced",
    }


def test_download_for_web_downloads_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDownloader()
    _install_fake(monkeypatch, fake)
    monkeypatch.setattr(
        models_cmd,
        "load_config_env",
        lambda: (None, {"ASR_BACKEND": "whisper-cpp"}),
    )
    root = tmp_path / "models-root"

    failures = models_cmd.download_for_web(models_root=root)

    assert failures == 0
    assert (root / "whisper-models" / "ggml-large-v3-turbo.bin").is_file()


def test_manager_with_fake_downloader_reports_progress(tmp_path: Path) -> None:
    entry = core_models.find_model("whisper-small")
    assert entry is not None
    bus = core_models.DownloadBus(heartbeat=0.01)
    manager = core_models.ModelDownloadManager(
        models_root=tmp_path,
        downloader=FakeDownloader(),
        resolve_target=lambda item: tmp_path / item.target_dir,
        bus=bus,
    )

    assert manager.start(entry, token=None) is True
    manager.wait(entry.id)

    state = manager.state(entry.id)
    assert state.status == core_models.STATUS_DONE
    assert state.bytes_done > 0
    statuses = [event["status"] for event in bus.history()]
    assert statuses[0] == core_models.STATUS_DOWNLOADING
    assert statuses[-1] == core_models.STATUS_DONE


def test_manager_reports_error(tmp_path: Path) -> None:
    entry = core_models.find_model("whisper-small")
    assert entry is not None
    manager = core_models.ModelDownloadManager(
        models_root=tmp_path,
        downloader=FakeDownloader(fail_times=1),
        resolve_target=lambda item: tmp_path / item.target_dir,
    )

    manager.start(entry, token=None)
    manager.wait(entry.id)

    state = manager.state(entry.id)
    assert state.status == core_models.STATUS_ERROR
    assert "сеть недоступна" in (state.error or "")


def test_local_status_and_delete_roundtrip(tmp_path: Path) -> None:
    entry = core_models.find_model("whisper-base")
    assert entry is not None
    target = tmp_path / "whisper-models"

    assert core_models.local_status(entry, target).present is False
    target.mkdir(parents=True)
    (target / "ggml-base.bin").write_bytes(b"b" * 7)
    status = core_models.local_status(entry, target)
    assert status.present is True
    assert status.size == 7

    assert core_models.delete_model_files(entry, target) is True
    assert core_models.delete_model_files(entry, target) is False


def test_resolve_target_new_whisper_entries(tmp_path: Path) -> None:
    entry = core_models.find_model("whisper-large-v3")
    assert entry is not None
    assert core_models.resolve_target(
        entry, models_root=tmp_path, settings=object()
    ) == (tmp_path / "whisper-models")


def test_catalog_includes_large_v3_and_base() -> None:
    ids = {entry.id for entry in core_models.MODEL_CATALOG}
    assert "whisper-large-v3" in ids
    assert "whisper-base" in ids
