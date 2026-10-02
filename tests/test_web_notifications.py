"""Тесты десктоп-уведомлений веб-очереди (issue #34).

Уведомления шлёт воркер веб-очереди при переводе задачи в терминальный статус
(``done``/``error``/``cancelled``). ``notify`` подменяется, чтобы тесты не
дёргали ``notify-send``; сам воркер работает на фиктивном конвейере.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.utils.exceptions import ProcessingCancelled
from audio_transcriber.web import runner as runner_module
from audio_transcriber.web.config import build_job_config
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.runner import JobRunner
from audio_transcriber.web.storage.jobs_db import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    JobsDB,
)

#: Сколько ждать завершения задачи/уведомления в тесте.
_TIMEOUT = 5.0


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


def _config_builder(web_paths: WebPaths, *, notifications: bool) -> Callable[..., AppConfig]:
    def build(job_id: str, source_path: Path) -> AppConfig:
        return AppConfig(
            input_file=source_path,
            output_dir=web_paths.results_dir / job_id,
            denoise=False,
            diarization_enabled=False,
            export_speaker_samples=False,
            notifications=notifications,
            timeline=False,
            protocol_auto=False,
            use_cache=False,
        )

    return build


def _ok_pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
    return TranscriptionResult(source_path=config.input_file, language="ru", duration=2.5)


def _failing_pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
    raise RuntimeError("сбой распознавания")


class _CancelledPipeline:
    """Конвейер, который сразу поднимает ``ProcessingCancelled``."""

    def __call__(self, config: AppConfig, *, on_progress=None, cancel_event=None):
        raise ProcessingCancelled("остановлено пользователем")


@pytest.fixture
def notify_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_notify(title: str, message: str) -> bool:
        calls.append((title, message))
        return True

    monkeypatch.setattr(runner_module, "notify", fake_notify)
    return calls


def _wait_for_calls(calls: list[tuple[str, str]], count: int) -> None:
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if len(calls) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"ожидалось {count} уведомлений, получено {len(calls)}: {calls}")


def _wait_for_status(store: JobsDB, job_id: str, status: str) -> None:
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job is not None and job.status == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"задача {job_id} не достигла статуса {status}")


def _seed_job(web_paths: WebPaths, store: JobsDB, name: str) -> Path:
    source = web_paths.input_dir / name
    source.write_bytes(b"x")
    store.create(name, source)
    return source


def _make_runner(
    web_paths: WebPaths,
    store: JobsDB,
    pipeline_fn,
    *,
    notifications: bool,
) -> JobRunner:
    return JobRunner(
        store,
        JobEventBus(heartbeat=0.05),
        web_paths,
        _config_builder(web_paths, notifications=notifications),
        pipeline_fn=pipeline_fn,
    )


def test_web_runner_notifies_on_done(
    web_paths: WebPaths, notify_calls: list[tuple[str, str]]
) -> None:
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    source = _seed_job(web_paths, store, "meeting.mp3")
    runner = _make_runner(web_paths, store, _ok_pipeline, notifications=True)

    runner.start()
    try:
        assert runner.submit("meeting.mp3", source) is True
        _wait_for_status(store, "meeting.mp3", STATUS_DONE)
        _wait_for_calls(notify_calls, 1)
    finally:
        runner.stop()

    title, message = notify_calls[0]
    assert title == "Транскрибация завершена"
    assert "meeting.mp3" in message
    assert "готово" in message


def test_web_runner_notifies_on_error(
    web_paths: WebPaths, notify_calls: list[tuple[str, str]]
) -> None:
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    source = _seed_job(web_paths, store, "broken.mp3")
    runner = _make_runner(web_paths, store, _failing_pipeline, notifications=True)

    runner.start()
    try:
        assert runner.submit("broken.mp3", source) is True
        _wait_for_status(store, "broken.mp3", STATUS_ERROR)
        _wait_for_calls(notify_calls, 1)
    finally:
        runner.stop()

    title, message = notify_calls[0]
    assert title == "Транскрибация не удалась"
    assert "broken.mp3" in message
    assert "ошибка" in message
    assert "сбой распознавания" in message


def test_web_runner_notifies_on_cancelled(
    web_paths: WebPaths, notify_calls: list[tuple[str, str]]
) -> None:
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    source = _seed_job(web_paths, store, "stopped.mp3")
    runner = _make_runner(web_paths, store, _CancelledPipeline(), notifications=True)

    runner.start()
    try:
        assert runner.submit("stopped.mp3", source) is True
        _wait_for_status(store, "stopped.mp3", STATUS_CANCELLED)
        _wait_for_calls(notify_calls, 1)
    finally:
        runner.stop()

    title, message = notify_calls[0]
    assert title == "Транскрибация отменена"
    assert "stopped.mp3" in message
    assert "отменено" in message


def test_web_runner_no_notify_when_disabled(
    web_paths: WebPaths, notify_calls: list[tuple[str, str]]
) -> None:
    """``NOTIFICATIONS=false`` — уведомление не отправляется."""
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    source = _seed_job(web_paths, store, "silent.mp3")
    runner = _make_runner(web_paths, store, _ok_pipeline, notifications=False)

    runner.start()
    try:
        assert runner.submit("silent.mp3", source) is True
        _wait_for_status(store, "silent.mp3", STATUS_DONE)
        time.sleep(0.2)
    finally:
        runner.stop()

    assert notify_calls == []


def test_web_runner_survives_notify_failure(
    web_paths: WebPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ошибка ``notify`` не роняет воркер: очередь продолжает работу."""

    def boom(title: str, message: str) -> bool:
        raise RuntimeError("notify-send сломан")

    monkeypatch.setattr(runner_module, "notify", boom)
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    first = _seed_job(web_paths, store, "first.mp3")
    second = _seed_job(web_paths, store, "second.mp3")
    runner = _make_runner(web_paths, store, _ok_pipeline, notifications=True)

    runner.start()
    try:
        assert runner.submit("first.mp3", first) is True
        assert runner.submit("second.mp3", second) is True
        _wait_for_status(store, "first.mp3", STATUS_DONE)
        _wait_for_status(store, "second.mp3", STATUS_DONE)
    finally:
        runner.stop()


def test_build_job_config_reads_notifications(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_job_config`` берёт флаг из ``NOTIFICATIONS`` (по умолчанию включён)."""
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    enabled = build_job_config(
        audio_file, output_dir=tmp_path / "out", data_dir=tmp_path / "data"
    )
    disabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"NOTIFICATIONS": "false"},
    )

    assert enabled.notifications is True
    assert disabled.notifications is False


def test_web_settings_notifications_override_env(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#35: сохранённый тумблер уведомлений переопределяет ``NOTIFICATIONS`` env."""
    from audio_transcriber.web.settings import SettingsStore, settings_from_mapping

    monkeypatch.setattr(
        "audio_transcriber.web.config.env_defaults", lambda: {"NOTIFICATIONS": "true"}
    )
    store = SettingsStore(tmp_path / "settings.json")
    saved = store.save(settings_from_mapping({"notifications": False}, base=store.load()))

    overrides = saved.env_overrides()
    assert overrides["NOTIFICATIONS"] == "false"

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides=overrides,
    )
    assert config.notifications is False
