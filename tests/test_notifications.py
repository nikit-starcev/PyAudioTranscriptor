"""Тесты модуля десктоп-уведомлений (:mod:`audio_transcriber.utils.notifications`)."""

from __future__ import annotations

import subprocess

import pytest

from audio_transcriber.utils import notifications


def test_notify_returns_false_when_notify_send_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notifications.shutil, "which", lambda _name: None)

    # Отсутствие notify-send — тихий no-op, без исключения.
    assert notifications.notify("Заголовок", "Текст") is False


def test_notify_invokes_notify_send(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(notifications.shutil, "which", lambda _name: "/usr/bin/notify-send")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    result = notifications.notify("Транскрибация завершена", "Обработано 2 файл(ов)")

    assert result is True
    args = captured["args"]
    assert isinstance(args, list)
    assert args[0] == "/usr/bin/notify-send"
    assert args[-2:] == ["Транскрибация завершена", "Обработано 2 файл(ов)"]


def test_notify_swallows_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        raise OSError("нет доступа")

    monkeypatch.setattr(notifications.shutil, "which", lambda _name: "/usr/bin/notify-send")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify("Заголовок", "Текст") is False


def test_notify_swallows_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=5)

    monkeypatch.setattr(notifications.shutil, "which", lambda _name: "/usr/bin/notify-send")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify("Заголовок", "Текст") is False
