"""Тесты автоподбора свободного порта веб-сервера (issue #27).

Реальный сервер не поднимается: проверка занятости порта и ``serve``
подменяются заглушками.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from audio_transcriber.cli import app as cli_app
from audio_transcriber.cli.app import app
from audio_transcriber.utils import net
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths

runner = CliRunner()


@pytest.fixture
def serve_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Подменяет ``serve`` и возвращает словарь последнего вызова."""
    calls: dict[str, Any] = {}

    def fake_serve(*, host: str, port: int, open_browser: bool, reload: bool) -> None:
        calls.update(host=host, port=port, open_browser=open_browser, reload=reload)

    monkeypatch.setattr("audio_transcriber.web.app.serve", fake_serve)
    return calls


def _combined_output(result: Any) -> str:
    return (result.output or "") + (getattr(result, "stderr", "") or "")


def _always_free(host: str, port: int) -> bool:
    return True


def _never_free(host: str, port: int) -> bool:
    return False


def _busy_default_port(host: str, port: int) -> bool:
    return port != net.DEFAULT_WEB_PORT


def _scan_returns(value: int | None) -> Callable[[str, int, int], int | None]:
    def find(host: str, start: int, limit: int) -> int | None:
        return value

    return find


def test_default_web_port_used_when_free(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app, "is_port_available", _always_free)

    result = runner.invoke(app, ["web", "--no-browser"])

    assert result.exit_code == 0, _combined_output(result)
    assert serve_calls["port"] == net.DEFAULT_WEB_PORT
    assert f"http://127.0.0.1:{net.DEFAULT_WEB_PORT}/" in result.output


def test_busy_default_port_picks_next_free(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app, "is_port_available", _busy_default_port)
    monkeypatch.setattr(cli_app, "find_available_port", _scan_returns(8767))

    result = runner.invoke(app, ["web", "--no-browser"])

    assert result.exit_code == 0, _combined_output(result)
    assert serve_calls["port"] == 8767
    assert "8767" in result.output
    assert "http://127.0.0.1:8767/" in result.output


def test_explicit_busy_port_fails_without_substitution(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app, "is_port_available", _never_free)
    monkeypatch.setattr(cli_app, "find_available_port", _scan_returns(8766))

    result = runner.invoke(app, ["web", "--port", "8765", "--no-browser"])

    assert result.exit_code == 1
    assert "занят" in _combined_output(result)
    # Порт не подменён молча: serve не вызывался, предложен явный --port.
    assert serve_calls == {}


def test_explicit_free_port_used(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app, "is_port_available", _always_free)

    result = runner.invoke(app, ["web", "--port", "9000", "--no-browser"])

    assert result.exit_code == 0, _combined_output(result)
    assert serve_calls["port"] == 9000


def test_no_free_port_in_range_fails(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_app, "is_port_available", _never_free)
    monkeypatch.setattr(cli_app, "find_available_port", _scan_returns(None))

    result = runner.invoke(app, ["web", "--no-browser"])

    assert result.exit_code == 1
    assert serve_calls == {}


def test_find_available_port_scans_range(monkeypatch: pytest.MonkeyPatch) -> None:
    occupied = {net.DEFAULT_WEB_PORT, net.DEFAULT_WEB_PORT + 1}

    def is_available(host: str, port: int) -> bool:
        return port not in occupied

    monkeypatch.setattr(net, "is_port_available", is_available)

    assert net.find_available_port("127.0.0.1") == net.DEFAULT_WEB_PORT + 2


def test_find_available_port_returns_none_when_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(net, "is_port_available", _never_free)

    assert net.find_available_port("127.0.0.1", start=8765, limit=2) is None


def test_health_reports_server_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/api/health`` отдаёт фактический адрес, проставленный ``serve``."""
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    web_app = create_app(paths=WebPaths(tmp_path / "web-data"), heartbeat=0.05)
    web_app.state.server_host = "127.0.0.1"
    web_app.state.server_port = 8767

    with TestClient(web_app) as client:
        payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 8767


def test_web_download_models_runs_before_serve(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--download-models`` скачивает модели до запуска сервера."""
    from audio_transcriber.cli import models_cmd

    calls: list[str] = []
    monkeypatch.setattr(cli_app, "is_port_available", _always_free)
    monkeypatch.setattr(
        models_cmd, "download_for_web", lambda **_: calls.append("download") or 0
    )

    result = runner.invoke(app, ["web", "--no-browser", "--download-models"])

    assert result.exit_code == 0, _combined_output(result)
    assert calls == ["download"]
    assert serve_calls["port"] == net.DEFAULT_WEB_PORT


def test_web_download_models_failure_aborts(
    serve_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """При неудачном скачивании сервер не стартует."""
    from audio_transcriber.cli import models_cmd

    monkeypatch.setattr(cli_app, "is_port_available", _always_free)
    monkeypatch.setattr(models_cmd, "download_for_web", lambda **_: 1)

    result = runner.invoke(app, ["web", "--no-browser", "--download-models"])

    assert result.exit_code == 1
    assert serve_calls == {}
