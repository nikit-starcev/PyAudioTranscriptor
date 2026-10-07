"""Тесты реестра и автоустановки опциональных пакетов (``/api/deps``, #66).

Реальный установщик не запускается: в приложение передаётся ``runner``-заглушка,
поэтому тесты быстрые и локальные. Проверяются реестр/allowlist, выбор команды
(uv в приоритете), эндпоинты запуска, конфликты, инвалидация кэша доктора и SSE
с ``Last-Event-ID``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.doctor import DoctorCheck
from audio_transcriber.web import deps as web_deps
from audio_transcriber.web import doctor_api
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


class FakeRunner:
    """Заглушка установщика: пишет строки и возвращает заданный код."""

    def __init__(
        self,
        *,
        code: int = 0,
        lines: tuple[str, ...] = ("Collecting onnx-asr", "Successfully installed"),
        block: threading.Event | None = None,
    ) -> None:
        self.code = code
        self.lines = lines
        self.block = block
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], on_line: Callable[[str], None]) -> int:
        self.commands.append(command)
        if self.block is not None:
            self.block.wait(timeout=5)
        for line in self.lines:
            on_line(line)
        return self.code


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует тесты от реального ``config.env`` и команд установки."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    monkeypatch.setattr(doctor_api, "load_config_env", lambda: (None, {}))
    # Фиксируем uv: команда и её выбор не зависят от машины теста.
    monkeypatch.setattr(web_deps, "find_uv", lambda: "/usr/local/bin/uv")


def _make_client(web_paths: WebPaths, runner: FakeRunner) -> TestClient:
    app = create_app(paths=web_paths, dep_install_runner=runner, heartbeat=0.05)
    return TestClient(app)


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def client(web_paths: WebPaths, runner: FakeRunner) -> Iterator[TestClient]:
    with _make_client(web_paths, runner) as test_client:
        yield test_client


# --- реестр и выбор установщика -------------------------------------------


def test_registry_is_nonempty_and_unique() -> None:
    keys = [spec.key for spec in web_deps.DEPENDENCIES]
    assert len(keys) == len(set(keys))
    assert {"gigaam", "sherpa"} <= set(keys)
    for spec in web_deps.DEPENDENCIES:
        assert spec.spec
        assert spec.module
        assert spec.check_id.startswith("dep:")
        assert web_deps.find_dependency(spec.key) is spec
    assert web_deps.find_dependency("nope") is None


def test_module_available_probes_without_import() -> None:
    assert web_deps.module_available("json") is True
    assert web_deps.module_available("definitely-not-real-module-xyz") is False


def test_resolve_install_command_prefers_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_deps, "find_uv", lambda: "/usr/local/bin/uv")
    command = web_deps.resolve_install_command("onnx-asr[cpu,hub]")

    assert command[:4] == ["/usr/local/bin/uv", "pip", "install", "--python"]
    assert command[4] == sys.executable
    assert command[-1] == "onnx-asr[cpu,hub]"


def test_resolve_install_command_falls_back_to_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_deps, "find_uv", lambda: None)
    monkeypatch.setattr(web_deps, "pip_available", lambda: True)

    assert web_deps.resolve_install_command("deepfilternet") == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "deepfilternet",
    ]
    assert web_deps.installer_name() == "pip"


def test_resolve_install_command_without_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_deps, "find_uv", lambda: None)
    monkeypatch.setattr(web_deps, "pip_available", lambda: False)

    with pytest.raises(web_deps.InstallerUnavailable, match="uv/pip"):
        web_deps.resolve_install_command("onnx-asr")
    assert web_deps.installer_name() is None
    assert web_deps.installer_available() is False


# --- GET /api/deps ---------------------------------------------------------


def test_list_deps_shape(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_deps, "installer_available", lambda: True)
    monkeypatch.setattr(web_deps, "installer_name", lambda: "uv")

    payload = client.get("/api/deps").json()

    assert payload["installer"] == "uv"
    by_key = {item["key"]: item for item in payload["deps"]}
    assert set(by_key) == {spec.key for spec in web_deps.DEPENDENCIES}
    gigaam = by_key["gigaam"]
    assert gigaam["spec"] == "onnx-asr[cpu,hub]"
    assert gigaam["check_id"] == "dep:onnx_asr"
    assert gigaam["installable"] is True
    assert gigaam["status"] == web_deps.DEP_IDLE
    assert isinstance(gigaam["installed"], bool)


def test_list_deps_reports_no_installer(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_deps, "installer_available", lambda: False)
    monkeypatch.setattr(web_deps, "installer_name", lambda: None)

    payload = client.get("/api/deps").json()

    assert payload["installer"] is None
    assert all(item["installable"] is False for item in payload["deps"])


# --- POST /api/deps/{key}/install -----------------------------------------


def test_install_unknown_key_404(client: TestClient) -> None:
    assert client.post("/api/deps/nope/install").status_code == 404


def test_install_runs_in_background_and_publishes_progress(
    client: TestClient, runner: FakeRunner
) -> None:
    response = client.post("/api/deps/gigaam/install")

    assert response.status_code == 202
    assert response.json() == {"key": "gigaam", "status": "running"}

    client.app.state.dependency_installer.wait(5)  # type: ignore[attr-defined]

    assert runner.commands == [
        [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "onnx-asr[cpu,hub]",
        ]
    ]
    state = client.app.state.dependency_installer.state("gigaam")  # type: ignore[attr-defined]
    assert state.status == web_deps.DEP_DONE
    assert state.error is None

    events = client.app.state.deps_bus.history()  # type: ignore[attr-defined]
    statuses = [event["status"] for event in events if event["key"] == "gigaam"]
    assert "running" in statuses
    assert statuses[-1] == "done"
    messages = [event["message"] for event in events if event["key"] == "gigaam"]
    assert "Collecting onnx-asr" in messages


def test_install_invalidates_doctor_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    checks = [
        DoctorCheck(
            key="dep:onnx_asr",
            label="Зависимость: onnx-asr",
            ok=True,
            critical=False,
            detail="доступна",
        )
    ]
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), checks)[1],
    )

    client.get("/api/doctor")
    assert len(calls) == 1

    assert client.post("/api/deps/gigaam/install").status_code == 202
    client.app.state.dependency_installer.wait(5)  # type: ignore[attr-defined]

    client.get("/api/doctor")
    assert len(calls) == 2  # кэш сброшен успешной установкой


def test_install_error_runner_reports_error(web_paths: WebPaths) -> None:
    runner = FakeRunner(code=1, lines=("ERROR: no matching distribution",))
    with _make_client(web_paths, runner) as test_client:
        assert test_client.post("/api/deps/gigaam/install").status_code == 202
        test_client.app.state.dependency_installer.wait(5)  # type: ignore[attr-defined]

        state = test_client.app.state.dependency_installer.state("gigaam")  # type: ignore[attr-defined]
        assert state.status == web_deps.DEP_ERROR
        assert state.error is not None

        events = test_client.app.state.deps_bus.history()  # type: ignore[attr-defined]
        last = [event for event in events if event["key"] == "gigaam"][-1]
        assert last["status"] == "error"


def test_install_twice_conflicts(web_paths: WebPaths) -> None:
    block = threading.Event()
    runner = FakeRunner(block=block)
    with _make_client(web_paths, runner) as test_client:
        assert test_client.post("/api/deps/gigaam/install").status_code == 202
        second = test_client.post("/api/deps/gigaam/install")
        assert second.status_code == 409
        # Вторая установка (даже другого пакета) тоже конфликтует: одна за раз.
        assert test_client.post("/api/deps/sherpa/install").status_code == 409
        block.set()
        test_client.app.state.dependency_installer.wait(5)  # type: ignore[attr-defined]


def test_install_without_installer_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_deps, "installer_available", lambda: False)

    response = client.post("/api/deps/gigaam/install")

    assert response.status_code == 400
    assert "uv/pip" in response.json()["detail"]


def test_install_start_race_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_deps, "installer_available", lambda: True)
    installer = client.app.state.dependency_installer  # type: ignore[attr-defined]
    monkeypatch.setattr(installer, "is_running", lambda: False)
    monkeypatch.setattr(installer, "start", lambda _key, _spec: False)

    assert client.post("/api/deps/gigaam/install").status_code == 409


# --- SSE /api/deps/events --------------------------------------------------


def _deps_events_route(app: object) -> object:
    """Находит APIRoute ``/api/deps/events`` (роутер FastAPI может быть вложен)."""
    stack = list(app.routes)  # type: ignore[attr-defined]
    while stack:
        route = stack.pop()
        if getattr(route, "path", None) == "/api/deps/events":
            return route
        nested = getattr(route, "routes", None)
        if nested:
            stack.extend(nested)
        original = getattr(route, "original_router", None)
        if original is not None:
            stack.extend(getattr(original, "routes", None) or [])
    raise AssertionError("маршрут /api/deps/events не найден")


def _read_deps_events(
    client: TestClient, *, last_event_id: str | None = None
) -> tuple[list[str], list[dict[str, object]]]:
    """Читает первый батч SSE (строки и события), закрывая генератор."""
    route = _deps_events_route(client.app)  # type: ignore[arg-type]
    endpoint = route.endpoint  # type: ignore[attr-defined]

    async def scenario() -> tuple[list[str], list[dict[str, object]]]:
        response = await endpoint(last_event_id=last_event_id)
        lines: list[str] = []
        events: list[dict[str, object]] = []
        agen = response.body_iterator
        try:
            async for chunk in agen:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("data:"):
                        events.append(json.loads(line[len("data:") :].strip()))
                        return lines, events
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await aclose()
        return lines, events

    return asyncio.run(scenario())


def test_deps_events_skips_history_before_last_event_id(client: TestClient) -> None:
    bus = client.app.state.deps_bus  # type: ignore[attr-defined]
    bus.clear()
    bus.publish({"key": "gigaam", "status": "running", "message": "a"})
    bus.publish({"key": "gigaam", "status": "done", "message": "b"})
    first_seq = int(bus.history()[0]["seq"])  # type: ignore[arg-type]

    lines, events = _read_deps_events(client, last_event_id=str(first_seq))

    assert [event["status"] for event in events] == ["done"]
    assert int(events[0]["seq"]) > first_seq  # type: ignore[arg-type]
    assert f"id: {events[0]['seq']}" in lines


def test_deps_events_fresh_client_gets_history(client: TestClient) -> None:
    bus = client.app.state.deps_bus  # type: ignore[attr-defined]
    bus.clear()
    bus.publish({"key": "gigaam", "status": "running", "message": "a"})
    bus.publish({"key": "gigaam", "status": "done", "message": "b"})

    lines, events = _read_deps_events(client)

    assert [event["status"] for event in events] == ["running"]
    assert any(line.startswith("id: ") for line in lines)
