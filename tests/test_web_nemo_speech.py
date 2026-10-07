"""Тесты автодетекта nemo-speech и скачивания Sortformer из веб-UI.

Реальный ``nemo-speech`` не запускается: в приложение передаётся ``runner``-
заглушка ``pull``, а проба бинарника подменяется. Проверяются эндпоинты
``/api/diarization/nemo-speech/{detect,model,model/download,model/events}``,
SSE с ``Last-Event-ID``, конфликт 409, деградация без бинарника и
инвалидация кэша доктора по успеху.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.diarization import nemo_speech_assets as assets
from audio_transcriber.doctor import DoctorCheck
from audio_transcriber.web import doctor_api
from audio_transcriber.web.app import create_app
from audio_transcriber.web.nemo_speech import (
    NemoPullTimeout,
    run_nemo_speech_pull,
)
from audio_transcriber.web.paths import WebPaths

MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"
DOWNLOAD_LINE = f"[model] downloading {MODEL}@abc123 (diarization, 140.3 MiB)"
EXPECTED_TOTAL = int(140.3 * 1024 * 1024)


class FakePullRunner:
    """Заглушка ``nemo-speech pull``: пишет строки и возвращает заданный код."""

    def __init__(
        self,
        *,
        code: int = 0,
        lines: tuple[str, ...] = (
            DOWNLOAD_LINE,
            "[model] verifying size and SHA-256...",
            "[model] ready: /tmp/models/sortformer.q8_0.gguf",
        ),
        block: threading.Event | None = None,
    ) -> None:
        self.code = code
        self.lines = lines
        self.block = block
        self.commands: list[list[str]] = []
        self.envs: list[Mapping[str, str]] = []

    def __call__(
        self, command: list[str], on_line: Callable[[str], None], env: Mapping[str, str]
    ) -> int:
        self.commands.append(command)
        self.envs.append(env)
        if self.block is not None:
            self.block.wait(timeout=5)
        for line in self.lines:
            on_line(line)
        return self.code


def _make_binary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return path


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    monkeypatch.setattr(doctor_api, "load_config_env", lambda: (None, {}))
    # Кэш модели nemo-speech — в tmp, чтобы не читать реальный ~/.cache.
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "nemo-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def runner() -> FakePullRunner:
    return FakePullRunner()


@pytest.fixture
def client(web_paths: WebPaths, runner: FakePullRunner) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        nemo_pull_runner=runner,
        nemo_poll_interval=None,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _configure_binary(client: TestClient, binary: Path, lib: Path | None = None) -> None:
    payload: dict[str, str] = {"nemo_speech_binary": str(binary)}
    if lib is not None:
        payload["nemo_speech_lib_path"] = str(lib)
    response = client.put("/api/settings", json=payload)
    assert response.status_code == 200


# --- GET /detect --------------------------------------------------------------


def test_detect_returns_candidates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = assets.NemoSpeechCandidate(
        binary="/opt/nemo-speech/bin/nemo-speech",
        lib_path="/opt/nemo-speech/lib",
        source="настройки",
        version="0.1.0",
        devices=("[0] gpu  AMD Radeon",),
        has_vulkan=True,
    )
    monkeypatch.setattr(
        assets, "detect_candidates", lambda *_args, **_kwargs: [candidate]
    )

    payload = client.get("/api/diarization/nemo-speech/detect").json()

    assert payload["found"] is True
    assert payload["recommended"] == candidate.binary
    assert payload["candidates"][0]["lib_path"] == candidate.lib_path
    assert payload["candidates"][0]["version"] == "0.1.0"
    assert payload["current"]["binary"]  # текущее значение настроек


def test_detect_without_candidates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assets, "detect_candidates", lambda *_args, **_kwargs: [])

    payload = client.get("/api/diarization/nemo-speech/detect").json()

    assert payload["found"] is False
    assert payload["candidates"] == []
    assert payload["recommended"] is None


# --- GET /model ---------------------------------------------------------------


def test_model_status_absent(client: TestClient) -> None:
    payload = client.get("/api/diarization/nemo-speech/model").json()

    assert payload["model"] == MODEL
    assert payload["present"] is False
    assert payload["download"]["status"] == "idle"
    assert payload["binary"]["available"] is False


def test_model_status_present(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "nemo-cache"
    gguf = root / MODEL / "abc" / "sortformer.q8_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"x" * 4096)
    Path(f"{gguf}.verified").write_text("sha256=...", encoding="utf-8")

    payload = client.get("/api/diarization/nemo-speech/model").json()

    assert payload["present"] is True
    assert payload["size"] == 4096
    assert payload["path"] == str(gguf)


# --- POST /model/download -----------------------------------------------------


def test_download_requires_binary_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.app.binary_available", lambda _b: False)

    response = client.post("/api/diarization/nemo-speech/model/download")

    assert response.status_code == 400
    assert "nemo-speech" in response.json()["detail"]


def test_download_runs_in_background_and_publishes_progress(
    client: TestClient, runner: FakePullRunner, tmp_path: Path
) -> None:
    binary = _make_binary(tmp_path / "bundle" / "bin" / "nemo-speech")
    _configure_binary(client, binary)

    response = client.post("/api/diarization/nemo-speech/model/download")

    assert response.status_code == 202
    assert response.json()["status"] == "downloading"

    client.app.state.nemo_downloader.wait(5)  # type: ignore[attr-defined]

    assert runner.commands == [[str(binary), "pull", MODEL]]
    state = client.app.state.nemo_downloader.state()  # type: ignore[attr-defined]
    assert state.status == "done"
    assert state.fraction == 1.0
    assert state.error is None

    events = client.app.state.nemo_bus.history()  # type: ignore[attr-defined]
    assert events[0]["status"] == "downloading"
    assert any(event["total"] == EXPECTED_TOTAL for event in events)
    assert any(
        str(event["message"]).startswith("Проверка") for event in events
    )
    assert events[-1]["status"] == "done"


def test_download_uses_safe_library_path(
    client: TestClient, runner: FakePullRunner, tmp_path: Path
) -> None:
    binary = _make_binary(tmp_path / "bundle" / "bin" / "nemo-speech")
    lib = tmp_path / "bundle" / "lib"
    lib.mkdir()
    (lib / "libnemo_speech.so").write_text("", encoding="utf-8")
    _configure_binary(client, binary, lib)

    assert client.post("/api/diarization/nemo-speech/model/download").status_code == 202
    client.app.state.nemo_downloader.wait(5)  # type: ignore[attr-defined]

    assert runner.envs[0]["LD_LIBRARY_PATH"].startswith(str(lib))


def test_download_conflict_409(
    web_paths: WebPaths, runner: FakePullRunner, tmp_path: Path
) -> None:
    block = threading.Event()
    runner.block = block
    binary = _make_binary(tmp_path / "bin" / "nemo-speech")
    app = create_app(
        paths=web_paths, nemo_pull_runner=runner, nemo_poll_interval=None, heartbeat=0.05
    )
    with TestClient(app) as client:
        _configure_binary(client, binary)
        assert client.post("/api/diarization/nemo-speech/model/download").status_code == 202
        assert client.post("/api/diarization/nemo-speech/model/download").status_code == 409
        block.set()
        client.app.state.nemo_downloader.wait(5)  # type: ignore[attr-defined]


def test_download_error_reports_error(
    web_paths: WebPaths, tmp_path: Path
) -> None:
    runner = FakePullRunner(code=1, lines=("curl failed while downloading",))
    binary = _make_binary(tmp_path / "bin" / "nemo-speech")
    app = create_app(
        paths=web_paths, nemo_pull_runner=runner, nemo_poll_interval=None, heartbeat=0.05
    )
    with TestClient(app) as client:
        _configure_binary(client, binary)
        assert client.post("/api/diarization/nemo-speech/model/download").status_code == 202
        client.app.state.nemo_downloader.wait(5)  # type: ignore[attr-defined]

        state = client.app.state.nemo_downloader.state()  # type: ignore[attr-defined]
        assert state.status == "error"
        assert state.error and "кодом 1" in state.error


def test_download_invalidates_doctor_cache(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    checks = [
        DoctorCheck(
            key="bin:nemo-speech", label="Бинарник nemo-speech", ok=True, critical=False
        )
    ]
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), checks)[1],
    )
    binary = _make_binary(tmp_path / "bin" / "nemo-speech")
    _configure_binary(client, binary)

    client.get("/api/doctor")
    assert len(calls) == 1

    assert client.post("/api/diarization/nemo-speech/model/download").status_code == 202
    client.app.state.nemo_downloader.wait(5)  # type: ignore[attr-defined]

    client.get("/api/doctor")
    assert len(calls) == 2


# --- SSE /model/events --------------------------------------------------------


def _events_route(app: object) -> object:
    stack = list(app.routes)  # type: ignore[attr-defined]
    while stack:
        route = stack.pop()
        if getattr(route, "path", None) == "/api/diarization/nemo-speech/model/events":
            return route
        nested = getattr(route, "routes", None)
        if nested:
            stack.extend(nested)
        original = getattr(route, "original_router", None)
        if original is not None:
            stack.extend(getattr(original, "routes", None) or [])
    raise AssertionError("маршрут SSE не найден")


def _read_events(
    client: TestClient, *, last_event_id: str | None = None
) -> tuple[list[str], list[dict[str, object]]]:
    route = _events_route(client.app)  # type: ignore[arg-type]
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


def test_events_skips_history_before_last_event_id(client: TestClient) -> None:
    bus = client.app.state.nemo_bus  # type: ignore[attr-defined]
    bus.clear()
    bus.publish({"status": "downloading", "message": "a"})
    bus.publish({"status": "done", "message": "b"})
    first_seq = int(bus.history()[0]["seq"])  # type: ignore[arg-type]

    lines, events = _read_events(client, last_event_id=str(first_seq))

    assert [event["status"] for event in events] == ["done"]
    assert f"id: {events[0]['seq']}" in lines


def test_events_fresh_client_gets_history(client: TestClient) -> None:
    bus = client.app.state.nemo_bus  # type: ignore[attr-defined]
    bus.clear()
    bus.publish({"status": "downloading", "message": "a"})
    bus.publish({"status": "done", "message": "b"})

    lines, events = _read_events(client)

    assert [event["status"] for event in events] == ["downloading"]
    assert any(line.startswith("id: ") for line in lines)


# --- #87: таймаут и реестр процесса nemo-speech pull ------------------------


def test_run_nemo_speech_pull_times_out() -> None:
    command = [sys.executable, "-c", "import time; time.sleep(5)"]

    with pytest.raises(NemoPullTimeout, match="процесс остановлен"):
        run_nemo_speech_pull(command, lambda _line: None, {}, timeout=0.5)


def test_nemo_pull_timeout_reported_as_error(web_paths: WebPaths) -> None:
    class TimeoutRunner:
        def __call__(
            self,
            command: list[str],
            on_line: Callable[[str], None],
            env: Mapping[str, str],
        ) -> int:
            raise NemoPullTimeout("nemo-speech pull не завершился за 1 с")

    app = create_app(
        paths=web_paths,
        nemo_pull_runner=TimeoutRunner(),
        nemo_poll_interval=None,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        binary = _make_binary(web_paths.data_dir / "bin" / "nemo-speech")
        _configure_binary(test_client, binary)
        assert (
            test_client.post("/api/diarization/nemo-speech/model/download").status_code
            == 202
        )
        test_client.app.state.nemo_downloader.wait(5)  # type: ignore[attr-defined]
        state = test_client.app.state.nemo_downloader.state()  # type: ignore[attr-defined]
        assert state.status == "error"
        assert state.error is not None and "не завершился" in state.error

