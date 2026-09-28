"""Тесты жизненного цикла и деградации LLM-клиента (без реальной модели).

Подменяем ``subprocess.Popen`` фиктивным процессом и отключаем ожидание
готовности, поэтому тесты детерминированы и не занимают VRAM/порт.
"""

from __future__ import annotations

import io
import socket
from pathlib import Path

import pytest

from audio_transcriber.llm import client as client_module
from audio_transcriber.llm.client import (
    DEFAULT_CONTEXT_SIZE,
    LlamaServerClient,
    _args_have_option,
    _port_in_use,
)
from audio_transcriber.utils.exceptions import LlmError


class FakeProc:
    """Минимальная замена ``subprocess.Popen`` для тестов."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    @property
    def stderr(self) -> io.StringIO:
        return io.StringIO("load_tensors: offloaded 29/29 layers to GPU\n")


class StubStderr:
    def __init__(self, text: str) -> None:
        self._text = text

    def full_text(self) -> str:
        return self._text

    def tail(self) -> str:
        return self._text[-200:]


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fake")
    return path


@pytest.fixture
def patch_popen(monkeypatch: pytest.MonkeyPatch):
    created: list[FakeProc] = []

    def fake_popen(cmd, **kwargs):
        proc = FakeProc(pid=10_000 + len(created))
        created.append(proc)
        return proc

    monkeypatch.setattr(client_module.subprocess, "Popen", fake_popen)
    # Не ждём реальный HTTP /health.
    monkeypatch.setattr(LlamaServerClient, "_wait_ready", lambda _self: None)
    return created


def test_default_context_size_is_4096() -> None:
    assert DEFAULT_CONTEXT_SIZE == 4096


def test_offload_attempts_go_from_full_gpu_to_cpu(model_file: Path) -> None:
    assert LlamaServerClient(model_file)._offload_attempts() == [99, 16, 0]


def test_offload_attempts_cpu_only(model_file: Path) -> None:
    assert LlamaServerClient(model_file, gpu=False)._offload_attempts() == [0]


def test_context_manager_terminates_process_and_unregisters(
    model_file: Path, patch_popen: list[FakeProc]
) -> None:
    with LlamaServerClient(model_file) as client:
        assert client.is_running
        pid = client._proc.pid if client._proc else None
        assert pid in client_module._active_processes

    assert not client.is_running
    assert patch_popen[0].terminated
    assert pid not in client_module._active_processes


def test_start_falls_back_to_cpu_when_gpu_attempts_fail(
    monkeypatch: pytest.MonkeyPatch, model_file: Path, patch_popen: list[FakeProc]
) -> None:
    attempts = {"count": 0}

    def failing_wait(self) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise LlmError("нехватка VRAM")

    monkeypatch.setattr(LlamaServerClient, "_wait_ready", failing_wait)

    client = LlamaServerClient(model_file, gpu=True)
    client.start()

    assert client.gpu_layers_used == 0
    assert len(patch_popen) == 3
    client.close()


def test_detect_device_reports_gpu(model_file: Path) -> None:
    client = LlamaServerClient(model_file)
    client._stderr = StubStderr("load_tensors: offloaded 25/25 layers to GPU")  # type: ignore[assignment]

    assert client._detect_device() == "GPU (25/25 слоёв)"


def test_detect_device_reports_cpu(model_file: Path) -> None:
    client = LlamaServerClient(model_file)
    client._stderr = StubStderr("load_tensors: layer 0 assigned to device CPU")  # type: ignore[assignment]

    assert client._detect_device() == "CPU"


def test_http_error_hint_for_context_overflow(model_file: Path) -> None:
    client = LlamaServerClient(model_file)

    hint = client._http_error_hint(400, "the prompt is too long: context length exceeded")

    assert "LLM_CONTEXT" in hint
    assert "размер входного файла" in hint


def test_http_error_hint_empty_for_other_errors(model_file: Path) -> None:
    client = LlamaServerClient(model_file)

    assert client._http_error_hint(400, "malformed json") == ""
    assert client._http_error_hint(500, "internal context error") == ""


def test_args_have_option_supports_both_forms() -> None:
    assert _args_have_option(["llama-server", "--port", "8080"], "--port", "8080")
    assert _args_have_option(["--port=8080"], "--port", "8080")
    assert not _args_have_option(["--port", "9090"], "--port", "8080")


def test_port_in_use_detects_listener() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])
    try:
        assert _port_in_use("127.0.0.1", port)
    finally:
        server.close()

    assert not _port_in_use("127.0.0.1", port)
