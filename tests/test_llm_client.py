"""Тесты жизненного цикла и деградации LLM-клиента (без реальной модели).

Подменяем ``subprocess.Popen`` фиктивным процессом и отключаем ожидание
готовности, поэтому тесты детерминированы и не занимают VRAM/порт.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.llm import client as client_module
from audio_transcriber.llm.client import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    LlamaServerClient,
    OpenAIClient,
    _args_have_option,
    _chat_completions_urls,
    _models_urls,
    _port_in_use,
    create_llm_client,
    probe_openai_server,
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


def test_request_timeout_defaults_to_constant(model_file: Path) -> None:
    assert LlamaServerClient(model_file)._request_timeout == DEFAULT_REQUEST_TIMEOUT


def test_create_llm_client_passes_request_timeout(model_file: Path) -> None:
    client = create_llm_client(
        model_path=model_file,
        binary="llama-server",
        library_path=None,
        gpu=False,
        request_timeout=12.5,
    )

    assert client is not None
    assert client._request_timeout == 12.5


class _FakeUrlopenResponse:
    def __enter__(self) -> _FakeUrlopenResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return b'{"choices": [{"message": {"content": "ok"}}]}'


def test_chat_uses_configured_request_timeout(
    monkeypatch: pytest.MonkeyPatch, model_file: Path
) -> None:
    client = LlamaServerClient(model_file, request_timeout=42.0)
    monkeypatch.setattr(client, "_ensure_started", lambda: None)
    client._base_url = "http://127.0.0.1:1"
    captured: dict[str, float | None] = {}

    def fake_urlopen(_request: object, timeout: float | None = None) -> _FakeUrlopenResponse:
        captured["timeout"] = timeout
        return _FakeUrlopenResponse()

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)

    assert client.chat([{"role": "user", "content": "привет"}]) == "ok"
    assert captured["timeout"] == 42.0


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


# ---------------------------------------------------------------------------
# Внешний провайдер (OpenAI-совместимый API)
# ---------------------------------------------------------------------------


class _OpenAIResponse:
    """Ответ ``urlopen``: JSON-строка, читаемая как bytes."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self) -> _OpenAIResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(url: str, code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url, code, f"HTTP {code}", hdrs=None, fp=io.BytesIO(body.encode("utf-8"))
    )


def test_chat_completions_url_variants() -> None:
    assert _chat_completions_urls("https://api.example.com") == [
        "https://api.example.com/v1/chat/completions",
        "https://api.example.com/chat/completions",
    ]
    assert _chat_completions_urls("http://localhost:11434/v1") == [
        "http://localhost:11434/v1/chat/completions",
        "http://localhost:11434/chat/completions",
    ]
    assert _chat_completions_urls("https://host/v1/chat/completions") == [
        "https://host/v1/chat/completions"
    ]


def test_openai_chat_sends_authorization_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        captured["data"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        captured["auth"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        captured["ctype"] = request.get_header("Content-type")  # type: ignore[attr-defined]
        return _OpenAIResponse('{"choices": [{"message": {"content": "привет"}}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient(
        "https://api.example.com/v1",
        "gpt-4o-mini",
        api_key="sk-secret",
        request_timeout=17.0,
    )

    answer = client.chat([{"role": "user", "content": "hi"}])

    assert answer == "привет"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["timeout"] == 17.0
    assert captured["auth"] == "Bearer sk-secret"
    assert captured["ctype"] == "application/json"
    data = captured["data"]
    assert data["model"] == "gpt-4o-mini"
    assert data["messages"] == [{"role": "user", "content": "hi"}]
    assert data["temperature"] == 0.0
    assert data["max_tokens"] == DEFAULT_MAX_TOKENS


def test_openai_chat_without_api_key_omits_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        captured["auth"] = request.get_header("Authorization")  # type: ignore[attr-defined]
        return _OpenAIResponse('{"choices": [{"message": {"content": "ok"}}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    OpenAIClient("http://localhost:11434/v1", "llama3.1").chat(
        [{"role": "user", "content": "hi"}]
    )

    assert captured["auth"] is None


def test_openai_falls_back_to_alternate_url_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        seen.append(request.full_url)  # type: ignore[attr-defined]
        if request.full_url.endswith("/v1/chat/completions"):  # type: ignore[attr-defined]
            raise _http_error(request.full_url, 404, "not found")  # type: ignore[attr-defined]
        return _OpenAIResponse('{"choices": [{"message": {"content": "ok"}}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("http://localhost:8080", "m", retry_backoff=0)

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert seen == [
        "http://localhost:8080/v1/chat/completions",
        "http://localhost:8080/chat/completions",
    ]


def test_openai_retries_on_server_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise _http_error(request.full_url, 503, "unavailable")  # type: ignore[attr-defined]
        return _OpenAIResponse('{"choices": [{"message": {"content": "ok"}}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("http://localhost:11434/v1", "m", max_retries=2, retry_backoff=0)

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls["count"] == 2


def test_openai_retries_on_network_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError("connection refused")
        return _OpenAIResponse('{"choices": [{"message": {"content": "ok"}}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("http://localhost:11434/v1", "m", max_retries=1, retry_backoff=0)

    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls["count"] == 2


def test_openai_exhausted_retries_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        calls["count"] += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("http://localhost:11434/v1", "m", max_retries=2, retry_backoff=0)

    with pytest.raises(LlmError, match="Не удалось обратиться"):
        client.chat([{"role": "user", "content": "hi"}])
    assert calls["count"] == 3


def test_openai_client_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        calls["count"] += 1
        raise _http_error(request.full_url, 401, "invalid api key")  # type: ignore[attr-defined]

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("http://localhost:11434/v1", "m", retry_backoff=0)

    with pytest.raises(LlmError, match="HTTP 401"):
        client.chat([{"role": "user", "content": "hi"}])
    assert calls["count"] == 1


def test_openai_unexpected_response_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        return _OpenAIResponse('{"error": "boom"}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("http://localhost:11434/v1", "m", retry_backoff=0)

    with pytest.raises(LlmError, match="Неожиданный ответ"):
        client.chat([{"role": "user", "content": "hi"}])


def test_openai_requires_base_url_and_model() -> None:
    with pytest.raises(ValueError):
        OpenAIClient("", "m")
    with pytest.raises(ValueError):
        OpenAIClient("http://localhost:11434/v1", "   ")


def test_create_llm_client_selects_openai_provider() -> None:
    client = create_llm_client(
        provider="openai",
        base_url="http://localhost:11434/v1",
        model_name="llama3.1",
        api_key="sk-x",
    )

    assert isinstance(client, OpenAIClient)
    assert client.model_name == "llama3.1"


def test_create_llm_client_openai_missing_params_returns_none() -> None:
    assert create_llm_client(provider="openai", base_url="http://x") is None
    assert create_llm_client(provider="openai", model_name="m") is None


def test_create_llm_client_defaults_to_llama(model_file: Path) -> None:
    client = create_llm_client(model_path=model_file, binary="llama-server", library_path=None, gpu=False)

    assert isinstance(client, LlamaServerClient)


def test_create_llm_client_from_config_openai(audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        llm_enabled=True,
        llm_provider="openai",
        llm_base_url="http://localhost:11434/v1",
        llm_model_name="llama3.1",
        llm_api_key="sk-config",
    )

    client = create_llm_client(config)

    assert isinstance(client, OpenAIClient)
    assert client.model_name == "llama3.1"


def test_create_llm_client_from_config_llama(audio_file: Path, model_file: Path) -> None:
    config = AppConfig(input_file=audio_file, llm_enabled=True, llm_model=model_file)

    client = create_llm_client(config)

    assert isinstance(client, LlamaServerClient)


def test_models_url_variants() -> None:
    assert _models_urls("http://host/v1") == ["http://host/v1/models"]
    assert _models_urls("http://host") == ["http://host/v1/models", "http://host/models"]
    assert _models_urls("") == []


def test_probe_openai_server_reports_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        return _OpenAIResponse('{"data": [{"id": "llama3.1"}, {"id": "qwen"}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)

    ok, message, models = probe_openai_server("http://localhost:11434/v1", api_key="sk")

    assert ok is True
    assert models == ["llama3.1", "qwen"]
    assert "2" in message


def test_probe_openai_server_falls_back_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        seen.append(request.full_url)  # type: ignore[attr-defined]
        if request.full_url.endswith("/v1/models"):  # type: ignore[attr-defined]
            raise _http_error(request.full_url, 404, "nf")  # type: ignore[attr-defined]
        return _OpenAIResponse('{"data": [{"id": "m"}]}')

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)

    ok, _message, models = probe_openai_server("http://localhost:8080")

    assert ok is True
    assert models == ["m"]
    assert seen == ["http://localhost:8080/v1/models", "http://localhost:8080/models"]


def test_probe_openai_server_reports_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float | None = None) -> _OpenAIResponse:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)

    ok, message, models = probe_openai_server("http://localhost:11434/v1")

    assert ok is False
    assert "Не удалось подключиться" in message
    assert models == []
