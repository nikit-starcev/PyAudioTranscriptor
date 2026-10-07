"""Тесты потоковой выдачи LLM-клиента (SSE, чат по стенограмме)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from audio_transcriber.llm import client as client_module
from audio_transcriber.llm.client import (
    LlamaServerClient,
    OpenAIClient,
    _iter_stream_deltas,
)
from audio_transcriber.utils.exceptions import LlmError


def _lines(*chunks: bytes) -> io.BytesIO:
    return io.BytesIO(b"\n".join(chunks))


def test_iter_stream_deltas_extracts_content() -> None:
    raw = _lines(
        'data: {"choices":[{"delta":{"content":"При"}}]}'.encode(),
        b": keep-alive",
        'data: {"choices":[{"delta":{"content":"вет"}}]}'.encode(),
        b"data: [DONE]",
    )
    assert list(_iter_stream_deltas(raw)) == ["При", "вет"]


def test_iter_stream_deltas_ignores_garbage_and_empty() -> None:
    raw = _lines(
        b"data: not json",
        b'data: {"choices":[]}',
        b'data: {"choices":[{"delta":{}}]}',
        b'data: {"choices":[{"delta":{"content":""}}]}',
    )
    assert list(_iter_stream_deltas(raw)) == []


def test_iter_stream_deltas_raises_on_error_event() -> None:
    raw = _lines(b'data: {"error": {"message": "boom"}}')
    with pytest.raises(LlmError):
        list(_iter_stream_deltas(raw))


class _StreamResponse:
    def __init__(self, body: bytes) -> None:
        self._buffer = io.BytesIO(body)

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._buffer)


def test_openai_chat_stream_yields_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"\n".join(
        [
            b'data: {"choices":[{"delta":{"content":"a"}}]}',
            b'data: {"choices":[{"delta":{"content":"b"}}]}',
            b"data: [DONE]",
        ]
    )
    captured: dict[str, str] = {}

    def fake_urlopen(request, timeout=None):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        return _StreamResponse(body)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    client = OpenAIClient("https://api.example.com", "model-x")
    assert list(client.chat_stream([{"role": "user", "content": "hi"}])) == ["a", "b"]
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert '"stream": true' in captured["body"]


def test_llama_chat_stream_yields_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]\n'
    client = LlamaServerClient(Path("/tmp/does-not-exist.gguf"))
    monkeypatch.setattr(client, "_ensure_started", lambda: None)
    client._base_url = "http://127.0.0.1:9"

    def fake_urlopen(request, timeout=None):  # type: ignore[no-untyped-def]
        return _StreamResponse(body)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    assert list(client.chat_stream([{"role": "user", "content": "hi"}])) == ["ok"]
