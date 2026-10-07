"""Тесты REST API чата по стенограмме (``/api/jobs/{id}/chat``, #54/#96).

LLM подменяется фиктивным стриминговым клиентом, конвейер — фиктивной функцией:
GPU/модели/сеть не нужны. Проверяются SSE-формат, цитаты, история и деградация
при выключенной LLM.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


class _StreamingClient:
    """Фиктивная LLM: отдаёт заранее заданные фрагменты и запоминает закрытие."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.closed = False

    def chat(self, messages: list[dict[str, str]]) -> str:
        return "".join(self._chunks)

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class _BlockingClient:
    """Фиктивная LLM без стриминга — проверяет мягкую деградацию к ``chat``."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        return "Только обычный ответ [0]"

    def close(self) -> None:
        pass


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        if on_progress is not None:
            on_progress(ProgressEvent("asr", "Распознавание речи", 0.5))
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entry = TranscriptEntry(start=0.0, end=1.0, text="договорились до пятницы", speaker=speaker)
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=2.5,
            entries=[entry],
            speakers=[speaker],
            low_confidence_threshold=-1.0,
        )

    return pipeline


def _config_builder(web_paths: WebPaths, *, llm_enabled: bool):
    def build(job_id: str, source_path: Path) -> AppConfig:
        return AppConfig(
            input_file=source_path,
            output_dir=web_paths.results_dir / job_id,
            denoise=False,
            diarization_enabled=False,
            export_speaker_samples=False,
            notifications=False,
            timeline=False,
            protocol_auto=False,
            use_cache=False,
            llm_enabled=llm_enabled,
        )

    return build


def _make_client(
    web_paths: WebPaths,
    fake_pipeline,
    *,
    llm_enabled: bool = True,
    llm_factory=None,
) -> TestClient:
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=_config_builder(web_paths, llm_enabled=llm_enabled),
        llm_factory=llm_factory,
        heartbeat=0.05,
    )
    return TestClient(app)


@pytest.fixture
def client(web_paths: WebPaths, fake_pipeline) -> Iterator[TestClient]:
    with _make_client(
        web_paths,
        fake_pipeline,
        llm_factory=lambda _config: _StreamingClient(["Договорились ", "[0]."]),
    ) as test_client:
        yield test_client


def _ready_job(client: TestClient) -> str:
    uploaded = client.post(
        "/api/files/upload", files={"file": ("sample.mp3", b"\x00\x01", "audio/mpeg")}
    )
    assert uploaded.status_code == 201
    path = uploaded.json()["path"]
    created = client.post("/api/jobs", json={"path": path})
    assert created.status_code == 201
    job_id = created.json()["id"]
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    return job_id


def _events(text: str) -> list[dict]:
    events: list[dict] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_chat_streams_tokens_and_citations(client: TestClient) -> None:
    job_id = _ready_job(client)
    response = client.post(f"/api/jobs/{job_id}/chat", json={"message": "о чём договорились?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert events[0] == {"type": "start"}
    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert "".join(tokens) == "Договорились [0]."
    done = next(event for event in events if event["type"] == "done")
    assert done["content"] == "Договорились [0]."
    assert done["citations"] == [
        {
            "index": 0,
            "start": 0.0,
            "end": 1.0,
            "speaker": "Иван",
            "text": "договорились до пятницы",
        }
    ]
    assert done["message"]["role"] == "assistant"


def test_chat_history_roundtrip_and_clear(client: TestClient) -> None:
    job_id = _ready_job(client)
    client.post(f"/api/jobs/{job_id}/chat", json={"message": "вопрос"})
    history = client.get(f"/api/jobs/{job_id}/chat").json()["messages"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == "вопрос"
    assert history[1]["citations"][0]["index"] == 0
    cleared = client.delete(f"/api/jobs/{job_id}/chat").json()
    assert cleared == {"cleared": 2}
    assert client.get(f"/api/jobs/{job_id}/chat").json()["messages"] == []


def test_chat_degrades_to_non_streaming(web_paths: WebPaths, fake_pipeline) -> None:
    with _make_client(
        web_paths, fake_pipeline, llm_factory=lambda _config: _BlockingClient()
    ) as test_client:
        job_id = _ready_job(test_client)
        response = test_client.post(f"/api/jobs/{job_id}/chat", json={"message": "вопрос"})
    events = _events(response.text)
    assert [event["type"] for event in events] == ["start", "token", "done"]
    assert events[1]["text"] == "Только обычный ответ [0]"
    assert events[2]["citations"][0]["index"] == 0


def test_chat_disabled_llm_returns_conflict(web_paths: WebPaths, fake_pipeline) -> None:
    with _make_client(
        web_paths, fake_pipeline, llm_enabled=False, llm_factory=lambda _config: _StreamingClient([])
    ) as test_client:
        job_id = _ready_job(test_client)
        response = test_client.post(f"/api/jobs/{job_id}/chat", json={"message": "вопрос"})
    assert response.status_code == 409
    assert "LLM отключена" in response.json()["detail"]


def test_chat_missing_result_returns_404(client: TestClient) -> None:
    uploaded = client.post(
        "/api/files/upload", files={"file": ("sample.mp3", b"\x00\x01", "audio/mpeg")}
    )
    created = client.post("/api/jobs", json={"path": uploaded.json()["path"]})
    job_id = created.json()["id"]
    assert client.post(f"/api/jobs/{job_id}/chat", json={"message": "вопрос"}).status_code == 404
    assert client.get(f"/api/jobs/{job_id}/chat").status_code == 404
