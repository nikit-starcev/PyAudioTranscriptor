"""Тесты OpenAI-совместимого API веб-интерфейса (#48).

Проверяются ``POST /v1/audio/transcriptions`` (все ``response_format``),
``GET /v1/models`` и строгая аутентификация по ключу. Конвейер подменяется
фиктивной функцией — GPU/модели не нужны.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import (
    Speaker,
    TranscriptEntry,
    TranscriptionResult,
    WordTimestamp,
)
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.secrets import SecretsStore, effective_api_key

API_KEY = "test-secret-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
FULL_TEXT = "привет мир"


def _fake_pipeline(*, fail: bool = False):
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        if fail:
            raise RuntimeError("ASR упал")
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entry = TranscriptEntry(
            start=0.0,
            end=1.5,
            text=FULL_TEXT,
            speaker=speaker,
            avg_logprob=-0.4,
            words=[
                WordTimestamp(text="привет", start=0.0, end=0.7),
                WordTimestamp(text="мир", start=0.7, end=1.5),
            ],
        )
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.5,
            entries=[entry],
            speakers=[speaker],
        )

    return pipeline


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def captured_configs() -> list[AppConfig]:
    return []


@pytest.fixture
def config_builder(web_paths: WebPaths, captured_configs: list[AppConfig]):
    def build(job_id: str, source_path: Path) -> AppConfig:
        config = AppConfig(
            input_file=source_path,
            output_dir=web_paths.results_dir / job_id,
            diarization_enabled=False,
            export_speaker_samples=False,
            notifications=False,
            use_cache=False,
        )
        captured_configs.append(config)
        return config

    return build


@pytest.fixture(autouse=True)
def isolate_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует тесты от реального ``config.env`` (там может быть API_KEY)."""
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})


def _make_client(
    web_paths: WebPaths,
    config_builder,
    pipeline,
    *,
    api_key: str | None = API_KEY,
    timeout: float = 5.0,
) -> TestClient:
    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
        openai_api_key=api_key,
        openai_timeout=timeout,
        openai_poll_interval=0.01,
    )
    return TestClient(app)


@pytest.fixture
def client(
    web_paths: WebPaths, config_builder, captured_configs: list[AppConfig]
) -> Iterator[TestClient]:
    with _make_client(web_paths, config_builder, _fake_pipeline()) as test_client:
        yield test_client


def _transcribe(client: TestClient, **form):
    data = {"model": "whisper-1", **form}
    return client.post(
        "/v1/audio/transcriptions",
        headers=AUTH,
        files={"file": ("sample.wav", b"\x00\x01\x02", "audio/wav")},
        data=data,
    )


def test_models_list_returns_whisper(client: TestClient) -> None:
    response = client.get("/v1/models", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    model = body["data"][0]
    assert model["id"] == "whisper-1"
    assert model["object"] == "model"
    assert model["owned_by"] == "local"
    assert isinstance(model["created"], int)


def test_auth_required_for_transcriptions_and_models(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401
    assert (
        client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"x", "audio/wav")},
        ).status_code
        == 401
    )

    bad = {"Authorization": "Bearer wrong-key"}
    assert client.get("/v1/models", headers=bad).status_code == 401
    assert (
        client.post(
            "/v1/audio/transcriptions", headers=bad, files={"file": ("a.wav", b"x", "audio/wav")}
        ).status_code
        == 401
    )


def test_unconfigured_key_refuses_even_with_header(
    web_paths: WebPaths, config_builder
) -> None:
    with _make_client(web_paths, config_builder, _fake_pipeline(), api_key="") as client:
        response = client.get("/v1/models", headers=AUTH)

    assert response.status_code == 401
    assert "API_KEY" in response.json()["error"]["message"]


def test_json_format_default(client: TestClient) -> None:
    response = _transcribe(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"text": FULL_TEXT}


def test_text_format(client: TestClient) -> None:
    response = _transcribe(client, response_format="text")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == FULL_TEXT


def test_srt_format(client: TestClient) -> None:
    response = _transcribe(client, response_format="srt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-subrip")
    assert response.text.startswith("1\n")
    assert "00:00:00,000 --> 00:00:01,500" in response.text
    assert "Иван: привет мир" in response.text


def test_vtt_format(client: TestClient) -> None:
    response = _transcribe(client, response_format="vtt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/vtt")
    assert response.text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in response.text


def test_verbose_json_format(client: TestClient) -> None:
    response = _transcribe(client, response_format="verbose_json")

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "ru"
    assert body["duration"] == 1.5
    assert body["text"] == FULL_TEXT
    segment = body["segments"][0]
    assert segment["id"] == 0
    assert segment["start"] == 0.0
    assert segment["end"] == 1.5
    assert segment["text"] == FULL_TEXT
    assert segment["speaker"] == "Иван"
    assert "words" not in segment


def test_verbose_json_word_granularity(client: TestClient) -> None:
    response = _transcribe(
        client,
        response_format="verbose_json",
        **{"timestamp_granularities[]": "word"},
    )

    assert response.status_code == 200
    words = response.json()["segments"][0]["words"]
    assert [word["word"] for word in words] == ["привет", "мир"]
    assert words[0]["start"] == 0.0
    assert words[1]["end"] == 1.5


def test_invalid_response_format_returns_400(client: TestClient) -> None:
    response = _transcribe(client, response_format="xml")

    assert response.status_code == 400
    assert "response_format" in response.json()["error"]["message"]


def test_missing_file_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=AUTH,
        data={"model": "whisper-1"},
    )

    assert response.status_code == 400
    assert "файл" in response.json()["error"]["message"]


def test_prompt_and_language_reach_config(
    client: TestClient, captured_configs: list[AppConfig]
) -> None:
    response = _transcribe(client, prompt="ОИБ, АРМ", language="en")

    assert response.status_code == 200
    assert captured_configs[-1].initial_prompt == "ОИБ, АРМ"
    assert captured_configs[-1].language == "en"


def test_pipeline_error_returns_500(web_paths: WebPaths, config_builder) -> None:
    with _make_client(web_paths, config_builder, _fake_pipeline(fail=True)) as client:
        response = _transcribe(client)

    assert response.status_code == 500
    assert "ASR упал" in response.json()["error"]["message"]


def test_timeout_returns_504(
    web_paths: WebPaths, config_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_client(
        web_paths, config_builder, _fake_pipeline(), timeout=0.05
    ) as client:
        # Задача ставится, но воркеру не передаётся — статус навсегда ``queued``.
        monkeypatch.setattr(client.app.state.runner, "submit", lambda *_args, **_kwargs: True)
        response = _transcribe(client)

    assert response.status_code == 504


def test_existing_api_and_unknown_v1_paths(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 200
    assert client.get("/v1/unknown").status_code == 404


def test_effective_api_key_prefers_secret(web_paths: WebPaths) -> None:
    store = SecretsStore(web_paths.secrets_json)
    assert effective_api_key(store, {}) is None
    assert effective_api_key(store, {"API_KEY": "from-env"}) == "from-env"
    store.set_api_key("from-secret")
    assert effective_api_key(store, {"API_KEY": "from-env"}) == "from-secret"
