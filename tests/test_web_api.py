"""Тесты REST API веб-интерфейса (FastAPI TestClient).

Конвейер подменяется фиктивной функцией, поэтому GPU/модели не нужны, а
настройки — собственным ``config_builder`` (не читает реальный ``config.env``).
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import (
    Speaker,
    TranscriptEntry,
    TranscriptionResult,
)
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.web.app import _resolve_upload_file, create_app
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.paths import WebPaths


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        if on_progress is not None:
            on_progress(ProgressEvent("asr", "Распознавание речи", 0.5))
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entry = TranscriptEntry(
            start=0.0,
            end=1.0,
            text="привет",
            speaker=speaker,
            avg_logprob=-2.0,
            overlap=True,
        )
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=2.5,
            entries=[entry],
            speakers=[speaker],
            low_confidence_threshold=-1.0,
        )

    return pipeline


@pytest.fixture
def config_builder(web_paths: WebPaths):
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
        )

    return build


@pytest.fixture
def client(
    web_paths: WebPaths, fake_pipeline, config_builder
) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str = "sample.mp3", data: bytes = b"\x00\x01") -> dict:
    response = client.post(
        "/api/files/upload", files={"file": (name, data, "audio/mpeg")}
    )
    assert response.status_code == 201
    return response.json()


def _run_job(client: TestClient, path: str) -> tuple[str, dict]:
    created = client.post("/api/jobs", json={"path": path})
    assert created.status_code == 201
    job_id = created.json()["id"]
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        details = client.get(f"/api/jobs/{job_id}").json()
        if details["status"] in {"done", "error"}:
            return job_id, details
        time.sleep(0.02)
    raise AssertionError("задача не завершилась за отведённое время")


def _wav_bytes(seconds: float = 0.05, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"]


def test_config_slice(client: TestClient) -> None:
    payload = client.get("/api/config").json()

    assert set(payload) == {
        "input_dir",
        "output_dir",
        "export_formats",
        "llm_enabled",
        "glossary_enabled",
        "voices_dir",
    }


def test_files_list_and_upload(client: TestClient) -> None:
    assert client.get("/api/files").json() == []

    uploaded = _upload(client, "rec.mp3", b"abc")

    assert uploaded["name"] == "rec.mp3"
    assert uploaded["size"] == 3
    files = client.get("/api/files").json()
    assert [item["name"] for item in files] == ["rec.mp3"]


def test_upload_deduplicates_names(client: TestClient) -> None:
    _upload(client, "rec.mp3", b"a")
    second = _upload(client, "rec.mp3", b"b")

    assert second["name"] == "rec (2).mp3"


def test_create_job_rejects_path_outside_uploads(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"x")

    response = client.post("/api/jobs", json={"path": str(outside)})

    assert response.status_code == 400


def test_files_path_roundtrip_to_job(client: TestClient) -> None:
    """``GET /api/files`` → ``POST /api/jobs`` с тем же ``path`` даёт 201."""
    uploaded = _upload(client, "round.mp3", b"abc")
    files = client.get("/api/files").json()
    assert [item["name"] for item in files] == ["round.mp3"]

    listed = files[0]["path"]
    assert Path(listed).is_absolute()

    created = client.post("/api/jobs", json={"path": listed})
    assert created.status_code == 201
    assert created.json()["name"] == uploaded["name"]


def test_files_path_roundtrip_with_relative_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_pipeline, config_builder
) -> None:
    """Круговой сценарий работает и при относительном ``web-data``.

    Именно этот случай ломался: ``/api/files`` отдавал относительный путь
    ``web-data/uploads/x.mp4``, а ``POST /api/jobs`` повторно добавлял каталог.
    """
    monkeypatch.chdir(tmp_path)
    paths = WebPaths(Path("web-data"))
    app = create_app(
        paths=paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        _upload(client, "round.mp3", b"abc")
        listed = client.get("/api/files").json()[0]["path"]
        assert Path(listed).is_absolute()
        assert client.post("/api/jobs", json={"path": listed}).status_code == 201
        # Обратная совместимость: относительная форма с каталогом загрузок.
        legacy = str(Path("web-data/uploads/round.mp3"))
        assert client.post("/api/jobs", json={"path": legacy}).status_code == 201


def test_delete_file(client: TestClient, web_paths: WebPaths) -> None:
    _upload(client, "gone.mp3", b"abc")

    response = client.delete("/api/files/gone.mp3")

    assert response.status_code == 200
    assert response.json() == {"deleted": "gone.mp3"}
    assert client.get("/api/files").json() == []
    assert not (web_paths.input_dir / "gone.mp3").exists()
    # Повторное удаление уже отсутствующего файла.
    assert client.delete("/api/files/gone.mp3").status_code == 404


def test_delete_missing_file(client: TestClient) -> None:
    assert client.delete("/api/files/nope.mp3").status_code == 404


def test_delete_file_rejects_non_media(client: TestClient, web_paths: WebPaths) -> None:
    web_paths.input_dir.mkdir(parents=True, exist_ok=True)
    (web_paths.input_dir / "notes.txt").write_text("x", encoding="utf-8")

    assert client.delete("/api/files/notes.txt").status_code == 400
    assert (web_paths.input_dir / "notes.txt").is_file()


def test_delete_file_rejects_path_outside_uploads(
    web_paths: WebPaths, tmp_path: Path
) -> None:
    web_paths.input_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"x")

    with pytest.raises(HTTPException) as absolute:
        _resolve_upload_file(web_paths, str(outside))
    assert absolute.value.status_code == 400

    with pytest.raises(HTTPException) as traversal:
        _resolve_upload_file(web_paths, "..")
    assert traversal.value.status_code == 400


def test_delete_file_used_by_running_job(
    client: TestClient, web_paths: WebPaths
) -> None:
    _upload(client, "busy.mp3", b"abc")
    job_id = client.post("/api/jobs", json={"path": "busy.mp3"}).json()["id"]
    client.app.state.store.update(job_id, status="running")

    response = client.delete("/api/files/busy.mp3")

    assert response.status_code == 409
    assert (web_paths.input_dir / "busy.mp3").is_file()


def test_run_job_and_result(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id, details = _run_job(client, uploaded["name"])

    assert details["status"] == "done"
    assert details["language"] == "ru"
    assert details["duration"] == 2.5
    assert details["summary"] == {"language": "ru", "duration": 2.5, "entries": 1, "speakers": 1, "samples": 0}

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    body = result.json()
    assert body["language"] == "ru"
    assert body["speakers"] == [{"id": "SPEAKER_00", "display_name": "Иван", "has_sample": False}]
    entry = body["entries"][0]
    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["text"] == "привет"
    assert entry["low_confidence"] is True
    assert entry["overlap"] is True
    assert entry["extra_speaker_ids"] == []
    assert entry["speaker_confidence"] is None
    assert entry["low_speaker_confidence"] is False
    assert {mark["key"] for mark in body["marks"]} == {
        "low_confidence",
        "speaker_uncertain",
        "overlap",
    }


def test_run_job_records_stage_times(web_paths: WebPaths, config_builder) -> None:
    """Раннер сохраняет тайминги стадий и общее время; API их отдаёт."""

    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        if on_progress is not None:
            on_progress(ProgressEvent("denoise", "Шумоподавление"))
            on_progress(ProgressEvent("asr", "Распознавание речи", detail="из кэша"))
            on_progress(ProgressEvent("merge", "Объединение сегментов"))
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=2.5,
            entries=[],
            speakers=[],
            low_confidence_threshold=-1.0,
        )

    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client)
        _job_id, details = _run_job(client, uploaded["name"])
        listing = client.get("/api/jobs").json()

    assert details["status"] == "done"
    assert [item["stage"] for item in details["stage_times"]] == ["denoise", "asr", "merge"]
    assert details["stage_times"][1]["cached"] is True
    assert all(item["seconds"] >= 0 for item in details["stage_times"])
    assert details["total_seconds"] is not None
    assert details["total_seconds"] >= 0
    assert listing[0]["stage_times"][0]["stage"] == "denoise"
    assert listing[0]["total_seconds"] is not None


def test_run_job_error_records_stage_times(web_paths: WebPaths, config_builder) -> None:
    """Даже при ошибке уже закрытые стадии сохраняются в таймингах."""

    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        if on_progress is not None:
            on_progress(ProgressEvent("asr", "Распознавание речи"))
        raise RuntimeError("сбой распознавания")

    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client)
        _job_id, details = _run_job(client, uploaded["name"])

    assert details["status"] == "error"
    assert [item["stage"] for item in details["stage_times"]] == ["asr"]
    assert details["total_seconds"] is not None


def test_jobs_listing(client: TestClient) -> None:
    uploaded = _upload(client)
    client.post("/api/jobs", json={"path": uploaded["name"]})

    jobs = client.get("/api/jobs").json()

    assert len(jobs) == 1
    assert jobs[0]["name"] == "sample.mp3"
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["num_speakers"] is None


def test_create_job_with_num_speakers_reaches_config(
    web_paths: WebPaths, config_builder
) -> None:
    """``num_speakers`` задачи сохраняется и попадает в ``AppConfig`` задачи."""
    captured: list[int | None] = []

    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        captured.append(config.num_speakers)
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[],
            speakers=[],
            low_confidence_threshold=-1.0,
        )

    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        uploaded = _upload(test_client)
        created = test_client.post(
            "/api/jobs", json={"path": uploaded["name"], "num_speakers": 3}
        )
        assert created.status_code == 201
        job_id = created.json()["id"]
        assert created.json()["num_speakers"] == 3
        assert test_client.get("/api/jobs").json()[0]["num_speakers"] == 3
        assert test_client.get(f"/api/jobs/{job_id}").json()["num_speakers"] == 3

        assert test_client.post(f"/api/jobs/{job_id}/run").status_code == 200

        # Авто-режим: поле не задано — конфиг получает ``None``.
        auto = test_client.post("/api/jobs", json={"path": uploaded["name"]})
        assert auto.status_code == 201
        auto_id = auto.json()["id"]
        assert auto.json()["num_speakers"] is None
        assert test_client.post(f"/api/jobs/{auto_id}/run").status_code == 200

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if test_client.get(f"/api/jobs/{auto_id}").json()["status"] in {"done", "error"}:
                break
            time.sleep(0.02)

    assert captured == [3, None]


def test_create_job_rejects_invalid_num_speakers(client: TestClient) -> None:
    uploaded = _upload(client)

    response = client.post("/api/jobs", json={"path": uploaded["name"], "num_speakers": 0})

    assert response.status_code == 422


def test_patch_job_num_speakers(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    updated = client.patch(f"/api/jobs/{job_id}", json={"num_speakers": 2})
    assert updated.status_code == 200
    assert updated.json()["num_speakers"] == 2

    reset = client.patch(f"/api/jobs/{job_id}", json={"num_speakers": None})
    assert reset.status_code == 200
    assert reset.json()["num_speakers"] is None

    # Пустое тело — обновлять нечего.
    assert client.patch(f"/api/jobs/{job_id}", json={}).status_code == 400
    # Невалидное значение.
    assert client.patch(f"/api/jobs/{job_id}", json={"num_speakers": 0}).status_code == 422
    # Неизвестная задача.
    assert client.patch("/api/jobs/unknown", json={"num_speakers": 2}).status_code == 404


def test_events_stream_terminates_after_done(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id, _ = _run_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text
    assert '"status": "done"' in response.text


def test_sample_endpoint_serves_wav(client: TestClient, web_paths: WebPaths) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    wav_bytes = _wav_bytes()
    sample_path = web_paths.samples_dir / "Иван.wav"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(wav_bytes)

    result_path = web_paths.results_dir / f"{job_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "language": "ru",
                "duration": 1.0,
                "speakers": [{"id": "SPEAKER_00", "display_name": "Иван", "has_sample": True}],
                "entries": [],
                "marks": [],
                "samples": {"SPEAKER_00": "samples/Иван.wav"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client.app.state.store.update(job_id, status="done", result_path=str(result_path))

    response = client.get(f"/api/jobs/{job_id}/samples/SPEAKER_00")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == wav_bytes

    assert client.get(f"/api/jobs/{job_id}/samples/UNKNOWN").status_code == 404


def test_audio_endpoint_supports_range(client: TestClient) -> None:
    uploaded = _upload(client, "range.mp3", b"0123456789")
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    partial = client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=2-5"})
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"

    full = client.get(f"/api/jobs/{job_id}/audio")
    assert full.status_code == 200
    assert full.content == b"0123456789"


def test_delete_job(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_delete_job_removes_artifacts(
    client: TestClient, web_paths: WebPaths
) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    result_path = web_paths.results_dir / f"{job_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{}", encoding="utf-8")
    stage_dir = web_paths.results_dir / job_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage.wav").write_bytes(b"wav")
    client.app.state.store.update(job_id, result_path=str(result_path))

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200

    assert not result_path.exists()
    assert not stage_dir.exists()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_delete_job_with_missing_artifacts(
    client: TestClient, web_paths: WebPaths
) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    # ``result_path`` указывает на несуществующий файл — удаление не падает.
    client.app.state.store.update(
        job_id, result_path=str(web_paths.results_dir / "ghost.json")
    )

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_missing_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/jobs/unknown").status_code == 404
    assert client.get("/api/jobs/unknown/result").status_code == 404


def test_event_bus_streams_and_finishes() -> None:
    bus = JobEventBus(heartbeat=0.05)
    received: list[dict | None] = []

    async def consume() -> None:
        async for event in bus.subscribe("j1"):
            received.append(event)
            if event is not None and event.get("status") == "done":
                break

    async def main() -> None:
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        bus.publish("j1", {"stage": "asr", "fraction": 0.5, "message": "ASR", "status": "running"})
        bus.publish("j1", {"stage": "done", "fraction": 1.0, "message": "Готово", "status": "done"})
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(main())

    assert received[-1] is not None
    assert received[-1]["status"] == "done"
    assert any(event and event.get("stage") == "asr" for event in received)


def test_event_bus_replays_history_to_late_subscriber() -> None:
    bus = JobEventBus(heartbeat=0.05)
    bus.publish("j1", {"stage": "asr", "fraction": 0.2, "message": "ASR", "status": "running"})
    bus.publish("j1", {"stage": "done", "fraction": 1.0, "message": "Готово", "status": "done"})

    async def collect() -> list[dict | None]:
        events: list[dict | None] = []
        async for event in bus.subscribe("j1"):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert [event["stage"] for event in events if event is not None] == ["asr", "done"]


def test_cli_web_help_lists_command() -> None:
    from typer.testing import CliRunner

    from audio_transcriber.cli.app import app as cli_app

    result = CliRunner().invoke(cli_app, ["web", "--help"])

    assert result.exit_code == 0
    assert "--port" in result.stdout


def test_spa_index_served(client: TestClient) -> None:
    from audio_transcriber.web.paths import STATIC_DIR

    if not (STATIC_DIR / "index.html").is_file():
        pytest.skip("SPA не собран")

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="root"' in response.text


def test_unknown_api_path_returns_404(client: TestClient) -> None:
    assert client.get("/api/does-not-exist").status_code == 404
