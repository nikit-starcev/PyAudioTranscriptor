"""Тесты REST API веб-интерфейса (FastAPI TestClient).

Конвейер подменяется фиктивной функцией, поэтому GPU/модели не нужны, а
настройки — собственным ``config_builder`` (не читает реальный ``config.env``).
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
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
from audio_transcriber.utils.exceptions import ProcessingCancelled
from audio_transcriber.web.app import _resolve_upload_file, create_app
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.runner import ORPHAN_ERROR_MESSAGE, JobRunner
from audio_transcriber.web.storage.jobs_db import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
    JobsDB,
    utc_now_iso,
)
from audio_transcriber.web.timings import StageTiming


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
        "llm_provider",
        "llm_external",
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
    # Созданная, но не поставленная воркеру задача не активна.
    assert jobs[0]["active"] is False


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


def test_create_job_with_speaker_range_reaches_config(
    web_paths: WebPaths, config_builder
) -> None:
    """``min_speakers``/``max_speakers`` задачи доходят до ``AppConfig``."""
    captured: list[tuple[int | None, int | None, int | None]] = []

    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        captured.append((config.num_speakers, config.min_speakers, config.max_speakers))
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
            "/api/jobs",
            json={"path": uploaded["name"], "min_speakers": 2, "max_speakers": 4},
        )
        assert created.status_code == 201
        job_id = created.json()["id"]
        assert created.json()["min_speakers"] == 2
        assert created.json()["max_speakers"] == 4
        assert test_client.get(f"/api/jobs/{job_id}").json()["min_speakers"] == 2

        assert test_client.post(f"/api/jobs/{job_id}/run").status_code == 200
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if test_client.get(f"/api/jobs/{job_id}").json()["status"] in {"done", "error"}:
                break
            time.sleep(0.02)

    assert captured == [(None, 2, 4)]


def test_create_job_rejects_inverted_speaker_range(client: TestClient) -> None:
    uploaded = _upload(client)

    response = client.post(
        "/api/jobs",
        json={"path": uploaded["name"], "min_speakers": 5, "max_speakers": 2},
    )

    assert response.status_code == 422


def test_patch_job_speaker_range(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    updated = client.patch(
        f"/api/jobs/{job_id}", json={"min_speakers": 2, "max_speakers": 4}
    )
    assert updated.status_code == 200
    assert updated.json()["min_speakers"] == 2
    assert updated.json()["max_speakers"] == 4

    # Инверсия проверяется по итоговому (слитому) диапазону.
    assert (
        client.patch(f"/api/jobs/{job_id}", json={"min_speakers": 6}).status_code == 422
    )
    # Пустое тело — обновлять нечего.
    assert client.patch(f"/api/jobs/{job_id}", json={}).status_code == 400


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


def test_delete_job_soft_hides_and_restore(client: TestClient) -> None:
    """Обычное удаление — мягкое: скрывает из списка, запись и артефакты целы."""
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200

    # Скрыта из обычного списка, но остаётся доступной по id как удалённая.
    assert client.get("/api/jobs").json() == []
    details = client.get(f"/api/jobs/{job_id}")
    assert details.status_code == 200
    assert details.json()["deleted"] is True
    listed = client.get("/api/jobs", params={"include_deleted": "true"}).json()
    assert [job["id"] for job in listed] == [job_id]
    assert listed[0]["deleted"] is True

    # Восстановление возвращает задачу в обычный список.
    assert client.post(f"/api/jobs/{job_id}/restore").status_code == 200
    restored = client.get(f"/api/jobs/{job_id}").json()
    assert restored["deleted"] is False
    assert [job["id"] for job in client.get("/api/jobs").json()] == [job_id]


def test_soft_deleted_job_cannot_run_until_restored(client: TestClient) -> None:
    """Мягко удалённую задачу нельзя запустить; после restore — можно."""
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200

    assert client.post(f"/api/jobs/{job_id}/run").status_code == 409

    assert client.post(f"/api/jobs/{job_id}/restore").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200


def test_purge_job_removes_artifacts(
    client: TestClient, web_paths: WebPaths
) -> None:
    """``?purge=true`` стирает запись и артефакты безвозвратно."""
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    result_path = web_paths.results_dir / f"{job_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{}", encoding="utf-8")
    stage_dir = web_paths.results_dir / job_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage.wav").write_bytes(b"wav")
    client.app.state.store.update(job_id, result_path=str(result_path))

    # Сначала мягкое удаление — артефакты должны остаться.
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert result_path.exists()
    assert stage_dir.exists()

    response = client.delete(f"/api/jobs/{job_id}", params={"purge": "true"})

    assert response.status_code == 200
    assert response.json()["purged"] == job_id
    assert not result_path.exists()
    assert not stage_dir.exists()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_purge_job_with_missing_artifacts(
    client: TestClient, web_paths: WebPaths
) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    # ``result_path`` указывает на несуществующий файл — очистка не падает.
    client.app.state.store.update(
        job_id, result_path=str(web_paths.results_dir / "ghost.json")
    )

    assert client.delete(f"/api/jobs/{job_id}", params={"purge": "true"}).status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_reconcile_marks_orphans_and_keeps_active(
    web_paths: WebPaths, config_builder
) -> None:
    """Реконсиляция: running/queued вне активного набора → error, активные — нет."""
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    active = store.create("active-job", web_paths.input_dir / "a.mp3")
    store.update(
        active.id,
        status=STATUS_RUNNING,
        stage="asr",
        fraction=0.4,
        stage_started_at=utc_now_iso(),
    )
    orphan = store.create("orphan-job", web_paths.input_dir / "b.mp3")
    store.update(
        orphan.id,
        status=STATUS_RUNNING,
        stage="diarization",
        fraction=0.3,
        stage_started_at=utc_now_iso(),
    )
    pending = store.create("pending-job", web_paths.input_dir / "c.mp3")

    runner = JobRunner(store, JobEventBus(heartbeat=0.05), web_paths, config_builder)
    assert runner.submit(active.id, Path(active.source_path)) is True

    reconciled = runner.reconcile_orphans()

    assert set(reconciled) == {"orphan-job", "pending-job"}
    kept = store.get(active.id)
    assert kept is not None
    assert kept.status == STATUS_RUNNING
    assert kept.stage == "asr"
    assert kept.fraction == 0.4
    assert runner.is_active(active.id) is True

    fixed = store.get(orphan.id)
    assert fixed is not None
    assert fixed.status == STATUS_ERROR
    assert fixed.error == ORPHAN_ERROR_MESSAGE
    assert fixed.finished_at is not None
    assert fixed.stage is None
    assert fixed.fraction is None
    assert fixed.stage_started_at is None
    assert fixed.stage_times == []

    queued = store.get(pending.id)
    assert queued is not None
    assert queued.status == STATUS_ERROR


def test_startup_reconciles_orphaned_jobs(
    web_paths: WebPaths, fake_pipeline, config_builder
) -> None:
    """Подвешенная задача прошлого процесса переводится в error при старте."""
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    orphan = store.create("stale", web_paths.input_dir / "stale.mp3")
    store.update(
        orphan.id,
        status=STATUS_RUNNING,
        stage="diarization",
        fraction=0.3,
        stage_started_at=utc_now_iso(),
    )

    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        listing = client.get("/api/jobs").json()

    assert listing[0]["status"] == STATUS_ERROR
    assert listing[0]["error"] == ORPHAN_ERROR_MESSAGE
    assert listing[0]["active"] is False
    assert listing[0]["stage"] is None


def test_delete_orphaned_running_job(client: TestClient) -> None:
    """Осиротевшую running-задачу (воркер её не ведёт) можно удалить."""
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    client.app.state.store.update(
        job_id,
        status=STATUS_RUNNING,
        stage="diarization",
        fraction=0.3,
        stage_started_at=utc_now_iso(),
    )

    response = client.delete(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["deleted"] is True
    assert client.get("/api/jobs").json() == []


def test_delete_active_running_job_conflicts(
    web_paths: WebPaths, config_builder
) -> None:
    """Реально выполняющуюся задачу удалить нельзя — 409."""
    started = threading.Event()
    release = threading.Event()

    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        started.set()
        release.wait(timeout=5.0)
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
    with TestClient(app) as client:
        uploaded = _upload(client)
        job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
        assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
        assert started.wait(timeout=5.0)

        listing = client.get("/api/jobs").json()
        assert listing[0]["active"] is True
        assert client.delete(f"/api/jobs/{job_id}").status_code == 409

        release.set()


def test_run_orphaned_running_job_reruns(
    web_paths: WebPaths, fake_pipeline, config_builder
) -> None:
    """Осиротевшую running-задачу можно запустить заново."""
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client)
        job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
        store = client.app.state.store
        store.update(job_id, status=STATUS_RUNNING, stage="diarization")

        assert client.post(f"/api/jobs/{job_id}/run").status_code == 200

        deadline = time.time() + 5.0
        while time.time() < deadline:
            details = client.get(f"/api/jobs/{job_id}").json()
            if details["status"] == "done":
                break
            time.sleep(0.02)
        assert details["status"] == "done"


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


def test_rerun_clears_stale_sse_history(
    web_paths: WebPaths, config_builder, fake_pipeline
) -> None:
    """Повторный прогон не должен реиграть события прошлого запуска (ошибку)."""
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    bus = JobEventBus(heartbeat=0.05)
    runner = JobRunner(store, bus, web_paths, config_builder, pipeline_fn=fake_pipeline)

    source = web_paths.input_dir / "rerun.mp3"
    source.write_bytes(b"x")
    store.create("rerun-job", source)
    # Событие прошлого прогона — как после реконсиляции осиротевшей задачи.
    bus.publish(
        "rerun-job",
        {"stage": "error", "message": ORPHAN_ERROR_MESSAGE, "status": STATUS_ERROR},
    )
    assert any(e.get("message") == ORPHAN_ERROR_MESSAGE for e in bus.history("rerun-job"))

    runner.start()
    try:
        assert runner.submit("rerun-job", source) is True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current = store.get("rerun-job")
            if current is not None and current.is_terminal:
                break
            time.sleep(0.05)
    finally:
        runner.stop()

    history = bus.history("rerun-job")
    assert not any(e.get("message") == ORPHAN_ERROR_MESSAGE for e in history), history
    assert not any(e.get("status") == STATUS_ERROR for e in history), history


# --- #15/#24: оценки прогресса, ETA и здоровье ---------------------------


def test_job_payload_includes_estimate_fields(client: TestClient) -> None:
    """API отдаёт процент/ETA/здоровье даже для незапущенной задачи."""
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    payload = client.get(f"/api/jobs/{job_id}").json()

    assert payload["progress_percent"] == 0.0
    assert payload["eta_seconds"] is None
    assert payload["eta_by_stage"] is None
    assert payload["health"] is None
    # Поле ``updated_at`` заполняется при создании.
    assert payload["updated_at"]


def test_running_job_estimates_from_history(
    client: TestClient, web_paths: WebPaths
) -> None:
    """Процент, ETA и здоровье считаются по таймингам завершённых задач."""
    store = client.app.state.store
    store.create("history", web_paths.input_dir / "h.mp3")
    store.update(
        "history",
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[
            StageTiming("denoise", 10.0),
            StageTiming("asr", 50.0),
            StageTiming("merge", 20.0),
        ],
    )
    live = store.create("live", web_paths.input_dir / "live.mp3")
    store.update(
        live.id,
        status=STATUS_RUNNING,
        stage="asr",
        fraction=0.25,
        duration=100.0,
        stage_started_at=utc_now_iso(),
        stage_times=[],
    )
    client.app.state.estimator.invalidate()

    payload = client.get(f"/api/jobs/{live.id}").json()

    # Веса: denoise 0.1, asr 0.5, merge 0.2, остальные — запасной 0.2; сумма 1.8.
    # denoise пройдена (позиция), asr — 0.5 * 0.25 = 0.125. Итого 0.225 / 1.8 → 12.5%.
    assert payload["progress_percent"] == pytest.approx(12.5)
    # asr: остаток 0.75 * 50 = 37.5; merge 20; пять стадий по 20 = 100 → 157.5.
    assert payload["eta_seconds"] == pytest.approx(157.5)
    assert payload["eta_by_stage"]["asr"] == pytest.approx(37.5)
    # Задачу никто не ведёт (== активного воркера нет) — она «зависла».
    assert payload["health"]["status"] == "stalled"
    assert payload["health"]["last_update_seconds"] is not None


def test_events_after_done_include_estimates(client: TestClient) -> None:
    """Терминальное SSE-событие тоже содержит процент/ETA/здоровье."""
    uploaded = _upload(client)
    job_id, _ = _run_job(client, uploaded["name"])

    response = client.get(f"/api/jobs/{job_id}/events")

    assert response.status_code == 200
    assert '"progress_percent": 100.0' in response.text
    assert '"eta_seconds": 0.0' in response.text
    assert '"health": null' in response.text


# --- #22: остановка и возобновление обработки -----------------------------


class _BlockingCancelPipeline:
    """Конвейер, ждущий отмены и затем поднимающий ``ProcessingCancelled``.

    Принимает ``cancel_event`` (как штатный ``run_pipeline``), поэтому воркер
    прокидывает в него флаг; на ``cancel`` он разблокируется и завершается
    штатным прерыванием.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.calls = 0

    def __call__(self, config: AppConfig, *, on_progress=None, cancel_event=None):
        self.calls += 1
        self.started.set()
        assert cancel_event is not None
        cancel_event.wait(timeout=5.0)
        raise ProcessingCancelled("Остановлено пользователем")


def _wait_for_status(
    client: TestClient, job_id: str, status: str, *, timeout: float = 5.0
) -> dict:
    deadline = time.time() + timeout
    details: dict = {}
    while time.time() < deadline:
        details = client.get(f"/api/jobs/{job_id}").json()
        if details["status"] == status and details["active"] is False:
            break
        time.sleep(0.02)
    return details


def _wait_for_status_direct(
    store: JobsDB, job_id: str, status: str, *, timeout: float = 5.0
) -> Job | None:
    deadline = time.time() + timeout
    job = store.get(job_id)
    while time.time() < deadline:
        job = store.get(job_id)
        if job is not None and job.status == status:
            break
        time.sleep(0.02)
    return job


def test_cancel_active_job_marks_cancelled_and_reruns(
    web_paths: WebPaths, config_builder
) -> None:
    """Активная задача отменяется в ``cancelled`` (не ``error``) и перезапускается."""
    pipeline = _BlockingCancelPipeline()
    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client)
        job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
        assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
        assert pipeline.started.wait(timeout=5.0)

        response = client.post(f"/api/jobs/{job_id}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == STATUS_CANCELLED

        details = _wait_for_status(client, job_id, STATUS_CANCELLED)
        assert details["status"] == STATUS_CANCELLED
        assert details["active"] is False
        assert details["finished_at"] is not None
        assert details["error"] is None

        # Терминальное событие отмены опубликовано в SSE-шину.
        history = client.app.state.bus.history(job_id)
        assert any(event.get("status") == STATUS_CANCELLED for event in history)

        # Отменённая задача перезапускаема (не 409).
        assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
        deadline = time.time() + 5.0
        while time.time() < deadline and pipeline.calls < 2:
            time.sleep(0.02)
        assert pipeline.calls >= 2
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        _wait_for_status(client, job_id, STATUS_CANCELLED)


def test_cancel_inactive_or_missing_job(client: TestClient) -> None:
    """Неактивную задачу отменить нельзя (409), отсутствующую — 404."""
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409
    assert client.post("/api/jobs/unknown/cancel").status_code == 404


def test_cancel_completed_job_conflicts(client: TestClient) -> None:
    """Завершённую задачу отменить нельзя — 409."""
    uploaded = _upload(client)
    job_id, _ = _run_job(client, uploaded["name"])

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_delete_cancelled_job(web_paths: WebPaths, config_builder) -> None:
    """Отменённую задачу можно удалить, как и любую терминальную."""
    pipeline = _BlockingCancelPipeline()
    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client)
        job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
        assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
        assert pipeline.started.wait(timeout=5.0)
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        _wait_for_status(client, job_id, STATUS_CANCELLED)

        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/jobs/{job_id}").json()["deleted"] is True


def test_runner_cancel_inactive_returns_false(
    web_paths: WebPaths, config_builder
) -> None:
    """``JobRunner.cancel`` для задачи, которую воркер не ведёт, — ``False``."""
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    runner = JobRunner(store, JobEventBus(heartbeat=0.05), web_paths, config_builder)

    assert runner.cancel("missing") is False


def test_runner_skips_soft_deleted_queued_job(
    web_paths: WebPaths, config_builder
) -> None:
    """Мягко удалённую задачу воркер не запускает: она остаётся ``queued`` (#30)."""
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    source = web_paths.input_dir / "deleted.mp3"
    source.write_bytes(b"x")
    job = store.create("deleted-job", source)
    assert store.soft_delete(job.id) is not None

    runner = JobRunner(store, JobEventBus(heartbeat=0.05), web_paths, config_builder)
    runner.start()
    try:
        assert runner.submit(job.id, source) is True
        deadline = time.monotonic() + 5.0
        while runner.is_active(job.id) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert runner.is_active(job.id) is False
        still = store.get(job.id)
        assert still is not None
        # Пропущенная задача не переходит в running/done и остаётся удалённой.
        assert still.status == STATUS_QUEUED
        assert still.deleted is True
    finally:
        runner.stop()


def test_reconcile_marks_soft_deleted_orphan(
    web_paths: WebPaths, config_builder
) -> None:
    """Осиротевшая мягко удалённая задача тоже помечается ``error`` (#30)."""
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    orphan = store.create("deleted-orphan", web_paths.input_dir / "a.mp3")
    store.update(
        orphan.id,
        status=STATUS_RUNNING,
        stage="asr",
        fraction=0.4,
        stage_started_at=utc_now_iso(),
    )
    assert store.soft_delete(orphan.id) is not None

    runner = JobRunner(store, JobEventBus(heartbeat=0.05), web_paths, config_builder)
    reconciled = runner.reconcile_orphans()

    assert orphan.id in reconciled
    fixed = store.get(orphan.id)
    assert fixed is not None
    assert fixed.status == STATUS_ERROR
    assert fixed.deleted is True


def test_cancel_queued_job_does_not_kill_current_processes(
    web_paths: WebPaths, config_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отмена ждущей в очереди задачи не гасит процессы текущей обработки."""
    from audio_transcriber.web import runner as runner_module

    calls: list[str] = []
    monkeypatch.setattr(
        runner_module, "terminate_all_processes", lambda: calls.append("terminate")
    )
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    runner = JobRunner(store, JobEventBus(heartbeat=0.05), web_paths, config_builder)
    first = web_paths.input_dir / "first.mp3"
    second = web_paths.input_dir / "second.mp3"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    store.create("first", first)
    store.create("second", second)
    # Обе задачи активны (в очереди), но ни одна ещё не обрабатывается.
    assert runner.submit("first", first) is True
    assert runner.submit("second", second) is True

    assert runner.cancel("second") is True
    # Воркер ещё не взял её в работу — процессы гасить нечего.
    assert calls == []


def test_reconcile_does_not_touch_cancelling_job(
    web_paths: WebPaths, config_builder
) -> None:
    """Отменяемая задача ещё активна — реконсиляция её не помечает как осиротевшую."""
    pipeline = _BlockingCancelPipeline()
    web_paths.ensure()
    store = JobsDB(web_paths.jobs_db)
    store.initialize()
    runner = JobRunner(
        store, JobEventBus(heartbeat=0.05), web_paths, config_builder, pipeline_fn=pipeline
    )
    source = web_paths.input_dir / "cancel-me.mp3"
    source.write_bytes(b"x")
    store.create("cancel-me", source)

    runner.start()
    try:
        assert runner.submit("cancel-me", source) is True
        assert pipeline.started.wait(timeout=5.0)
        assert runner.cancel("cancel-me") is True

        assert "cancel-me" not in runner.reconcile_orphans()
        current = store.get("cancel-me")
        assert current is not None
        assert current.status in (STATUS_RUNNING, STATUS_CANCELLED)

        final = _wait_for_status_direct(store, "cancel-me", STATUS_CANCELLED)
        assert final is not None and final.status == STATUS_CANCELLED
    finally:
        runner.stop()


# --- #16: обработанные файлы убираются из списка «Файлы» -------------------


def _file_names(client: TestClient, *, include_processed: bool = False) -> list[str]:
    params = {"include_processed": "true"} if include_processed else None
    return [item["name"] for item in client.get("/api/files", params=params).json()]


def test_successful_run_hides_file_but_keeps_it_accessible(
    client: TestClient, web_paths: WebPaths
) -> None:
    """После ``done`` файл уходит из ``/api/files``, но аудио и маркер на месте."""
    uploaded = _upload(client, "done.mp3", b"abc")
    job_id, details = _run_job(client, uploaded["name"])

    assert details["status"] == STATUS_DONE
    # Основной список пуст, полный (с обработанными) — содержит файл.
    assert _file_names(client) == []
    full = client.get("/api/files", params={"include_processed": "true"}).json()
    assert [item["name"] for item in full] == ["done.mp3"]
    assert full[0]["processed"] is True
    # Файл физически остался, marker создан.
    assert (web_paths.input_dir / "done.mp3").is_file()
    assert (web_paths.input_dir / "done.mp3.processed").is_file()
    # Аудио по-прежнему отдаётся.
    audio = client.get(f"/api/jobs/{job_id}/audio")
    assert audio.status_code == 200
    assert audio.content == b"abc"


def test_error_run_keeps_file_in_list(web_paths: WebPaths, config_builder) -> None:
    """Ошибочный прогон не убирает файл из списка (пометки нет)."""

    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        raise RuntimeError("сбой распознавания")

    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client, "err.mp3", b"abc")
        _job_id, details = _run_job(client, uploaded["name"])

        assert details["status"] == STATUS_ERROR
        files = client.get("/api/files").json()
        assert [item["name"] for item in files] == ["err.mp3"]
        assert files[0]["processed"] is False
        assert not (web_paths.input_dir / "err.mp3.processed").exists()


def test_cancelled_run_keeps_file_in_list(
    web_paths: WebPaths, config_builder
) -> None:
    """Отменённый прогон файл не убирает."""
    pipeline = _BlockingCancelPipeline()
    app = create_app(
        paths=web_paths,
        pipeline_fn=pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as client:
        uploaded = _upload(client, "cancel.mp3", b"abc")
        job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
        assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
        assert pipeline.started.wait(timeout=5.0)
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        _wait_for_status(client, job_id, STATUS_CANCELLED)

        files = client.get("/api/files").json()
        assert [item["name"] for item in files] == ["cancel.mp3"]
        assert files[0]["processed"] is False
        assert not (web_paths.input_dir / "cancel.mp3.processed").exists()


def test_restore_returns_file_to_main_list(client: TestClient, web_paths: WebPaths) -> None:
    """``POST /api/files/{name}/restore`` снимает метку; вызов идемпотентен."""
    uploaded = _upload(client, "restore.mp3", b"abc")
    _run_job(client, uploaded["name"])
    assert _file_names(client) == []

    response = client.post("/api/files/restore.mp3/restore")
    assert response.status_code == 200
    assert response.json() == {"restored": "restore.mp3", "processed": False}

    listing = client.get("/api/files").json()
    assert [item["name"] for item in listing] == ["restore.mp3"]
    assert listing[0]["processed"] is False
    assert not (web_paths.input_dir / "restore.mp3.processed").exists()
    # Повторный возврат — не ошибка; отсутствующий файл — 404.
    assert client.post("/api/files/restore.mp3/restore").status_code == 200
    assert client.post("/api/files/nope.mp3/restore").status_code == 404


def test_processed_file_rerun_keeps_it_hidden(
    client: TestClient, web_paths: WebPaths
) -> None:
    """Повторный запуск задачи по убранному файлу работает и не «возвращает» его."""
    uploaded = _upload(client, "again.mp3", b"abc")
    job_id, details = _run_job(client, uploaded["name"])
    assert details["status"] == STATUS_DONE
    assert _file_names(client) == []

    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    rerun = _wait_for_status(client, job_id, STATUS_DONE)

    assert rerun["status"] == STATUS_DONE
    assert _file_names(client) == []
    assert _file_names(client, include_processed=True) == ["again.mp3"]


def test_one_file_multiple_jobs_stays_hidden_after_success(
    client: TestClient, web_paths: WebPaths
) -> None:
    """Один файл на несколько задач: файл скрыт после первого успеха, вторая задача работает."""
    uploaded = _upload(client, "multi.mp3", b"abc")
    first = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]
    second = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    assert client.post(f"/api/jobs/{first}/run").status_code == 200
    _wait_for_status(client, first, STATUS_DONE)
    assert _file_names(client) == []

    # Вторая задача по тому же файлу не сломана: исходник на месте.
    assert client.post(f"/api/jobs/{second}/run").status_code == 200
    _wait_for_status(client, second, STATUS_DONE)
    assert (web_paths.input_dir / "multi.mp3.processed").is_file()


def test_delete_processed_file_removes_marker(
    client: TestClient, web_paths: WebPaths
) -> None:
    """Удаление выделенного файла уносит и sidecar-маркер."""
    uploaded = _upload(client, "delproc.mp3", b"abc")
    _run_job(client, uploaded["name"])
    marker = web_paths.input_dir / "delproc.mp3.processed"
    assert marker.is_file()

    assert client.delete("/api/files/delproc.mp3").status_code == 200

    assert not marker.exists()
    assert not (web_paths.input_dir / "delproc.mp3").exists()
    assert _file_names(client, include_processed=True) == []


def test_upload_clears_stale_processed_marker(
    client: TestClient, web_paths: WebPaths
) -> None:
    """Новый файл с именем прежде обработанного не наследует метку."""
    web_paths.input_dir.mkdir(parents=True, exist_ok=True)
    stale = web_paths.input_dir / "stale.mp3.processed"
    stale.write_text("", encoding="utf-8")

    uploaded = _upload(client, "stale.mp3", b"abc")

    assert uploaded["processed"] is False
    assert not stale.exists()
    assert _file_names(client) == ["stale.mp3"]

