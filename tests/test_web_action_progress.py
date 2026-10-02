"""Тесты прогресса длительных действий веб-интерфейса (#58).

Проверяются:

* шина событий действий (:class:`ActionEventBus`): трансляция этапов,
  закрытие потока конечным событием, реигра историей и вытеснение старых;
* публикация этапов четырьмя эндпоинтами при наличии заголовка ``X-Action-Id``
  (apply-names/enrollment, apply-glossary, correct-text, protocol);
* обратная совместимость: без заголовка эндпоинты работают как раньше.

Реальные модели и LLM не запускаются: enrollment, протокол и конвейер
подменяются фейками, эмитирующими события ``ProgressEvent``.
"""

from __future__ import annotations

import asyncio
import json
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.enrollment import EnrollmentOutcome
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.protocol import ProtocolArtifacts
from audio_transcriber.storage.glossary_db import GlossaryDB
from audio_transcriber.web.actions import (
    ACTION_CORRECTION,
    ACTION_ENROLLMENT,
    ACTION_GLOSSARY,
    ACTION_PROTOCOL,
    ActionEventBus,
    sanitize_action_id,
)
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


def _write_wav(path: Path, seconds: float = 0.2, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))


# --- юнит-тесты шины действий ----------------------------------------------


def test_action_bus_streams_stages_and_closes_on_done() -> None:
    bus = ActionEventBus(heartbeat=0.05)
    received: list[dict | None] = []

    async def collect() -> None:
        async for event in bus.subscribe("a1"):
            received.append(event)

    async def main() -> None:
        task = asyncio.create_task(collect())
        await asyncio.sleep(0.01)
        bus.publish(
            "a1",
            {"action": "glossary", "stage": "process", "message": "…", "status": "running"},
        )
        bus.publish(
            "a1",
            {"action": "glossary", "stage": "done", "message": "Готово", "status": "done"},
        )
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(main())

    assert received[-1] is not None
    assert received[-1]["status"] == "done"
    assert any(event and event.get("stage") == "process" for event in received)


def test_action_bus_replays_history_to_late_subscriber() -> None:
    bus = ActionEventBus(heartbeat=0.05)
    bus.publish("a1", {"stage": "prepare", "message": "Подготовка", "status": "running"})
    bus.publish("a1", {"stage": "done", "message": "Готово", "status": "done"})

    async def collect() -> list[dict | None]:
        events: list[dict | None] = []
        async for event in bus.subscribe("a1"):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert [event["stage"] for event in events if event is not None] == ["prepare", "done"]


def test_action_bus_evicts_oldest_actions() -> None:
    bus = ActionEventBus(heartbeat=0.05, max_actions=2)
    bus.publish("a", {"stage": "done", "status": "done"})
    bus.publish("b", {"stage": "done", "status": "done"})
    bus.publish("c", {"stage": "done", "status": "done"})

    assert bus.history("a") == []
    assert bus.history("b")
    assert bus.history("c")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("123e4567-e89b-12d3-a456-426614174000", "123e4567-e89b-12d3-a456-426614174000"),
        ("  action_1 ", "action_1"),
        ("", None),
        (None, None),
        ("плохой id", None),
        ("a" * 65, None),
    ],
)
def test_sanitize_action_id(value: str | None, expected: str | None) -> None:
    assert sanitize_action_id(value) == expected


# --- интеграционные тесты эндпоинтов ---------------------------------------


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "voices"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def glossary_db(tmp_path: Path) -> Path:
    path = tmp_path / "glossary.db"
    with GlossaryDB(path) as db:
        db.add_entry("ОИБ", variant="АИБ")
    return path


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        speaker = Speaker(id="SPEAKER_00", display_name="Иван")
        entries = [
            TranscriptEntry(start=0.0, end=1.0, text="АИБ безопасность", speaker=speaker),
            TranscriptEntry(start=1.0, end=2.0, text="привет ,мир", speaker=speaker),
            TranscriptEntry(
                start=2.0, end=3.0, text="Конфиденциальнасть мы не нашли.", speaker=speaker
            ),
        ]
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=3.0,
            entries=entries,
            speakers=[speaker],
        )

    return pipeline


@pytest.fixture
def protocol_fn():
    def build(config: AppConfig, result: TranscriptionResult, **kwargs) -> ProtocolArtifacts:
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress(ProgressEvent("llm", "Резюме встречи", 0.5))
            on_progress(ProgressEvent("export", "Экспорт txt", 1.0))
        config.ensure_output_dir()
        target = config.output_dir / f"{config.input_file.stem}.txt"
        target.write_text("тело протокола", encoding="utf-8")
        return ProtocolArtifacts(paths=(target,), summary="Резюме встречи")

    return build


@pytest.fixture
def config_builder(web_paths: WebPaths, glossary_db: Path):
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
            glossary_db=glossary_db,
        )

    return build


@pytest.fixture
def client(
    web_paths: WebPaths,
    voices_dir: Path,
    fake_pipeline,
    config_builder,
    protocol_fn,
) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        voices_dir=voices_dir,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        protocol_fn=protocol_fn,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _prepared_job(client: TestClient) -> str:
    upload = client.post(
        "/api/files/upload", files={"file": ("sample.mp3", b"\x00\x01", "audio/mpeg")}
    )
    assert upload.status_code == 201
    created = client.post("/api/jobs", json={"path": upload.json()["name"]})
    assert created.status_code == 201
    job_id = created.json()["id"]
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        details = client.get(f"/api/jobs/{job_id}").json()
        if details["status"] in {"done", "error"}:
            assert details["status"] == "done", details.get("error")
            return job_id
        time.sleep(0.02)
    raise AssertionError("задача не завершилась за отведённое время")


def _events(client: TestClient, action_id: str) -> list[dict]:
    response = client.get(f"/api/actions/{action_id}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events: list[dict] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _post_with_action(
    client: TestClient, url: str, action_id: str, payload: dict | None = None
):
    response = client.post(
        url,
        json=payload if payload is not None else {},
        headers={"X-Action-Id": action_id},
    )
    assert response.status_code == 200, response.text
    return response


def test_glossary_publishes_action_events(client: TestClient) -> None:
    job_id = _prepared_job(client)
    action_id = "glossary-action"

    response = _post_with_action(
        client, f"/api/jobs/{job_id}/apply-glossary", action_id, {"respect_edited": True}
    )
    assert response.json()["error"] is None

    events = _events(client, action_id)
    assert events, "ожидались события прогресса"
    assert {event["action"] for event in events} == {ACTION_GLOSSARY}
    assert any(event["stage"] == "prepare" for event in events)
    assert any(event["stage"] == "process" for event in events)
    assert events[-1]["status"] == "done"
    assert events[-1]["message"].startswith("Готово")
    assert all(isinstance(event["elapsed"], (int, float)) for event in events)


def test_correct_text_publishes_action_events(client: TestClient) -> None:
    job_id = _prepared_job(client)
    action_id = "correction-action"

    response = _post_with_action(
        client,
        f"/api/jobs/{job_id}/correct-text",
        action_id,
        {"dry_run": True, "fix_common": True, "check_spelling": True},
    )
    assert response.json()["suggestions"]

    events = _events(client, action_id)
    assert {event["action"] for event in events} == {ACTION_CORRECTION}
    assert any(event["stage"] == "analyze" for event in events)
    assert events[-1]["status"] == "done"


def test_protocol_publishes_action_events(client: TestClient) -> None:
    job_id = _prepared_job(client)
    action_id = "protocol-action"

    response = _post_with_action(client, f"/api/jobs/{job_id}/protocol", action_id)
    assert response.json()["summary"] == "Резюме встречи"

    events = _events(client, action_id)
    assert {event["action"] for event in events} == {ACTION_PROTOCOL}
    stages = [event["stage"] for event in events]
    assert "prepare" in stages
    assert "llm" in stages
    assert "export" in stages
    assert events[-1]["status"] == "done"


def test_apply_names_publishes_enrollment_events(
    client: TestClient, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _prepared_job(client)
    _write_wav(voices_dir / "Иван.wav")

    def fake_enroll(**kwargs) -> EnrollmentOutcome:
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress(ProgressEvent("samples", "Загрузка образцов голоса", 0.0))
            on_progress(ProgressEvent("embeddings", "Эмбеддинги образцов", 0.5))
            on_progress(ProgressEvent("matching", "Сопоставление", 1.0))
        return EnrollmentOutcome(
            mapping={"SPEAKER_00": "Иван"}, best_candidates={}, speaker_count=1
        )

    monkeypatch.setattr("audio_transcriber.web.speakers.enroll_speakers", fake_enroll)
    action_id = "enrollment-action"

    response = _post_with_action(
        client, f"/api/jobs/{job_id}/apply-names", action_id, {}
    )
    assert response.json()["matched"] == {"SPEAKER_00": "Иван"}

    events = _events(client, action_id)
    assert {event["action"] for event in events} == {ACTION_ENROLLMENT}
    stages = [event["stage"] for event in events]
    assert {"samples", "embeddings", "matching", "apply"} <= set(stages)
    assert stages.index("matching") < stages.index("apply")
    assert events[-1]["status"] == "done"


def test_apply_names_soft_error_publishes_error_event(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _prepared_job(client)
    action_id = "enrollment-error"

    # Библиотека пуста и явные образцы не заданы — мягкая ошибка без enrollment.
    response = _post_with_action(
        client, f"/api/jobs/{job_id}/apply-names", action_id, {}
    )
    assert response.json()["error"]

    events = _events(client, action_id)
    assert events[-1]["status"] == "error"


def test_endpoints_work_without_action_header(client: TestClient) -> None:
    job_id = _prepared_job(client)

    glossary = client.post(f"/api/jobs/{job_id}/apply-glossary", json={})
    assert glossary.status_code == 200
    assert glossary.json()["error"] is None

    correction = client.post(
        f"/api/jobs/{job_id}/correct-text", json={"dry_run": True}
    )
    assert correction.status_code == 200

    protocol = client.post(f"/api/jobs/{job_id}/protocol")
    assert protocol.status_code == 200


def test_invalid_action_id_falls_back_and_events_endpoint_rejects(
    client: TestClient,
) -> None:
    job_id = _prepared_job(client)

    # Некорректный заголовок не ломает действие — прогресс просто не ведётся.
    response = client.post(
        f"/api/jobs/{job_id}/apply-glossary",
        json={},
        headers={"X-Action-Id": "bad!id"},
    )
    assert response.status_code == 200

    assert client.get("/api/actions/bad!id/events").status_code == 400
