"""Тесты REST API архива стенограмм (``/api/archive``, #99).

Конвейер подменяется фиктивной функцией: GPU/модели/сеть не нужны. Проверяются
список записей, правка метаданных, полнотекстовый поиск с фильтрами и ленивая
переиндексация после правки стенограммы.
"""

from __future__ import annotations

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


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        if on_progress is not None:
            on_progress(ProgressEvent("asr", "Распознавание речи", 0.5))
        ivan = Speaker(id="SPEAKER_00", display_name="Иван")
        maria = Speaker(id="SPEAKER_01", display_name="Мария")
        entries = [
            TranscriptEntry(
                start=0.0, end=1.0, text="Договорились до пятницы по поставке", speaker=ivan
            ),
            TranscriptEntry(start=1.0, end=2.0, text="Пятница выходной", speaker=maria),
        ]
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=2.0,
            entries=entries,
            speakers=[ivan, maria],
            low_confidence_threshold=-1.0,
        )

    return pipeline


def _config_builder(web_paths: WebPaths):
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
def client(web_paths: WebPaths, fake_pipeline) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        pipeline_fn=fake_pipeline,
        config_builder=_config_builder(web_paths),
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _ready_job(client: TestClient, name: str = "sample.mp3") -> str:
    uploaded = client.post(
        "/api/files/upload", files={"file": (name, b"\x00\x01", "audio/mpeg")}
    )
    assert uploaded.status_code == 201, uploaded.text
    created = client.post("/api/jobs", json={"path": uploaded.json()["path"]})
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] in {"done", "error"}:
            break
        time.sleep(0.02)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    return job_id


def test_list_archive_records(client: TestClient) -> None:
    first = _ready_job(client, "a.mp3")
    second = _ready_job(client, "b.mp3")

    payload = client.get("/api/archive").json()
    assert {record["job_id"] for record in payload["records"]} == {first, second}
    assert set(payload["speakers"]) == {"Иван", "Мария"}
    record = next(item for item in payload["records"] if item["job_id"] == first)
    assert record["name"] == "a.mp3"
    assert record["meeting_date"] is not None
    assert record["participants"] == []
    assert record["title"] is None


def test_patch_archive_record(client: TestClient) -> None:
    job_id = _ready_job(client)

    patched = client.patch(
        f"/api/archive/{job_id}",
        json={
            "title": "Планёрка по поставке",
            "meeting_date": "2026-06-04",
            "participants": ["Иван", "Мария"],
            "notes": "Обсудили сроки поставки",
        },
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["title"] == "Планёрка по поставке"
    assert body["meeting_date"] == "2026-06-04"
    assert body["participants"] == ["Иван", "Мария"]
    assert body["notes"] == "Обсудили сроки поставки"

    fetched = client.get(f"/api/archive/{job_id}").json()
    assert fetched["title"] == "Планёрка по поставке"

    # Очистка полей пустыми значениями.
    cleared = client.patch(f"/api/archive/{job_id}", json={"title": "", "notes": ""}).json()
    assert cleared["title"] is None
    assert cleared["notes"] is None
    assert cleared["participants"] == ["Иван", "Мария"]


def test_patch_invalid_date_returns_400(client: TestClient) -> None:
    job_id = _ready_job(client)
    response = client.patch(f"/api/archive/{job_id}", json={"meeting_date": "04.06.2026"})
    assert response.status_code == 400


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/archive/nope").status_code == 404
    assert client.patch("/api/archive/nope", json={"title": "x"}).status_code == 404


def test_search_by_text_and_filters(client: TestClient) -> None:
    first = _ready_job(client, "a.mp3")
    second = _ready_job(client, "b.mp3")
    client.patch(f"/api/archive/{first}", json={"meeting_date": "2026-06-04"})
    client.patch(f"/api/archive/{second}", json={"meeting_date": "2026-05-01"})

    found = client.get("/api/archive/search", params={"q": "пятниц"}).json()
    assert found["total"] == 4  # по 2 реплики в каждой задаче
    assert {hit["job_id"] for hit in found["hits"]} == {first, second}
    assert all("пятниц" in hit["snippet"].lower() for hit in found["hits"])

    # Фильтр по говорящему.
    by_speaker = client.get(
        "/api/archive/search", params={"q": "пятниц", "speaker": "Мария"}
    ).json()
    assert by_speaker["total"] == 2
    assert {hit["entry_index"] for hit in by_speaker["hits"]} == {1}

    # Фильтр по диапазону дат.
    by_date = client.get(
        "/api/archive/search",
        params={"q": "пятниц", "date_from": "2026-06-01", "date_to": "2026-06-30"},
    ).json()
    assert by_date["total"] == 2
    assert {hit["job_id"] for hit in by_date["hits"]} == {first}


def test_search_empty_query_returns_400(client: TestClient) -> None:
    _ready_job(client)
    assert client.get("/api/archive/search", params={"q": "  "}).status_code == 400


def test_index_refreshed_after_transcript_edit(client: TestClient) -> None:
    job_id = _ready_job(client)
    assert client.get("/api/archive/search", params={"q": "совершенно"}).json()["total"] == 0

    edited = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "Совершенно новый текст реплики"}]},
    )
    assert edited.status_code == 200

    fresh = client.get("/api/archive/search", params={"q": "совершенно"}).json()
    assert fresh["total"] == 1
    assert fresh["hits"][0]["entry_index"] == 0
    # Старый текст первой реплики больше не находится (остался только у второй
    # реплики слова нет), поэтому «поставке» исчезает из выдачи.
    assert client.get("/api/archive/search", params={"q": "поставке"}).json()["total"] == 0


def test_date_filter_list(client: TestClient) -> None:
    first = _ready_job(client, "a.mp3")
    second = _ready_job(client, "b.mp3")
    client.patch(f"/api/archive/{first}", json={"meeting_date": "2026-06-04"})
    client.patch(f"/api/archive/{second}", json={"meeting_date": "2026-05-01"})

    payload = client.get(
        "/api/archive", params={"date_from": "2026-06-01", "date_to": "2026-06-30"}
    ).json()
    assert [record["job_id"] for record in payload["records"]] == [first]
