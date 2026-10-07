"""Тесты разрезания реплики между говорявшими и второго говорящего (#78).

Конвейер подменяется, поэтому GPU/модели не нужны. Проверяем доменные операции
через веб-API: сплит по пословным таймкодам, сохранение ручной правки текста
(#26), добавление/удаление второго говорящего, запись в файл результата и
одношаговую отмену.
"""

from __future__ import annotations

import json
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.samples import samples_directory
from audio_transcriber.domain.models import (
    Speaker,
    TranscriptEntry,
    TranscriptionResult,
    WordTimestamp,
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


def _words(*items: tuple[str, float, float]) -> list[WordTimestamp]:
    return [WordTimestamp(text=text, start=start, end=end) for text, start, end in items]


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "voices"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def fake_pipeline():
    def pipeline(config: AppConfig, *, on_progress=None) -> TranscriptionResult:
        speaker_a = Speaker(id="SPEAKER_00", display_name="Иван")
        speaker_b = Speaker(id="SPEAKER_01", display_name="Пётр")
        entries = [
            TranscriptEntry(
                start=0.0,
                end=4.0,
                text="привет как дела хорошо",
                speaker=speaker_a,
                avg_logprob=-2.0,
                words=_words(
                    ("привет", 0.0, 1.0),
                    ("как", 1.0, 2.0),
                    ("дела", 2.0, 3.0),
                    ("хорошо", 3.0, 4.0),
                ),
            ),
            TranscriptEntry(
                start=4.0,
                end=8.0,
                text="пока всё",
                speaker=speaker_b,
                avg_logprob=-0.5,
                overlap=True,
                extra_speakers=[speaker_a],
                speaker_confidence=0.3,
                words=_words(("пока", 4.0, 6.0), ("всё", 6.0, 8.0)),
            ),
            TranscriptEntry(
                start=8.0, end=12.0, text="ага конечно", speaker=speaker_a, avg_logprob=-2.0
            ),
        ]
        directory = samples_directory(config.output_dir, config.input_file)
        for display_name in ("Иван", "Пётр"):
            _write_wav(directory / f"{display_name}.wav")
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=12.0,
            entries=entries,
            speakers=[speaker_a, speaker_b],
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
            export_speaker_samples=True,
            notifications=False,
            timeline=False,
            protocol_auto=False,
            use_cache=False,
        )

    return build


@pytest.fixture
def client(
    web_paths: WebPaths, voices_dir: Path, fake_pipeline, config_builder
) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        voices_dir=voices_dir,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str = "sample.mp3") -> dict:
    response = client.post(
        "/api/files/upload", files={"file": (name, b"\x00\x01", "audio/mpeg")}
    )
    assert response.status_code == 201
    return response.json()


def _run_job(client: TestClient, path: str) -> str:
    created = client.post("/api/jobs", json={"path": path})
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


def _prepared_job(client: TestClient) -> tuple[str, dict]:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    result = client.get(f"/api/jobs/{job_id}/result").json()
    return job_id, result


def _split_url(job_id: str) -> str:
    return f"/api/jobs/{job_id}/transcript/split"


def _extra_url(job_id: str) -> str:
    return f"/api/jobs/{job_id}/transcript/extra-speaker"


def test_result_has_word_timestamps(client: TestClient) -> None:
    _, result = _prepared_job(client)

    assert [word["text"] for word in result["entries"][0]["words"]] == [
        "привет",
        "как",
        "дела",
        "хорошо",
    ]


def test_split_entry_by_words(client: TestClient, web_paths: WebPaths) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        _split_url(job_id),
        json={
            "index": 0,
            "boundary": 2.5,
            "first": {"speaker_id": "SPEAKER_00"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["first_speaker_id"] == "SPEAKER_00"
    assert body["second_speaker_id"] == "SPEAKER_01"
    assert body["created_speakers"] == []
    entries = body["result"]["entries"]
    assert len(entries) == 4
    assert entries[0]["text"] == "привет как дела"
    assert entries[0]["speaker_id"] == "SPEAKER_00"
    assert entries[0]["end"] == 2.5
    assert entries[1]["text"] == "хорошо"
    assert entries[1]["speaker_id"] == "SPEAKER_01"
    assert entries[1]["start"] == 2.5
    # Вычисленный флаг низкой уверенности перенесён по времени на обе части.
    assert entries[0]["low_confidence"] is True
    assert entries[1]["low_confidence"] is True
    # Следующая реплика не тронута.
    assert entries[2]["text"] == "пока всё"

    on_disk = json.loads((web_paths.results_dir / f"{job_id}.json").read_text("utf-8"))
    assert on_disk["entries"][0]["text"] == "привет как дела"
    assert on_disk["entries"][1]["speaker_id"] == "SPEAKER_01"


def test_split_creates_new_speaker_for_second_part(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        _split_url(job_id),
        json={
            "index": 0,
            "boundary": 2.0,
            "first": {"speaker_id": "SPEAKER_00"},
            "second": {"new_name": "Анна"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["created_speakers"] == [{"id": "SPEAKER_02", "display_name": "Анна"}]
    assert body["second_speaker_id"] == "SPEAKER_02"
    assert body["result"]["entries"][1]["speaker_id"] == "SPEAKER_02"
    names = {speaker["id"]: speaker["display_name"] for speaker in body["result"]["speakers"]}
    assert names["SPEAKER_02"] == "Анна"


def test_split_both_new_speakers_get_distinct_ids(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        _split_url(job_id),
        json={
            "index": 0,
            "boundary": 2.0,
            "first": {"new_name": "Анна"},
            "second": {"new_name": "Борис"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    created = {speaker["display_name"]: speaker["id"] for speaker in body["created_speakers"]}
    assert created == {"Анна": "SPEAKER_02", "Борис": "SPEAKER_03"}
    assert body["first_speaker_id"] == "SPEAKER_02"
    assert body["second_speaker_id"] == "SPEAKER_03"


def test_split_preserves_manual_text_edit(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    edited = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "правка текста"}]},
    )
    assert edited.status_code == 200

    response = client.post(
        _split_url(job_id),
        json={
            "index": 0,
            "boundary": 2.0,
            "first": {"speaker_id": "SPEAKER_00"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    )

    assert response.status_code == 200
    entries = response.json()["result"]["entries"]
    assert entries[0]["text"] == "правка"
    assert entries[1]["text"] == "текста"
    assert entries[0]["edited"] is True
    assert entries[1]["edited"] is True
    assert entries[0]["original_text"] == "привет как"
    assert entries[1]["original_text"] == "дела хорошо"


def test_split_undo_restores_result(client: TestClient) -> None:
    job_id, original = _prepared_job(client)
    client.post(
        _split_url(job_id),
        json={
            "index": 0,
            "boundary": 2.5,
            "first": {"speaker_id": "SPEAKER_00"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    )

    undone = client.post(f"/api/jobs/{job_id}/speakers/undo")

    assert undone.status_code == 200
    assert undone.json()["entries"] == original["entries"]
    assert undone.json()["speakers"] == original["speakers"]


def test_split_validation_errors(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    url = _split_url(job_id)

    assert client.post(
        url,
        json={
            "index": 99,
            "boundary": 1.0,
            "first": {"speaker_id": "SPEAKER_00"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    ).status_code == 400

    for boundary in (0.0, 10.0):
        assert client.post(
            url,
            json={
                "index": 0,
                "boundary": boundary,
                "first": {"speaker_id": "SPEAKER_00"},
                "second": {"speaker_id": "SPEAKER_01"},
            },
        ).status_code == 400

    assert client.post(
        url,
        json={
            "index": 0,
            "boundary": 2.0,
            "first": {"speaker_id": "SPEAKER_00", "new_name": "X"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    ).status_code == 400

    assert client.post(
        url,
        json={
            "index": 0,
            "boundary": 2.0,
            "first": {"speaker_id": "SPEAKER_99"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    ).status_code == 404

    assert client.post(
        _split_url("nonexistent"),
        json={
            "index": 0,
            "boundary": 2.0,
            "first": {"speaker_id": "SPEAKER_00"},
            "second": {"speaker_id": "SPEAKER_01"},
        },
    ).status_code == 404


def test_add_extra_speaker_existing(client: TestClient, web_paths: WebPaths) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        _extra_url(job_id),
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["removed"] is False
    assert body["target_speaker_id"] == "SPEAKER_01"
    assert body["created_speaker"] is None
    entry = body["result"]["entries"][0]
    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["extra_speaker_ids"] == ["SPEAKER_01"]
    assert entry["overlap"] is True
    assert body["changes"][0]["after_extra_ids"] == ["SPEAKER_01"]

    on_disk = json.loads((web_paths.results_dir / f"{job_id}.json").read_text("utf-8"))
    assert on_disk["entries"][0]["extra_speaker_ids"] == ["SPEAKER_01"]


def test_add_extra_speaker_creates_new(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        _extra_url(job_id), json={"indexes": [0], "new_name": "Анна"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["created_speaker"] == {"id": "SPEAKER_02", "display_name": "Анна"}
    assert body["result"]["entries"][0]["extra_speaker_ids"] == ["SPEAKER_02"]
    names = {speaker["id"]: speaker["display_name"] for speaker in body["result"]["speakers"]}
    assert names["SPEAKER_02"] == "Анна"


def test_add_extra_speaker_preserves_manual_edit(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "исправлено"}]},
    )

    response = client.post(
        _extra_url(job_id),
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    entry = response.json()["result"]["entries"][0]
    assert entry["text"] == "исправлено"
    assert entry["edited"] is True
    assert entry["original_text"] == "привет как дела хорошо"
    assert entry["extra_speaker_ids"] == ["SPEAKER_01"]


def test_remove_extra_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.post(
        _extra_url(job_id),
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01"},
    )

    response = client.post(
        _extra_url(job_id),
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01", "remove": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["removed"] is True
    entry = body["result"]["entries"][0]
    assert entry["extra_speaker_ids"] == []
    assert entry["overlap"] is False


def test_extra_speaker_undo(client: TestClient) -> None:
    job_id, original = _prepared_job(client)
    client.post(
        _extra_url(job_id),
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01"},
    )

    undone = client.post(f"/api/jobs/{job_id}/speakers/undo")

    assert undone.status_code == 200
    assert undone.json()["entries"] == original["entries"]


def test_extra_speaker_validation_errors(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    url = _extra_url(job_id)

    assert client.post(url, json={"target_speaker_id": "SPEAKER_01"}).status_code == 400
    assert client.post(url, json={"indexes": [99], "new_name": "X"}).status_code == 400
    assert client.post(url, json={"indexes": [0]}).status_code == 400
    assert (
        client.post(
            url,
            json={"indexes": [0], "target_speaker_id": "SPEAKER_01", "new_name": "X"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            url, json={"indexes": [0], "target_speaker_id": "SPEAKER_99"}
        ).status_code
        == 404
    )

    # Реплика 0 — уже основной SPEAKER_00: добавить его вторым нечего.
    assert (
        client.post(
            url, json={"indexes": [0], "target_speaker_id": "SPEAKER_00"}
        ).status_code
        == 400
    )
    # У реплики 0 нет второго говорящего SPEAKER_01 до добавления.
    assert (
        client.post(
            url,
            json={"indexes": [0], "target_speaker_id": "SPEAKER_01", "remove": True},
        ).status_code
        == 400
    )
