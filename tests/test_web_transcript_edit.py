"""Тесты ручной правки текста реплик (#26) и её устойчивости.

Конвейер и enrollment подменяются, поэтому GPU/модели не нужны. Проверяем, что
правка сохраняется в JSON результата, не трогает таймкоды/говорящего, переживает
переименование/слияние говорящих, «Применить имена» и повторный прогон из
кэша, а также попадает в экспорт.
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
from audio_transcriber.diarization.enrollment import EnrollmentOutcome
from audio_transcriber.diarization.samples import samples_directory
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths


def _write_wav(path: Path, seconds: float = 0.2, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))


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
                start=0.0, end=1.0, text="привет", speaker=speaker_a, avg_logprob=-2.0
            ),
            TranscriptEntry(
                start=1.0,
                end=2.0,
                text="пока",
                speaker=speaker_b,
                avg_logprob=-0.5,
                overlap=True,
                extra_speakers=[speaker_a],
                speaker_confidence=0.3,
            ),
            TranscriptEntry(
                start=2.0, end=3.0, text="ага", speaker=speaker_a, avg_logprob=-2.0
            ),
        ]
        directory = samples_directory(config.output_dir, config.input_file)
        for display_name in ("Иван", "Пётр"):
            _write_wav(directory / f"{display_name}.wav")
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=3.0,
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
    _rerun_job(client, job_id)
    return job_id


def _rerun_job(client: TestClient, job_id: str) -> None:
    assert client.post(f"/api/jobs/{job_id}/run").status_code == 200
    deadline = time.time() + 5.0
    while time.time() < deadline:
        details = client.get(f"/api/jobs/{job_id}").json()
        if details["status"] in {"done", "error"}:
            assert details["status"] == "done", details.get("error")
            return
        time.sleep(0.02)
    raise AssertionError("задача не завершилась за отведённое время")


def _prepared_job(client: TestClient) -> tuple[str, dict]:
    uploaded = _upload(client)
    job_id = _run_job(client, uploaded["name"])
    result = client.get(f"/api/jobs/{job_id}/result").json()
    return job_id, result


def test_result_entries_have_edited_flag_by_default(client: TestClient) -> None:
    _, result = _prepared_job(client)

    assert all(entry["edited"] is False for entry in result["entries"])
    assert all(entry["original_text"] is None for entry in result["entries"])


def test_edit_text_sets_flag_and_persists(client: TestClient, web_paths: WebPaths) -> None:
    job_id, result = _prepared_job(client)
    before = result["entries"][0]

    response = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "Привет, мир!"}]},
    )

    assert response.status_code == 200
    body = response.json()
    edited = body["entries"][0]
    assert edited["text"] == "Привет, мир!"
    assert edited["edited"] is True
    assert edited["original_text"] == "привет"
    # Таймкоды и говорящий не меняются.
    assert edited["start"] == before["start"]
    assert edited["end"] == before["end"]
    assert edited["speaker_id"] == before["speaker_id"]
    # Остальные реплики не затронуты.
    assert body["entries"][1]["text"] == "пока"
    assert body["entries"][1]["edited"] is False

    # GET /result отдаёт правку, JSON на диске перезаписан.
    fetched = client.get(f"/api/jobs/{job_id}/result").json()
    assert fetched["entries"][0]["text"] == "Привет, мир!"
    on_disk = json.loads((web_paths.results_dir / f"{job_id}.json").read_text("utf-8"))
    assert on_disk["entries"][0]["edited"] is True
    assert on_disk["entries"][0]["original_text"] == "привет"


def test_reset_restores_original_text(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 1, "text": "исправлено"}]},
    )

    response = client.patch(f"/api/jobs/{job_id}/transcript", json={"resets": [1]})

    assert response.status_code == 200
    entry = response.json()["entries"][1]
    assert entry["text"] == "пока"
    assert entry["edited"] is False
    assert entry["original_text"] is None


def test_edit_keeping_text_does_not_mark_edited(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "привет"}]},
    )

    assert response.status_code == 200
    entry = response.json()["entries"][0]
    assert entry["text"] == "привет"
    assert entry["edited"] is False
    assert entry["original_text"] is None


def test_edit_keeps_first_original_text(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "первое"}]},
    )

    response = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "второе"}]},
    )

    entry = response.json()["entries"][0]
    assert entry["text"] == "второе"
    assert entry["original_text"] == "привет"


def test_edit_invalid_index_returns_400(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 99, "text": "нет"}]},
    )

    assert response.status_code == 400


def test_edit_without_changes_returns_400(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(f"/api/jobs/{job_id}/transcript", json={})

    assert response.status_code == 400


def test_edit_survives_rename_and_merge(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "правка"}]},
    )

    renamed = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_00": "Иван Иванов"}},
    )
    assert renamed.status_code == 200
    assert renamed.json()["entries"][0]["text"] == "правка"
    assert renamed.json()["entries"][0]["edited"] is True

    merged = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"merges": [{"source": "SPEAKER_01", "target": "SPEAKER_00"}]},
    )
    assert merged.status_code == 200
    assert merged.json()["entries"][0]["text"] == "правка"
    assert merged.json()["entries"][0]["original_text"] == "привет"


def test_edit_survives_apply_names(
    client: TestClient, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "правка"}]},
    )
    _write_wav(voices_dir / "Иван Иванов.wav")

    def fake_enroll(**kwargs) -> EnrollmentOutcome:
        return EnrollmentOutcome(
            mapping={"SPEAKER_00": "Иван Иванов"},
            best_candidates={},
            speaker_count=2,
        )

    monkeypatch.setattr("audio_transcriber.web.speakers.enroll_speakers", fake_enroll)

    response = client.post(f"/api/jobs/{job_id}/apply-names", json={})

    assert response.status_code == 200
    entry = response.json()["result"]["entries"][0]
    assert entry["text"] == "правка"
    assert entry["edited"] is True
    assert entry["original_text"] == "привет"


def test_edit_survives_rerun_from_cache(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 2, "text": "исправлено"}]},
    )

    _rerun_job(client, job_id)

    entry = client.get(f"/api/jobs/{job_id}/result").json()["entries"][2]
    assert entry["text"] == "исправлено"
    assert entry["edited"] is True
    assert entry["original_text"] == "ага"


def test_export_contains_edited_text(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "Привет, мир!"}]},
    )

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": "txt"})

    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert "Привет, мир!" in body
    assert "привет" not in body


def test_assign_speaker_to_selected_entries(
    client: TestClient, web_paths: WebPaths
) -> None:
    """Принудительное назначение существующего говорящего выбранным репликам (#59)."""
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [0, 2], "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_speaker_id"] == "SPEAKER_01"
    assert body["created_speaker"] is None
    assert body["indexes"] == [0, 2]
    assert [change["index"] for change in body["changes"]] == [0, 2]
    assert [entry["speaker_id"] for entry in body["result"]["entries"]] == [
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_01",
    ]
    # Список говорящих не меняется и не prune-ится.
    assert [s["id"] for s in body["result"]["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]

    on_disk = json.loads((web_paths.results_dir / f"{job_id}.json").read_text("utf-8"))
    assert on_disk["entries"][0]["speaker_id"] == "SPEAKER_01"
    assert on_disk["entries"][2]["speaker_id"] == "SPEAKER_01"


def test_assign_speaker_creates_new_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [0], "new_name": "Анна"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_speaker_id"] == "SPEAKER_02"
    assert body["created_speaker"] == {"id": "SPEAKER_02", "display_name": "Анна"}
    entry = body["result"]["entries"][0]
    assert entry["speaker_id"] == "SPEAKER_02"
    names = {s["id"]: s["display_name"] for s in body["result"]["speakers"]}
    assert names["SPEAKER_02"] == "Анна"
    assert body["result"]["speakers"][-1]["has_sample"] is False


def test_assign_speaker_reuses_existing_by_name(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [0], "new_name": "Пётр"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_speaker_id"] == "SPEAKER_01"
    assert body["created_speaker"] is None
    assert body["result"]["entries"][0]["speaker_id"] == "SPEAKER_01"


def test_assign_speaker_removes_target_from_extras(client: TestClient) -> None:
    """Целевой говорящий, бывший доп. участником, не дублируется (#59)."""
    job_id, _ = _prepared_job(client)
    # Реплика 1: основной SPEAKER_01 (Пётр), доп. SPEAKER_00 (Иван).
    assert client.get(f"/api/jobs/{job_id}/result").json()["entries"][1][
        "extra_speaker_ids"
    ] == ["SPEAKER_00"]

    response = client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [1], "target_speaker_id": "SPEAKER_00"},
    )

    assert response.status_code == 200
    entry = response.json()["result"]["entries"][1]
    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["extra_speaker_ids"] == []
    change = response.json()["changes"][0]
    assert change["before_speaker_id"] == "SPEAKER_01"
    assert change["before_extra_ids"] == ["SPEAKER_00"]
    assert change["after_extra_ids"] == []


def test_assign_speaker_co_speaker_keeps_primary(client: TestClient) -> None:
    """``co_speaker=True`` добавляет цель участником наложения (#59)."""
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01", "co_speaker": True},
    )

    assert response.status_code == 200
    entry = response.json()["result"]["entries"][0]
    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["extra_speaker_ids"] == ["SPEAKER_01"]


def test_assign_speaker_preserves_manual_edit(client: TestClient) -> None:
    """Назначение говорящего не теряет ручную правку текста (#26)."""
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "исправлено вручную"}]},
    )

    response = client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    entry = response.json()["result"]["entries"][0]
    assert entry["speaker_id"] == "SPEAKER_01"
    assert entry["text"] == "исправлено вручную"
    assert entry["edited"] is True
    assert entry["original_text"] == "привет"
    assert entry["low_confidence"] is True


def test_assign_speaker_undo_restores_previous_result(client: TestClient) -> None:
    job_id, original = _prepared_job(client)
    client.post(
        f"/api/jobs/{job_id}/transcript/assign-speaker",
        json={"indexes": [0, 2], "target_speaker_id": "SPEAKER_01"},
    )

    undone = client.post(f"/api/jobs/{job_id}/speakers/undo")

    assert undone.status_code == 200
    body = undone.json()
    assert body["entries"] == original["entries"]
    assert body["speakers"] == original["speakers"]
    assert client.post(f"/api/jobs/{job_id}/speakers/undo").status_code == 404


def test_assign_speaker_validation_errors(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    url = f"/api/jobs/{job_id}/transcript/assign-speaker"

    no_indexes = client.post(url, json={"target_speaker_id": "SPEAKER_01"})
    assert no_indexes.status_code == 400

    bad_index = client.post(
        url, json={"indexes": [99], "target_speaker_id": "SPEAKER_01"}
    )
    assert bad_index.status_code == 400

    neither = client.post(url, json={"indexes": [0]})
    assert neither.status_code == 400

    both = client.post(
        url,
        json={"indexes": [0], "target_speaker_id": "SPEAKER_01", "new_name": "X"},
    )
    assert both.status_code == 400

    unknown_target = client.post(
        url, json={"indexes": [0], "target_speaker_id": "SPEAKER_99"}
    )
    assert unknown_target.status_code == 404

    # Реплика уже принадлежит этому говорящему — менять нечего.
    entry = client.get(f"/api/jobs/{job_id}/result").json()["entries"][0]
    same = client.post(
        url,
        json={"indexes": [0], "target_speaker_id": entry["speaker_id"]},
    )
    assert same.status_code == 400


def test_assign_speaker_missing_result_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/jobs/nonexistent/transcript/assign-speaker",
        json={"indexes": [0], "new_name": "X"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("fmt", ["txt", "json", "srt", "docx"])
def test_export_all_formats_contain_edited_text(client: TestClient, fmt: str) -> None:
    job_id, _ = _prepared_job(client)
    client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "Привет, мир!"}]},
    )

    response = client.get(f"/api/jobs/{job_id}/export", params={"fmt": fmt})

    assert response.status_code == 200
    if fmt == "docx":
        import io

        from docx import Document

        document = Document(io.BytesIO(response.content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    else:
        text = response.content.decode("utf-8")
    assert "Привет, мир!" in text
    assert "привет" not in text

