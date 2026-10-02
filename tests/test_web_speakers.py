"""Тесты веб-API этапа 2: правка говорящих, применение имён, библиотека голосов.

Конвейер и enrollment подменяются, поэтому GPU/модели не нужны. Каталоги
данных и библиотеки голосов изолированы во временных путях.
"""

from __future__ import annotations

import json
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

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


def _wav_bytes(seconds: float = 0.2, rate: int = 16000) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


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
def client(web_paths: WebPaths, voices_dir: Path, fake_pipeline, config_builder) -> Iterator[TestClient]:
    app = create_app(
        paths=web_paths,
        voices_dir=voices_dir,
        pipeline_fn=fake_pipeline,
        config_builder=config_builder,
        heartbeat=0.05,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str = "sample.mp3", data: bytes = b"\x00\x01") -> dict:
    response = client.post("/api/files/upload", files={"file": (name, data, "audio/mpeg")})
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


def test_result_has_samples_and_meta(client: TestClient) -> None:
    _, result = _prepared_job(client)

    assert result["samples"]
    assert {speaker["id"] for speaker in result["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(speaker["has_sample"] for speaker in result["speakers"])

    meta = client.get(f"/api/jobs/{client.get('/api/jobs').json()[0]['id']}/samples").json()
    assert {item["speaker_id"] for item in meta} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(item["duration"] > 0 for item in meta)


def test_result_includes_extra_speakers_and_speaker_confidence(client: TestClient) -> None:
    _, result = _prepared_job(client)

    entry = result["entries"][1]
    assert entry["extra_speaker_ids"] == ["SPEAKER_00"]
    assert entry["speaker_confidence"] == pytest.approx(0.3)
    assert entry["low_speaker_confidence"] is True

    # Доп. говорящие присутствуют в speakers — id разрешаются во имя фронтендом.
    speaker_ids = {speaker["id"] for speaker in result["speakers"]}
    assert set(entry["extra_speaker_ids"]) <= speaker_ids
    assert {mark["key"] for mark in result["marks"]} == {
        "low_confidence",
        "speaker_uncertain",
        "overlap",
    }


def test_rename_of_extra_speaker_resolves_to_new_name(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_00": "Иван Иванов"}},
    )

    assert response.status_code == 200
    body = response.json()
    names = {speaker["id"]: speaker["display_name"] for speaker in body["speakers"]}
    entry = body["entries"][1]
    # id доп. говорящего сохранён, а имя разрешается по актуальному speakers.
    assert entry["extra_speaker_ids"] == ["SPEAKER_00"]
    assert names[entry["extra_speaker_ids"][0]] == "Иван Иванов"
    # Оценка уверенности говорящего переживает пересборку payload.
    assert entry["speaker_confidence"] == pytest.approx(0.3)
    assert entry["low_speaker_confidence"] is True


def test_merge_removes_duplicate_extra_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"merges": [{"source": "SPEAKER_01", "target": "SPEAKER_00"}]},
    )

    assert response.status_code == 200
    body = response.json()
    # SPEAKER_01 был основным в реплике 2 и источником слияния: после
    # переназначения на SPEAKER_00 он не дублируется как доп. говорящий.
    assert body["entries"][1]["speaker_id"] == "SPEAKER_00"
    assert body["entries"][1]["extra_speaker_ids"] == []


def test_patch_rename_updates_result_json_and_sample(
    client: TestClient, web_paths: WebPaths
) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_00": "Иван Иванов"}},
    )

    assert response.status_code == 200
    body = response.json()
    names = {speaker["id"]: speaker["display_name"] for speaker in body["speakers"]}
    assert names["SPEAKER_00"] == "Иван Иванов"
    # Флаги низкой уверенности сохраняются при пересборке.
    assert body["entries"][0]["low_confidence"] is True
    assert body["entries"][1]["low_confidence"] is False

    sample_rel = body["samples"]["SPEAKER_00"]
    assert sample_rel.endswith("Иван Иванов.wav")
    assert (web_paths.data_dir / sample_rel).is_file()
    assert client.get(f"/api/jobs/{job_id}/samples/SPEAKER_00").status_code == 200

    # JSON на диске перезаписан.
    on_disk = json.loads((web_paths.results_dir / f"{job_id}.json").read_text("utf-8"))
    assert on_disk["speakers"][0]["display_name"] == "Иван Иванов"


def test_patch_merge_reassigns_entries_and_samples(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"merges": [{"source": "SPEAKER_01", "target": "SPEAKER_00"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert [speaker["id"] for speaker in body["speakers"]] == ["SPEAKER_00"]
    assert {entry["speaker_id"] for entry in body["entries"]} == {"SPEAKER_00"}
    assert set(body["samples"]) == {"SPEAKER_00"}


def test_patch_ignores_unknown_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_99": "Никто"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert {speaker["id"] for speaker in body["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_patch_without_result_returns_404(client: TestClient) -> None:
    uploaded = _upload(client)
    job_id = client.post("/api/jobs", json={"path": uploaded["name"]}).json()["id"]

    response = client.patch(
        f"/api/jobs/{job_id}/speakers", json={"renames": {"SPEAKER_00": "Икс"}}
    )
    assert response.status_code == 404


def test_apply_names_matches_and_reports(
    client: TestClient, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _ = _prepared_job(client)
    _write_wav(voices_dir / "Иван Иванов.wav")

    def fake_enroll(**kwargs) -> EnrollmentOutcome:
        return EnrollmentOutcome(
            mapping={"SPEAKER_00": "Иван Иванов"},
            best_candidates={"SPEAKER_01": ("Пётр", 0.35)},
            speaker_count=2,
        )

    monkeypatch.setattr("audio_transcriber.web.speakers.enroll_speakers", fake_enroll)

    response = client.post(f"/api/jobs/{job_id}/apply-names", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["matched"] == {"SPEAKER_00": "Иван Иванов"}
    assert body["best_candidates"] == {"SPEAKER_01": {"name": "Пётр", "score": 0.35}}
    assert body["threshold"] == pytest.approx(0.6)
    names = {speaker["id"]: speaker["display_name"] for speaker in body["result"]["speakers"]}
    assert names["SPEAKER_00"] == "Иван Иванов"


def test_apply_names_soft_error_when_model_unavailable(
    client: TestClient, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, original = _prepared_job(client)
    _write_wav(voices_dir / "Кто-то.wav")

    def boom(**kwargs) -> EnrollmentOutcome:
        raise RuntimeError("модель недоступна")

    monkeypatch.setattr("audio_transcriber.web.speakers.enroll_speakers", boom)

    response = client.post(f"/api/jobs/{job_id}/apply-names", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["matched"] == {}
    assert body["error"] and "недоступ" in body["error"]
    assert body["result"]["speakers"] == original["speakers"]


def test_apply_names_without_references_is_soft(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(f"/api/jobs/{job_id}/apply-names", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["matched"] == {}
    assert body["error"]


def test_voices_crud_and_audio(client: TestClient, voices_dir: Path) -> None:
    assert client.get("/api/voices").json() == []

    created = client.post(
        "/api/voices",
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        data={"name": "Анна"},
    )
    assert created.status_code == 201
    entry = created.json()
    assert entry["name"] == "Анна"
    assert entry["filename"] == "Анна.wav"
    assert entry["duration"] > 0

    listing = client.get("/api/voices").json()
    assert [item["name"] for item in listing] == ["Анна"]

    audio = client.get("/api/voices/Анна/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content == (voices_dir / "Анна.wav").read_bytes()

    envelope = client.get("/api/voices/Анна/envelope", params={"columns": 24}).json()
    assert envelope["columns"] == 24
    assert len(envelope["envelope"]) == 24

    deleted = client.delete("/api/voices/Анна")
    assert deleted.status_code == 200
    assert client.get("/api/voices").json() == []
    assert client.delete("/api/voices/Анна").status_code == 404


def test_voices_upload_requires_name(client: TestClient) -> None:
    response = client.post(
        "/api/voices",
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        data={"name": "   "},
    )
    assert response.status_code == 400


def test_speaker_to_library_copies_sample(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/to-library", json={"name": "Иван Клон"}
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Иван Клон"
    assert any(item["name"] == "Иван Клон" for item in client.get("/api/voices").json())

    missing = client.post(
        f"/api/jobs/{job_id}/speakers/UNKNOWN/to-library", json={"name": "X"}
    )
    assert missing.status_code == 404


def test_voices_upload_reports_quality_warnings(client: TestClient) -> None:
    created = client.post(
        "/api/voices",
        files={"file": ("voice.wav", _wav_bytes(seconds=0.2), "audio/wav")},
        data={"name": "Тихий"},
    )

    assert created.status_code == 201
    quality = created.json()["quality"]
    assert quality["too_short"] is True
    assert quality["warnings"]


def test_speaker_to_library_reports_quality(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/to-library", json={"name": "Клон"}
    )

    assert response.status_code == 201
    quality = response.json()["quality"]
    assert "warnings" in quality
    assert "speech_seconds" in quality


def test_voice_delete_cannot_escape_library(
    client: TestClient, voices_dir: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "secret.wav"
    _write_wav(outside)

    response = client.delete("/api/voices/secret")

    assert response.status_code == 404
    assert outside.is_file()
    assert not (voices_dir / "secret.wav").exists()


# --- несколько образцов на человека (#19) ----------------------------------


def test_voices_list_groups_duplicates(client: TestClient, voices_dir: Path) -> None:
    _write_wav(voices_dir / "Иван.wav")
    _write_wav(voices_dir / "Иван (2).wav")
    _write_wav(voices_dir / "Мария.wav")

    groups = client.get("/api/voices").json()

    assert [group["name"] for group in groups] == ["Иван", "Мария"]
    ivan = next(group for group in groups if group["name"] == "Иван")
    assert ivan["count"] == 2
    assert [sample["filename"] for sample in ivan["samples"]] == ["Иван.wav", "Иван (2).wav"]


def test_voices_upload_appends_to_existing_person(client: TestClient) -> None:
    first = client.post(
        "/api/voices",
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        data={"name": "Анна"},
    )
    second = client.post(
        "/api/voices",
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        data={"name": "Анна"},
    )

    assert first.json()["filename"] == "Анна.wav"
    assert second.json()["filename"] == "Анна (2).wav"
    groups = client.get("/api/voices").json()
    assert len(groups) == 1
    assert groups[0]["count"] == 2


def test_voice_delete_sample_by_filename_and_person(client: TestClient, voices_dir: Path) -> None:
    for name in ("Иван.wav", "Иван (2).wav", "Иван (3).wav"):
        _write_wav(voices_dir / name)

    one = client.delete(f"/api/voices/samples/{quote('Иван (2).wav', safe='')}")
    assert one.status_code == 200
    assert one.json()["deleted"] == "Иван (2).wav"
    assert not (voices_dir / "Иван (2).wav").exists()

    remaining = client.get("/api/voices").json()
    assert remaining[0]["count"] == 2

    all_deleted = client.delete("/api/voices/people/Иван")
    assert all_deleted.status_code == 200
    assert all_deleted.json()["count"] == 2
    assert client.get("/api/voices").json() == []
    assert client.delete("/api/voices/people/Иван").status_code == 404


def test_apply_names_uses_all_samples_of_person(
    client: TestClient, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _ = _prepared_job(client)
    _write_wav(voices_dir / "Иван.wav")
    _write_wav(voices_dir / "Иван (2).wav")
    captured: dict[str, object] = {}

    def fake_enroll(**kwargs) -> EnrollmentOutcome:
        captured.update(kwargs)
        return EnrollmentOutcome(mapping={}, best_candidates={}, speaker_count=2)

    monkeypatch.setattr("audio_transcriber.web.speakers.enroll_speakers", fake_enroll)
    response = client.post(f"/api/jobs/{job_id}/apply-names", json={})

    assert response.status_code == 200
    references = captured["references"]
    assert isinstance(references, dict)
    assert len(references["Иван"]) == 2


# --- варианты прослушивания (#25) ------------------------------------------


def test_speaker_variants_fall_back_without_audio(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.get(f"/api/jobs/{job_id}/speakers/SPEAKER_00/variants")

    assert response.status_code == 200
    body = response.json()
    assert body["speaker_id"] == "SPEAKER_00"
    assert [(item["start"], item["end"]) for item in body["variants"]] == [(0.0, 1.0), (2.0, 3.0)]
    assert all(item["duration"] > 0 for item in body["variants"])


def test_speaker_variants_unknown_speaker_is_404(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.get(f"/api/jobs/{job_id}/speakers/SPEAKER_99/variants")

    assert response.status_code == 404


def test_to_library_from_variant_window(client: TestClient, voices_dir: Path) -> None:
    uploaded = _upload(client, "source.wav", _wav_bytes(1.0))
    job_id = _run_job(client, uploaded["name"])

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/to-library",
        json={"name": "Тест", "start": 0.2, "end": 0.6},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Тест"
    assert body["filename"] == "Тест.wav"
    assert body["duration"] == pytest.approx(0.4, abs=0.05)
    assert (voices_dir / "Тест.wav").is_file()


def test_to_library_rejects_invalid_window(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/to-library",
        json={"name": "Тест", "start": 0.6, "end": 0.2},
    )

    assert response.status_code == 400


# --- переназначение окна другому/новому говорящему (#40/#41) ----------------


def test_reassign_window_to_existing_speaker(
    client: TestClient, web_paths: WebPaths
) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_speaker_id"] == "SPEAKER_01"
    assert body["created_speaker"] is None
    assert [change["index"] for change in body["changes"]] == [0]
    assert body["changes"][0]["before_speaker_id"] == "SPEAKER_00"
    assert body["changes"][0]["after_speaker_id"] == "SPEAKER_01"
    assert body["result"]["entries"][0]["speaker_id"] == "SPEAKER_01"
    # Не пересекающаяся с окном реплика не тронута.
    assert body["result"]["entries"][2]["speaker_id"] == "SPEAKER_00"

    on_disk = json.loads((web_paths.results_dir / f"{job_id}.json").read_text("utf-8"))
    assert on_disk["entries"][0]["speaker_id"] == "SPEAKER_01"


def test_reassign_partial_overlap_moves_touching_entries(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.5, "end": 2.5, "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    body = response.json()
    # Реплика 1 уже принадлежит цели (без изменений), 0 и 2 заходят в окно краем.
    assert [change["index"] for change in body["changes"]] == [0, 2]
    assert {entry["speaker_id"] for entry in body["result"]["entries"]} == {"SPEAKER_01"}


def test_reassign_creates_new_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.0, "end": 1.0, "new_name": "Новый"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["created_speaker"] == {"id": "SPEAKER_02", "display_name": "Новый"}
    assert body["target_speaker_id"] == "SPEAKER_02"
    assert body["result"]["entries"][0]["speaker_id"] == "SPEAKER_02"
    names = {speaker["id"]: speaker["display_name"] for speaker in body["result"]["speakers"]}
    assert names["SPEAKER_02"] == "Новый"


def test_reassign_new_name_reuses_existing_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.0, "end": 1.0, "new_name": "Пётр"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["created_speaker"] is None
    assert body["target_speaker_id"] == "SPEAKER_01"
    assert body["result"]["entries"][0]["speaker_id"] == "SPEAKER_01"


def test_reassign_split_marks_co_speaker(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.5, "end": 1.0, "new_name": "Новый", "split": True},
    )

    assert response.status_code == 200
    body = response.json()
    entry = body["result"]["entries"][0]
    # Основной говорящий сохранён, целевой добавлен сов-участником.
    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["extra_speaker_ids"] == ["SPEAKER_02"]
    assert body["changes"][0]["after_extra_ids"] == ["SPEAKER_02"]


def test_reassign_preserves_manual_text_edit(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    edited = client.patch(
        f"/api/jobs/{job_id}/transcript",
        json={"edits": [{"index": 0, "text": "исправлено вручную"}]},
    )
    assert edited.status_code == 200

    response = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_01"},
    )

    assert response.status_code == 200
    entry = response.json()["result"]["entries"][0]
    assert entry["text"] == "исправлено вручную"
    assert entry["edited"] is True
    assert entry["original_text"] == "привет"
    assert entry["low_confidence"] is True


def test_reassign_undo_restores_previous_result(client: TestClient) -> None:
    job_id, original = _prepared_job(client)
    client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_01"},
    )

    undone = client.post(f"/api/jobs/{job_id}/speakers/undo")

    assert undone.status_code == 200
    body = undone.json()
    assert body["entries"] == original["entries"]
    assert body["speakers"] == original["speakers"]
    # Отменять больше нечего.
    assert client.post(f"/api/jobs/{job_id}/speakers/undo").status_code == 404


def test_other_speaker_edit_clears_undo(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign",
        json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_01"},
    )
    client.patch(
        f"/api/jobs/{job_id}/speakers",
        json={"renames": {"SPEAKER_00": "Иван Иванов"}},
    )

    assert client.post(f"/api/jobs/{job_id}/speakers/undo").status_code == 404


def test_reassign_validation_errors(client: TestClient) -> None:
    job_id, _ = _prepared_job(client)
    url = f"/api/jobs/{job_id}/speakers/SPEAKER_00/reassign"

    bad_window = client.post(url, json={"start": 0.6, "end": 0.2, "target_speaker_id": "SPEAKER_01"})
    assert bad_window.status_code == 400

    neither = client.post(url, json={"start": 0.0, "end": 1.0})
    assert neither.status_code == 400

    both = client.post(
        url,
        json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_01", "new_name": "X"},
    )
    assert both.status_code == 400

    unknown_target = client.post(
        url, json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_99"}
    )
    assert unknown_target.status_code == 404

    unknown_source = client.post(
        f"/api/jobs/{job_id}/speakers/SPEAKER_99/reassign",
        json={"start": 0.0, "end": 1.0, "target_speaker_id": "SPEAKER_01"},
    )
    assert unknown_source.status_code == 404

    no_overlap = client.post(
        url, json={"start": 10.0, "end": 11.0, "target_speaker_id": "SPEAKER_01"}
    )
    assert no_overlap.status_code == 400

