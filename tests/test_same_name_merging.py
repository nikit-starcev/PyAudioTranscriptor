"""Автослияние кластеров диаризации с одинаковым итоговым именем (#13).

Проверяем: кластеры с одним уверенным именем сворачиваются в одного
говорящего (реплики и сов-говорящие наложения переназначаются), безымянные
«Спикер N» и разные имена не сливаются, слияние работает в конвейере и в
веб-слое (переименование/применение имён), образцы голоса не «висят».
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    Speaker,
    TranscriptEntry,
    TranscriptionResult,
    TranscriptionSegment,
)
from audio_transcriber.merging.same_name import (
    merge_result_same_name_speakers,
    merge_same_name_speakers,
    same_name_merge_pairs,
)
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.web.speakers import apply_speaker_changes


def _speaker(speaker_id: str, name: str) -> Speaker:
    return Speaker(id=speaker_id, display_name=name)


# --- чистое слияние списков ------------------------------------------------


def test_three_clusters_same_name_collapse_to_one() -> None:
    artem = _speaker("SPEAKER_00", "Артем Ванян")
    artem_2 = _speaker("SPEAKER_08", "Артем Ванян")
    artem_3 = _speaker("SPEAKER_10", "Артем Ванян")
    miroslava = _speaker("SPEAKER_01", "Мирослава Колупаева")
    miroslava_2 = _speaker("SPEAKER_05", "Мирослава Колупаева")
    entries = [
        TranscriptEntry(0.0, 1.0, "раз", artem, extra_speakers=[artem_2]),
        TranscriptEntry(1.0, 2.0, "два", artem_3),
        TranscriptEntry(2.0, 3.0, "три", miroslava, extra_speakers=[miroslava_2]),
    ]

    merged_entries, merged_speakers = merge_same_name_speakers(entries, [
        artem, miroslava, artem_2, miroslava_2, artem_3
    ])

    # Дубликаты убраны, остались по одному говорящему на имя.
    assert [speaker.id for speaker in merged_speakers] == ["SPEAKER_00", "SPEAKER_01"]
    assert [speaker.display_name for speaker in merged_speakers] == [
        "Артем Ванян",
        "Мирослава Колупаева",
    ]
    # Реплики переназначены на выживший id; сов-говорящий «сам с собой» убран.
    assert merged_entries[0].speaker is not None
    assert merged_entries[0].speaker.id == "SPEAKER_00"
    assert merged_entries[0].extra_speakers == []
    assert merged_entries[1].speaker is not None
    assert merged_entries[1].speaker.id == "SPEAKER_00"
    assert merged_entries[2].speaker is not None
    assert merged_entries[2].speaker.id == "SPEAKER_01"
    assert merged_entries[2].extra_speakers == []


def test_same_name_merge_pairs_are_deterministic_first_wins() -> None:
    speakers = [
        _speaker("SPEAKER_03", "Иван"),
        _speaker("SPEAKER_07", "Иван"),
        _speaker("SPEAKER_09", "Иван"),
    ]

    assert same_name_merge_pairs(speakers) == [
        ("SPEAKER_07", "SPEAKER_03"),
        ("SPEAKER_09", "SPEAKER_03"),
    ]


def test_merge_result_wrapper_does_not_mutate_source() -> None:
    first = _speaker("SPEAKER_00", "Иван")
    second = _speaker("SPEAKER_01", "Иван")
    result = TranscriptionResult(
        source_path=Path("/tmp/a.wav"),
        language="ru",
        duration=2.0,
        entries=[
            TranscriptEntry(0.0, 1.0, "а", first),
            TranscriptEntry(1.0, 2.0, "б", second),
        ],
        speakers=[first, second],
    )

    merged = merge_result_same_name_speakers(result)

    assert [speaker.id for speaker in merged.speakers] == ["SPEAKER_00"]
    assert merged.entries[1].speaker is not None
    assert merged.entries[1].speaker.id == "SPEAKER_00"
    # Исходный результат не изменяется (функция чистая).
    assert len(result.speakers) == 2


def test_unnamed_speakers_are_not_merged() -> None:
    first = _speaker("SPEAKER_00", "Спикер 1")
    second = _speaker("SPEAKER_01", "Спикер 1")
    auto_id = _speaker("SPEAKER_02", "SPEAKER_03")
    empty = _speaker("SPEAKER_03", "   ")

    assert same_name_merge_pairs([first, second, auto_id, empty]) == []
    entries = [TranscriptEntry(0.0, 1.0, "а", first), TranscriptEntry(1.0, 2.0, "б", second)]
    merged_entries, merged_speakers = merge_same_name_speakers(entries, [first, second])
    assert merged_entries == entries
    assert merged_speakers == [first, second]


def test_different_and_partial_names_are_not_merged() -> None:
    artem = _speaker("SPEAKER_00", "Артем")
    artem_full = _speaker("SPEAKER_01", "Артем Ванян")
    case_mismatch = _speaker("SPEAKER_02", "артем")

    assert same_name_merge_pairs([artem, artem_full, case_mismatch]) == []


# --- интеграция в конвейер -------------------------------------------------


class _TwoSameNameMerger:
    """Фиктивный объединитель: два кластера с одинаковым именем."""

    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        first = _speaker("SPEAKER_00", "Иван")
        second = _speaker("SPEAKER_01", "Иван")
        entries = [
            TranscriptEntry(0.0, 1.0, "раз", first),
            TranscriptEntry(1.0, 2.0, "два", second),
        ]
        return entries, [first, second]


class _FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return ([TranscriptionSegment(0.0, 1.0, "привет")], "ru", 1.0)


class _NoDiarizer:
    def diarize(self, *args, **kwargs):
        return []


def _config(audio_file: Path, tmp_path: Path, **kwargs: object) -> AppConfig:
    return AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        diarization_enabled=False,
        export_speaker_samples=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_pipeline_merges_same_name_speakers(audio_file: Path, tmp_path: Path) -> None:
    result = run_pipeline(
        _config(audio_file, tmp_path),
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_NoDiarizer(),
        merger=_TwoSameNameMerger(),
    )

    assert len(result.speakers) == 1
    assert result.speakers[0].display_name == "Иван"
    assert {entry.speaker.id for entry in result.entries if entry.speaker} == {"SPEAKER_00"}


def test_pipeline_can_disable_same_name_merging(audio_file: Path, tmp_path: Path) -> None:
    result = run_pipeline(
        _config(audio_file, tmp_path, merge_same_name_speakers=False),
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_NoDiarizer(),
        merger=_TwoSameNameMerger(),
    )

    assert len(result.speakers) == 2


# --- веб-слой: переименование двух кластеров в одно имя --------------------


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"wav")


def test_web_rename_to_same_name_merges_and_syncs_samples(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_file(data_dir / "samples" / "a.wav")
    _write_file(data_dir / "samples" / "b.wav")
    payload = {
        "language": "ru",
        "duration": 2.0,
        "speakers": [
            {"id": "SPEAKER_00", "display_name": "SPEAKER_00", "has_sample": True},
            {"id": "SPEAKER_05", "display_name": "SPEAKER_05", "has_sample": True},
        ],
        "samples": {"SPEAKER_00": "samples/a.wav", "SPEAKER_05": "samples/b.wav"},
        "entries": [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker_id": "SPEAKER_00",
                "extra_speaker_ids": [],
                "text": "раз",
                "edited": False,
                "original_text": None,
            },
            {
                "start": 1.0,
                "end": 2.0,
                "speaker_id": "SPEAKER_05",
                "extra_speaker_ids": [],
                "text": "два",
                "edited": True,
                "original_text": "два!",
            },
        ],
    }

    updated = apply_speaker_changes(
        payload,
        source_path=tmp_path / "audio.wav",
        renames={"SPEAKER_00": "Иван", "SPEAKER_05": "Иван"},
        merges=(),
        samples=payload["samples"],  # type: ignore[arg-type]
        data_dir=data_dir,
    )

    assert [speaker["id"] for speaker in updated["speakers"]] == ["SPEAKER_00"]  # type: ignore[index]
    assert [speaker["display_name"] for speaker in updated["speakers"]] == ["Иван"]  # type: ignore[index]
    assert [entry["speaker_id"] for entry in updated["entries"]] == [  # type: ignore[index]
        "SPEAKER_00",
        "SPEAKER_00",
    ]
    # Образец лишнего кластера не остаётся «висячим».
    assert updated["samples"] == {"SPEAKER_00": "samples/Иван.wav"}  # type: ignore[index]
    # Ручная правка текста переживает слияние.
    assert updated["entries"][1]["edited"] is True  # type: ignore[index]
    assert updated["entries"][1]["original_text"] == "два!"  # type: ignore[index]


def test_web_merge_keeps_sample_shared_by_source_and_target(tmp_path: Path) -> None:
    """Если источник и цель ссылаются на один файл образца — его не удаляем."""
    data_dir = tmp_path / "data"
    shared = data_dir / "samples" / "shared.wav"
    _write_file(shared)
    payload = {
        "language": "ru",
        "duration": 2.0,
        "speakers": [
            {"id": "SPEAKER_00", "display_name": "Иван", "has_sample": True},
            {"id": "SPEAKER_05", "display_name": "Иван", "has_sample": True},
        ],
        "samples": {"SPEAKER_00": "samples/shared.wav", "SPEAKER_05": "samples/shared.wav"},
        "entries": [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "а"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_05", "text": "б"},
        ],
    }

    updated = apply_speaker_changes(
        payload,
        source_path=tmp_path / "audio.wav",
        renames={},
        merges=(("SPEAKER_05", "SPEAKER_00"),),
        samples=payload["samples"],  # type: ignore[arg-type]
        data_dir=data_dir,
    )

    assert updated["samples"] == {"SPEAKER_00": "samples/shared.wav"}  # type: ignore[index]
    assert shared.is_file()


@pytest.mark.parametrize("name", ["Спикер 1", "SPEAKER_05", ""])
def test_web_unnamed_renames_are_not_merged(tmp_path: Path, name: str) -> None:
    data_dir = tmp_path / "data"
    payload = {
        "speakers": [
            {"id": "SPEAKER_00", "display_name": "SPEAKER_00"},
            {"id": "SPEAKER_05", "display_name": "SPEAKER_05"},
        ],
        "entries": [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "а"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_05", "text": "б"},
        ],
    }

    updated = apply_speaker_changes(
        payload,
        source_path=tmp_path / "audio.wav",
        renames={"SPEAKER_00": name, "SPEAKER_05": name},
        merges=(),
        samples={},
        data_dir=data_dir,
    )

    # Обе метки — заглушки/пустые, слияния нет (переименования сохранены как есть).
    assert len(updated["speakers"]) == 2  # type: ignore[arg-type]
