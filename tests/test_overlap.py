"""Тесты пометки наложения речи: вычисление зон, модель и интеграция."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio_transcriber.cache.serialization import (
    diarization_from_payload,
    diarization_payload,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.overlap import compute_overlap_regions
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    Speaker,
    SpeakerOverlap,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
    WordTimestamp,
)
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.merging.overlap import (
    apply_overlap_regions,
    mark_overlap_entries,
    trim_artifact_overlaps,
)
from audio_transcriber.pipeline import run_pipeline

# --- вычисление зон ----------------------------------------------------------


def test_compute_overlap_regions_finds_shared_interval() -> None:
    segments = [
        SpeakerSegment(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=2.0, end=5.0, speaker_id="SPEAKER_01"),
    ]

    assert compute_overlap_regions(segments) == [
        SpeakerOverlap(
            start=2.0, end=3.0, speaker_ids=("SPEAKER_00", "SPEAKER_01")
        )
    ]


def test_compute_overlap_regions_empty_without_overlap() -> None:
    segments = [
        SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
    ]

    assert compute_overlap_regions(segments) == []


def test_compute_overlap_regions_three_speakers() -> None:
    segments = [
        SpeakerSegment(start=0.0, end=4.0, speaker_id="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=5.0, speaker_id="SPEAKER_01"),
        SpeakerSegment(start=2.0, end=3.0, speaker_id="SPEAKER_02"),
    ]

    assert compute_overlap_regions(segments) == [
        SpeakerOverlap(
            start=1.0,
            end=4.0,
            speaker_ids=("SPEAKER_00", "SPEAKER_01", "SPEAKER_02"),
        )
    ]


def test_compute_overlap_regions_ignores_degenerate() -> None:
    segments = [SpeakerSegment(start=1.0, end=1.0, speaker_id="SPEAKER_00")]

    assert compute_overlap_regions(segments) == []


# --- пометка реплик ----------------------------------------------------------


def test_mark_overlap_entries_marks_intersecting_entry() -> None:
    entries = [
        TranscriptEntry(start=0.0, end=1.0, text="вне"),
        TranscriptEntry(start=1.0, end=2.0, text="внутри"),
    ]
    regions = [SpeakerOverlap(start=0.5, end=1.5)]

    result = mark_overlap_entries(entries, regions)

    assert [entry.overlap for entry in result] == [True, True]
    assert result[0].text == "вне"


def test_mark_overlap_entries_leaves_non_intersecting() -> None:
    entries = [TranscriptEntry(start=5.0, end=6.0, text="тишина")]

    result = mark_overlap_entries(entries, [SpeakerOverlap(start=0.0, end=1.0)])

    assert result[0].overlap is False


def test_mark_overlap_entries_returns_same_list_without_regions() -> None:
    entries = [TranscriptEntry(start=0.0, end=1.0, text="текст")]

    result = mark_overlap_entries(entries, [])

    assert result is entries


# --- движок pyannote ---------------------------------------------------------


class _FakeTurns:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, *, yield_label: bool = False):
        for start, end, speaker in self._turns:
            yield SimpleNamespace(start=start, end=end), None, speaker


class _FakeDiarizationOutput:
    def __init__(self, *, with_regular: bool) -> None:
        self.exclusive_speaker_diarization = _FakeTurns(
            [(0.0, 2.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01")]
        )
        if with_regular:
            self.speaker_diarization = _FakeTurns(
                [(0.0, 3.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01")]
            )


class _FakePipeline:
    def __init__(self, output: object) -> None:
        self._output = output

    def to(self, *args: object, **kwargs: object) -> _FakePipeline:
        return self

    def __call__(self, audio: object, *, num_speakers=None, hook=None):
        if hook is not None:
            hook("segmentation", None, file={"uri": "test"}, total=1, completed=1)
        return self._output


def _run_diarizer(output: object, monkeypatch: pytest.MonkeyPatch) -> PyannoteSpeakerDiarizer:
    diarizer = PyannoteSpeakerDiarizer(Device.CPU)
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: _FakePipeline(output))
    monkeypatch.setattr(
        "audio_transcriber.diarization.pyannote_engine.load_waveform",
        lambda _path, **_kwargs: np.zeros(16000, dtype=np.float32),
    )
    diarizer.diarize(Path("audio.wav"))
    return diarizer


def test_overlap_regions_from_regular_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    diarizer = _run_diarizer(_FakeDiarizationOutput(with_regular=True), monkeypatch)

    assert diarizer.overlap_regions() == [
        SpeakerOverlap(
            start=2.0, end=3.0, speaker_ids=("SPEAKER_00", "SPEAKER_01")
        )
    ]


def test_overlap_regions_empty_without_regular_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diarizer = _run_diarizer(_FakeDiarizationOutput(with_regular=False), monkeypatch)

    assert diarizer.overlap_regions() == []


def test_overlap_regions_empty_before_diarize() -> None:
    assert PyannoteSpeakerDiarizer(Device.CPU).overlap_regions() == []


# --- интеграция с конвейером -------------------------------------------------


class _FakeRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return ([TranscriptionSegment(start=0.0, end=1.0, text="привет")], "ru", 1.0)


class _OverlapDiarizer:
    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: np.ndarray | None = None,
    ):
        return [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    def overlap_regions(self):
        return [SpeakerOverlap(start=0.5, end=1.5)]


class _SingleEntryMerger:
    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        return [TranscriptEntry(start=0.0, end=2.0, text="спор")], []


def test_pipeline_marks_overlap_entries(audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_OverlapDiarizer(),
        merger=_SingleEntryMerger(),
    )

    assert result.entries[0].overlap is True
    content = (output_dir / "sample.txt").read_text(encoding="utf-8")
    assert "[наложение речи]" in content


def test_pipeline_skips_overlap_when_disabled(audio_file: Path, tmp_path: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        mark_overlap=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_OverlapDiarizer(),
        merger=_SingleEntryMerger(),
    )

    assert result.entries[0].overlap is False


def test_pipeline_degrades_without_overlap_method(audio_file: Path, tmp_path: Path) -> None:
    class _PlainDiarizer:
        def diarize(
            self,
            audio_path: Path,
            *,
            num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
            waveform: np.ndarray | None = None,
        ):
            return [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_FakeRecognizer(),
        diarizer=_PlainDiarizer(),
        merger=_SingleEntryMerger(),
    )

    assert result.entries[0].overlap is False


# --- участники зон наложения (speaker_ids) -----------------------------------


def test_diarization_payload_roundtrip_preserves_speaker_ids() -> None:
    segments = [SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00")]
    overlaps = [
        SpeakerOverlap(
            start=0.2, end=0.8, speaker_ids=("SPEAKER_00", "SPEAKER_01")
        )
    ]

    restored_segments, restored_overlaps = diarization_from_payload(
        diarization_payload(segments, overlaps)
    )

    assert restored_segments == segments
    assert restored_overlaps == overlaps


def test_diarization_from_payload_tolerates_old_overlap_format() -> None:
    # Старый кэш: у зоны нет поля ``speaker_ids`` — разбор не должен падать.
    data = {"segments": [], "overlaps": [{"start": 0.2, "end": 0.8}]}

    _, restored_overlaps = diarization_from_payload(data)

    assert restored_overlaps == [SpeakerOverlap(start=0.2, end=0.8, speaker_ids=())]


def test_apply_overlap_regions_names_extra_speakers() -> None:
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    entries = [TranscriptEntry(start=0.0, end=4.0, text="спор", speaker=main)]
    regions = [
        SpeakerOverlap(
            start=1.0, end=3.0, speaker_ids=("SPEAKER_00", "SPEAKER_01")
        )
    ]

    result, speakers = apply_overlap_regions(
        entries, [main], regions, known_speakers={"SPEAKER_01": "Боря"}
    )

    assert result[0].overlap is True
    assert [speaker.id for speaker in result[0].extra_speakers] == ["SPEAKER_01"]
    assert result[0].speaker_label == "Аня + Боря"
    assert {speaker.id for speaker in speakers} == {"SPEAKER_00", "SPEAKER_01"}


def test_apply_overlap_regions_unions_with_existing_extra_speakers() -> None:
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    sweep_extra = Speaker(id="SPEAKER_09", display_name="Спикер 9")
    entries = [
        TranscriptEntry(
            start=0.0,
            end=4.0,
            text="спор",
            speaker=main,
            extra_speakers=[sweep_extra],
            overlap=True,
        )
    ]
    regions = [
        SpeakerOverlap(
            start=1.0, end=3.0, speaker_ids=("SPEAKER_00", "SPEAKER_01")
        )
    ]

    result, _ = apply_overlap_regions(entries, [main], regions)

    # extras из заметания сохраняются, участник зоны добавляется без дублей.
    assert [speaker.id for speaker in result[0].extra_speakers] == [
        "SPEAKER_09",
        "SPEAKER_01",
    ]


def test_apply_overlap_regions_old_format_marks_without_names() -> None:
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    entries = [TranscriptEntry(start=0.0, end=2.0, text="x", speaker=main)]

    result, _ = apply_overlap_regions(
        entries, [main], [SpeakerOverlap(start=0.5, end=1.5)]
    )

    assert result[0].overlap is True
    assert result[0].extra_speakers == []


def test_apply_overlap_regions_orders_by_accumulated_overlap() -> None:
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    entries = [TranscriptEntry(start=0.0, end=10.0, text="общий", speaker=main)]
    regions = [
        SpeakerOverlap(start=0.0, end=4.0, speaker_ids=("SPEAKER_00", "SPEAKER_01")),
        SpeakerOverlap(start=5.0, end=6.0, speaker_ids=("SPEAKER_00", "SPEAKER_02")),
    ]

    result, _ = apply_overlap_regions(entries, [main], regions)

    # SPEAKER_01 даёт 4 с, SPEAKER_02 — 1 с.
    assert [speaker.id for speaker in result[0].extra_speakers] == [
        "SPEAKER_01",
        "SPEAKER_02",
    ]


def test_apply_overlap_regions_below_threshold_marks_but_no_names() -> None:
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    entries = [TranscriptEntry(start=0.0, end=10.0, text="микро", speaker=main)]
    regions = [
        SpeakerOverlap(start=3.0, end=3.1, speaker_ids=("SPEAKER_00", "SPEAKER_01"))
    ]

    result, _ = apply_overlap_regions(entries, [main], regions)

    assert result[0].overlap is True  # зона пересекает реплику
    assert result[0].extra_speakers == []  # 0.1 с < порога 0.3


def test_apply_overlap_regions_no_regions_is_noop() -> None:
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    entries = [TranscriptEntry(start=0.0, end=2.0, text="x", speaker=main)]

    same_entries, same_speakers = apply_overlap_regions(entries, [main], [])

    assert same_entries is entries
    assert same_speakers == [main]


# --- интеграция: extras из зон наложения (эксклюзивные сегменты) --------------


class _DisjointOverlapDiarizer:
    """Эксклюзивные сегменты не пересекаются, но зона наложения — с участниками."""

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: np.ndarray | None = None,
    ):
        return [
            SpeakerSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
            SpeakerSegment(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
        ]

    def overlap_regions(self):
        return [
            SpeakerOverlap(
                start=0.5, end=1.5, speaker_ids=("SPEAKER_00", "SPEAKER_01")
            )
        ]


class _TwoSecondRecognizer:
    def transcribe(self, audio_path: Path, *, language: str | None = None):
        return ([TranscriptionSegment(start=0.0, end=2.0, text="спор")], "ru", 2.0)


def test_pipeline_extra_speakers_from_overlap_regions(
    audio_file: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT, ExportFormat.JSON),
        denoise=False,
        speaker_names={"SPEAKER_00": "Аня", "SPEAKER_01": "Боря"},
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_TwoSecondRecognizer(),
        diarizer=_DisjointOverlapDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    entry = result.entries[0]
    assert entry.speaker is not None
    assert entry.speaker.id == "SPEAKER_00"  # тай-брейк при равном перекрытии
    assert [speaker.id for speaker in entry.extra_speakers] == ["SPEAKER_01"]
    assert entry.overlap is True
    assert entry.speaker_label == "Аня + Боря"

    content = (output_dir / "sample.txt").read_text(encoding="utf-8")
    assert "Аня + Боря" in content

    payload = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert payload["entries"][0]["extra_speakers"] == ["SPEAKER_01"]


def test_pipeline_old_cache_overlaps_mark_without_names(
    audio_file: Path, tmp_path: Path
) -> None:
    # Старый формат зон (без участников): реплика помечается, extras пусты.
    class _OldFormatDiarizer:
        def diarize(
            self,
            audio_path: Path,
            *,
            num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
            waveform: np.ndarray | None = None,
        ):
            return [SpeakerSegment(start=0.0, end=2.0, speaker_id="SPEAKER_00")]

        def overlap_regions(self):
            return [SpeakerOverlap(start=0.5, end=1.5)]

    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=_TwoSecondRecognizer(),
        diarizer=_OldFormatDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    assert result.entries[0].overlap is True
    assert result.entries[0].extra_speakers == []


# --- артефактные пересечения соседних реплик (#94) ---------------------------


def _speaker(speaker_id: str) -> Speaker:
    return Speaker(id=speaker_id, display_name=speaker_id)


def test_trim_artifact_overlaps_clamps_unflagged_pair() -> None:
    first = TranscriptEntry(
        start=0.0,
        end=2.0,
        text="еще надо посмотреть",
        speaker=_speaker("A"),
        words=[
            WordTimestamp("еще", 1.0, 1.3),
            WordTimestamp("посмотреть", 1.3, 2.0),
        ],
    )
    second = TranscriptEntry(start=1.5, end=3.0, text="давайте смотреть", speaker=_speaker("B"))

    result = trim_artifact_overlaps([first, second])

    assert result[0].end == pytest.approx(1.5)
    assert result[1].end == pytest.approx(3.0)
    # Слово, вылезавшее за новый конец, подрезано.
    assert result[0].words[-1].end == pytest.approx(1.5)


def test_trim_artifact_overlaps_keeps_flagged_overlap() -> None:
    first = TranscriptEntry(start=0.0, end=2.0, text="первый", speaker=_speaker("A"))
    second = TranscriptEntry(
        start=1.5,
        end=3.0,
        text="второй",
        speaker=_speaker("B"),
        overlap=True,
        extra_speakers=[_speaker("A")],
    )

    result = trim_artifact_overlaps([first, second])

    assert result[0].end == pytest.approx(2.0)


def test_trim_artifact_overlaps_keeps_extra_speakers() -> None:
    first = TranscriptEntry(
        start=0.0,
        end=2.0,
        text="первый",
        speaker=_speaker("A"),
        extra_speakers=[_speaker("B")],
    )
    second = TranscriptEntry(start=1.5, end=3.0, text="второй", speaker=_speaker("B"))

    result = trim_artifact_overlaps([first, second])

    assert result[0].end == pytest.approx(2.0)


def test_trim_artifact_overlaps_leaves_non_overlapping() -> None:
    first = TranscriptEntry(start=0.0, end=1.0, text="первый", speaker=_speaker("A"))
    second = TranscriptEntry(start=1.5, end=3.0, text="второй", speaker=_speaker("B"))

    result = trim_artifact_overlaps([first, second])

    assert result[0].end == pytest.approx(1.0)


# --- интервальный индекс зон наложения (#89) ---------------------------------


def _many_regions() -> list[SpeakerOverlap]:
    """Много зон: часть пересекает реплики, часть — «мимо»."""
    regions: list[SpeakerOverlap] = []
    for index in range(200):
        start = index * 10.0
        end = start + 0.4
        # Зоны с чётным индексом несут участников, с нечётным — старый формат.
        if index % 2 == 0:
            regions.append(
                SpeakerOverlap(
                    start=start,
                    end=end,
                    speaker_ids=("SPEAKER_00", f"SPEAKER_{index % 7 + 1:02d}"),
                )
            )
        else:
            regions.append(SpeakerOverlap(start=start, end=end))
    return regions


def _brute_mark(
    entries: list[TranscriptEntry], regions: list[SpeakerOverlap]
) -> list[bool]:
    return [
        entry.overlap
        or any(entry.start < region.end and entry.end > region.start for region in regions)
        for entry in entries
    ]


def test_mark_overlap_entries_index_matches_brute_force() -> None:
    regions = _many_regions()
    entries = [
        TranscriptEntry(start=value, end=value + 0.6, text=str(index))
        for index, value in enumerate(
            [0.0, 9.5, 20.1, 50.0, 100.3, 199.9, 500.0, 1000.0]
        )
    ]

    result = mark_overlap_entries(entries, regions)

    assert [entry.overlap for entry in result] == _brute_mark(entries, regions)


def _brute_apply(
    entries: list[TranscriptEntry],
    speakers: list[Speaker],
    regions: list[SpeakerOverlap],
    *,
    known_speakers: dict[str, str] | None = None,
    overlap_min_seconds: float = 0.3,
) -> tuple[list[list[str]], set[str]]:
    """Эталон прежнего перебора: extras и множество добавленных говорящих."""
    names = known_speakers or {}
    by_id = {speaker.id: speaker for speaker in speakers}
    extras: list[list[str]] = []
    for entry in entries:
        main_id = entry.speaker.id if entry.speaker is not None else None
        durations: dict[str, float] = {}
        for region in regions:
            overlap = min(entry.end, region.end) - max(entry.start, region.start)
            if overlap <= 0.0:
                continue
            if overlap < overlap_min_seconds:
                continue
            for speaker_id in region.speaker_ids:
                if speaker_id == main_id:
                    continue
                durations[speaker_id] = durations.get(speaker_id, 0.0) + overlap
        candidates = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
        extra_ids = [speaker.id for speaker in entry.extra_speakers]
        seen = set(extra_ids)
        for speaker_id, _seconds in candidates:
            if speaker_id in seen:
                continue
            if speaker_id not in by_id:
                by_id[speaker_id] = Speaker(
                    id=speaker_id,
                    display_name=names.get(speaker_id, f"Спикер {len(by_id) + 1}"),
                )
            extra_ids.append(speaker_id)
            seen.add(speaker_id)
        extras.append(extra_ids)
    return extras, set(by_id)


def test_apply_overlap_regions_index_matches_brute_force() -> None:
    regions = _many_regions()
    main = Speaker(id="SPEAKER_00", display_name="Аня")
    entries = [
        TranscriptEntry(start=0.0, end=2.0, text="а", speaker=main),
        TranscriptEntry(start=20.0, end=21.0, text="б", speaker=main),
        TranscriptEntry(start=100.2, end=100.5, text="в", speaker=main),
        TranscriptEntry(start=5000.0, end=5001.0, text="г", speaker=main),
    ]

    result, speakers = apply_overlap_regions(entries, [main], regions)
    expected_extras, expected_speakers = _brute_apply(entries, [main], regions)

    assert [[speaker.id for speaker in entry.extra_speakers] for entry in result] == (
        expected_extras
    )
    assert {speaker.id for speaker in speakers} == expected_speakers
    # «в» короче порога (0.3 с) — помечена, но участников не получает.
    assert result[2].overlap is True
    assert result[2].extra_speakers == []
    assert result[3].overlap is False
