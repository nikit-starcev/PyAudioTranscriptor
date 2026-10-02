"""Согласованность правки говорящих: основной и дополнительные участники.

Проверяется, что переименование и объединение обновляют не только основного
говорящего реплики, но и ``extra_speakers`` (участников наложения), чтобы метка
``speaker_label`` не «застревала» со старым именем.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.editing import (
    merge_speakers,
    next_speaker_id,
    reassign_window,
    rename_speaker,
)
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult


def _result() -> TranscriptionResult:
    anya = Speaker(id="SPEAKER_00", display_name="Аня")
    boris = Speaker(id="SPEAKER_01", display_name="Боря")
    vasya = Speaker(id="SPEAKER_02", display_name="Вася")
    return TranscriptionResult(
        source_path=Path("call.mp3"),
        language="ru",
        duration=2.0,
        entries=[
            TranscriptEntry(
                start=0.0,
                end=1.0,
                text="да",
                speaker=anya,
                overlap=True,
                extra_speakers=[boris],
                speaker_confidence=0.4,
            ),
            TranscriptEntry(
                start=1.0,
                end=2.0,
                text="хором",
                speaker=vasya,
                overlap=True,
                extra_speakers=[anya, boris],
                speaker_confidence=0.9,
            ),
        ],
        speakers=[anya, boris, vasya],
    )


def test_rename_main_speaker_updates_extra_labels() -> None:
    renamed = rename_speaker(_result(), "SPEAKER_00", "Анна")

    assert renamed.entries[0].speaker_label == "Анна + Боря"
    assert renamed.entries[1].speaker_label == "Вася + Анна + Боря"


def test_rename_extra_speaker_updates_extra_labels() -> None:
    renamed = rename_speaker(_result(), "SPEAKER_01", "Борис")

    assert renamed.entries[0].speaker_label == "Аня + Борис"
    assert renamed.entries[1].speaker_label == "Вася + Аня + Борис"


def test_rename_does_not_mutate_original() -> None:
    renamed = rename_speaker(_result(), "SPEAKER_00", "Анна")

    assert renamed.entries[0].speaker_label == "Анна + Боря"
    assert _result().entries[0].speaker_label == "Аня + Боря"


def test_merge_extra_source_into_main_dedupes() -> None:
    merged = merge_speakers(_result(), "SPEAKER_01", "SPEAKER_00")

    # Боря (доп.) слит с Аней (основной): дубликат убирается.
    assert merged.entries[0].speaker_label == "Аня"
    # Трио: основной Вася, доп. [Аня, Боря→Аня] → [Аня].
    assert merged.entries[1].speaker_label == "Вася + Аня"
    assert [speaker.id for speaker in merged.speakers] == ["SPEAKER_00", "SPEAKER_02"]


def test_merge_main_source_into_extra_target() -> None:
    merged = merge_speakers(_result(), "SPEAKER_00", "SPEAKER_01")

    # Основной Аня слит с Борей (был доп. в той же реплике) — без дубля.
    assert merged.entries[0].speaker_label == "Боря"
    assert merged.entries[1].speaker_label == "Вася + Боря"


def test_merge_extra_source_into_other_extra() -> None:
    merged = merge_speakers(_result(), "SPEAKER_01", "SPEAKER_02")

    # Вася — цель и основной в трио; доп. Боря→Вася убирается как дубль.
    assert merged.entries[1].speaker_label == "Вася + Аня"


# --- переназначение окна другому говорящему (#40/#41) -----------------------


def test_next_speaker_id_uses_first_free_after_max() -> None:
    assert next_speaker_id(_result()) == "SPEAKER_03"


def test_next_speaker_id_without_auto_ids() -> None:
    result = TranscriptionResult(
        source_path=Path("call.mp3"),
        language="ru",
        duration=1.0,
        entries=[],
        speakers=[Speaker(id="Аня", display_name="Аня")],
    )
    assert next_speaker_id(result) == "SPEAKER_00"


def test_reassign_window_moves_fully_contained_entry() -> None:
    target = Speaker(id="SPEAKER_02", display_name="Вася")
    updated, changes = reassign_window(_result(), start=0.0, end=1.0, target=target)

    assert updated.entries[0].speaker_label == "Вася + Боря"
    # Вторая реплика не пересекается с окном — не тронута.
    assert updated.entries[1].speaker_label == "Вася + Аня + Боря"
    assert [change.index for change in changes] == [0]
    assert changes[0].before_speaker_id == "SPEAKER_00"
    assert changes[0].after_speaker_id == "SPEAKER_02"


def test_reassign_window_counts_partial_overlap() -> None:
    target = Speaker(id="SPEAKER_07", display_name="Новый")
    # Окно [0.5, 1.5) лишь частично накрывает обе реплики — обе переназначаются.
    updated, changes = reassign_window(_result(), start=0.5, end=1.5, target=target)

    assert updated.entries[0].speaker is not None
    assert updated.entries[0].speaker.id == "SPEAKER_07"
    assert updated.entries[1].speaker is not None
    assert updated.entries[1].speaker.id == "SPEAKER_07"
    assert [change.index for change in changes] == [0, 1]


def test_reassign_window_dedupes_target_from_extra() -> None:
    target = Speaker(id="SPEAKER_01", display_name="Боря")
    # Боря уже доп. участник второй реплики: став основным, он не дублируется.
    updated, _ = reassign_window(_result(), start=1.0, end=2.0, target=target)

    assert updated.entries[1].speaker_label == "Боря + Аня"


def test_reassign_window_adds_new_speaker_to_result() -> None:
    target = Speaker(id="SPEAKER_07", display_name="Новый")
    updated, _ = reassign_window(_result(), start=0.0, end=1.0, target=target)

    assert [speaker.id for speaker in updated.speakers][-1] == "SPEAKER_07"


def test_reassign_window_split_marks_co_speaker_on_partial_overlap() -> None:
    target = Speaker(id="SPEAKER_02", display_name="Вася")
    # Окно [0.5, 1.0) накрывает только хвост первой реплики — при split она
    # становится общей, а основной говорящий сохраняется.
    updated, changes = reassign_window(
        _result(), start=0.5, end=1.0, target=target, split=True
    )

    assert updated.entries[0].speaker_label == "Аня + Боря + Вася"
    assert updated.entries[1].speaker_label == "Вася + Аня + Боря"
    assert [change.index for change in changes] == [0]
    assert changes[0].before_extra_ids == ("SPEAKER_01",)
    assert changes[0].after_extra_ids == ("SPEAKER_01", "SPEAKER_02")


def test_reassign_window_split_moves_fully_contained_entry() -> None:
    target = Speaker(id="SPEAKER_02", display_name="Вася")
    updated, _ = reassign_window(_result(), start=0.0, end=1.0, target=target, split=True)

    # Целиком в окне — переназначается основным даже в режиме split.
    assert updated.entries[0].speaker_label == "Вася + Боря"


def test_reassign_window_without_overlap_returns_no_changes() -> None:
    target = Speaker(id="SPEAKER_02", display_name="Вася")
    updated, changes = reassign_window(_result(), start=5.0, end=6.0, target=target)

    assert changes == []
    assert updated.entries == _result().entries


def test_reassign_window_does_not_mutate_original() -> None:
    target = Speaker(id="SPEAKER_02", display_name="Вася")
    reassign_window(_result(), start=0.0, end=1.0, target=target)

    assert _result().entries[0].speaker_label == "Аня + Боря"

