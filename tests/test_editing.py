"""Согласованность правки говорящих: основной и дополнительные участники.

Проверяется, что переименование и объединение обновляют не только основного
говорящего реплики, но и ``extra_speakers`` (участников наложения), чтобы метка
``speaker_label`` не «застревала» со старым именем.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.editing import merge_speakers, rename_speaker
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
