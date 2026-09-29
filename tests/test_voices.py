"""Тесты библиотеки образцов голоса (``diarization.voices``)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.diarization.voices import (
    collect_voice_library,
    delete_voice_sample,
    merge_references,
    save_speaker_sample,
)


def test_collect_voice_library_uses_file_stem(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    ivan = voices / "Иван.wav"
    maria = voices / "Мария.WAV"
    ivan.write_bytes(b"")
    maria.write_bytes(b"")
    (voices / "readme.txt").write_text("не образец", encoding="utf-8")

    library = collect_voice_library(voices)

    assert library == {"Иван": (ivan,), "Мария": (maria,)}


def test_collect_voice_library_groups_same_name(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    first = voices / "ivan.wav"
    second = voices / "Иван.wav"
    first.write_bytes(b"")
    second.write_bytes(b"")

    library = collect_voice_library(voices)

    assert library == {"ivan": (first,), "Иван": (second,)}


def test_collect_voice_library_missing_dir_is_empty(tmp_path: Path) -> None:
    assert collect_voice_library(tmp_path / "absent") == {}


def test_collect_voice_library_none_is_empty() -> None:
    assert collect_voice_library(None) == {}


def test_merge_references_adds_library_to_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.wav"
    library_file = tmp_path / "library.wav"

    merged = merge_references(
        {"Иван": (explicit,)},
        {"Иван": (library_file,), "Мария": (library_file,)},
    )

    assert merged == {
        "Иван": (explicit, library_file),
        "Мария": (library_file,),
    }


def test_merge_references_deduplicates_identical_paths(tmp_path: Path) -> None:
    same = tmp_path / "voice.wav"

    merged = merge_references({"Иван": (same,)}, {"Иван": (same,)})

    assert merged == {"Иван": (same,)}
    assert len(merged["Иван"]) == 1


def test_save_speaker_sample_copies_and_sanitizes(tmp_path: Path) -> None:
    source = tmp_path / "generated.wav"
    source.write_bytes(b"audio")
    voices = tmp_path / "voices"

    target = save_speaker_sample(source, voices, "Иван/Тест: 1")

    assert target == voices / "Иван_Тест_ 1.wav"
    assert target.read_bytes() == b"audio"


# --- Удаление образцов из библиотеки ---------------------------------------


def test_delete_voice_sample_inside_library(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    sample = voices / "Иван.wav"
    sample.write_bytes(b"audio")

    assert delete_voice_sample(sample, voices) is True
    assert not sample.exists()


def test_delete_voice_sample_outside_library_refused(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")

    assert delete_voice_sample(outside, voices) is False
    assert outside.exists()


def test_delete_voice_sample_rejects_non_wav(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    other = voices / "notes.txt"
    other.write_text("не образец", encoding="utf-8")

    assert delete_voice_sample(other, voices) is False
    assert other.exists()


def test_delete_voice_sample_missing_file_refused(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()

    assert delete_voice_sample(voices / "absent.wav", voices) is False


def test_delete_voice_sample_defaults_to_parent_dir(tmp_path: Path) -> None:
    sample = tmp_path / "Иван.wav"
    sample.write_bytes(b"audio")

    assert delete_voice_sample(sample) is True
    assert not sample.exists()
