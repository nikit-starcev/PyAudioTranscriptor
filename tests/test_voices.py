"""Тесты библиотеки образцов голоса (``diarization.voices``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization.reference import ReferencePrepareOptions
from audio_transcriber.diarization.voices import (
    base_sample_name,
    collect_voice_library,
    delete_voice_sample,
    delete_voice_samples,
    merge_references,
    sample_index,
    save_reference_sample,
    save_speaker_sample,
    unique_sample_path,
)
from audio_transcriber.utils.audio import load_waveform, write_wav


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


def test_collect_voice_library_groups_duplicates_by_person(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    first = voices / "Иван.wav"
    second = voices / "Иван (2).wav"
    third = voices / "Иван (3).wav"
    maria = voices / "Мария.wav"
    for path in (first, second, third, maria):
        path.write_bytes(b"")

    library = collect_voice_library(voices)

    assert library["Иван"] == (first, second, third)
    assert library["Мария"] == (maria,)


def test_base_sample_name_and_index() -> None:
    assert base_sample_name("Иван") == "Иван"
    assert base_sample_name("Иван (2)") == "Иван"
    assert base_sample_name("Иван (12)") == "Иван"
    assert sample_index("Иван") == 1
    assert sample_index("Иван (2)") == 2
    assert sample_index("Иван (12)") == 12


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


def test_save_speaker_sample_appends_duplicates(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    first_source = tmp_path / "first.wav"
    second_source = tmp_path / "second.wav"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")

    first = save_speaker_sample(first_source, voices, "Иван")
    second = save_speaker_sample(second_source, voices, "Иван")
    third = save_speaker_sample(second_source, voices, "Иван")

    assert first == voices / "Иван.wav"
    assert second == voices / "Иван (2).wav"
    assert third == voices / "Иван (3).wav"
    # Существующие образцы не перезаписываются.
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_unique_sample_path_skips_taken_names(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "Иван.wav").write_bytes(b"")
    (voices / "Иван (2).wav").write_bytes(b"")

    assert unique_sample_path(voices, "Иван") == voices / "Иван (3).wav"
    assert unique_sample_path(voices, "Пётр") == voices / "Пётр.wav"


def test_save_reference_sample_prepares_and_reports_quality(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    waveform = np.zeros(8 * 16000, dtype=np.float32)
    waveform[2 * 16000 : 5 * 16000] = 0.5  # 3 с речи, остальное — тишина
    write_wav(source, waveform)

    target, quality = save_reference_sample(source, tmp_path / "voices", "Иван")

    assert target == tmp_path / "voices" / "Иван.wav"
    prepared = load_waveform(target)
    # Обрезка до речи: длительность заметно меньше исходных 8 с.
    assert prepared.size < waveform.size
    assert quality is not None
    assert quality.speech_seconds == pytest.approx(3.0, abs=0.2)
    assert quality.ok is True


def test_save_reference_sample_passthrough_when_disabled(tmp_path: Path) -> None:
    source = tmp_path / "raw.wav"
    waveform = np.zeros(8 * 16000, dtype=np.float32)
    waveform[2 * 16000 : 5 * 16000] = 0.5
    write_wav(source, waveform)

    target, quality = save_reference_sample(
        source,
        tmp_path / "voices",
        "Иван",
        options=ReferencePrepareOptions(enabled=False),
    )

    assert quality is None
    assert target.read_bytes() == source.read_bytes()


def test_delete_voice_samples_removes_whole_group(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    for name in ("Иван.wav", "Иван (2).wav", "Иван (3).wav", "Мария.wav"):
        (voices / name).write_bytes(b"audio")

    deleted = delete_voice_samples(voices, "Иван")

    assert deleted == 3
    assert not (voices / "Иван.wav").exists()
    assert not (voices / "Иван (2).wav").exists()
    assert (voices / "Мария.wav").is_file()
    # Имя с суффиксом указывает на ту же группу, повторное удаление — не ошибка.
    assert delete_voice_samples(voices, "Иван (2)") == 0


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
