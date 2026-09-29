"""Тесты авто-извлечения образцов голоса говорящих (``diarization.samples``).

Реальное аудио не требуется: декодирование подменяется (``load_waveform``),
а запись идёт настоящим ``write_wav`` — записанный WAV читается обратно через
стандартный ``wave``, чтобы проверить формат (16 кГц, моно).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import samples
from audio_transcriber.diarization.samples import (
    extract_speaker_samples,
    find_speaker_samples,
    samples_directory,
    select_sample_segment,
    slice_waveform,
)
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.utils.audio import SAMPLE_RATE

IVAN = Speaker(id="SPEAKER_00", display_name="Иван")
MARIA = Speaker(id="SPEAKER_01", display_name="Мария")


def _entry(start: float, end: float, speaker: Speaker | None, *, overlap: bool = False) -> TranscriptEntry:
    return TranscriptEntry(
        start=start, end=end, text="речь", speaker=speaker, overlap=overlap
    )


def _result(audio_file: Path, entries: list[TranscriptEntry]) -> TranscriptionResult:
    return TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=60.0,
        entries=entries,
        speakers=[IVAN, MARIA],
    )


# --- выбор сегмента --------------------------------------------------------


def test_select_sample_segment_prefers_longest_clean() -> None:
    entries = [_entry(0.0, 2.0, IVAN), _entry(5.0, 12.0, IVAN), _entry(20.0, 21.0, IVAN)]

    assert select_sample_segment(entries, "SPEAKER_00") == (5.0, 12.0)


def test_select_sample_segment_skips_overlap() -> None:
    entries = [
        _entry(0.0, 2.0, IVAN),
        _entry(5.0, 12.0, IVAN, overlap=True),
    ]

    assert select_sample_segment(entries, "SPEAKER_00") == (0.0, 2.0)


def test_select_sample_segment_truncates_to_max_duration() -> None:
    entries = [_entry(0.0, 20.0, IVAN)]

    assert select_sample_segment(entries, "SPEAKER_00", max_duration=8.0) == (0.0, 8.0)


def test_select_sample_segment_does_not_cross_next_entry() -> None:
    # Внутри длинной реплики начинается другая (наложение) — не залезаем в неё.
    entries = [_entry(0.0, 12.0, IVAN), _entry(6.0, 7.0, MARIA)]

    assert select_sample_segment(entries, "SPEAKER_00") == (0.0, 6.0)


def test_select_sample_segment_returns_none_without_clean_speech() -> None:
    entries = [_entry(0.0, 5.0, IVAN, overlap=True), _entry(6.0, 8.0, MARIA)]

    assert select_sample_segment(entries, "SPEAKER_00") is None


def test_select_sample_segment_ignores_speaker_without_entries() -> None:
    assert select_sample_segment([_entry(0.0, 5.0, IVAN)], "SPEAKER_01") is None


def test_slice_waveform_clamps_to_bounds() -> None:
    waveform = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)

    assert slice_waveform(waveform, 1.0, 2.0).shape[0] == SAMPLE_RATE
    # Хвост за пределами аудио обрезается.
    assert slice_waveform(waveform, 9.0, 99.0).shape[0] == SAMPLE_RATE
    assert slice_waveform(waveform, 5.0, 5.0).size == 0


# --- запись образцов -------------------------------------------------------


@pytest.fixture
def fake_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет декодирование: 60 секунд моно сигнала на 16 кГц."""

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return np.linspace(-0.5, 0.5, 60 * sample_rate, dtype=np.float32)

    monkeypatch.setattr(samples, "load_waveform", loader)


def _read_wav(path: Path) -> tuple[int, int, int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    return channels, rate, width, np.frombuffer(frames, dtype=np.int16)


def test_extract_speaker_samples_writes_one_wav_per_speaker(
    audio_file: Path, tmp_path: Path, fake_audio
) -> None:
    result = _result(
        audio_file,
        [_entry(0.0, 10.0, IVAN), _entry(12.0, 20.0, MARIA)],
    )
    output_dir = tmp_path / "out"

    written = extract_speaker_samples(result, audio_path=audio_file, output_dir=output_dir)

    assert set(written) == {"SPEAKER_00", "SPEAKER_01"}
    assert written["SPEAKER_00"] == output_dir / "sample.speakers" / "Иван.wav"
    assert written["SPEAKER_01"] == output_dir / "sample.speakers" / "Мария.wav"

    channels, rate, width, frames = _read_wav(written["SPEAKER_00"])
    assert (channels, rate, width) == (1, SAMPLE_RATE, 2)
    # 8 секунд при 16 кГц.
    assert frames.shape[0] == 8 * SAMPLE_RATE
    assert np.any(frames != 0)


def test_extract_speaker_samples_uses_shorter_clean_segment(
    audio_file: Path, tmp_path: Path, fake_audio
) -> None:
    result = _result(audio_file, [_entry(0.0, 1.5, IVAN)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    assert "SPEAKER_00" in written
    _, _, _, frames = _read_wav(written["SPEAKER_00"])
    assert frames.shape[0] == int(1.5 * SAMPLE_RATE)


def test_extract_speaker_samples_skips_speaker_without_speech(
    audio_file: Path, tmp_path: Path, fake_audio
) -> None:
    result = _result(audio_file, [_entry(0.0, 5.0, IVAN, overlap=True)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    assert written == {}


def test_extract_speaker_samples_sanitizes_filename(
    audio_file: Path, tmp_path: Path, fake_audio
) -> None:
    bad = Speaker(id="SPEAKER_09", display_name="Иван/Тест: 1")
    result = TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=10.0,
        entries=[_entry(0.0, 5.0, bad)],
        speakers=[bad],
    )

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    assert written["SPEAKER_09"].name == "Иван_Тест_ 1.wav"


def test_extract_speaker_samples_degrades_when_audio_unreadable(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        raise RuntimeError("битый аудиофайл")

    monkeypatch.setattr(samples, "load_waveform", broken_loader)
    result = _result(audio_file, [_entry(0.0, 5.0, IVAN)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    assert written == {}


def test_samples_directory_uses_source_stem(audio_file: Path, tmp_path: Path) -> None:
    assert samples_directory(tmp_path, audio_file) == tmp_path / "sample.speakers"


def test_find_speaker_samples_matches_sanitized_names(
    audio_file: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "sample.speakers"
    directory.mkdir()
    ivan = directory / "Иван.wav"
    ivan.write_bytes(b"")
    (directory / "лишний.wav").write_bytes(b"")
    result = _result(audio_file, [])

    found = find_speaker_samples(result, directory)

    assert found == {"SPEAKER_00": ivan}


def test_find_speaker_samples_missing_dir_is_empty(audio_file: Path, tmp_path: Path) -> None:
    assert find_speaker_samples(_result(audio_file, []), tmp_path / "nope") == {}
