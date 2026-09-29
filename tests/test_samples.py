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
    normalize_sample,
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


def _tone_waveform(
    duration: float = 60.0,
    *,
    windows: tuple[tuple[float, float], ...] = (),
    amplitude: float = 0.6,
) -> np.ndarray:
    """Синтетическое аудио: тишина, в заданных окнах — тон 220 Гц."""
    samples = np.zeros(round(duration * SAMPLE_RATE), dtype=np.float32)
    times = np.arange(samples.size, dtype=np.float64) / SAMPLE_RATE
    for start, end in windows:
        first = round(start * SAMPLE_RATE)
        last = round(end * SAMPLE_RATE)
        samples[first:last] = amplitude * np.sin(2 * np.pi * 220.0 * times[first:last])
    return samples


def _loader_for(waveform: np.ndarray):
    """Загрузчик сигнала с фиксированным содержимым (для monkeypatch)."""

    def loader(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return waveform

    return loader


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


def test_select_sample_segment_prefers_high_energy_window() -> None:
    # Длинная реплика 0–20 с, но речь (тон) только с 10 по 18 с — паузы не берём.
    entries = [_entry(0.0, 20.0, IVAN)]
    waveform = _tone_waveform(windows=((10.0, 18.0),))

    segment = select_sample_segment(
        entries, "SPEAKER_00", max_duration=8.0, waveform=waveform
    )

    assert segment == pytest.approx((10.0, 18.0))


def test_select_sample_segment_returns_none_for_silence() -> None:
    entries = [_entry(0.0, 20.0, IVAN)]
    waveform = np.zeros(60 * SAMPLE_RATE, dtype=np.float32)

    assert (
        select_sample_segment(entries, "SPEAKER_00", max_duration=8.0, waveform=waveform)
        is None
    )


def test_select_sample_segment_energy_does_not_cross_foreign_interval() -> None:
    entries = [_entry(0.0, 20.0, IVAN), _entry(12.0, 13.0, MARIA)]
    waveform = _tone_waveform(windows=((5.0, 12.0),))

    segment = select_sample_segment(
        entries, "SPEAKER_00", max_duration=8.0, waveform=waveform
    )

    assert segment is not None
    start, end = segment
    # Окно обрезано до речи (5–12 с) и упирается в начало реплики Марии.
    assert start == pytest.approx(5.0, abs=0.05)
    assert end <= 12.0 + 1e-6


def test_select_sample_segment_trims_to_speech_not_silence() -> None:
    # Тон в середине длинной тишины: образец — почти только речь без пауз.
    entries = [_entry(0.0, 60.0, IVAN)]
    waveform = _tone_waveform(duration=60.0, windows=((25.0, 29.0),))

    segment = select_sample_segment(
        entries, "SPEAKER_00", max_duration=8.0, waveform=waveform
    )

    assert segment == pytest.approx((25.0, 29.0), abs=0.05)


def test_select_sample_segment_pads_tiny_speech_to_minimum() -> None:
    # Очень короткая «речь» дополняется до минимума ~1.5 с вокруг пика.
    entries = [_entry(0.0, 60.0, IVAN)]
    waveform = _tone_waveform(duration=60.0, windows=((30.0, 30.4),))

    segment = select_sample_segment(
        entries, "SPEAKER_00", max_duration=8.0, waveform=waveform
    )

    assert segment is not None
    start, end = segment
    assert end - start >= 1.5 - 1e-6
    assert start <= 30.0 <= end



def test_select_sample_segment_without_waveform_falls_back_to_longest() -> None:
    entries = [_entry(0.0, 2.0, IVAN), _entry(5.0, 12.0, IVAN)]

    assert select_sample_segment(entries, "SPEAKER_00") == (5.0, 12.0)


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


def test_extract_speaker_samples_uses_energetic_window(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waveform = _tone_waveform(windows=((10.0, 18.0),))
    monkeypatch.setattr(
        samples, "load_waveform", _loader_for(waveform)
    )
    result = _result(audio_file, [_entry(0.0, 20.0, IVAN)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    assert set(written) == {"SPEAKER_00"}
    _, _, _, frames = _read_wav(written["SPEAKER_00"])
    # Взято 8 секунд реальной речи (тон), а не пауза.
    assert frames.shape[0] == 8 * SAMPLE_RATE
    assert np.abs(frames).max() > 0.9 * 32767


def test_extract_speaker_samples_trims_pauses_from_sample(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Речь — только 4 секунды в середине: образец почти без тишины (не 8 секунд).
    waveform = _tone_waveform(duration=60.0, windows=((25.0, 29.0),))
    monkeypatch.setattr(samples, "load_waveform", _loader_for(waveform))
    result = _result(audio_file, [_entry(0.0, 60.0, IVAN)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    _, _, _, frames = _read_wav(written["SPEAKER_00"])
    assert 3.8 * SAMPLE_RATE <= frames.shape[0] <= 4.2 * SAMPLE_RATE



def test_extract_speaker_samples_skips_silent_speaker(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waveform = np.zeros(60 * SAMPLE_RATE, dtype=np.float32)
    monkeypatch.setattr(
        samples, "load_waveform", _loader_for(waveform)
    )
    result = _result(audio_file, [_entry(0.0, 20.0, IVAN)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    assert written == {}


def test_extract_speaker_samples_normalizes_peak(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Тихая речь усиливается до целевого пика, но не клиппится.
    waveform = _tone_waveform(windows=((0.0, 20.0),), amplitude=0.1)
    monkeypatch.setattr(
        samples, "load_waveform", _loader_for(waveform)
    )
    result = _result(audio_file, [_entry(0.0, 20.0, IVAN)])

    written = extract_speaker_samples(
        result, audio_path=audio_file, output_dir=tmp_path / "out"
    )

    _, _, _, frames = _read_wav(written["SPEAKER_00"])
    peak = np.abs(frames).max() / 32767
    assert peak == pytest.approx(0.97, abs=0.01)


def test_normalize_sample_scales_quiet_and_loud_without_clipping() -> None:
    quiet = np.full(100, 0.2, dtype=np.float32)
    assert np.abs(normalize_sample(quiet)).max() == pytest.approx(0.97, abs=1e-6)

    loud = np.full(100, 2.0, dtype=np.float32)
    normalized = normalize_sample(loud)
    assert np.abs(normalized).max() <= 1.0
    assert np.abs(normalized).max() == pytest.approx(0.97, abs=1e-6)


def test_normalize_sample_limits_gain_for_near_silence() -> None:
    tiny = np.full(100, 1e-4, dtype=np.float32)

    normalized = normalize_sample(tiny)

    # Усиление ограничено 10x, сигнал не «разгоняется» безмерно.
    assert np.abs(normalized).max() == pytest.approx(1e-3, rel=1e-3)


def test_normalize_sample_keeps_true_silence() -> None:
    silence = np.zeros(100, dtype=np.float32)

    assert np.array_equal(normalize_sample(silence), silence)


def test_find_speaker_samples_missing_dir_is_empty(audio_file: Path, tmp_path: Path) -> None:
    assert find_speaker_samples(_result(audio_file, []), tmp_path / "nope") == {}
