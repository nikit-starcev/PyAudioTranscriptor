"""Тесты поиска дубликатов библиотеки голосов (``diarization.dedup``, #39).

Аудио синтетическое (реальные тоны через ``write_wav``): одинаковые файлы дают
точные дубликаты, тот же тон иной громкости/с шумом — почти одинаковые, разные
тоны — не совпадают. Аудио-отпечаток связывает только образцы одного имени;
разные имена — лишь по speaker-эмбеддингам (issue #116). Speaker-эмбеддинги
подменяются лёгкими фейками.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization.dedup import (
    audio_fingerprint,
    cosine_similarity,
    file_digest,
    find_duplicate_groups,
    find_similar_samples,
)
from audio_transcriber.diarization.voices import collect_voice_library, merge_voice_people
from audio_transcriber.utils.audio import SAMPLE_RATE, write_wav

DURATION = 3.0


def _tone(
    frequency: float,
    *,
    amplitude: float = 0.5,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Синтетический тон (несколько секунд) в float32."""
    times = np.arange(round(DURATION * SAMPLE_RATE)) / SAMPLE_RATE
    samples = amplitude * np.sin(2 * np.pi * frequency * times)
    if noise:
        samples = samples + np.random.default_rng(seed).normal(0.0, noise, size=times.size)
    return samples.astype(np.float32)


def _write(directory: Path, name: str, waveform: np.ndarray) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, waveform)
    return path


class _FakeEmbedder:
    """Эмбеддер-заглушка: низкие частоты — один вектор, высокие — другой."""

    window_seconds = 5.0

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        samples = np.asarray(waveform, dtype=np.float64).reshape(-1)
        spectrum = np.abs(np.fft.rfft(samples))
        dominant = float(np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)[np.argmax(spectrum)])
        return np.array([1.0, 0.0] if dominant < 2000 else [0.0, 1.0], dtype=np.float32)


class _CorrelatedEmbedder:
    """Эмбеддер-заглушка с косинусом 0.9 между низкими и высокими частотами."""

    window_seconds = 5.0

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        samples = np.asarray(waveform, dtype=np.float64).reshape(-1)
        spectrum = np.abs(np.fft.rfft(samples))
        dominant = float(np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)[np.argmax(spectrum)])
        vector = [1.0, 0.0] if dominant < 2000 else [0.9, 0.4358899]
        return np.array(vector, dtype=np.float32)


# --- низкоуровневые примитивы -------------------------------------------------


def test_file_digest_same_content_matches(tmp_path: Path) -> None:
    left = tmp_path / "a.wav"
    right = tmp_path / "b.wav"
    left.write_bytes(b"hello")
    right.write_bytes(b"hello")

    assert file_digest(left) == file_digest(right)


def test_audio_fingerprint_volume_invariant(tmp_path: Path) -> None:
    loud = audio_fingerprint(_tone(220.0))
    quiet = audio_fingerprint(_tone(220.0, amplitude=0.1))

    assert loud is not None and quiet is not None
    assert cosine_similarity(loud, quiet) == pytest.approx(1.0, abs=1e-3)


def test_audio_fingerprint_silence_is_none() -> None:
    assert audio_fingerprint(np.zeros(SAMPLE_RATE, dtype=np.float32)) is None


# --- точные дубликаты ---------------------------------------------------------


def test_exact_duplicates_grouped(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Иван (2).wav", _tone(220.0))
    _write(voices, "Мария.wav", _tone(880.0))

    groups = find_duplicate_groups(voices)

    assert len(groups) == 1
    group = groups[0]
    assert group.kind == "exact"
    assert group.score == pytest.approx(1.0)
    assert {member.filename for member in group.members} == {"Иван.wav", "Иван (2).wav"}
    assert group.names == ("Иван",)
    assert group.keep.filename == "Иван.wav"


# --- почти одинаковые ---------------------------------------------------------


def test_near_duplicates_grouped_by_audio(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Иван (2).wav", _tone(220.0, amplitude=0.3))
    _write(voices, "Мария.wav", _tone(880.0))

    groups = find_duplicate_groups(voices)

    assert len(groups) == 1
    group = groups[0]
    assert group.kind == "audio"
    assert {member.filename for member in group.members} == {"Иван.wav", "Иван (2).wav"}
    # Один человек под одним именем — предлагаем объединение образцов.
    assert group.names == ("Иван",)


def test_cross_name_audio_similarity_not_grouped(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    # Одинаковая форма спектра (отпечаток инвариантен к громкости), но разные
    # имена: без эмбеддингов это разные люди — группировать нельзя (#116).
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Мария.wav", _tone(220.0, amplitude=0.3))

    assert find_duplicate_groups(voices) == []


def test_exact_duplicates_cross_name_grouped(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    source = _write(voices, "Иван.wav", _tone(220.0))
    clone = voices / "Мария.wav"
    clone.write_bytes(source.read_bytes())

    groups = find_duplicate_groups(voices)

    assert len(groups) == 1
    group = groups[0]
    assert group.kind == "exact"
    assert {member.filename for member in group.members} == {"Иван.wav", "Мария.wav"}
    # Точный дубль сводит имена: одинаковые байты — один и тот же файл.
    assert set(group.names) == {"Иван", "Мария"}


def test_near_duplicates_by_embedding(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    # Разные тоны: аудио-отпечаток их не сблизит, а фейковый эмбеддер — да.
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Иван Клон.wav", _tone(880.0))

    groups = find_duplicate_groups(
        voices,
        near_threshold=0.999,
        embedder=_FakeEmbedder(),
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.kind == "embedding"
    assert set(group.names) == {"Иван", "Иван Клон"}


def test_cross_name_embedding_respects_raised_threshold(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Мария.wav", _tone(3000.0))
    embedder = _CorrelatedEmbedder()

    # Косинус 0.9 ниже поднятого порога 0.95 — разные имена не склеиваются.
    assert (
        find_duplicate_groups(voices, near_threshold=0.999, embedder=embedder) == []
    )

    # Явно пониженный порог 0.9 — то же совпадение принимается.
    grouped = find_duplicate_groups(
        voices,
        near_threshold=0.999,
        embedding_threshold=0.9,
        embedder=embedder,
    )
    assert len(grouped) == 1
    assert grouped[0].kind == "embedding"
    assert set(grouped[0].names) == {"Иван", "Мария"}


def test_no_false_positives_for_different_voices(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Мария.wav", _tone(880.0))
    _write(voices, "Пётр.wav", _tone(3000.0))

    assert find_duplicate_groups(voices) == []


def test_empty_and_single_library_have_no_groups(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    assert find_duplicate_groups(voices) == []
    _write(voices, "Иван.wav", _tone(220.0))
    assert find_duplicate_groups(voices) == []


# --- предупреждение при добавлении --------------------------------------------


def test_find_similar_samples_reports_match(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Мария.wav", _tone(880.0))
    fresh = _write(tmp_path, "Иван (2).wav", _tone(220.0, amplitude=0.25))

    similar = find_similar_samples(fresh, voices)

    assert [item.filename for item in similar] == ["Иван.wav"]
    assert similar[0].kind == "audio"
    assert similar[0].score >= 0.9


def test_find_similar_samples_cross_name_audio_ignored(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    # Похожий тембр, но другое имя: без эмбеддингов предупреждения нет (#116).
    fresh = _write(tmp_path, "Мария.wav", _tone(220.0, amplitude=0.3))

    assert find_similar_samples(fresh, voices) == []


def test_find_similar_samples_reports_exact_cross_name(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    source = _write(voices, "Иван.wav", _tone(220.0))
    fresh = tmp_path / "Мария.wav"
    fresh.write_bytes(source.read_bytes())

    similar = find_similar_samples(fresh, voices)

    assert [item.filename for item in similar] == ["Иван.wav"]
    assert similar[0].kind == "exact"
    assert similar[0].score == pytest.approx(1.0)


def test_find_similar_samples_ignores_self(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    path = _write(voices, "Иван.wav", _tone(220.0))

    assert find_similar_samples(path, voices) == []


# --- объединение людей --------------------------------------------------------


def test_merge_voice_people_moves_all_samples(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))
    _write(voices, "Иван (2).wav", _tone(240.0))
    _write(voices, "Мария.wav", _tone(880.0))

    moved = merge_voice_people(voices, "Иван", "Мария")

    assert len(moved) == 2
    library = collect_voice_library(voices)
    assert set(library) == {"Мария"}
    assert len(library["Мария"]) == 3
    assert (voices / "Мария.wav").is_file()


def test_merge_voice_people_same_or_absent_is_noop(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    _write(voices, "Иван.wav", _tone(220.0))

    assert merge_voice_people(voices, "Иван", "Иван") == []
    assert merge_voice_people(voices, "Пётр", "Иван") == []
    assert (voices / "Иван.wav").is_file()
