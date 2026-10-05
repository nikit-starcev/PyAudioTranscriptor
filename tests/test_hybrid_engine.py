"""Тесты гибридной диаризации (#64, часть 2).

Реальный ``nemo-speech`` и модели не запускаются: подменяются ``diarize_audio``
и эмбеддер. Проверяются нарезка окон (границы/перекрытие), глобальная
кластеризация, склейка без дублей/пропусков, перекрытия, прогресс и мягкая
деградация.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import hybrid_engine
from audio_transcriber.diarization.hybrid_engine import (
    AnalysisWindow,
    HybridDiarizationError,
    HybridSpeakerDiarizer,
    window_is_overloaded,
)
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.utils.audio import SAMPLE_RATE

SR = SAMPLE_RATE

_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_C = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_D = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _overloaded_segments(count: int = 12) -> list[SpeakerSegment]:
    """Окно с занятыми всеми 4 головами и почти каждой сменой говорящего."""
    return [
        SpeakerSegment(start=i * 0.5, end=(i + 1) * 0.5, speaker_id=f"spk{i % 4}")
        for i in range(count)
    ]


class _SequenceEmbedder:
    """Возвращает заранее заданные векторы по порядку вызовов."""

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self._vectors = list(vectors)
        self._index = 0

    def embed(self, _waveform: np.ndarray) -> np.ndarray:
        vector = self._vectors[self._index]
        self._index += 1
        return vector


def _patch_windows(
    monkeypatch: pytest.MonkeyPatch, windows: list[AnalysisWindow]
) -> None:
    monkeypatch.setattr(hybrid_engine, "plan_windows", lambda *_a, **_k: windows)


def _patch_diarize(
    monkeypatch: pytest.MonkeyPatch, results: list[list[SpeakerSegment] | None]
) -> None:
    calls = {"count": 0}

    def fake_diarize(_audio_path: Path, **_kwargs: object) -> list[SpeakerSegment] | None:
        index = calls["count"]
        calls["count"] += 1
        return results[index] if index < len(results) else None

    monkeypatch.setattr(hybrid_engine, "diarize_audio", fake_diarize)


def _diarizer(
    *,
    embedder: object,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: object,
) -> HybridSpeakerDiarizer:
    monkeypatch.setattr(hybrid_engine, "binary_available", lambda _binary: True)
    return HybridSpeakerDiarizer(embedder=embedder, **kwargs)


# --- нарезка окон ------------------------------------------------------------


def test_plan_windows_covers_recording_without_gaps_or_duplicates() -> None:
    total = 300 * SR

    windows = hybrid_engine.plan_windows(
        total, window_seconds=90.0, overlap_seconds=2.0
    )

    assert len(windows) > 1
    assert windows[0].own_start == 0
    assert windows[-1].own_end == total
    for previous, current in pairwise(windows):
        # Зоны владения стыкуются встык — ни пропусков, ни дублей.
        assert current.own_start == previous.own_end
        # Соседние окна перекрываются ровно на заданную величину.
        assert previous.end - current.start == 2 * SR


def test_plan_windows_short_audio_is_single_window() -> None:
    total = 30 * SR

    windows = hybrid_engine.plan_windows(
        total, window_seconds=90.0, overlap_seconds=2.0
    )

    assert len(windows) == 1
    assert (windows[0].start, windows[0].end) == (0, total)
    assert (windows[0].own_start, windows[0].own_end) == (0, total)


def test_plan_windows_empty_audio() -> None:
    assert hybrid_engine.plan_windows(0, window_seconds=90.0, overlap_seconds=2.0) == []


def test_plan_windows_snaps_boundary_to_pause() -> None:
    total = 200 * SR
    samples = np.ones(total, dtype=np.float32)
    samples[int(100 * SR) : int(104 * SR)] = 0.0  # «пауза» вокруг границы

    windows = hybrid_engine.plan_windows(
        total, window_seconds=100.0, overlap_seconds=0.0, samples=samples
    )

    second = windows[1]
    # Граница сдвинулась к паузе, но осталась рядом с исходной (в пределах ±3 с).
    assert abs(second.start - 100 * SR) <= 3 * SR
    assert second.start >= windows[0].own_end
    assert windows[-1].own_end == total


# --- диаризация, кластеризация, склейка --------------------------------------


def test_hybrid_recovers_more_than_four_speakers(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(200 * SR, 0.1, dtype=np.float32)
    _patch_windows(
        monkeypatch,
        [
            AnalysisWindow(0, 100 * SR, 0, 100 * SR),
            AnalysisWindow(100 * SR, 200 * SR, 100 * SR, 200 * SR),
        ],
    )
    _patch_diarize(
        monkeypatch,
        [
            [SpeakerSegment(0.0, 10.0, "local_0"), SpeakerSegment(10.0, 20.0, "local_1")],
            [SpeakerSegment(0.0, 10.0, "local_0"), SpeakerSegment(10.0, 20.0, "local_1")],
        ],
    )
    diarizer = _diarizer(
        embedder=_SequenceEmbedder([_A, _B, _C, _A]), monkeypatch=monkeypatch
    )

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples)

    # A (окна 0 и 1), B (окно 0), C (окно 1) — три глобальных говорящих.
    assert {segment.speaker_id for segment in segments} == {
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02",
    }
    assert [(segment.start, segment.speaker_id) for segment in segments] == [
        (0.0, "SPEAKER_00"),
        (10.0, "SPEAKER_01"),
        (100.0, "SPEAKER_02"),
        (110.0, "SPEAKER_00"),
    ]


def test_hybrid_soft_estimate_does_not_force_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.full(150 * SR, 0.5, dtype=np.float32)
    _patch_windows(monkeypatch, [AnalysisWindow(0, 150 * SR, 0, 150 * SR)])
    _patch_diarize(
        monkeypatch,
        [[SpeakerSegment(0.0, 5.0, "a"), SpeakerSegment(5.0, 10.0, "b")]],
    )
    diarizer = _diarizer(
        embedder=_SequenceEmbedder([_A, _B]),
        monkeypatch=monkeypatch,
        expected_speakers=1,
    )

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples)

    # Оценка N=1 — мягкий ориентир и НЕ форсирует один кластер: кластеризация
    # идёт по порогу, поэтому два ортогональных эмбеддинга дают двух говорящих.
    assert {segment.speaker_id for segment in segments} == {"SPEAKER_00", "SPEAKER_01"}


def test_hybrid_explicit_num_speakers_forces_exact_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.full(150 * SR, 0.5, dtype=np.float32)
    _patch_windows(monkeypatch, [AnalysisWindow(0, 150 * SR, 0, 150 * SR)])
    _patch_diarize(
        monkeypatch,
        [[SpeakerSegment(0.0, 5.0, "a"), SpeakerSegment(5.0, 10.0, "b")]],
    )
    diarizer = _diarizer(embedder=_SequenceEmbedder([_A, _B]), monkeypatch=monkeypatch)

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples, num_speakers=1)

    # Явное num_speakers=1 кластеризует ровно в один кластер.
    assert {segment.speaker_id for segment in segments} == {"SPEAKER_00"}


def test_hybrid_soft_estimate_bounds_runaway_oversegmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.full(150 * SR, 0.5, dtype=np.float32)
    _patch_windows(monkeypatch, [AnalysisWindow(0, 150 * SR, 0, 150 * SR)])
    _patch_diarize(
        monkeypatch,
        [
            [
                SpeakerSegment(i * 2.0, i * 2.0 + 2.0, f"s{i}")
                for i in range(10)
            ]
        ],
    )
    vectors = np.eye(10, dtype=np.float32)
    diarizer = _diarizer(
        embedder=_SequenceEmbedder([vectors[i] for i in range(10)]),
        monkeypatch=monkeypatch,
        expected_speakers=1,
    )

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples)

    # Десять ортогональных эмбеддингов по порогу дали бы 10 кластеров; мягкая
    # оценка 1 ограничивает сверху запасом headroom (4) → не больше 5.
    assert len({segment.speaker_id for segment in segments}) <= 5


def test_hybrid_window_overlap_produces_no_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.full(150 * SR, 0.5, dtype=np.float32)
    _patch_windows(
        monkeypatch,
        [
            AnalysisWindow(0, 100 * SR, 0, 70 * SR),
            AnalysisWindow(40 * SR, 140 * SR, 70 * SR, 140 * SR),
        ],
    )
    # Физический сегмент [60, 80] виден в обоих окнах со своим смещением.
    _patch_diarize(
        monkeypatch,
        [
            [SpeakerSegment(60.0, 80.0, "A")],
            [SpeakerSegment(20.0, 40.0, "A")],
        ],
    )
    diarizer = _diarizer(embedder=_SequenceEmbedder([_A, _A]), monkeypatch=monkeypatch)

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples)

    # Перекрытие окон вырезано зонами владения, затем сегменты склеены в один.
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (60.0, 80.0)
    assert segments[0].speaker_id == "SPEAKER_00"


def test_hybrid_reports_overlap_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(50 * SR, 0.2, dtype=np.float32)
    _patch_windows(monkeypatch, [AnalysisWindow(0, 50 * SR, 0, 50 * SR)])
    _patch_diarize(
        monkeypatch,
        [[SpeakerSegment(0.0, 10.0, "a"), SpeakerSegment(5.0, 15.0, "b")]],
    )
    diarizer = _diarizer(embedder=_SequenceEmbedder([_A, _B]), monkeypatch=monkeypatch)

    diarizer.diarize(Path("audio.wav"), waveform=samples)

    overlaps = diarizer.overlap_regions()
    assert overlaps
    assert overlaps[0].start == pytest.approx(5.0)
    assert overlaps[0].end == pytest.approx(10.0)
    assert overlaps[0].speaker_ids == ("SPEAKER_00", "SPEAKER_01")


# --- прогресс и деградация ---------------------------------------------------


def test_hybrid_emits_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(200 * SR, 0.1, dtype=np.float32)
    _patch_windows(
        monkeypatch,
        [
            AnalysisWindow(0, 100 * SR, 0, 100 * SR),
            AnalysisWindow(100 * SR, 200 * SR, 100 * SR, 200 * SR),
        ],
    )
    _patch_diarize(
        monkeypatch,
        [
            [SpeakerSegment(0.0, 10.0, "a")],
            [SpeakerSegment(0.0, 10.0, "a")],
        ],
    )
    events: list[object] = []
    diarizer = _diarizer(
        embedder=_SequenceEmbedder([_A, _A]),
        monkeypatch=monkeypatch,
        on_progress=events.append,
    )

    diarizer.diarize(Path("audio.wav"), waveform=samples)

    assert events
    assert any(getattr(event, "fraction", None) == 1.0 for event in events)
    assert any("окно" in getattr(event, "detail", "") for event in events)


def test_hybrid_skips_failed_window(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(200 * SR, 0.1, dtype=np.float32)
    _patch_windows(
        monkeypatch,
        [
            AnalysisWindow(0, 100 * SR, 0, 100 * SR),
            AnalysisWindow(100 * SR, 200 * SR, 100 * SR, 200 * SR),
        ],
    )
    _patch_diarize(
        monkeypatch,
        [None, [SpeakerSegment(0.0, 10.0, "a")]],
    )
    diarizer = _diarizer(embedder=_SequenceEmbedder([_A]), monkeypatch=monkeypatch)

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples)

    # Сбой первого окна пропущен; результат — из второго (со смещением 100 с).
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(100.0)


def test_hybrid_raises_when_no_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(100 * SR, 0.1, dtype=np.float32)
    _patch_windows(monkeypatch, [AnalysisWindow(0, 100 * SR, 0, 100 * SR)])
    _patch_diarize(monkeypatch, [[]])
    diarizer = _diarizer(embedder=_SequenceEmbedder([]), monkeypatch=monkeypatch)

    with pytest.raises(HybridDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=samples)


def test_hybrid_raises_when_speakers_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(100 * SR, 0.1, dtype=np.float32)
    _patch_windows(monkeypatch, [AnalysisWindow(0, 100 * SR, 0, 100 * SR)])
    _patch_diarize(monkeypatch, [[SpeakerSegment(0.0, 0.5, "a")]])
    diarizer = _diarizer(
        embedder=_SequenceEmbedder([]),
        monkeypatch=monkeypatch,
        min_speaker_seconds=1.5,
    )

    with pytest.raises(HybridDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=samples)


def test_hybrid_raises_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hybrid_engine, "binary_available", lambda _binary: False)
    diarizer = HybridSpeakerDiarizer(embedder=_SequenceEmbedder([]))

    with pytest.raises(HybridDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32))


def test_hybrid_raises_when_embedder_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hybrid_engine, "binary_available", lambda _binary: True)
    monkeypatch.setattr(hybrid_engine.embedding_utils, "sherpa_available", lambda: False)
    diarizer = HybridSpeakerDiarizer()

    with pytest.raises(HybridDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32))


def test_hybrid_raises_on_empty_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hybrid_engine, "binary_available", lambda _binary: True)
    diarizer = HybridSpeakerDiarizer(embedder=_SequenceEmbedder([]))

    with pytest.raises(HybridDiarizationError):
        diarizer.diarize(Path("audio.wav"), waveform=np.zeros(0, dtype=np.float32))


# --- детект «перегруженного» окна и переобработка (#68) ----------------------


def test_window_is_overloaded_true_on_all_heads_with_churn() -> None:
    assert window_is_overloaded(_overloaded_segments()) is True


def test_window_is_overloaded_false_without_all_heads() -> None:
    segments = [
        SpeakerSegment(start=i * 0.5, end=(i + 1) * 0.5, speaker_id=f"spk{i % 3}")
        for i in range(12)
    ]

    assert window_is_overloaded(segments) is False


def test_window_is_overloaded_false_with_few_segments() -> None:
    segments = [
        SpeakerSegment(start=float(i), end=float(i) + 1, speaker_id=f"spk{i % 4}")
        for i in range(4)
    ]

    assert window_is_overloaded(segments) is False


def test_window_is_overloaded_false_on_calm_four_speakers() -> None:
    # Четыре говорящих, но длинные блоки без частой смены — окно не «путаное».
    segments = [
        SpeakerSegment(start=float(i), end=float(i) + 1, speaker_id=f"spk{i // 4}")
        for i in range(16)
    ]

    assert window_is_overloaded(segments) is False


def test_plan_subwindows_cover_parent_own_zone() -> None:
    total = 60 * SR
    samples = np.zeros(total, dtype=np.float32)
    parent = AnalysisWindow(0, total, 0, total)

    subs = hybrid_engine._plan_subwindows(
        parent, samples, subwindow_seconds=30.0, overlap_seconds=2.0
    )

    assert len(subs) > 1
    assert subs[0].start == parent.start
    assert subs[-1].end == parent.end
    assert subs[0].own_start == parent.own_start
    assert subs[-1].own_end == parent.own_end
    for previous, current in pairwise(subs):
        # Зоны владения мелких окон стыкуются встык: без пропусков и дублей.
        assert current.own_start == previous.own_end


def test_hybrid_reprocesses_overloaded_window_with_subwindows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.full(60 * SR, 0.5, dtype=np.float32)
    calls = {"count": 0}

    def fake_diarize(_audio_path: Path, **_kwargs: object) -> list[SpeakerSegment]:
        index = calls["count"]
        calls["count"] += 1
        if index == 0:
            return _overloaded_segments()
        return [SpeakerSegment(0.0, 10.0, "local_0")]

    monkeypatch.setattr(hybrid_engine, "diarize_audio", fake_diarize)
    monkeypatch.setattr(hybrid_engine, "binary_available", lambda _binary: True)
    diarizer = HybridSpeakerDiarizer(embedder=_SequenceEmbedder([_A, _B, _C]))

    segments = diarizer.diarize(Path("audio.wav"), waveform=samples)

    # Одно «перегруженное» окно (60 с) + три мелких (по 30 с с перекрытием).
    assert calls["count"] == 4
    assert len({segment.speaker_id for segment in segments}) == 3


def test_hybrid_overload_split_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.full(60 * SR, 0.5, dtype=np.float32)
    calls = {"count": 0}

    def fake_diarize(_audio_path: Path, **_kwargs: object) -> list[SpeakerSegment]:
        calls["count"] += 1
        return _overloaded_segments()

    monkeypatch.setattr(hybrid_engine, "diarize_audio", fake_diarize)
    monkeypatch.setattr(hybrid_engine, "binary_available", lambda _binary: True)
    diarizer = HybridSpeakerDiarizer(
        embedder=_SequenceEmbedder([_A, _B, _C, _D]),
        overload_split=False,
    )

    diarizer.diarize(Path("audio.wav"), waveform=samples)

    # Дробления нет — «nemo-speech» вызван один раз для единственного окна.
    assert calls["count"] == 1
