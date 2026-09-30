"""Тесты замера длительности стадий веб-интерфейса (``web/timings.py``)."""

from __future__ import annotations

from audio_transcriber.progress import ProgressEvent
from audio_transcriber.web.timings import StageTimer


class FakeClock:
    """Управляемые монотонные часы: время двигается только явно."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _timer() -> tuple[StageTimer, FakeClock]:
    clock = FakeClock()
    return StageTimer(clock=clock), clock


def test_stage_timer_measures_each_stage_in_order() -> None:
    timer, clock = _timer()

    timer.observe(ProgressEvent("denoise"))
    clock.advance(1.5)
    timer.observe(ProgressEvent("asr"))
    clock.advance(2.0)
    timer.observe(ProgressEvent("done"))

    timings = timer.timings()
    assert [timing.stage for timing in timings] == ["denoise", "asr"]
    assert timings[0].seconds == 1.5
    assert timings[1].seconds == 2.0
    assert all(timing.cached is False for timing in timings)


def test_stage_timer_keeps_single_timing_for_repeated_stage() -> None:
    timer, clock = _timer()

    # Диаризация шлёт несколько событий подряд (сегментация → эмбеддинги).
    timer.observe(ProgressEvent("diarization", "Определение говорящих"))
    clock.advance(0.5)
    timer.observe(ProgressEvent("diarization", "Сопоставление голосов"))
    clock.advance(0.5)
    timer.observe(ProgressEvent("merge"))

    timings = timer.timings()
    assert [timing.stage for timing in timings] == ["diarization"]
    assert timings[0].seconds == 1.0


def test_stage_timer_flags_cached_stage() -> None:
    timer, clock = _timer()

    timer.observe(ProgressEvent("asr", "Распознавание речи", detail="из кэша"))
    clock.advance(0.2)
    timer.observe(ProgressEvent("merge"))

    assert timer.timings()[0].cached is True


def test_stage_timer_cached_marker_on_repeat_event() -> None:
    """Денойз сообщает о кэш-хите отдельным повторным событием стадии."""
    timer, clock = _timer()

    timer.observe(ProgressEvent("denoise"))
    timer.observe(ProgressEvent("denoise", detail="из кэша"))
    clock.advance(0.3)
    timer.observe(ProgressEvent("asr"))

    denoise = timer.timings()[0]
    assert denoise.stage == "denoise"
    assert denoise.cached is True


def test_stage_timer_elapsed_and_current_elapsed() -> None:
    timer, clock = _timer()

    clock.advance(1.0)
    timer.observe(ProgressEvent("asr"))
    clock.advance(3.0)

    assert timer.elapsed() == 4.0
    assert timer.current_elapsed() == 3.0

    timer.close()
    assert timer.current_elapsed() == 0.0
    assert timer.elapsed() == 4.0


def test_stage_timer_close_is_idempotent() -> None:
    timer, clock = _timer()

    timer.observe(ProgressEvent("asr"))
    clock.advance(1.0)
    timer.close()
    timer.close()

    assert [timing.stage for timing in timer.timings()] == ["asr"]


def test_stage_timer_snapshot_is_json_ready() -> None:
    timer, clock = _timer()

    timer.observe(ProgressEvent("asr", detail="из кэша"))
    clock.advance(1.23456)
    timer.observe(ProgressEvent("done"))

    assert timer.snapshot() == [{"stage": "asr", "seconds": 1.235, "cached": True}]
