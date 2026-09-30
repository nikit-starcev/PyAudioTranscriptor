"""Тесты прогресса faster-whisper: fraction по обработанным сегментам.

Реальная модель не загружается — ``_load_model`` подменяется заглушкой,
возвращающей итератор сегментов и объект ``info`` с длительностью.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_transcriber.domain.enums import Device
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer


def _segment(start: float, end: float, text: str, avg_logprob: float = -0.1) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=avg_logprob)


class _FakeModel:
    def __init__(self, segments: list[SimpleNamespace], info: SimpleNamespace) -> None:
        self._segments = segments
        self._info = info
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio: str, **kwargs: object) -> tuple[object, SimpleNamespace]:
        self.calls.append(kwargs)
        return iter(self._segments), self._info


def _recognizer(
    monkeypatch: pytest.MonkeyPatch,
    model: _FakeModel,
    events: list[object],
) -> WhisperSpeechRecognizer:
    recognizer = WhisperSpeechRecognizer("tiny", Device.CPU, on_progress=events.append)
    monkeypatch.setattr(recognizer, "_load_model", lambda: model)
    return recognizer


def test_progress_fraction_grows_with_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(
        [_segment(0.0, 2.0, " раз"), _segment(2.0, 5.0, " два"), _segment(5.0, 10.0, " три")],
        SimpleNamespace(language="ru", duration=20.0),
    )
    events: list = []

    recognizer = _recognizer(monkeypatch, model, events)
    segments, language, duration = recognizer.transcribe(Path("audio.wav"))

    assert language == "ru"
    assert duration == 20.0
    assert [segment.text for segment in segments] == ["раз", "два", "три"]

    fractions = [event.fraction for event in events]
    # Первое событие — старт (0.0), затем fraction по концу каждого сегмента.
    assert fractions[0] == 0.0
    assert fractions[1:] == [pytest.approx(0.1), pytest.approx(0.25), pytest.approx(0.5)]
    assert all(event.stage == "asr" for event in events)


def test_progress_fraction_is_monotonic_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Конец последнего сегмента выходит за длительность — fraction ограничен 1.0.
    model = _FakeModel(
        [_segment(0.0, 4.0, "а"), _segment(4.0, 25.0, "б")],
        SimpleNamespace(language="ru", duration=20.0),
    )
    events: list = []

    recognizer = _recognizer(monkeypatch, model, events)
    recognizer.transcribe(Path("audio.wav"))

    fractions = [event.fraction for event in events if event.fraction is not None]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert all(0.0 <= fraction <= 1.0 for fraction in fractions)


def test_progress_without_known_duration_has_no_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(
        [_segment(0.0, 4.0, "а")],
        SimpleNamespace(language="en", duration=None),
    )
    events: list = []

    recognizer = _recognizer(monkeypatch, model, events)
    recognizer.transcribe(Path("audio.wav"))

    assert events
    assert all(event.fraction is None for event in events)


def test_progress_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(
        [_segment(0.0, 4.0, "а")],
        SimpleNamespace(language="ru", duration=10.0),
    )
    recognizer = WhisperSpeechRecognizer("tiny", Device.CPU)
    monkeypatch.setattr(recognizer, "_load_model", lambda: model)

    segments, _, _ = recognizer.transcribe(Path("audio.wav"))

    assert len(segments) == 1
