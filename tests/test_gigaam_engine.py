"""Тесты движка GigaAM (#46) — без реальной загрузки модели.

``onnx_asr`` подменяется фейковым модулем, а декодирование аудио — заглушкой.
Реальная модель не скачивается.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audio_transcriber.domain.enums import Device
from audio_transcriber.transcription import gigaam_engine
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.transcription.gigaam_engine import GigaAmRecognizer
from audio_transcriber.utils.exceptions import TranscriptionError

SAMPLE_RATE = 16000


class _SegmentResult:
    """Аналог ``onnx_asr.vad.TimestampedSegmentResult``."""

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        logprobs: list[float] | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.logprobs = logprobs


class _TimestampedResult:
    """Аналог ``onnx_asr.asr.TimestampedResult`` (без VAD)."""

    def __init__(
        self,
        text: str,
        timestamps: list[float] | None = None,
        tokens: list[str] | None = None,
        logprobs: list[float] | None = None,
    ) -> None:
        self.text = text
        self.timestamps = timestamps
        self.tokens = tokens
        self.logprobs = logprobs


class _FakeAdapter:
    """Цепочка адаптера onnx-asr: with_vad/with_timestamps/recognize."""

    def __init__(self, results: Any) -> None:
        self._results = results
        self.vad_kwargs: dict[str, object] | None = None
        self.timestamps_requested = False
        self.recognize_args: tuple[Any, int] | None = None

    def with_vad(self, vad: object, **kwargs: object) -> _FakeAdapter:
        self.vad = vad
        self.vad_kwargs = kwargs
        return self

    def with_timestamps(self) -> _FakeAdapter:
        self.timestamps_requested = True
        return self

    def recognize(self, waveform: Any, *, sample_rate: int = SAMPLE_RATE) -> Any:
        self.recognize_args = (waveform, sample_rate)
        if isinstance(self._results, list):
            return iter(self._results)
        return self._results


def _install_onnx_asr(
    monkeypatch: pytest.MonkeyPatch,
    adapter: _FakeAdapter,
    *,
    load_model: Any | None = None,
    load_vad: Any | None = None,
) -> None:
    module = types.ModuleType("onnx_asr")
    module.load_model = load_model or (lambda *_args, **_kwargs: adapter)
    module.load_vad = load_vad or (lambda *_args, **_kwargs: object())
    monkeypatch.setitem(sys.modules, "onnx_asr", module)


def _patch_waveform(
    monkeypatch: pytest.MonkeyPatch, *, seconds: float = 10.0
) -> np.ndarray:
    waveform = np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)
    monkeypatch.setattr(gigaam_engine, "load_waveform", lambda _path: waveform)
    return waveform


def test_recognizer_satisfies_protocol() -> None:
    recognizer = GigaAmRecognizer()
    assert isinstance(recognizer, SpeechRecognizer)


def test_transcribe_with_vad_maps_segments_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(
        [
            _SegmentResult(0.0, 3.0, " Привет ", logprobs=[-0.1, -0.2]),
            _SegmentResult(4.0, 5.0, "   ", logprobs=[-0.1]),
            _SegmentResult(6.0, 10.0, "мир", logprobs=[-0.5]),
        ]
    )
    _install_onnx_asr(monkeypatch, adapter)
    _patch_waveform(monkeypatch, seconds=10.0)
    events: list = []

    recognizer = GigaAmRecognizer(on_progress=events.append)
    segments, language, duration = recognizer.transcribe(Path("audio.wav"))

    assert language == "ru"
    assert duration == pytest.approx(10.0)
    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 3.0, "Привет"),
        (6.0, 10.0, "мир"),
    ]
    assert segments[0].avg_logprob == pytest.approx(-0.15)
    assert segments[1].avg_logprob == pytest.approx(-0.5)
    # Пустой сегмент отброшен.
    assert all(s.text for s in segments)

    assert adapter.timestamps_requested
    assert adapter.vad_kwargs is not None
    assert adapter.recognize_args is not None
    assert adapter.recognize_args[1] == SAMPLE_RATE

    fractions = [event.fraction for event in events]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fractions if f is not None)
    assert all(event.stage == "asr" for event in events)


def test_transcribe_without_vad_uses_token_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(
        _TimestampedResult(
            "готово", timestamps=[1.0, 1.5, 2.0], logprobs=[-0.2, -0.2, -0.2]
        )
    )
    _install_onnx_asr(monkeypatch, adapter)
    _patch_waveform(monkeypatch, seconds=5.0)

    recognizer = GigaAmRecognizer(use_vad=False)
    segments, language, duration = recognizer.transcribe(Path("audio.wav"))

    assert language == "ru"
    assert duration == pytest.approx(5.0)
    assert len(segments) == 1
    # Конец последнего токена продлён на средний шаг между токенами.
    assert segments[0].start == pytest.approx(1.0)
    assert segments[0].end == pytest.approx(2.5)
    assert segments[0].avg_logprob == pytest.approx(-0.2)


def test_missing_onnx_asr_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_waveform(monkeypatch)
    # ``None`` в sys.modules заставляет ``import onnx_asr`` бросить ImportError.
    monkeypatch.setitem(sys.modules, "onnx_asr", None)

    recognizer = GigaAmRecognizer()
    with pytest.raises(TranscriptionError) as excinfo:
        recognizer.transcribe(Path("audio.wav"))
    assert "onnx-asr" in str(excinfo.value)


def test_model_load_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("нет сети")

    adapter = _FakeAdapter([])
    _install_onnx_asr(monkeypatch, adapter, load_model=_boom)
    _patch_waveform(monkeypatch)

    recognizer = GigaAmRecognizer()
    with pytest.raises(TranscriptionError) as excinfo:
        recognizer._load_adapter()
    assert "GigaAM" in str(excinfo.value)


def test_vad_load_failure_degrades_to_no_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(
        _TimestampedResult("да", timestamps=[0.0, 0.5, 1.0], logprobs=[-0.1, -0.1, -0.1])
    )

    def _no_vad(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("vad недоступен")

    _install_onnx_asr(monkeypatch, adapter, load_vad=_no_vad)
    _patch_waveform(monkeypatch, seconds=1.0)

    recognizer = GigaAmRecognizer(use_vad=True)
    segments, _, _ = recognizer.transcribe(Path("audio.wav"))

    # VAD не загрузился, но распознавание всё равно прошло (адаптер без VAD):
    # результат — один сегмент по границам токенов.
    assert [s.text for s in segments] == ["да"]
    assert recognizer._vad_active is False
    assert adapter.vad_kwargs is None


def test_resolve_providers_cpu_is_none() -> None:
    assert gigaam_engine.resolve_onnx_providers(Device.CPU) is None
    assert gigaam_engine.resolve_onnx_providers(Device.AUTO) is None


def test_empty_waveform_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter([])
    _install_onnx_asr(monkeypatch, adapter)
    monkeypatch.setattr(
        gigaam_engine, "load_waveform", lambda _path: np.zeros(0, dtype=np.float32)
    )

    segments, language, duration = GigaAmRecognizer().transcribe(Path("audio.wav"))
    assert segments == []
    assert language == "ru"
    assert duration == 0.0
