"""Тесты гибридного ASR (#57) — детекция «плохих» сегментов и склейка.

Реальные модели не нужны: основной и резервный движки — фейки, а резервный
читает временный WAV, чтобы проверить нарезку с контекстом.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.cache.serialization import asr_from_payload, asr_payload
from audio_transcriber.domain.models import TranscriptionSegment
from audio_transcriber.transcription import hybrid as hybrid_module
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.transcription.hybrid import (
    BadSegment,
    HybridOptions,
    HybridSpeechRecognizer,
    classify_segment,
    detect_bad_segments,
    segment_rms,
)
from audio_transcriber.utils.exceptions import TranscriptionError

SAMPLE_RATE = 16000


class _ScriptedRecognizer:
    """Фейковый движок, отдающий заранее заданный результат."""

    def __init__(
        self,
        segments: list[TranscriptionSegment],
        *,
        language: str = "ru",
        duration: float | None = None,
        fail: bool = False,
    ) -> None:
        self.segments = segments
        self.language = language
        self.duration = (
            duration if duration is not None else (segments[-1].end if segments else 0.0)
        )
        self.fail = fail
        self.calls = 0

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        self.calls += 1
        if self.fail:
            raise TranscriptionError("резервный движок недоступен")
        return list(self.segments), self.language, self.duration


class _ChunkFallback:
    """Резервный движок, строящий ответ по длительности полученного куска."""

    def __init__(self, builder) -> None:
        self._builder = builder
        self.chunk_seconds: list[float] = []

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        with wave.open(str(audio_path), "rb") as wf:
            seconds = wf.getnframes() / wf.getframerate()
        self.chunk_seconds.append(seconds)
        return self._builder(seconds), "ru", seconds


def _segment(
    start: float,
    end: float,
    text: str,
    *,
    avg_logprob: float | None = -0.2,
    no_speech_prob: float | None = None,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        start=start,
        end=end,
        text=text,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


def _patch_waveform(monkeypatch: pytest.MonkeyPatch, waveform: np.ndarray) -> None:
    monkeypatch.setattr(hybrid_module, "load_waveform", lambda _path: waveform)


def _loud(seconds: float) -> np.ndarray:
    return np.full(round(seconds * SAMPLE_RATE), 0.1, dtype=np.float32)


# --- Детекция «плохих» сегментов -------------------------------------------


def test_segment_rms_handles_empty_and_out_of_range() -> None:
    waveform = np.full(100, 0.5, dtype=np.float32)
    assert segment_rms(waveform, 0.0, 0.001) > 0.0
    assert segment_rms(waveform, 1.0, 1.0) == 0.0
    assert segment_rms(np.zeros(0, dtype=np.float32), 0.0, 1.0) == 0.0
    assert segment_rms(waveform, 5.0, 10.0) == 0.0


def test_classify_segment_detects_each_reason() -> None:
    options = HybridOptions()
    loud = _loud(3.0)
    silent = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)

    good = _segment(0.0, 1.0, "привет", avg_logprob=-0.2)
    assert classify_segment(good, waveform=loud, options=options) == ()

    assert classify_segment(
        _segment(0.0, 1.0, "   ", avg_logprob=-0.2), waveform=loud, options=options
    ) == ("empty",)
    assert "short" in classify_segment(
        _segment(0.0, 0.2, "да", avg_logprob=-0.2), waveform=loud, options=options
    )
    assert "low_logprob" in classify_segment(
        _segment(0.0, 1.0, "шум", avg_logprob=-3.0), waveform=loud, options=options
    )
    assert "no_speech" in classify_segment(
        _segment(0.0, 1.0, "шум", avg_logprob=-0.2, no_speech_prob=0.9),
        waveform=loud,
        options=options,
    )
    assert "silence" in classify_segment(
        _segment(0.0, 1.0, "тихо", avg_logprob=-0.2), waveform=silent, options=options
    )


def test_detect_bad_segments_returns_indices_and_reasons() -> None:
    segments = [
        _segment(0.0, 2.0, "хорошо"),
        _segment(2.0, 3.0, "плохо", avg_logprob=-4.0),
        _segment(3.0, 3.2, "обрывок"),
    ]
    bad = detect_bad_segments(
        segments, waveform=_loud(4.0), options=HybridOptions(), sample_rate=SAMPLE_RATE
    )

    assert [item.index for item in bad] == [1, 2]
    assert bad[0].reasons == ("low_logprob",)
    assert bad[1].reasons == ("short",)
    assert all(isinstance(item, BadSegment) for item in bad)


# --- Гибрид: склейка и абсолютные смещения ---------------------------------


def test_hybrid_refines_only_bad_segment_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _ScriptedRecognizer(
        [
            _segment(0.0, 2.0, "ввод"),
            _segment(10.0, 11.0, "неразборчиво", avg_logprob=-3.0),
        ],
        duration=12.0,
    )
    # Куску с контекстом [9.6, 11.4] (1.8 с) соответствует результат Whisper.
    fallback = _ChunkFallback(
        lambda _seconds: [_segment(0.0, 1.8, "исправлено", avg_logprob=-0.1)]
    )
    _patch_waveform(monkeypatch, _loud(12.0))

    hybrid = HybridSpeechRecognizer(primary, fallback, options=HybridOptions())
    segments, language, duration = hybrid.transcribe(Path("audio.wav"))

    assert language == "ru"
    assert duration == pytest.approx(12.0)
    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 2.0, "ввод"),
        (10.0, 11.0, "исправлено"),
    ]
    # Резервный движок получил ровно один кусок с контекстом 0.4 с с каждой
    # стороны: 1.0 + 0.8 = 1.8 с.
    assert primary.calls == 1
    assert len(fallback.chunk_seconds) == 1
    assert fallback.chunk_seconds[0] == pytest.approx(1.8, abs=0.01)

    assert hybrid.last_refined_segments == 1
    assert hybrid.last_refined_seconds == pytest.approx(1.0)
    assert hybrid.last_refined_fraction == pytest.approx(1.0 / 12.0)


def test_hybrid_timestamps_are_absolute_and_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _ScriptedRecognizer(
        [_segment(100.0, 102.0, "плохой", avg_logprob=-5.0)], duration=120.0
    )
    # Кусок = [99.6, 102.4]; результат Whisper [0.05, 1.95] смещается в
    # [99.65, 101.55] и обрезается исходным окном [100, 102].
    fallback = _ChunkFallback(
        lambda _seconds: [_segment(0.05, 1.95, "абсолют", avg_logprob=-0.2)]
    )
    _patch_waveform(monkeypatch, _loud(120.0))

    hybrid = HybridSpeechRecognizer(primary, fallback)
    segments, _, _ = hybrid.transcribe(Path("audio.wav"))

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(100.0)
    assert segments[0].end == pytest.approx(101.55, abs=0.01)
    assert segments[0].text == "абсолют"


def test_hybrid_drops_fallback_text_outside_original_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _ScriptedRecognizer(
        [_segment(10.0, 11.0, "исходный", avg_logprob=-9.0)], duration=30.0
    )
    # Результат Whisper целиком лежит в контексте до исходного окна — отбрасываем.
    fallback = _ChunkFallback(lambda _seconds: [_segment(0.0, 0.2, "лишнее")])
    _patch_waveform(monkeypatch, _loud(30.0))

    hybrid = HybridSpeechRecognizer(primary, fallback)
    segments, _, _ = hybrid.transcribe(Path("audio.wav"))

    assert [(s.start, s.end, s.text) for s in segments] == [(10.0, 11.0, "исходный")]


def test_hybrid_keeps_original_when_fallback_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _ScriptedRecognizer(
        [_segment(1.0, 2.0, "тишина", avg_logprob=-9.0)], duration=10.0
    )
    fallback = _ChunkFallback(lambda _seconds: [])
    _patch_waveform(monkeypatch, _loud(10.0))

    hybrid = HybridSpeechRecognizer(primary, fallback)
    segments, _, _ = hybrid.transcribe(Path("audio.wav"))

    assert [s.text for s in segments] == ["тишина"]


def test_hybrid_keeps_original_on_fallback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _ScriptedRecognizer(
        [_segment(1.0, 2.0, "ошибка", avg_logprob=-9.0)], duration=10.0
    )
    fallback = _ScriptedRecognizer([], fail=True)
    _patch_waveform(monkeypatch, _loud(10.0))

    hybrid = HybridSpeechRecognizer(primary, fallback)
    segments, _, _ = hybrid.transcribe(Path("audio.wav"))

    assert [s.text for s in segments] == ["ошибка"]


def test_hybrid_skips_refinement_when_all_good(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = _ScriptedRecognizer([_segment(0.0, 4.0, "хорошо")], duration=10.0)
    fallback = _ChunkFallback(lambda _seconds: [])
    _patch_waveform(monkeypatch, _loud(10.0))

    hybrid = HybridSpeechRecognizer(primary, fallback)
    segments, _, _ = hybrid.transcribe(Path("audio.wav"))

    assert [s.text for s in segments] == ["хорошо"]
    assert fallback.chunk_seconds == []
    assert hybrid.last_refined_fraction == 0.0
    assert hybrid.last_refined_segments == 0


def test_hybrid_degrades_when_audio_cannot_be_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _ScriptedRecognizer([_segment(0.0, 1.0, "текст", avg_logprob=-9.0)])

    def _boom(_path: Path) -> np.ndarray:
        raise TranscriptionError("не декодируется")

    monkeypatch.setattr(hybrid_module, "load_waveform", _boom)
    fallback = _ChunkFallback(lambda _seconds: [])

    hybrid = HybridSpeechRecognizer(primary, fallback)
    segments, _, _ = hybrid.transcribe(Path("audio.wav"))

    assert [s.text for s in segments] == ["текст"]
    assert fallback.chunk_seconds == []


def test_hybrid_satisfies_protocol() -> None:
    hybrid = HybridSpeechRecognizer(
        _ScriptedRecognizer([]), _ScriptedRecognizer([])
    )
    assert isinstance(hybrid, SpeechRecognizer)


# --- Совместимость кэша ASR с no_speech_prob --------------------------------


def test_asr_payload_roundtrip_preserves_no_speech_prob() -> None:
    segments = [_segment(0.0, 1.0, "текст", no_speech_prob=0.42)]
    payload = asr_payload(segments, "ru", 1.0)

    restored, language, duration = asr_from_payload(payload)

    assert language == "ru"
    assert duration == pytest.approx(1.0)
    assert restored[0].no_speech_prob == pytest.approx(0.42)


def test_asr_payload_tolerates_legacy_without_no_speech_prob() -> None:
    payload = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "текст", "avg_logprob": -0.1}],
        "language": "ru",
        "duration": 1.0,
    }
    restored, _, _ = asr_from_payload(payload)
    assert restored[0].no_speech_prob is None
