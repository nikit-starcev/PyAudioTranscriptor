"""Тесты чанкинга длинных файлов в whisper.cpp (без реальных моделей).

Реальный бинарник не запускается: ``subprocess.Popen`` подменяется заглушкой,
которая пишет JSON-ответ в путь из ``-of``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.domain.models import TranscriptionSegment
from audio_transcriber.transcription import whisper_cpp_engine as whisper_module
from audio_transcriber.transcription.whisper_cpp_engine import (
    WhisperCppRecognizer,
    _choose_chunk_bounds,
    _ChunkedSegment,
    _deduplicate_chunk_segments,
    _min_energy_sample,
    _segments_are_duplicates,
)
from audio_transcriber.utils.audio import AudioProbe

SAMPLE_RATE = 16000


class _FakeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.pid = 5150
        self.returncode = returncode
        self.stderr = iter(["progress = 100%\n"])

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _segment_payload(text: str, start_ms: int = 0, end_ms: int = 1000) -> dict:
    return {
        "offsets": {"from": start_ms, "to": end_ms},
        "text": f" {text}",
        "tokens": [{"text": f" {text}", "p": 0.9}],
    }


def _install_popen(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[list[str]],
    payload_for: Callable[[int], list[dict]],
) -> None:
    def _fake_popen(cmd: list[str], *_args: object, **_kwargs: object) -> _FakeProc:
        captured.append(list(cmd))
        base = Path(cmd[cmd.index("-of") + 1])
        suffix = base.name.rsplit("_", 1)[-1]
        index = int(suffix) if suffix.isdigit() else 0
        payload = {"result": {"language": "ru"}, "transcription": payload_for(index)}
        Path(str(base) + ".json").write_text(json.dumps(payload), encoding="utf-8")
        return _FakeProc()

    monkeypatch.setattr(whisper_module.subprocess, "Popen", _fake_popen)


def _prepare_long_audio(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_seconds: float,
    waveform: np.ndarray | None = None,
) -> tuple[Path, list[Path]]:
    """Готовит «длинный» native WAV: probe и load_waveform подменяются."""

    probe = AudioProbe("wav", "pcm_s16le", SAMPLE_RATE, 1, duration_seconds)
    monkeypatch.setattr(whisper_module, "probe_audio", lambda _p: probe)
    data = (
        waveform
        if waveform is not None
        else np.zeros(round(duration_seconds * SAMPLE_RATE), dtype=np.float32)
    )
    monkeypatch.setattr(whisper_module, "load_waveform", lambda _p: data)
    written: list[Path] = []
    monkeypatch.setattr(
        whisper_module, "write_wav", lambda path, _waveform: written.append(Path(path))
    )
    return Path("input.wav"), written


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    return model


def test_long_audio_is_split_into_chunks_with_shifted_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    input_path, written = _prepare_long_audio(monkeypatch, duration_seconds=65.0)
    _install_popen(
        monkeypatch,
        captured,
        lambda index: [_segment_payload(f"кусок{index}")],
    )

    recognizer = WhisperCppRecognizer(
        _model(tmp_path), chunk_seconds=30.0, chunk_overlap=2.0
    )
    segments, language, duration = recognizer.transcribe(input_path)

    assert language == "ru"
    assert duration == pytest.approx(65.0)
    # 65 с при куске 30 с и перекрытии 2 с: [0,30], [28,58], [56,65].
    assert len(captured) == 3
    assert len(written) == 3
    assert [segment.text for segment in segments] == ["кусок0", "кусок1", "кусок2"]
    assert [round(segment.start, 3) for segment in segments] == [0.0, 28.0, 56.0]


def test_chunking_can_be_disabled_by_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    input_path, written = _prepare_long_audio(monkeypatch, duration_seconds=65.0)
    _install_popen(monkeypatch, captured, lambda _index: [_segment_payload("весь файл")])

    recognizer = WhisperCppRecognizer(_model(tmp_path), chunk_seconds=0.0)
    segments, _language, _duration = recognizer.transcribe(input_path)

    assert len(captured) == 1
    assert written == []
    assert [segment.text for segment in segments] == ["весь файл"]


def test_short_audio_is_not_chunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    input_path, written = _prepare_long_audio(monkeypatch, duration_seconds=10.0)
    _install_popen(monkeypatch, captured, lambda _index: [_segment_payload("коротко")])

    recognizer = WhisperCppRecognizer(_model(tmp_path), chunk_seconds=30.0)
    recognizer.transcribe(input_path)

    assert len(captured) == 1
    assert written == []


def test_overlap_duplicate_is_kept_once_from_more_central_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Одинаковая реплика на стыке кусков остаётся один раз — из центрального куска."""

    captured: list[list[str]] = []
    input_path, _written = _prepare_long_audio(monkeypatch, duration_seconds=65.0)

    def payload_for(index: int) -> list[dict]:
        if index == 0:
            # Ближе к концу куска 0 [0,30] — далеко от центра (15 с).
            return [_segment_payload("у нас не получилось залочить", start_ms=28500, end_ms=29500)]
        if index == 1:
            # Ближе к началу куска 1 [28,58] — далеко от центра (43 с), но
            # чуть центральнее, чем вариант из куска 0.
            return [_segment_payload("у нас не получилось залочить", start_ms=700, end_ms=1600)]
        return [_segment_payload("дальше")]

    _install_popen(monkeypatch, captured, payload_for)

    recognizer = WhisperCppRecognizer(
        _model(tmp_path), chunk_seconds=30.0, chunk_overlap=2.0
    )
    segments, _language, _duration = recognizer.transcribe(input_path)

    texts = [segment.text for segment in segments]
    assert texts.count("у нас не получилось залочить") == 1
    # Остался вариант из более центрального куска 1 (28.7..29.6), а не 28.5..29.5.
    duplicate = next(s for s in segments if s.text == "у нас не получилось залочить")
    assert duplicate.start == pytest.approx(28.7)


def test_chunk_overlap_boundaries_snap_to_quiet_point() -> None:
    total = 40 * SAMPLE_RATE
    waveform = np.full(total, 0.5, dtype=np.float32)
    # Впадина около целевой точки разреза второго куска (28 с = 30 - 2 перекрытия).
    dip_start = 28 * SAMPLE_RATE - 4000
    dip_end = 28 * SAMPLE_RATE + 4000
    waveform[dip_start:dip_end] = 0.0

    bounds = _choose_chunk_bounds(
        waveform, chunk_seconds=30.0, overlap_seconds=2.0, sample_rate=SAMPLE_RATE
    )

    assert len(bounds) == 2
    # Начало второго куска попало в «тихую» впадину, а не разрубило сигнал.
    assert dip_start <= bounds[1][0] <= dip_end
    assert bounds[0][0] == 0
    assert bounds[-1][1] == total


def test_min_energy_sample_returns_target_without_radius() -> None:
    waveform = np.ones(10 * SAMPLE_RATE, dtype=np.float32)
    assert _min_energy_sample(waveform, target=5 * SAMPLE_RATE, radius=0) == 5 * SAMPLE_RATE


def test_dedup_ignores_segments_without_time_overlap() -> None:
    assert not _segments_are_duplicates(
        TranscriptionSegment(10.0, 10.5, "да"),
        TranscriptionSegment(11.0, 11.5, "да"),
    )


def test_dedup_requires_text_similarity() -> None:
    assert not _segments_are_duplicates(
        TranscriptionSegment(10.0, 12.0, "совершенно другая фраза"),
        TranscriptionSegment(10.0, 12.0, "абсолютно иной текст"),
    )
    assert _segments_are_duplicates(
        TranscriptionSegment(10.0, 12.0, "одна и та же фраза"),
        TranscriptionSegment(10.1, 12.1, "одна и та же фраза"),
    )


def test_dedup_keeps_both_non_duplicate_overlapping_segments() -> None:
    items = [
        _ChunkedSegment(TranscriptionSegment(28.0, 29.0, "первый"), 15.0, 0),
        _ChunkedSegment(TranscriptionSegment(28.0, 29.0, "второй"), 43.0, 1),
    ]

    result = _deduplicate_chunk_segments(items)

    assert [segment.text for segment in result] == ["первый", "второй"]
