"""Тесты интеграции кэша в конвейер (``run_pipeline``).

Проверяют попадание/промах по стадиям, инвалидацию по параметрам и файлу,
возобновление после сбоя, мягкую деградацию на битом кэше и переиспользование
результата шумоподавления.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import (
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)
from audio_transcriber.pipeline import run_pipeline


class RecordingRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Path, *, language: str | None = None):
        self.calls += 1
        return ([TranscriptionSegment(0.0, 1.0, "привет", -0.2)], "ru", 1.0)


class RecordingDiarizer:
    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, audio_path: Path, *, num_speakers: int | None = None):
        self.calls += 1
        return [SpeakerSegment(0.0, 1.0, "SPEAKER_00")]


class RaisingDiarizer:
    def diarize(self, *args, **kwargs):
        raise RuntimeError("сбой диаризации")


class SimpleMerger:
    def merge(self, transcription_segments, speaker_segments, known_speakers=None):
        entries = [
            TranscriptEntry(start=segment.start, end=segment.end, text=segment.text)
            for segment in transcription_segments
        ]
        return entries, []


class RecordingDenoiser:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.calls = 0

    def denoise(self, input_path: Path) -> Path:
        self.calls += 1
        return self.output

    def close(self) -> None:
        pass


def _config(audio_file: Path, tmp_path: Path, **overrides) -> AppConfig:
    options = {
        "input_file": audio_file,
        "output_dir": tmp_path / "out",
        "export_formats": (ExportFormat.TXT,),
        "denoise": False,
    }
    options.update(overrides)
    return AppConfig(**options)


def _run(config: AppConfig, recognizer, diarizer, **kwargs):
    return run_pipeline(
        config,
        device=Device.CPU,
        recognizer=recognizer,
        diarizer=diarizer,
        merger=SimpleMerger(),
        **kwargs,
    )


def test_cache_hit_skips_asr_and_diarization(audio_file: Path, tmp_path: Path) -> None:
    config = _config(audio_file, tmp_path)

    first_rec, first_dia = RecordingRecognizer(), RecordingDiarizer()
    _run(config, first_rec, first_dia)
    second_rec, second_dia = RecordingRecognizer(), RecordingDiarizer()
    _run(config, second_rec, second_dia)

    assert first_rec.calls == 1 and first_dia.calls == 1
    assert second_rec.calls == 0
    assert second_dia.calls == 0


def test_asr_cache_hit_but_diarization_recomputed_when_missing(
    audio_file: Path, tmp_path: Path
) -> None:
    config = _config(audio_file, tmp_path)

    # Первый прогон падает на диаризации: ASR-результат успевает закэшироваться.
    with pytest.raises(RuntimeError):
        _run(config, RecordingRecognizer(), RaisingDiarizer())

    resumed_rec = RecordingRecognizer()
    resumed_dia = RecordingDiarizer()
    result = _run(config, resumed_rec, resumed_dia)

    # Возобновление: ASR не пересчитывается, диаризация выполняется вновь.
    assert resumed_rec.calls == 0
    assert resumed_dia.calls == 1
    assert len(result.entries) == 1


def test_parameter_change_invalidates_asr_cache(audio_file: Path, tmp_path: Path) -> None:
    config = _config(audio_file, tmp_path)
    _run(config, RecordingRecognizer(), RecordingDiarizer())

    changed = replace(config, language="en")
    recognizer = RecordingRecognizer()
    _run(changed, recognizer, RecordingDiarizer())

    assert recognizer.calls == 1


def test_file_change_invalidates_cache(audio_file: Path, tmp_path: Path) -> None:
    config = _config(audio_file, tmp_path)
    _run(config, RecordingRecognizer(), RecordingDiarizer())

    audio_file.write_bytes(b"changed content")

    recognizer, diarizer = RecordingRecognizer(), RecordingDiarizer()
    _run(config, recognizer, diarizer)

    assert recognizer.calls == 1
    assert diarizer.calls == 1


def test_broken_cache_file_is_recomputed(audio_file: Path, tmp_path: Path) -> None:
    config = _config(audio_file, tmp_path)
    _run(config, RecordingRecognizer(), RecordingDiarizer())

    asr_files = list((tmp_path / "out" / ".cache").glob("asr-*.json"))
    assert len(asr_files) == 1
    asr_files[0].write_text("{broken", encoding="utf-8")

    recognizer, diarizer = RecordingRecognizer(), RecordingDiarizer()
    _run(config, recognizer, diarizer)

    assert recognizer.calls == 1  # пересчитан
    assert diarizer.calls == 0  # диаризационный кэш цел


def test_structurally_invalid_cache_is_recomputed(audio_file: Path, tmp_path: Path) -> None:
    config = _config(audio_file, tmp_path)
    _run(config, RecordingRecognizer(), RecordingDiarizer())

    asr_file = next((tmp_path / "out" / ".cache").glob("asr-*.json"))
    asr_file.write_text(
        json.dumps({"cache_version": 1, "stage": "asr", "data": {"segments": "oops"}}),
        encoding="utf-8",
    )

    recognizer = RecordingRecognizer()
    _run(config, recognizer, RecordingDiarizer())

    assert recognizer.calls == 1


def test_cache_disabled_always_recomputes(audio_file: Path, tmp_path: Path) -> None:
    config = _config(audio_file, tmp_path, use_cache=False)
    _run(config, RecordingRecognizer(), RecordingDiarizer())

    recognizer = RecordingRecognizer()
    _run(config, recognizer, RecordingDiarizer())

    assert recognizer.calls == 1
    assert not (tmp_path / "out" / ".cache").exists()


def test_custom_cache_dir_is_used(audio_file: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "custom-cache"
    config = _config(audio_file, tmp_path, cache_dir=cache_dir)
    _run(config, RecordingRecognizer(), RecordingDiarizer())

    assert cache_dir.is_dir()
    assert list(cache_dir.glob("asr-*.json"))


def test_denoise_result_is_reused(audio_file: Path, tmp_path: Path) -> None:
    denoised = tmp_path / "denoised.wav"
    denoised.write_bytes(b"RIFF-cleaned-audio")
    config = _config(audio_file, tmp_path, denoise=True, diarization_enabled=False)

    first_denoiser = RecordingDenoiser(denoised)
    first_rec = RecordingRecognizer()
    _run(config, first_rec, RecordingDiarizer(), denoiser=first_denoiser)

    second_denoiser = RecordingDenoiser(denoised)
    second_rec = RecordingRecognizer()
    _run(config, second_rec, RecordingDiarizer(), denoiser=second_denoiser)

    assert first_denoiser.calls == 1
    assert second_denoiser.calls == 0  # кэш денойза
    assert second_rec.calls == 0  # кэш ASR


def test_soft_denoise_degradation_is_not_cached(audio_file: Path, tmp_path: Path) -> None:
    # Денойзер вернул исходный путь (движок недоступен) — неудачу не кэшируем.
    config = _config(audio_file, tmp_path, denoise=True, diarization_enabled=False)

    class SkippingDenoiser(RecordingDenoiser):
        def denoise(self, input_path: Path) -> Path:
            self.calls += 1
            return input_path

    first = SkippingDenoiser(audio_file)
    _run(config, RecordingRecognizer(), RecordingDiarizer(), denoiser=first)
    second = SkippingDenoiser(audio_file)
    _run(config, RecordingRecognizer(), RecordingDiarizer(), denoiser=second)

    assert first.calls == 1
    assert second.calls == 1
