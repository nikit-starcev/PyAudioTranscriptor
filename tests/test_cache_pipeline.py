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

from audio_transcriber.cache.store import compute_cache_key
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.overlap import DIARIZATION_IMPL_VERSION
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.domain.models import (
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)
from audio_transcriber.pipeline import (
    _asr_cache_params,
    _build_recognizer,
    _diarization_cache_params,
    _whisper_cpp_chunk_settings,
    run_pipeline,
)
from audio_transcriber.transcription.whisper_cpp_engine import (
    ASR_IMPL_VERSION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SECONDS,
    WhisperCppRecognizer,
)


class RecordingRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Path, *, language: str | None = None):
        self.calls += 1
        return ([TranscriptionSegment(0.0, 1.0, "привет", -0.2)], "ru", 1.0)


class RecordingDiarizer:
    def __init__(self) -> None:
        self.calls = 0

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
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


def _whisper_cpp_config(audio_file: Path, tmp_path: Path) -> AppConfig:
    return AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        export_formats=(ExportFormat.TXT,),
        denoise=False,
        asr_backend=AsrBackend.WHISPER_CPP,
        whisper_cpp_model=tmp_path / "model.bin",
    )


def test_asr_cache_params_include_impl_version(audio_file: Path, tmp_path: Path) -> None:
    config = _whisper_cpp_config(audio_file, tmp_path)
    recognizer = WhisperCppRecognizer(config.whisper_cpp_model)  # type: ignore[arg-type]

    params = _asr_cache_params(config, Device.CPU, recognizer)

    assert params["asr_impl_version"] == ASR_IMPL_VERSION


def test_asr_cache_key_changes_with_impl_version(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _whisper_cpp_config(audio_file, tmp_path)
    recognizer = WhisperCppRecognizer(config.whisper_cpp_model)  # type: ignore[arg-type]

    before = compute_cache_key(
        "asr", audio_file, _asr_cache_params(config, Device.CPU, recognizer)
    )
    monkeypatch.setattr("audio_transcriber.pipeline.ASR_IMPL_VERSION", ASR_IMPL_VERSION + 1)
    after = compute_cache_key(
        "asr", audio_file, _asr_cache_params(config, Device.CPU, recognizer)
    )

    assert before != after


def test_faster_whisper_cache_params_have_no_whisper_cpp_salt(
    audio_file: Path, tmp_path: Path
) -> None:
    config = _config(audio_file, tmp_path)  # бэкенд по умолчанию — faster-whisper

    params = _asr_cache_params(config, Device.CPU, RecordingRecognizer())

    assert "asr_impl_version" not in params


def test_asr_cache_params_include_chunk_settings(audio_file: Path, tmp_path: Path) -> None:
    config = _whisper_cpp_config(audio_file, tmp_path)
    recognizer = WhisperCppRecognizer(
        config.whisper_cpp_model,  # type: ignore[arg-type]
        chunk_seconds=15.0,
        chunk_overlap=3.0,
    )

    params = _asr_cache_params(config, Device.CPU, recognizer)

    assert params["whisper_cpp_chunk_seconds"] == 15.0
    assert params["whisper_cpp_chunk_overlap"] == 3.0


def test_whisper_cpp_chunk_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WHISPER_CPP_CHUNK_SECONDS", raising=False)
    monkeypatch.delenv("WHISPER_CPP_CHUNK_OVERLAP", raising=False)

    assert _whisper_cpp_chunk_settings() == (DEFAULT_CHUNK_SECONDS, DEFAULT_CHUNK_OVERLAP)


def test_whisper_cpp_chunk_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_CPP_CHUNK_SECONDS", "45")
    monkeypatch.setenv("WHISPER_CPP_CHUNK_OVERLAP", "3")

    assert _whisper_cpp_chunk_settings() == (45.0, 3.0)


def test_whisper_cpp_chunk_settings_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_CPP_CHUNK_SECONDS", "0")

    chunk_seconds, _overlap = _whisper_cpp_chunk_settings()

    assert chunk_seconds == 0.0


def test_whisper_cpp_chunk_settings_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHISPER_CPP_CHUNK_SECONDS", "не число")
    monkeypatch.setenv("WHISPER_CPP_CHUNK_OVERLAP", "-5")

    assert _whisper_cpp_chunk_settings() == (DEFAULT_CHUNK_SECONDS, DEFAULT_CHUNK_OVERLAP)


def test_whisper_cpp_chunk_settings_caps_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_CPP_CHUNK_SECONDS", "10")
    monkeypatch.setenv("WHISPER_CPP_CHUNK_OVERLAP", "20")

    assert _whisper_cpp_chunk_settings() == (10.0, 5.0)


def test_build_recognizer_reads_chunk_env(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WHISPER_CPP_CHUNK_SECONDS", "12")
    monkeypatch.setenv("WHISPER_CPP_CHUNK_OVERLAP", "1.5")
    config = _whisper_cpp_config(audio_file, tmp_path)

    recognizer = _build_recognizer(config, Device.CPU)

    assert isinstance(recognizer, WhisperCppRecognizer)
    assert recognizer.chunk_seconds == 12.0
    assert recognizer.chunk_overlap == 1.5


def test_diarization_cache_params_include_impl_version(
    audio_file: Path, tmp_path: Path
) -> None:
    config = _config(audio_file, tmp_path)

    params = _diarization_cache_params(config, Device.CPU, RecordingDiarizer())

    assert params["diarization_impl_version"] == DIARIZATION_IMPL_VERSION


def test_diarization_cache_key_changes_with_impl_version(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(audio_file, tmp_path)

    before = compute_cache_key(
        "diarization", audio_file, _diarization_cache_params(config, Device.CPU, RecordingDiarizer())
    )
    monkeypatch.setattr(
        "audio_transcriber.pipeline.DIARIZATION_IMPL_VERSION",
        DIARIZATION_IMPL_VERSION + 1,
    )
    after = compute_cache_key(
        "diarization", audio_file, _diarization_cache_params(config, Device.CPU, RecordingDiarizer())
    )

    assert before != after


def test_diarization_cache_params_include_hyperparameters(
    audio_file: Path, tmp_path: Path
) -> None:
    config = _config(
        audio_file,
        tmp_path,
        diarization_min_duration_off=0.7,
        diarization_clustering_threshold=0.6,
        diarization_clustering_fb=1.5,
        min_speakers=2,
        max_speakers=5,
    )

    params = _diarization_cache_params(config, Device.CPU, RecordingDiarizer())

    assert params["min_duration_off"] == pytest.approx(0.7)
    assert params["clustering_threshold"] == pytest.approx(0.6)
    assert params["clustering_fb"] == pytest.approx(1.5)
    assert params["min_speakers"] == 2
    assert params["max_speakers"] == 5


def test_diarization_cache_key_changes_with_min_duration_off(
    audio_file: Path, tmp_path: Path
) -> None:
    base = _config(audio_file, tmp_path, diarization_min_duration_off=0.0)
    changed = _config(audio_file, tmp_path, diarization_min_duration_off=0.5)

    before = compute_cache_key(
        "diarization", audio_file, _diarization_cache_params(base, Device.CPU, RecordingDiarizer())
    )
    after = compute_cache_key(
        "diarization",
        audio_file,
        _diarization_cache_params(changed, Device.CPU, RecordingDiarizer()),
    )

    assert before != after


def test_diarization_cache_key_changes_with_speaker_range(
    audio_file: Path, tmp_path: Path
) -> None:
    base = _config(audio_file, tmp_path)
    changed = _config(audio_file, tmp_path, min_speakers=2, max_speakers=4)

    before = compute_cache_key(
        "diarization", audio_file, _diarization_cache_params(base, Device.CPU, RecordingDiarizer())
    )
    after = compute_cache_key(
        "diarization",
        audio_file,
        _diarization_cache_params(changed, Device.CPU, RecordingDiarizer()),
    )

    assert before != after
