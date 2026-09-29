"""Тесты компонента шумоподавления (без реальной модели DeepFilterNet).

Проверяется мягкая деградация (движок недоступен или аудио не декодируется —
конвейер не падает) и запись временного WAV частоты конвейера. Сама модель
не загружается: ``_ensure_model``/``_enhance`` подменяются заглушками.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from audio_transcriber.denoising import deepfilter as df_module
from audio_transcriber.denoising.deepfilter import DF_SAMPLE_RATE, DeepFilterDenoiser
from audio_transcriber.utils.audio import SAMPLE_RATE, write_wav


def _valid_wav(path: Path) -> Path:
    """Создаёт валидный моно WAV 48 кГц длительностью 1 секунда."""
    samples = np.zeros(DF_SAMPLE_RATE, dtype=np.float32)
    write_wav(path, samples, sample_rate=DF_SAMPLE_RATE)
    return path


def test_denoiser_returns_input_for_undecodable_audio(audio_file: Path) -> None:
    # Пустой файл не декодируется — денойз мягко пропускается, путь не меняется.
    with DeepFilterDenoiser() as denoiser:
        assert denoiser.denoise(audio_file) == audio_file


def test_denoiser_soft_skips_when_engine_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    valid = _valid_wav(tmp_path / "input.wav")
    monkeypatch.setattr(df_module, "_MODEL_CACHE", None)
    monkeypatch.setattr(df_module, "_MODEL_FAILED", False)
    monkeypatch.setattr(df_module, "_load_deepfilter", lambda: None)

    with DeepFilterDenoiser() as denoiser:
        # Движок недоступен — возвращаем исходный путь без исключения.
        assert denoiser.denoise(valid) == valid


def test_denoiser_writes_temp_wav_at_pipeline_rate(
    tmp_path: Path, monkeypatch
) -> None:
    valid = _valid_wav(tmp_path / "input.wav")
    # Модель «загружена» (заглушки), реальный DeepFilterNet не нужен.
    monkeypatch.setattr(df_module, "_MODEL_CACHE", (object(), object()))
    monkeypatch.setattr(df_module, "_MODEL_FAILED", False)

    def _fake_enhance(_self, _waveform, _model, _df_state):
        return np.zeros(DF_SAMPLE_RATE, dtype=np.float32)

    monkeypatch.setattr(DeepFilterDenoiser, "_enhance", _fake_enhance)

    denoiser = DeepFilterDenoiser()
    output = denoiser.denoise(valid)

    assert output != valid
    assert output.exists()
    with wave.open(str(output), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1

    denoiser.close()
    assert not output.exists()


def test_denoiser_close_is_idempotent() -> None:
    denoiser = DeepFilterDenoiser()
    denoiser.close()
    denoiser.close()  # повторный вызов не должен падать


def test_denoiser_context_manager_returns_self() -> None:
    with DeepFilterDenoiser() as denoiser:
        assert isinstance(denoiser, DeepFilterDenoiser)
