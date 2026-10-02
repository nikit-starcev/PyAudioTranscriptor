"""Тесты компонента шумоподавления (без реальной модели DeepFilterNet).

Проверяется мягкая деградация (движок недоступен или аудио не декодируется —
конвейер не падает), запись временного WAV частоты конвейера, а также
потоковая чанковая обработка: overlap-add без пропусков/дублей, синхронность
``last_waveform`` с записанным WAV и ограниченный пик памяти. Сама модель не
загружается: ``_ensure_model``/``_enhance`` подменяются заглушками.
"""

from __future__ import annotations

import tracemalloc
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.denoising import deepfilter as df_module
from audio_transcriber.denoising.deepfilter import DF_SAMPLE_RATE, DeepFilterDenoiser
from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    encode_pcm16,
    load_waveform,
    resample_waveform,
    write_wav,
)


def _valid_wav(path: Path) -> Path:
    """Создаёт валидный моно WAV 48 кГц длительностью 1 секунда."""
    samples = np.zeros(DF_SAMPLE_RATE, dtype=np.float32)
    write_wav(path, samples, sample_rate=DF_SAMPLE_RATE)
    return path


def _write_signal(path: Path, seconds: float, *, rate: int, seed: int = 0) -> np.ndarray:
    """Пишет детерминированный моно-сигнал и возвращает его float32-версию."""
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(round(seconds * rate)) * 0.2).astype(np.float32)
    write_wav(path, samples, sample_rate=rate)
    return samples


def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Помещает в кэш модели заглушки, чтобы реальный DeepFilterNet не грузился."""
    monkeypatch.setattr(df_module, "_MODEL_CACHE", (object(), object()))
    monkeypatch.setattr(df_module, "_MODEL_FAILED", False)


def _identity_enhance(_self, waveform, _model, _df_state):
    """Денойз-заглушка: возвращает вход без изменений (проверяем склейку, не модель)."""
    return np.array(waveform, dtype=np.float32, copy=True)


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


def test_denoiser_rejects_overlap_not_smaller_than_half_chunk() -> None:
    with pytest.raises(ValueError):
        DeepFilterDenoiser(chunk_seconds=1.0, overlap_seconds=0.6)


def test_chunked_denoise_reconstructs_full_signal_without_gaps_or_duplicates(
    tmp_path: Path, monkeypatch
) -> None:
    """Overlap-add на 48 кГц: идентичный денойз даёт ровно входной сигнал.

    Длина и сэмплы совпадают точно, значит нет ни пропусков, ни дублей на стыках.
    """
    _write_signal(tmp_path / "input.wav", 5.5, rate=DF_SAMPLE_RATE, seed=1)
    decoded = load_waveform(tmp_path / "input.wav", sample_rate=DF_SAMPLE_RATE)
    _fake_model(monkeypatch)
    monkeypatch.setattr(DeepFilterDenoiser, "_enhance", _identity_enhance)

    denoiser = DeepFilterDenoiser(
        output_sample_rate=DF_SAMPLE_RATE, chunk_seconds=1.0, overlap_seconds=0.25
    )
    output = denoiser.denoise(tmp_path / "input.wav")
    result = denoiser.last_waveform
    assert result is not None

    assert output != tmp_path / "input.wav"
    assert result.shape[0] == decoded.shape[0]
    # Разница только машинная (float32-округление кроссфейда).
    assert np.max(np.abs(result - decoded)) < 1e-6


def test_chunked_denoise_matches_full_file_resample(
    tmp_path: Path, monkeypatch
) -> None:
    """Чанкованный конвейер 48 → 16 кГц эквивалентен полнофайловому ресемплингу."""
    input_path = tmp_path / "input.wav"
    _write_signal(input_path, 4.3, rate=DF_SAMPLE_RATE, seed=2)
    decoded = load_waveform(input_path, sample_rate=DF_SAMPLE_RATE)
    _fake_model(monkeypatch)
    monkeypatch.setattr(DeepFilterDenoiser, "_enhance", _identity_enhance)

    denoiser = DeepFilterDenoiser(chunk_seconds=1.0, overlap_seconds=0.25)
    output = denoiser.denoise(input_path)
    result = denoiser.last_waveform
    assert result is not None
    # WAV на диске — та же 16-кГц дорожка.
    on_disk = load_waveform(output)
    denoiser.close()

    full_file = resample_waveform(
        decoded, source_rate=DF_SAMPLE_RATE, target_rate=SAMPLE_RATE
    )
    assert result.shape[0] == full_file.shape[0]
    assert np.max(np.abs(result - full_file)) < 1e-4
    assert on_disk.shape[0] == full_file.shape[0]


def test_last_waveform_matches_written_wav(tmp_path: Path, monkeypatch) -> None:
    """``last_waveform`` синхронен записанному WAV (побитово, до s16-квантования)."""
    input_path = tmp_path / "input.wav"
    _write_signal(input_path, 3.2, rate=DF_SAMPLE_RATE, seed=3)
    _fake_model(monkeypatch)
    monkeypatch.setattr(DeepFilterDenoiser, "_enhance", _identity_enhance)

    denoiser = DeepFilterDenoiser(chunk_seconds=1.0, overlap_seconds=0.25)
    output = denoiser.denoise(input_path)
    result = denoiser.last_waveform
    assert result is not None

    with wave.open(str(output), "rb") as wf:
        written = wf.readframes(wf.getnframes())
    denoiser.close()

    # encode_pcm16 — поточечная, поэтому WAV побитово равен кодировке всего массива.
    assert written == encode_pcm16(result)


def test_denoise_peak_memory_bounded_by_chunk(tmp_path: Path, monkeypatch) -> None:
    """Пик памяти денойза не растёт с длиной файла, а ограничен чанком + 16 кГц.

    Синтетический вход длиннее многих чанков: если бы файл грузился целиком в
    48 кГц, пик был бы как минимум размером полного 48-кГц массива (и с запасом
    больше из-за копий). Здесь проверяем, что пик заметно ниже.
    """
    seconds = 120.0
    _write_signal(tmp_path / "input.wav", seconds, rate=SAMPLE_RATE, seed=4)
    _fake_model(monkeypatch)
    monkeypatch.setattr(DeepFilterDenoiser, "_enhance", _identity_enhance)

    denoiser = DeepFilterDenoiser(chunk_seconds=5.0, overlap_seconds=0.5)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        denoiser.denoise(tmp_path / "input.wav")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    denoiser.close()

    full_48k_bytes = int(seconds * DF_SAMPLE_RATE * 4)
    out_16k_bytes = int(seconds * SAMPLE_RATE * 4)
    chunk_bytes = int(5.0 * DF_SAMPLE_RATE * 4)

    assert peak < full_48k_bytes
    # С запасом на 16-кГц результат + несколько чанков + накладные расходы.
    assert peak < out_16k_bytes + 6 * chunk_bytes + 5_000_000
