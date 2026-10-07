"""Тесты компонента шумоподавления (внешний CLI ``deep-filter``, без реального бинарника).

Проверяется мягкая деградация (бинарник не найден, аудио не декодируется,
процесс падает — конвейер не падает), точный вызов CLI, запись временного WAV
частоты конвейера, а также потоковая чанковая обработка: overlap-add без
пропусков/дублей, синхронность ``last_waveform`` с записанным WAV и
ограниченный пик памяти. Реальный процесс не запускается: ``subprocess.Popen``
подменяется заглушкой, которая копирует вход в выход (identity-денойз).
"""

from __future__ import annotations

import shutil
import tracemalloc
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.cache.denoiser import CachingDenoiser
from audio_transcriber.cache.store import StageCache
from audio_transcriber.denoising import deepfilter as df_module
from audio_transcriber.denoising.deepfilter import (
    DENOISE_IMPL_VERSION,
    DF_SAMPLE_RATE,
    DeepFilterDenoiser,
)
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    encode_pcm16,
    load_waveform,
    resample_waveform,
    write_wav,
)


def _fake_binary(tmp_path: Path) -> Path:
    """Создаёт файл-заглушку «бинарника»: важно лишь, что путь существует."""
    path = tmp_path / "deep-filter"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return path


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


class _FakeProcess:
    """Минимальная замена ``subprocess.Popen`` для тестов денойза."""

    def __init__(self, command: list[str], *, returncode: int = 0, stderr: str = "") -> None:
        self.command = command
        self.pid = 987654
        self.returncode = returncode
        self._stderr = stderr

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        return "", self._stderr

    def poll(self) -> int:
        return self.returncode


def _install_fake_popen(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stderr: str = "",
    copy_output: bool = True,
) -> list[list[str]]:
    """Подменяет ``Popen``: собирает команды и (по желанию) копирует вход в выход."""
    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        commands.append(command)
        if copy_output and returncode == 0:
            out_index = command.index("-o")
            input_wav = Path(command[out_index - 1])
            output_dir = Path(command[out_index + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_wav, output_dir / input_wav.name)
        return _FakeProcess(command, returncode=returncode, stderr=stderr)

    monkeypatch.setattr(df_module.subprocess, "Popen", fake_popen)
    return commands


def test_resolve_binary_finds_file_and_path(tmp_path: Path) -> None:
    fake = _fake_binary(tmp_path)
    assert df_module._resolve_binary(str(fake)) == str(fake)
    assert df_module._resolve_binary("definitely-no-such-binary-xyz") is None
    assert df_module._resolve_binary("") is None


def test_denoiser_soft_skips_when_binary_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = _valid_wav(tmp_path / "input.wav")
    monkeypatch.setattr(df_module, "_resolve_binary", lambda _binary: None)

    with DeepFilterDenoiser(binary="deep-filter") as denoiser:
        # Бинарник не найден — возвращаем исходный путь без исключения.
        assert denoiser.denoise(valid) == valid


def test_denoiser_returns_input_for_undecodable_audio(tmp_path: Path) -> None:
    # Пустой файл не декодируется — денойз мягко пропускается, путь не меняется.
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"")
    binary = _fake_binary(tmp_path)
    with DeepFilterDenoiser(binary=str(binary)) as denoiser:
        assert denoiser.denoise(audio_file) == audio_file


def test_denoiser_soft_skips_when_process_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = _valid_wav(tmp_path / "input.wav")
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch, returncode=1, stderr="boom", copy_output=False)

    with DeepFilterDenoiser(binary=str(binary)) as denoiser:
        assert denoiser.denoise(valid) == valid


def test_transient_process_failure_does_not_disable_denoise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Транзиентный сбой ``deep-filter`` не отключает денойз навсегда (#91).

    После разового сбоя (мягкая деградация → исходный путь) следующий вызов
    должен снова запускать CLI и получать результат: неудача не кэшируется как
    постоянная недоступность.
    """
    valid = _valid_wav(tmp_path / "input.wav")
    binary = _fake_binary(tmp_path)
    calls = {"count": 0}

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        calls["count"] += 1
        out_index = command.index("-o")
        input_wav = Path(command[out_index - 1])
        output_dir = Path(command[out_index + 1])
        if calls["count"] == 1:
            return _FakeProcess(command, returncode=1, stderr="transient")
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_wav, output_dir / input_wav.name)
        return _FakeProcess(command)

    monkeypatch.setattr(df_module.subprocess, "Popen", fake_popen)

    with DeepFilterDenoiser(binary=str(binary)) as denoiser:
        first = denoiser.denoise(valid)
        assert first == valid  # сбой — мягкая деградация, без исключения
        second = denoiser.denoise(valid)
        assert second != valid  # повторный запуск не отключён
        assert second.exists()
    assert calls["count"] == 2


def test_denoiser_invokes_cli_with_expected_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_wav = _valid_wav(tmp_path / "input.wav")
    binary = _fake_binary(tmp_path)
    commands = _install_fake_popen(monkeypatch)

    denoiser = DeepFilterDenoiser(binary=str(binary))
    output = denoiser.denoise(input_wav)

    assert output != input_wav
    assert len(commands) == 1
    command = commands[0]
    assert command[0] == str(binary)
    # ``-D`` обязателен: компенсация задержки STFT/модели (~30 мс).
    assert command[1] == "-D"
    assert command[2].endswith(".wav")
    assert command[3] == "-o"
    assert Path(command[4]).is_dir()
    denoiser.close()


def test_enhance_pads_shorter_cli_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``-D`` отдаёт выход на ~30 мс короче; недостающее добивается нулями."""
    binary = _fake_binary(tmp_path)
    cut = DF_SAMPLE_RATE * 30 // 1000  # задержка STFT/модели на 48 кГц

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        out_index = command.index("-o")
        input_wav = Path(command[out_index - 1])
        output_dir = Path(command[out_index + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        with wave.open(str(input_wav), "rb") as wf:
            rate, data = wf.getframerate(), wf.readframes(wf.getnframes())
        with wave.open(str(output_dir / input_wav.name), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(data[: len(data) - cut * 2])
        return _FakeProcess(command)

    monkeypatch.setattr(df_module.subprocess, "Popen", fake_popen)

    rng = np.random.default_rng(0)
    segment = (rng.standard_normal(DF_SAMPLE_RATE) * 0.1).astype(np.float32)
    denoiser = DeepFilterDenoiser(binary=str(binary))
    try:
        result = denoiser._enhance(segment, str(binary))
    finally:
        denoiser.close()

    assert result.shape[0] == segment.shape[0]
    assert np.all(result[-cut:] == 0.0)


def test_denoiser_writes_temp_wav_at_pipeline_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = _valid_wav(tmp_path / "input.wav")
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch)

    denoiser = DeepFilterDenoiser(binary=str(binary))
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


def test_denoiser_rejects_empty_binary() -> None:
    denoiser = DeepFilterDenoiser(binary="   ")
    assert denoiser.binary == ""


def test_chunked_denoise_reconstructs_full_signal_without_gaps_or_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlap-add на 48 кГц: идентичный денойз даёт ровно входной сигнал.

    Длина и сэмплы совпадают точно, значит нет ни пропусков, ни дублей на стыках.
    """
    _write_signal(tmp_path / "input.wav", 5.5, rate=DF_SAMPLE_RATE, seed=1)
    decoded = load_waveform(tmp_path / "input.wav", sample_rate=DF_SAMPLE_RATE)
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch)

    denoiser = DeepFilterDenoiser(
        binary=str(binary),
        output_sample_rate=DF_SAMPLE_RATE,
        chunk_seconds=1.0,
        overlap_seconds=0.25,
    )
    output = denoiser.denoise(tmp_path / "input.wav")
    result = denoiser.last_waveform
    assert result is not None

    assert output != tmp_path / "input.wav"
    assert result.shape[0] == decoded.shape[0]
    # Разница только машинная (float32-округление кроссфейда).
    assert np.max(np.abs(result - decoded)) < 1e-6


def test_chunked_denoise_matches_full_file_resample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Чанкованный конвейер 48 → 16 кГц эквивалентен полнофайловому ресемплингу."""
    input_path = tmp_path / "input.wav"
    _write_signal(input_path, 4.3, rate=DF_SAMPLE_RATE, seed=2)
    decoded = load_waveform(input_path, sample_rate=DF_SAMPLE_RATE)
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch)

    denoiser = DeepFilterDenoiser(
        binary=str(binary), chunk_seconds=1.0, overlap_seconds=0.25
    )
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


def test_last_waveform_matches_written_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``last_waveform`` синхронен записанному WAV (побитово, до s16-квантования)."""
    input_path = tmp_path / "input.wav"
    _write_signal(input_path, 3.2, rate=DF_SAMPLE_RATE, seed=3)
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch)

    denoiser = DeepFilterDenoiser(
        binary=str(binary), chunk_seconds=1.0, overlap_seconds=0.25
    )
    output = denoiser.denoise(input_path)
    result = denoiser.last_waveform
    assert result is not None

    with wave.open(str(output), "rb") as wf:
        written = wf.readframes(wf.getnframes())
    denoiser.close()

    # encode_pcm16 — поточечная, поэтому WAV побитово равен кодировке всего массива.
    assert written == encode_pcm16(result)


def test_denoiser_emits_chunked_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Длинный денойз эмитит растущую долю обработанного аудио и финальные 100 %.

    Раньше стадия отдавала одно событие ``fraction=None`` и молчала минуты —
    теперь по каждому чанку приходит реальный процент (#см. баг «статусы не
    приходят»).
    """
    input_path = tmp_path / "input.wav"
    _write_signal(input_path, 6.0, rate=DF_SAMPLE_RATE, seed=5)
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch)

    events: list[ProgressEvent] = []
    denoiser = DeepFilterDenoiser(
        binary=str(binary),
        chunk_seconds=1.0,
        overlap_seconds=0.25,
        on_progress=events.append,
    )
    denoiser.denoise(input_path)
    denoiser.close()

    denoise_events = [event for event in events if event.stage == "denoise"]
    # Больше одного чанка → несколько промежуточных событий, затем 100 %.
    assert len(denoise_events) > 1
    fractions = [event.fraction for event in denoise_events]
    assert all(fraction is not None for fraction in fractions)
    assert fractions == sorted(fractions)
    assert fractions[0] is not None and 0.0 < fractions[0] < 1.0
    assert fractions[-1] == 1.0


def test_denoise_peak_memory_bounded_by_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пик памяти денойза не растёт с длиной файла, а ограничен чанком + 16 кГц.

    Синтетический вход длиннее многих чанков: если бы файл грузился целиком в
    48 кГц, пик был бы как минимум размером полного 48-кГц массива (и с запасом
    больше из-за копий). Здесь проверяем, что пик заметно ниже.
    """
    seconds = 120.0
    _write_signal(tmp_path / "input.wav", seconds, rate=SAMPLE_RATE, seed=4)
    binary = _fake_binary(tmp_path)
    _install_fake_popen(monkeypatch)

    denoiser = DeepFilterDenoiser(
        binary=str(binary), chunk_seconds=5.0, overlap_seconds=0.5
    )
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


def test_caching_denoiser_key_includes_binary_and_impl_version(
    tmp_path: Path, audio_file: Path
) -> None:
    """Ключ кэша денойза учитывает бинарник и версию реализации (#50)."""
    cache = StageCache(tmp_path / "cache", enabled=True)
    inner = DeepFilterDenoiser(binary="deep-filter")
    wrapper = CachingDenoiser(inner, cache, source=audio_file)

    params = wrapper._params()
    assert params["binary"] == "deep-filter"
    assert params["impl_version"] == DENOISE_IMPL_VERSION
    assert params["chunk_seconds"] == df_module.DEFAULT_CHUNK_SECONDS
    assert params["overlap_seconds"] == df_module.DEFAULT_OVERLAP_SECONDS
    assert wrapper.cache_key is None

    other = DeepFilterDenoiser(binary="other-deep-filter")
    assert other.binary != inner.binary
