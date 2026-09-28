"""Декодирование аудиофайлов через PyAV.

pyannote.audio по умолчанию использует torchcodec для чтения аудио с диска,
который на многих системах (в частности Windows без полной установки
ffmpeg) не работает. PyAV уже используется faster-whisper и декодирует
аудио надёжно на всех платформах, поэтому диаризация получает готовый
waveform вместо пути к файлу.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from audio_transcriber.utils.exceptions import AudioFileError

SAMPLE_RATE = 16000


def load_waveform(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Декодирует аудиофайл в моно waveform float32 с заданной частотой дискретизации."""

    try:
        container = av.open(str(path))
    except Exception as exc:
        raise AudioFileError(f"Не удалось открыть аудиофайл {path}: {exc}") from exc

    try:
        stream = container.streams.audio[0]
    except IndexError as exc:
        container.close()
        raise AudioFileError(f"В файле {path} не найдена аудиодорожка") from exc

    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(stream):
            chunks.extend(resampled.to_ndarray() for resampled in resampler.resample(frame))
    except Exception as exc:
        raise AudioFileError(f"Не удалось декодировать аудиофайл {path}: {exc}") from exc
    finally:
        container.close()

    if not chunks:
        raise AudioFileError(f"Аудиофайл {path} не содержит звуковых данных")

    return np.concatenate(chunks, axis=1)[0]


def write_wav(path: Path, waveform: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> None:
    """Записывает моно waveform float32 как 16-битный PCM WAV.

    Используется для передачи аудио внешним инструментам (whisper-cli),
    которые не декодируют все форматы (например, WebM), но принимают WAV.
    """

    import wave

    samples = np.clip(waveform, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
