"""Декодирование аудиофайлов через PyAV.

pyannote.audio по умолчанию использует torchcodec для чтения аудио с диска,
который на многих системах (в частности Windows без полной установки
ffmpeg) не работает. PyAV уже используется faster-whisper и декодирует
аудио надёжно на всех платформах, поэтому диаризация получает готовый
waveform вместо пути к файлу.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

from audio_transcriber.utils.exceptions import AudioFileError

SAMPLE_RATE = 16000


class WaveformAccumulator:
    """Одномерный float32-буфер, собираемый потоково без двойной аллокации.

    ``np.concatenate`` над списком чанков держит в памяти одновременно список
    и результат (пик — двойная длина). Этот буфер предвыделяется по оценке
    длительности и растёт геометрически лишь при недооценке, поэтому память
    ограничена итоговой длиной, а не её удвоением. ``finish`` возвращает
    contiguous-срез без копирования.
    """

    def __init__(self, expected_samples: int = 0) -> None:
        self._data = np.empty(max(expected_samples, 1), dtype=np.float32)
        self._size = 0

    @property
    def size(self) -> int:
        """Число записанных сэмплов."""
        return self._size

    def append(self, samples: np.ndarray) -> None:
        """Добавляет моно-сэмплы, растит буфер при необходимости (геометрически)."""
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        count = int(arr.shape[0])
        if count == 0:
            return
        needed = self._size + count
        capacity = int(self._data.shape[0])
        if needed > capacity:
            grown = np.empty(max(needed, capacity * 2), dtype=np.float32)
            grown[: self._size] = self._data[: self._size]
            self._data = grown
        self._data[self._size : needed] = arr
        self._size = needed

    def finish(self) -> np.ndarray:
        """Возвращает накопленный waveform как contiguous-срез без копии."""
        return self._data[: self._size]


def estimate_sample_count(container: Any, stream: Any, sample_rate: int) -> int:
    """Оценивает число сэмплов потока по заголовку (без декодирования).

    Используется для предвыделения буфера. Даёт верхнюю оценку с запасом
    (1 секунда + 1 %), чтобы ресемплер не вышел за границы. ``0`` — длительность
    неизвестна, тогда буфер растёт геометрически.
    """

    duration: float | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration = container.duration / av.time_base
    if duration is None or duration <= 0:
        return 0
    estimate = duration * sample_rate
    return int(estimate) + sample_rate + int(estimate * 0.01)


@dataclass(frozen=True)
class AudioProbe:
    """Параметры первого аудиопотока, полученные без полного декодирования."""

    #: Имя контейнера PyAV (``wav``, ``mov,mp4,...`` и т.п.) или ``None``.
    format_name: str | None
    #: Имя кодека потока (``pcm_s16le``, ``aac`` и т.п.) или ``None``.
    codec: str | None
    sample_rate: int
    channels: int
    #: Длительность потока в секундах, если её удалось определить из заголовка.
    duration_seconds: float | None


def probe_audio(path: Path) -> AudioProbe:
    """Читает параметры первого аудиопотока, не декодируя сэмплы.

    Нужно, чтобы решить, можно ли отдать файл внешнему инструменту напрямую
    (whisper-cli читает WAV сам), не перекодируя его во временный WAV.
    """

    try:
        container = av.open(str(path))
    except Exception as exc:
        raise AudioFileError(f"Не удалось открыть аудиофайл {path}: {exc}") from exc

    try:
        try:
            stream = container.streams.audio[0]
        except IndexError as exc:
            raise AudioFileError(f"В файле {path} не найдена аудиодорожка") from exc

        codec_context = stream.codec_context
        # Длительность берём из заголовка (без декодирования): для WAV она
        # равна числу сэмплов, делённому на частоту, и совпадает с длиной
        # декодированного waveform.
        duration: float | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base

        return AudioProbe(
            format_name=container.format.name or None,
            codec=codec_context.name,
            sample_rate=int(codec_context.sample_rate or 0),
            channels=int(codec_context.channels or 0),
            duration_seconds=duration,
        )
    finally:
        container.close()


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
    accumulator = WaveformAccumulator(estimate_sample_count(container, stream, sample_rate))
    try:
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                accumulator.append(resampled.to_ndarray()[0])
        # Сбрасываем внутренний буфер ресемплера — иначе теряется «хвост»
        # потока (и всё аудио оказывается сдвинуто на задержку фильтра).
        # Такое же сбрасывание уже делают ``_iter_decoded_frames`` и
        # ``resample_waveform``; без него декодированная дорожка короче
        # исходной, и вычисленные по ней таймкоды не совпадают со шкалой
        # исходника (#14).
        for resampled in resampler.resample(None):
            accumulator.append(resampled.to_ndarray()[0])
    except Exception as exc:
        raise AudioFileError(f"Не удалось декодировать аудиофайл {path}: {exc}") from exc
    finally:
        container.close()

    if accumulator.size == 0:
        raise AudioFileError(f"Аудиофайл {path} не содержит звуковых данных")

    return accumulator.finish()


def resample_waveform(
    waveform: np.ndarray, *, source_rate: int, target_rate: int
) -> np.ndarray:
    """Меняет частоту дискретизации моно waveform float32 (без обращения к диску).

    Используется шумоподавлением: DeepFilterNet работает на 48 кГц, а конвейер
    распознавания и диаризации — на 16 кГц.
    """

    if source_rate == target_rate:
        return waveform

    if source_rate <= 0 or target_rate <= 0:
        raise AudioFileError("Частоты дискретизации должны быть положительными")

    try:
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(waveform, dtype=np.float32).reshape(1, -1),
            format="fltp",
            layout="mono",
        )
        frame.sample_rate = source_rate
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=target_rate
        )
        chunks: list[np.ndarray] = [
            resampled.to_ndarray() for resampled in (resampler.resample(frame) or ())
        ]
        # Сбрасываем внутренний буфер ресемплера, чтобы не потерять «хвост».
        chunks.extend(
            resampled.to_ndarray() for resampled in (resampler.resample(None) or ())
        )
    except Exception as exc:
        raise AudioFileError(f"Не удалось изменить частоту дискретизации: {exc}") from exc

    if not chunks:
        raise AudioFileError("Ресемплинг не вернул ни одного сэмпла")

    return np.concatenate(chunks, axis=1)[0]


def encode_pcm16(waveform: np.ndarray) -> bytes:
    """Кодирует моно waveform float32 в 16-битный PCM (little-endian).

    Масштаб 32768 и округление к ближайшему дают точный round-trip
    s16 → float32 → s16: PyAV декодирует s16 делением на 32768, поэтому
    обратное умножение на то же число восстанавливает исходные сэмплы без
    ошибки. Масштаб 32767 с усечением вносил бы разницу до 1 LSB, из-за
    которой whisper.cpp мог «срываться» и терять речь.
    """

    samples = np.clip(waveform, -1.0, 1.0) * 32768.0
    pcm = np.clip(np.rint(samples), -32768.0, 32767.0).astype(np.int16)
    return pcm.tobytes()


def write_wav(path: Path, waveform: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> None:
    """Записывает моно waveform float32 как 16-битный PCM WAV.

    Используется для передачи аудио внешним инструментам (whisper-cli),
    которые не декодируют все форматы (например, WebM), но принимают WAV.
    """

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(encode_pcm16(waveform))
