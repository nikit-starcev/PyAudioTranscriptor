"""Локальное проигрывание аудио-образцов без блокировки интерфейса.

Используется TUI, чтобы прослушать образец голоса выбранного говорящего.
Плеер выбирается из установленных в системе (ffplay / paplay / aplay / mpv /
afplay) и запускается отдельным процессом — интерфейс не блокируется. Если
плеера нет или запуск не удался, функция возвращает ``None``/``False``, а
вызывающий код показывает уведомление. Исключения наружу не пробрасываются.

Кроме запуска модуль умеет:
* :func:`read_duration` — длительность WAV (через stdlib :mod:`wave`);
* :func:`amplitude_envelope` — нормированную амплитуду (RMS по окнам) для
  визуализации «где звук» в интерфейсе;
* :class:`PlaybackHandle` — дескриптор запущенного процесса с ``elapsed``,
  ``is_running()`` и безопасным ``stop()``.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Поддерживаемые плееры в порядке предпочтения: (бинарник, аргументы).
_PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
    ("paplay", ()),
    ("aplay", ("-q",)),
    ("mpv", ("--no-video", "--really-quiet")),
    ("afplay", ()),
)

#: Порог RMS, ниже которого образец считается «тишина/нет голоса».
SILENCE_RMS_THRESHOLD = 0.01


def find_player() -> tuple[str, tuple[str, ...]] | None:
    """Возвращает первый доступный плеер (бинарник и аргументы) или ``None``."""
    for binary, options in _PLAYERS:
        if shutil.which(binary):
            return binary, options
    return None


def read_duration(path: Path) -> float:
    """Длительность WAV в секундах (``0.0`` для не-WAV и при ошибке).

    Читается только заголовок через stdlib :mod:`wave` — полное декодирование
    не выполняется, поэтому вызов дешёвый и его можно делать из GUI.
    """
    if Path(path).suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return 0.0
            return handle.getnframes() / rate
    except (OSError, EOFError, wave.Error) as exc:
        logger.debug("Не удалось прочитать длительность %s: %s", path, exc)
        return 0.0


def _read_wav_samples(path: Path) -> np.ndarray | None:
    """Читает моно float32 сэмплы из WAV через stdlib :mod:`wave` (или ``None``)."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        logger.debug("Не удалось прочитать WAV %s: %s", path, exc)
        return None

    if not frames:
        return None
    if width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    if channels > 1 and data.size >= channels:
        data = data[: data.size - data.size % channels].reshape(-1, channels).mean(axis=1)
    return data


def _load_samples(path: Path) -> np.ndarray | None:
    """Возвращает моно float32 сэмплы файла или ``None`` (мягкая деградация).

    Для WAV используется stdlib (быстро и без тяжёлых зависимостей), для прочих
    форматов — общий декодер :func:`audio_transcriber.utils.audio.load_waveform`.
    """
    if Path(path).suffix.lower() == ".wav":
        samples = _read_wav_samples(path)
        if samples is not None:
            return samples
    try:
        from audio_transcriber.utils.audio import load_waveform

        return np.asarray(load_waveform(Path(path)), dtype=np.float32).reshape(-1)
    except Exception as exc:  # noqa: BLE001 — любой сбой декодирования = нет данных
        logger.debug("Не удалось декодировать %s для амплитуды: %s", path, exc)
        return None


def amplitude_envelope(path: Path, columns: int = 50) -> list[float]:
    """Амплитуда (RMS по окнам) для визуализации «где звук»: список 0..1.

    Файл делится на ``columns`` равных окон, в каждом считается RMS, после чего
    значения нормируются к максимуму (пику) — так форма речи читается независимо
    от громкости записи. Если пик ниже :data:`SILENCE_RMS_THRESHOLD` (тишина/нет
    голоса), возвращаются все нули, а не «раздутая» нормировка. Пустой файл,
    ошибка декодирования или ``columns <= 0`` дают ``[]``.
    """
    if columns <= 0:
        return []
    samples = _load_samples(path)
    if samples is None or samples.size == 0:
        return []

    edges = np.linspace(0, samples.size, columns + 1).astype(np.int64)
    envelope: list[float] = []
    for index in range(columns):
        first, last = int(edges[index]), int(edges[index + 1])
        if last <= first:
            envelope.append(0.0)
            continue
        window = samples[first:last]
        envelope.append(float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))))

    peak = max(envelope)
    if peak < SILENCE_RMS_THRESHOLD:
        return [0.0] * columns
    return [min(1.0, value / peak) for value in envelope]


@dataclass
class PlaybackHandle:
    """Дескриптор запущенного процесса плеера с длительностью и таймингом."""

    process: subprocess.Popen[bytes] | None
    duration: float
    started_at: float

    @property
    def elapsed(self) -> float:
        """Сколько секунд прошло с момента запуска (не отрицательное)."""
        return max(0.0, time.monotonic() - self.started_at)

    def is_running(self) -> bool:
        """``True``, пока процесс плеера ещё жив."""
        process = self.process
        if process is None:
            return False
        try:
            return process.poll() is None
        except Exception:  # noqa: BLE001 — опрос процесса не должен ронять UI
            return False

    def stop(self) -> None:
        """Останавливает плеер (terminate → kill). Исключения не пробрасываются."""
        process = self.process
        if process is None:
            return
        with contextlib.suppress(Exception):  # плеер мог уже завершиться
            process.terminate()
        try:
            process.wait(timeout=2)
        except Exception:  # noqa: BLE001 — таймаут/сбой опроса не пробрасываем
            with contextlib.suppress(Exception):
                process.kill()


def start_playback(path: Path) -> PlaybackHandle | None:
    """Запускает неблокирующее проигрывание файла и возвращает дескриптор.

    ``None`` — плеер не найден или запуск не удался. Процесс отвязывается от
    TUI (``start_new_session``), его вывод глушится. В дескриптор кладётся
    длительность файла (для полосы прогресса).
    """
    player = find_player()
    if player is None:
        logger.warning("Аудио-плеер не найден — образец %s не проигран", path.name)
        return None
    binary, options = player
    try:
        process = subprocess.Popen(
            [binary, *options, str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        logger.warning("Не удалось запустить плеер %s: %s", binary, exc)
        return None
    return PlaybackHandle(
        process=process, duration=read_duration(path), started_at=time.monotonic()
    )


def play_audio_file(path: Path) -> bool:
    """Запускает неблокирующее проигрывание файла; ``False`` — не получилось.

    Тонкая обёртка над :func:`start_playback` для обратной совместимости.
    """
    return start_playback(path) is not None
