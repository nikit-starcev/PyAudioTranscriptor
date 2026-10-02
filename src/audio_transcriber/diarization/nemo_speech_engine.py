"""Реализация диаризации на NeMo-Speech.cpp (``nemo-speech``).

Движок оборачивает внешний бинарник ``nemo-speech diarize`` (модель
Sortformer streaming, EEND, 4 спикера) и приводит его вывод в формате RTTM к
доменным :class:`~audio_transcriber.domain.models.SpeakerSegment`. Публичный
контракт совпадает с :class:`~audio_transcriber.diarization.pyannote_engine.PyannoteSpeakerDiarizer`:
вход — путь к аудио и/или готовый 16-кГц моно waveform, выход — сегменты
говорящих и интервалы наложения речи.

Особенности движка:

* ``nemo-speech`` читает только WAV, поэтому waveform (переданный или
  декодированный) записывается во временный 16-кГц моно WAV.
* Каталог ``lib/`` бандла подмешивается в ``LD_LIBRARY_PATH`` **только если в
  нём нет** ``libstdc++.so.6``/``libgcc_s.so.1``: бандл-версии затеняют
  системные и ломают загрузку Vulkan-ICD на Radeon. У самого бинарника есть
  RUNPATH ``$ORIGIN/../lib``, поэтому отсутствие ``LD_LIBRARY_PATH`` не мешает.
* У EEND-модели **нет** per-speaker эмбеддингов, но enrollment по образцам
  от движка диаризации не зависит: конвейер использует отдельный embedding-
  движок, поэтому имена по образцам присваиваются и с nemo-speech.
* Ошибки (нет бинарника, сбой subprocess, битый вывод) приводят к мягкой
  деградации: предупреждение в лог и пустой список сегментов — конвейер
  продолжается без разметки говорящих.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_NEMO_SPEECH_BINARY,
    DEFAULT_NEMO_SPEECH_DEVICE,
    DEFAULT_NEMO_SPEECH_MODEL,
    NEMO_SPEECH_FORBIDDEN_LIBS,
    NEMO_SPEECH_MAX_SPEAKERS,
)
from audio_transcriber.diarization.overlap import compute_overlap_regions
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import load_waveform, write_wav
from audio_transcriber.utils.env import (
    binary_available,
    effective_library_path,
    with_library_path,
)

logger = logging.getLogger(__name__)

#: Таймаут одного вызова ``nemo-speech diarize`` (секунды). С запасом покрывает
#: длинные записи на CPU, но не даёт процессу висеть бесконечно.
DEFAULT_NEMO_SPEECH_TIMEOUT = 3600.0

#: Формат RTTM: минимум полей до имени говорящего включительно.
_RTTM_MIN_FIELDS = 8
#: Индексы полей RTTM (0-based): тип, файл, канал, начало, длительность, ..., имя.
_RTTM_FIELD_START = 3
_RTTM_FIELD_DURATION = 4
_RTTM_FIELD_SPEAKER = 7


def _effective_library_path(lib_path: str | None) -> str | None:
    """Каталог библиотек nemo-speech без затенения системных libstdc++/libgcc_s."""
    return effective_library_path(lib_path, forbidden_libs=NEMO_SPEECH_FORBIDDEN_LIBS)


def normalize_speaker_label(raw: str) -> str:
    """Приводит метку говорящего nemo-speech к формату конвейера.

    ``nemo-speech``/Sortformer нумерует спикеров с единицы (``speaker_1``),
    а конвейер (как и pyannote, ``--speaker-name 0=...``) ожидает
    ``SPEAKER_00``. Уже нормализованные метки (``SPEAKER_00``) не трогаем.
    """
    value = raw.strip()
    # Сопоставляем строго нижний регистр (формат nemo-speech), чтобы уже
    # нормализованные метки вида ``SPEAKER_00`` не сдвигались повторно.
    if value.startswith("speaker_") and value[8:].isdigit():
        index = int(value[8:])
        if index >= 1:
            return f"SPEAKER_{index - 1:02d}"
    return value


def parse_rttm(
    text: str,
    *,
    recording_id: str | None = None,
) -> list[SpeakerSegment]:
    """Разбирает вывод ``nemo-speech diarize --format rttm`` в сегменты.

    Формат строки::

        SPEAKER <rec> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>

    Пустые строки, комментарии и строки других типов пропускаются. Некорректные
    числовые поля в строке ``SPEAKER`` также пропускаются (частичный вывод не
    роняет разбор). ``recording_id`` — при задании оставляет только строки этого
    файла (полезно при пакетном выводе).
    """
    segments: list[SpeakerSegment] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if not fields or fields[0] != "SPEAKER" or len(fields) < _RTTM_MIN_FIELDS:
            continue
        if recording_id is not None and fields[1] != recording_id:
            continue
        speaker = fields[_RTTM_FIELD_SPEAKER].strip()
        if not speaker or speaker == "<NA>":
            continue
        try:
            start = float(fields[_RTTM_FIELD_START])
            duration = float(fields[_RTTM_FIELD_DURATION])
        except (TypeError, ValueError):
            logger.debug("nemo-speech: пропущена некорректная RTTM-строка: %r", stripped)
            continue
        end = start + duration
        if duration <= 0.0 or end <= start:
            continue
        segments.append(
            SpeakerSegment(
                start=start,
                end=end,
                speaker_id=normalize_speaker_label(speaker),
            )
        )
    return segments


def _diarize_command(
    wav_path: Path, *, binary: str, device: str, model: str
) -> list[str]:
    """Команда запуска ``nemo-speech diarize`` в формате RTTM."""
    return [
        binary,
        "diarize",
        str(wav_path),
        "--model",
        model,
        "--device",
        device,
        "--format",
        "rttm",
    ]


def _prepare_wav(audio_path: Path, waveform: np.ndarray | None) -> Path:
    """Готовит 16-кГц моно WAV для ``nemo-speech``.

    Возвращает путь к временному файлу; вызывающий обязан его удалить.
    Переданный waveform переиспользуется — повторного декодирования нет;
    при ``None`` файл декодируется через :func:`load_waveform`.
    """
    if waveform is None:
        waveform = load_waveform(audio_path)
    fd, raw_path = tempfile.mkstemp(prefix="nemo-speech-", suffix=".wav")
    os.close(fd)
    wav_path = Path(raw_path)
    write_wav(wav_path, waveform)
    return wav_path


def diarize_audio(
    audio_path: Path,
    *,
    binary: str = DEFAULT_NEMO_SPEECH_BINARY,
    device: str = DEFAULT_NEMO_SPEECH_DEVICE,
    model: str = DEFAULT_NEMO_SPEECH_MODEL,
    lib_path: str | None = None,
    timeout: float = DEFAULT_NEMO_SPEECH_TIMEOUT,
    waveform: np.ndarray | None = None,
) -> list[SpeakerSegment] | None:
    """Готовит 16-кГц моно WAV, запускает ``nemo-speech diarize`` и парсит RTTM.

    Переиспользуемая обёртка: движок :class:`NemoSpeechSpeakerDiarizer` и
    гибридная диаризация (#64) запускают один и тот же бинарник с одинаковым
    окружением. Возвращает список сегментов либо ``None`` при любой ошибке
    (нет бинарника, сбой subprocess, битый вывод) — вызывающий сам решает, как
    деградировать.
    """
    if not binary_available(binary):
        logger.warning(
            "Бинарник nemo-speech не найден (%r) — диаризация пропущена. "
            "Задайте NEMO_SPEECH_BINARY/--nemo-speech-binary.",
            binary,
        )
        return None

    wav_path: Path | None = None
    try:
        try:
            wav_path = _prepare_wav(audio_path, waveform)
        except Exception as exc:  # noqa: BLE001 — мягкая деградация
            logger.warning(
                "nemo-speech: не удалось подготовить аудио %s: %s — диаризация пропущена",
                audio_path,
                exc,
            )
            return None

        env = with_library_path(os.environ, _effective_library_path(lib_path))
        try:
            proc = subprocess.run(
                _diarize_command(wav_path, binary=binary, device=device, model=model),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "nemo-speech: не удалось запустить диаризацию (%s) — пропущена",
                exc,
            )
            return None

        if proc.returncode != 0:
            logger.warning(
                "nemo-speech завершился с кодом %d — диаризация пропущена. stderr: %s",
                proc.returncode,
                (proc.stderr or "").strip()[-500:],
            )
            return None

        return parse_rttm(proc.stdout or "")
    finally:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Не удалось удалить временный WAV %s", wav_path)


class NemoSpeechSpeakerDiarizer:
    """Диаризация через NeMo-Speech.cpp. Реализует протокол ``SpeakerDiarizer``.

    :param device: устройство ``nemo-speech`` (``auto``/``vulkan``/``cpu``).
    :param binary: путь или имя бинарника ``nemo-speech``.
    :param lib_path: каталог разделяемых библиотек бандла (``lib/``).
    :param model: имя модели каталога, HF-репозиторий или путь к ``.gguf``.
    :param on_progress: приёмник событий прогресса (мягко необязателен).
    :param timeout: таймаут одного вызова subprocess (секунды).
    """

    def __init__(
        self,
        device: str = DEFAULT_NEMO_SPEECH_DEVICE,
        *,
        binary: str = DEFAULT_NEMO_SPEECH_BINARY,
        lib_path: str | None = None,
        model: str = DEFAULT_NEMO_SPEECH_MODEL,
        on_progress: ProgressCallback | None = None,
        timeout: float = DEFAULT_NEMO_SPEECH_TIMEOUT,
    ) -> None:
        self._device = device
        self._binary = binary
        self._lib_path = lib_path
        self._model = model
        self._on_progress = on_progress
        self._timeout = timeout
        self._overlaps: list[SpeakerOverlap] = []

    def _emit(self, message: str, *, fraction: float | None, detail: str = "") -> None:
        emit = self._on_progress
        if emit is not None:
            emit(
                ProgressEvent(
                    "diarization",
                    message=message,
                    fraction=fraction,
                    detail=detail,
                )
            )

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: np.ndarray | None = None,
    ) -> list[SpeakerSegment]:
        self._overlaps = []

        for value in (num_speakers, min_speakers, max_speakers):
            if value is not None and value > NEMO_SPEECH_MAX_SPEAKERS:
                logger.warning(
                    "nemo-speech (Sortformer) поддерживает не более %d спикеров, "
                    "но запрошено %d — результат может содержать не более %d говорящих",
                    NEMO_SPEECH_MAX_SPEAKERS,
                    value,
                    NEMO_SPEECH_MAX_SPEAKERS,
                )

        self._emit("nemo-speech", fraction=None)
        segments = diarize_audio(
            audio_path,
            binary=self._binary,
            device=self._device,
            model=self._model,
            lib_path=self._lib_path,
            timeout=self._timeout,
            waveform=waveform,
        )
        if segments is None:
            return []

        self._overlaps = compute_overlap_regions(segments)
        self._emit("nemo-speech", fraction=1.0)
        return segments

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """Интервалы наложения речи из последнего вызова :meth:`diarize`."""
        return list(self._overlaps)
