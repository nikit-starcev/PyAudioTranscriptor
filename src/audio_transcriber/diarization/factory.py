"""Выбор и создание движка диаризации (#62, #64).

Диаризацию можно выполнять на pyannote.audio (точно, без лимита говорящих, но
медленно на CPU) или на NeMo-Speech.cpp (``nemo-speech``, EEND Sortformer,
GPU через Vulkan, быстро, но модель рассчитана на 4 спикера).

Режим ``auto`` выбирает движок **по числу говорящих**:

* ``num_speakers``, иначе ``max_speakers``, иначе дешёвая оценка
  :func:`~audio_transcriber.diarization.speaker_count.estimate_speaker_count`
  (sherpa-onnx, секунды);
* ``N <= DIARIZATION_ROUTE_MAX_SPEAKERS`` → ``nemo-speech`` (если бинарник
  доступен) — быстро;
* ``N`` больше лимита → ``hybrid`` (оконный nemo-speech + глобальная склейка
  говорящих по эмбеддингам), если доступны бинарник и эмбеддер; иначе
  ``pyannote`` — точно;
* ``N`` неизвестно (``None``) → ``pyannote`` (безопасно);
* если предпочтительный движок недоступен, берётся другой; если недоступны
  оба — возвращается предпочтительный, а движок деградирует сам.

Явно заданные ``pyannote``/``nemo-speech``/``hybrid`` маршрутизацию не
проходят — поведение не меняется. Решение логируется и отдаётся в прогресс.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import DEFAULT_DIARIZATION_ENGINE
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization import embeddings as embedding_utils
from audio_transcriber.diarization.base import SpeakerDiarizer
from audio_transcriber.diarization.hybrid_engine import HybridSpeakerDiarizer
from audio_transcriber.diarization.nemo_speech_engine import NemoSpeechSpeakerDiarizer
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.diarization.speaker_count import estimate_speaker_count
from audio_transcriber.domain.enums import Device
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.env import binary_available

logger = logging.getLogger(__name__)

PYANNOTE_ENGINE = "pyannote"
NEMO_SPEECH_ENGINE = "nemo-speech"
HYBRID_ENGINE = "hybrid"


@dataclass(frozen=True, slots=True)
class DiarizationDecision:
    """Решение о движке диаризации: движок, число говорящих и причина."""

    engine: str
    speaker_count: int | None
    reason: str
    estimated: bool = False


def _module_available(name: str) -> bool:
    """Доступен ли модуль (без импорта — ``find_spec``), как в докторе."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def pyannote_available() -> bool:
    """Доступен ли движок pyannote.audio (модуль + torch)."""
    return _module_available("pyannote.audio") and _module_available("torch")


def nemo_speech_available(config: AppConfig) -> bool:
    """Доступен ли бинарник nemo-speech."""
    return binary_available(config.nemo_speech_binary)


def hybrid_available(config: AppConfig) -> bool:
    """Доступна ли гибридная диаризация (оконный EEND + эмбеддинги).

    Нужны: включённый флаг, бинарник ``nemo-speech``, ``sherpa-onnx`` и уже
    скачанная модель эмбеддингов. Проверка **не** скачивает модель: если её
    нет на диске, гибрид не выбирается (маршрут уходит на pyannote), но в
    режиме ``auto`` оценщик числа говорящих обычно скачивает её раньше.
    """
    if not config.diarization_hybrid_enabled:
        return False
    if not nemo_speech_available(config):
        return False
    return embedding_utils.embedder_available(config.diarization_estimate_model)


def _estimate(
    config: AppConfig,
    audio_path: Path | None,
    waveform: np.ndarray | None,
    on_progress: ProgressCallback | None,
) -> tuple[int | None, str]:
    """Оценивает ``N`` оценщиком; возвращает ``(N, источник)``."""
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                "diarization",
                message="Оценка числа говорящих",
                fraction=None,
                detail="sherpa-onnx",
            )
        )
    started = time.monotonic()
    count = estimate_speaker_count(
        audio_path,
        waveform=waveform,
        max_seconds=config.diarization_estimate_seconds,
        threshold=config.diarization_estimate_threshold,
        model=config.diarization_estimate_model,
        on_progress=on_progress,
    )
    elapsed = time.monotonic() - started
    if count is None:
        return None, f"оценка не удалась за {elapsed:.1f} с"
    return count, f"N≈{count} (оценка за {elapsed:.1f} с)"


def decide_diarization(
    config: AppConfig,
    *,
    audio_path: Path | None = None,
    waveform: np.ndarray | None = None,
    on_progress: ProgressCallback | None = None,
) -> DiarizationDecision:
    """Выбирает движок диаризации по настройкам и (для ``auto``) числу говорящих."""
    engine = (config.diarization_engine or DEFAULT_DIARIZATION_ENGINE).strip().casefold()
    if engine in (PYANNOTE_ENGINE, NEMO_SPEECH_ENGINE, HYBRID_ENGINE):
        return DiarizationDecision(
            engine=engine,
            speaker_count=config.num_speakers,
            reason="движок задан явно",
        )

    # --- auto: определяем N ---
    estimated = False
    if config.num_speakers is not None:
        count: int | None = config.num_speakers
        origin = "задано пользователем (num_speakers)"
    elif config.max_speakers is not None:
        count = config.max_speakers
        origin = "задано пользователем (max_speakers)"
    elif config.diarization_estimate_enabled:
        count, origin = _estimate(config, audio_path, waveform, on_progress)
        estimated = count is not None
    else:
        count = None
        origin = "оценка отключена"

    nemo_ok = nemo_speech_available(config)
    hybrid_ok = hybrid_available(config)
    pyannote_ok = pyannote_available()
    cap = config.diarization_route_max_speakers

    if count is None:
        if pyannote_ok:
            engine = PYANNOTE_ENGINE
            reason = f"число говорящих неизвестно ({origin}) — pyannote (безопасно)"
        elif nemo_ok:
            engine = NEMO_SPEECH_ENGINE
            reason = (
                f"число говорящих неизвестно ({origin}), pyannote недоступен — "
                "nemo-speech"
            )
        else:
            engine = PYANNOTE_ENGINE
            reason = f"число говорящих неизвестно ({origin}); доступных движков нет"
    elif count <= cap:
        if nemo_ok:
            engine = NEMO_SPEECH_ENGINE
            reason = f"{origin} ≤ {cap} — nemo-speech (быстро)"
        elif pyannote_ok:
            engine = PYANNOTE_ENGINE
            reason = f"{origin} ≤ {cap}, но nemo-speech недоступен — pyannote"
        else:
            engine = NEMO_SPEECH_ENGINE
            reason = f"{origin} ≤ {cap}; доступных движков нет — nemo-speech"
    else:
        if hybrid_ok:
            engine = HYBRID_ENGINE
            reason = (
                f"{origin} > {cap} — hybrid (быстро, обход лимита {cap} "
                "говорящих включён)"
            )
        elif pyannote_ok:
            engine = PYANNOTE_ENGINE
            reason = f"{origin} > {cap} — pyannote (точно)"
        elif nemo_ok:
            engine = NEMO_SPEECH_ENGINE
            reason = (
                f"{origin} > {cap}, pyannote/hybrid недоступны — nemo-speech "
                f"(лимит {cap} говорящих)"
            )
        else:
            engine = PYANNOTE_ENGINE
            reason = f"{origin} > {cap}; доступных движков нет"

    return DiarizationDecision(
        engine=engine,
        speaker_count=count,
        reason=reason,
        estimated=estimated,
    )


def resolve_diarization_engine(
    config: AppConfig,
    *,
    audio_path: Path | None = None,
    waveform: np.ndarray | None = None,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Разрешает эффективный движок диаризации с учётом режима ``auto``.

    Для ``auto`` может понадобиться оценка числа говорящих — тогда передайте
    ``audio_path``/``waveform``. Без аудио оценка даёт ``None``, и выбирается
    pyannote (безопасно).
    """
    return decide_diarization(
        config,
        audio_path=audio_path,
        waveform=waveform,
        on_progress=on_progress,
    ).engine


def create_diarizer(
    config: AppConfig,
    device: Device,
    *,
    audio_path: Path | None = None,
    waveform: np.ndarray | None = None,
    on_progress: ProgressCallback | None = None,
) -> SpeakerDiarizer:
    """Создаёт движок диаризации, выбранный настройками ``config``."""
    decision = decide_diarization(
        config,
        audio_path=audio_path,
        waveform=waveform,
        on_progress=on_progress,
    )
    logger.info("Движок диаризации: %s — %s", decision.engine, decision.reason)
    if on_progress is not None:
        on_progress(
            ProgressEvent(
                "diarization",
                message=f"Движок диаризации: {decision.engine}",
                fraction=None,
                detail=decision.reason,
            )
        )

    if decision.engine == NEMO_SPEECH_ENGINE:
        logger.info(
            "nemo-speech: модель %s, устройство %s",
            config.nemo_speech_model,
            config.nemo_speech_device,
        )
        return NemoSpeechSpeakerDiarizer(
            config.nemo_speech_device,
            binary=config.nemo_speech_binary,
            lib_path=config.nemo_speech_lib_path,
            model=config.nemo_speech_model,
            on_progress=on_progress,
        )
    if decision.engine == HYBRID_ENGINE:
        logger.info(
            "hybrid: окно %.1f с, перекрытие %.1f с, эмбеддер %s",
            config.diarization_hybrid_window_seconds,
            config.diarization_hybrid_overlap_seconds,
            config.diarization_estimate_model,
        )
        return HybridSpeakerDiarizer(
            config.nemo_speech_device,
            binary=config.nemo_speech_binary,
            lib_path=config.nemo_speech_lib_path,
            model=config.nemo_speech_model,
            on_progress=on_progress,
            window_seconds=config.diarization_hybrid_window_seconds,
            overlap_seconds=config.diarization_hybrid_overlap_seconds,
            min_speaker_seconds=config.diarization_hybrid_min_speaker_seconds,
            overload_split=config.diarization_hybrid_overload_split,
            subwindow_seconds=config.diarization_hybrid_subwindow_seconds,
            max_split_depth=config.diarization_hybrid_max_split_depth,
            embedding_model=config.diarization_estimate_model,
            threshold=config.diarization_hybrid_threshold,
            linkage=config.diarization_hybrid_linkage,
            # Мягкий ориентир передаём гибриду только когда число реально
            # оценено (``auto``). Явные ``num_speakers``/``min/max_speakers``
            # пользователь задаёт точно — они уходят в ``diarize`` из конвейера
            # и там применяются как жёсткое число/границы, а не как оценка.
            expected_speakers=decision.speaker_count if decision.estimated else None,
        )
    return PyannoteSpeakerDiarizer(
        device,
        hf_token=config.hf_token,
        local_model_path=config.pyannote_local_model,
        on_progress=on_progress,
        min_duration_off=config.diarization_min_duration_off,
        clustering_threshold=config.diarization_clustering_threshold,
        clustering_fb=config.diarization_clustering_fb,
    )
