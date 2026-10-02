"""Выбор и создание движка диаризации (#62).

Диаризацию можно выполнять на pyannote.audio (по умолчанию) или на
NeMo-Speech.cpp (``nemo-speech``, EEND Sortformer, GPU через Vulkan). Режим
``auto`` предпочитает ``nemo-speech``, когда его бинарник доступен
(найден по пути или в ``PATH``), иначе откатывается к pyannote — существующий
путь не меняется.
"""

from __future__ import annotations

import logging

from audio_transcriber.config.defaults import DEFAULT_DIARIZATION_ENGINE
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.base import SpeakerDiarizer
from audio_transcriber.diarization.nemo_speech_engine import NemoSpeechSpeakerDiarizer
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device
from audio_transcriber.progress import ProgressCallback
from audio_transcriber.utils.env import binary_available

logger = logging.getLogger(__name__)

PYANNOTE_ENGINE = "pyannote"
NEMO_SPEECH_ENGINE = "nemo-speech"


def resolve_diarization_engine(config: AppConfig) -> str:
    """Разрешает эффективный движок диаризации с учётом режима ``auto``.

    ``pyannote``/``nemo-speech`` возвращаются как есть. Для ``auto`` выбирается
    ``nemo-speech``, если его бинарник доступен, иначе ``pyannote``.
    """
    engine = (config.diarization_engine or DEFAULT_DIARIZATION_ENGINE).strip().casefold()
    if engine in (PYANNOTE_ENGINE, NEMO_SPEECH_ENGINE):
        return engine
    if binary_available(config.nemo_speech_binary):
        return NEMO_SPEECH_ENGINE
    return PYANNOTE_ENGINE


def create_diarizer(
    config: AppConfig,
    device: Device,
    *,
    on_progress: ProgressCallback | None = None,
) -> SpeakerDiarizer:
    """Создаёт движок диаризации, выбранный настройками ``config``."""
    engine = resolve_diarization_engine(config)
    if engine == NEMO_SPEECH_ENGINE:
        logger.info(
            "Движок диаризации: nemo-speech (модель %s, устройство %s)",
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
    logger.info("Движок диаризации: pyannote.audio")
    return PyannoteSpeakerDiarizer(
        device,
        hf_token=config.hf_token,
        local_model_path=config.pyannote_local_model,
        on_progress=on_progress,
        min_duration_off=config.diarization_min_duration_off,
        clustering_threshold=config.diarization_clustering_threshold,
        clustering_fb=config.diarization_clustering_fb,
    )
