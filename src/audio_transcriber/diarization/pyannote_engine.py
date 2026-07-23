"""Реализация диаризации на pyannote.audio."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.exceptions import DiarizationError

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"


class PyannoteSpeakerDiarizer:
    """Определяет говорящих через pyannote.audio. Реализует протокол ``SpeakerDiarizer``."""

    def __init__(
        self,
        device: Device,
        *,
        pipeline_name: str = DEFAULT_PIPELINE,
        hf_token: str | None = None,
    ) -> None:
        self._device = device
        self._pipeline_name = pipeline_name
        self._hf_token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from pyannote.audio import Pipeline

        logger.debug("Загрузка модели диаризации '%s'", self._pipeline_name)
        try:
            pipeline = Pipeline.from_pretrained(self._pipeline_name, token=self._hf_token)
        except Exception as exc:
            raise DiarizationError(
                f"Не удалось загрузить модель диаризации '{self._pipeline_name}'. "
                "Убедитесь, что вы приняли условия использования модели на "
                "huggingface.co и указали действующий токен доступа "
                f"(--hf-token или переменная окружения HF_TOKEN). Исходная ошибка: {exc}"
            ) from exc

        if pipeline is None:
            raise DiarizationError(
                f"Модель диаризации '{self._pipeline_name}' не найдена или недоступна"
            )

        pipeline.to(torch.device(self._device.value))
        self._pipeline = pipeline
        return self._pipeline

    def diarize(
        self, audio_path: Path, *, num_speakers: int | None = None
    ) -> list[SpeakerSegment]:
        pipeline = self._load_pipeline()

        import torch
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        waveform = load_waveform(audio_path)
        audio_input = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }

        try:
            # ProgressHook показывает прогресс внутренних этапов (сегментация,
            # эмбеддинги, кластеризация) — без него диаризация длинных записей
            # выглядит как зависание, так как не даёт промежуточного вывода.
            with ProgressHook() as hook:
                output = pipeline(audio_input, num_speakers=num_speakers, hook=hook)
        except Exception as exc:
            raise DiarizationError(
                f"Ошибка при определении говорящих в файле {audio_path}: {exc}"
            ) from exc

        annotation = getattr(output, "exclusive_speaker_diarization", output)

        return [
            SpeakerSegment(start=turn.start, end=turn.end, speaker_id=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
