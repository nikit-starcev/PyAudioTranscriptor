"""Реализация диаризации на pyannote.audio."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
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
        local_model_path: Path | str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._device = device
        self._pipeline_name = pipeline_name
        self._local_model_path = Path(local_model_path) if local_model_path else None
        self._on_progress = on_progress
        self._hf_token = (
            hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self._pipeline: Any = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from pyannote.audio import Pipeline

        try:
            if self._local_model_path is not None and self._local_model_path.is_dir():
                # Локальная копия модели: грузится напрямую с диска, без
                # обращения к Hugging Face (токен и сеть не нужны).
                logger.debug(
                    "Загрузка локальной модели диаризации из '%s'",
                    self._local_model_path,
                )
                pipeline = Pipeline.from_pretrained(self._local_model_path)
            else:
                logger.debug("Загрузка модели диаризации '%s'", self._pipeline_name)
                pipeline = Pipeline.from_pretrained(self._pipeline_name, token=self._hf_token)
        except Exception as exc:
            if self._local_model_path is not None and self._local_model_path.is_dir():
                raise DiarizationError(
                    f"Не удалось загрузить локальную модель диаризации "
                    f"'{self._local_model_path}': {exc}"
                ) from exc
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

    def diarize(self, audio_path: Path, *, num_speakers: int | None = None) -> list[SpeakerSegment]:
        pipeline = self._load_pipeline()

        import torch
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        waveform = load_waveform(audio_path)
        audio_input = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }

        try:
            emit = self._on_progress
            if emit is not None:

                def _hook(
                    step_name: object,
                    _step_artifact: object,
                    file: object = None,
                    total: float | None = None,
                    completed: float | None = None,
                ) -> None:
                    # pyannote передаёт completed/total в разных комбинациях:
                    # completed=None — одноразовый шаг (без прогресса),
                    # completed может превышать total (batch_size > num_chunks),
                    # поэтому прогресс нормализуется в диапазон [0; 1].
                    del file  # параметр обязателен (pyannote зовёт hook(file=...)), не используется
                    if completed is None or total is None:
                        fraction = None
                    elif completed >= total:
                        fraction = 1.0
                    else:
                        fraction = max(0.0, min(1.0, completed / total))
                    emit(
                        ProgressEvent(
                            "diarization",
                            message=str(step_name),
                            fraction=fraction,
                            detail=str(step_name),
                        )
                    )

                output = pipeline(audio_input, num_speakers=num_speakers, hook=_hook)
            else:
                # ProgressHook показывает прогресс внутренних этапов
                # (сегментация, эмбеддинги, кластеризация) в консоли.
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
