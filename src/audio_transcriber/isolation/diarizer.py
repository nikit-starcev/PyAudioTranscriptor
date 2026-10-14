"""Обёртка диаризации через воркер-процесс (#143).

Сохраняет контракт :class:`~audio_transcriber.diarization.base.SpeakerDiarizer`,
но не грузит pyannote/torch в основной процесс: модель загружается воркером и
умирает вместе с ним, поэтому после стадии RSS основного процесса не содержит
torch/CTranslate2-арен (важно на Windows/macOS, где ``malloc_trim`` из #141
недоступен).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np

from audio_transcriber.cache.serialization import diarization_from_payload
from audio_transcriber.config.defaults import DEFAULT_DIARIZATION_MIN_DURATION_OFF
from audio_transcriber.diarization.pyannote_engine import (
    DEFAULT_PIPELINE,
)
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment
from audio_transcriber.isolation.protocol import (
    DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    KIND_DIARIZATION,
    WorkerRequest,
)
from audio_transcriber.isolation.runner import run_worker
from audio_transcriber.progress import ProgressCallback
from audio_transcriber.utils.exceptions import DiarizationError, EngineIsolationError


class SubprocessPyannoteDiarizer:
    """pyannote.audio в отдельном процессе. Реализует ``SpeakerDiarizer``.

    Конструктор повторяет :class:`PyannoteSpeakerDiarizer`, чтобы фабрика могла
    подменять движок без изменения остального кода.
    """

    def __init__(
        self,
        device: Device,
        *,
        pipeline_name: str = DEFAULT_PIPELINE,
        hf_token: str | None = None,
        local_model_path: Path | str | None = None,
        on_progress: ProgressCallback | None = None,
        min_duration_off: float = DEFAULT_DIARIZATION_MIN_DURATION_OFF,
        clustering_threshold: float | None = None,
        clustering_fb: float | None = None,
        timeout: float | None = DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    ) -> None:
        self._device = device
        self._pipeline_name = pipeline_name
        self._hf_token = (
            hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self._local_model_path = Path(local_model_path) if local_model_path else None
        self._on_progress = on_progress
        self._min_duration_off = min_duration_off
        self._clustering_threshold = clustering_threshold
        self._clustering_fb = clustering_fb
        self._timeout = timeout
        self._overlaps: list[SpeakerOverlap] = []

    def _options(self) -> dict[str, object]:
        return {
            "device": self._device.value,
            "pipeline_name": self._pipeline_name,
            "hf_token": self._hf_token,
            "local_model_path": (
                str(self._local_model_path) if self._local_model_path else None
            ),
            "min_duration_off": self._min_duration_off,
            "clustering_threshold": self._clustering_threshold,
            "clustering_fb": self._clustering_fb,
        }

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
        waveform_path: Path | None = None
        try:
            if waveform is not None:
                fd, raw_path = tempfile.mkstemp(prefix="isolation-wav-", suffix=".npy")
                os.close(fd)
                waveform_path = Path(raw_path)
                np.save(waveform_path, waveform)

            options = self._options()
            options.update(
                {
                    "num_speakers": num_speakers,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }
            )
            request = WorkerRequest(
                kind=KIND_DIARIZATION,
                audio_path=str(audio_path),
                waveform_path=str(waveform_path) if waveform_path else None,
                options=options,
            )
            try:
                result = run_worker(
                    request, on_progress=self._on_progress, timeout=self._timeout
                )
            except EngineIsolationError as exc:
                raise DiarizationError(
                    f"Изолированная диаризация не удалась ({audio_path}): {exc}"
                ) from exc
            if not result.ok:
                raise DiarizationError(
                    f"Изолированная диаризация завершилась ошибкой "
                    f"({audio_path}): {result.error}"
                )
            segments, overlaps = diarization_from_payload(result.payload or {})
            self._overlaps = overlaps
            return segments
        finally:
            if waveform_path is not None:
                with contextlib.suppress(OSError):
                    waveform_path.unlink()

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """Интервалы наложения речи из последнего вызова :meth:`diarize`."""
        return list(self._overlaps)

    def close(self) -> None:
        """В основном процессе модель не живёт — освобождать нечего (#143)."""
        self._overlaps = []
