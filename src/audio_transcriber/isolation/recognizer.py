"""Обёртка распознавания речи через воркер-процесс (#143).

Сохраняет контракт :class:`~audio_transcriber.transcription.base.SpeechRecognizer`,
но не грузит faster-whisper в основной процесс: модель загружается воркером
(``audio_transcriber.isolation.worker``) и умирает вместе с ним.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from audio_transcriber.cache.serialization import asr_from_payload
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import TranscriptionSegment
from audio_transcriber.isolation.protocol import (
    DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    KIND_ASR,
    WorkerRequest,
)
from audio_transcriber.isolation.runner import run_worker
from audio_transcriber.progress import ProgressCallback
from audio_transcriber.utils.exceptions import EngineIsolationError, TranscriptionError


class SubprocessWhisperRecognizer:
    """faster-whisper в отдельном процессе. Реализует ``SpeechRecognizer``."""

    def __init__(
        self,
        model_name: str,
        device: Device,
        *,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress: ProgressCallback | None = None,
        word_timestamps: bool = True,
        timeout: float | None = DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._initial_prompt = initial_prompt
        self._hotwords = hotwords
        self._on_progress = on_progress
        self._word_timestamps = word_timestamps
        self._timeout = timeout

    def _options(self) -> dict[str, object]:
        return {
            "model_name": self._model_name,
            "device": self._device.value,
            "initial_prompt": self._initial_prompt,
            "hotwords": self._hotwords,
            "word_timestamps": self._word_timestamps,
        }

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        request = WorkerRequest(
            kind=KIND_ASR,
            audio_path=str(audio_path),
            language=language,
            options=self._options(),
        )
        try:
            result = run_worker(request, on_progress=self._on_progress, timeout=self._timeout)
        except EngineIsolationError as exc:
            raise TranscriptionError(
                f"Изолированное распознавание речи не удалось ({audio_path}): {exc}"
            ) from exc
        if not result.ok:
            raise TranscriptionError(
                f"Изолированное распознавание речи завершилось ошибкой "
                f"({audio_path}): {result.error}"
            )
        segments, detected_language, duration = asr_from_payload(result.payload or {})
        return segments, cast(str, detected_language), duration

    def close(self) -> None:
        """В основном процессе модель не живёт — освобождать нечего (#143)."""
