"""Реализация распознавания речи на faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import TranscriptionSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.exceptions import TranscriptionError
from audio_transcriber.utils.hotwords import truncate_hotwords_by_tokens

logger = logging.getLogger(__name__)


class WhisperSpeechRecognizer:
    """Распознаёт речь через faster-whisper. Реализует протокол ``SpeechRecognizer``."""

    def __init__(
        self,
        model_name: str,
        device: Device,
        *,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._initial_prompt = initial_prompt
        self._hotwords = hotwords
        self._on_progress = on_progress
        self._model: Any = None

    def _emit(self, fraction: float | None = None, detail: str = "") -> None:
        if self._on_progress is not None:
            self._on_progress(
                ProgressEvent("asr", "Распознавание речи", fraction=fraction, detail=detail)
            )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        logger.debug(
            "Загрузка модели faster-whisper '%s' на %s", self._model_name, self._device.value
        )
        try:
            # "default" поручает CTranslate2 самому выбрать быстрый тип
            # вычислений, который реально поддерживается конкретным GPU/CPU
            # (например, старые GPU без эффективного float16 получат int8).
            self._model = WhisperModel(
                self._model_name,
                device=self._device.value,
                compute_type="default",
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Не удалось загрузить модель распознавания '{self._model_name}': {exc}"
            ) from exc

        return self._model

    def _prepare_hotwords(self, model: Any) -> str | None:
        """Обрезает hotwords по токенам модели, чтобы prompt не переполнял контекст."""
        if not self._hotwords:
            return None

        def encode(text: str) -> list[int]:
            return model.hf_tokenizer.encode(text).ids

        truncated, dropped = truncate_hotwords_by_tokens(self._hotwords, encode)
        if dropped:
            logger.warning(
                "Словарь терминов обрезан по лимиту токенов модели "
                "(%d терминов не учтено). Расположите самые важные в начале файла.",
                len(dropped),
            )
            logger.debug("Отброшенные термины: %s", ", ".join(dropped))
        return truncated or None

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        model = self._load_model()
        hotwords = self._prepare_hotwords(model)

        try:
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                # Отсекает тишину/не-речь и уточняет границы сегментов по
                # словам — снижает число "придуманных" фраз и повышает
                # точность таймкодов.
                vad_filter=True,
                word_timestamps=True,
                # Подсказка модели и "горячие слова" помогают правильно
                # распознавать имена и специфичные термины конкретного
                # разговора вместо похожих по звучанию слов.
                initial_prompt=self._initial_prompt,
                hotwords=hotwords,
            )
            # Длительность известна сразу (до порождения сегментов); по ней
            # нормируем прогресс. ``None``/0 — длительность неизвестна, тогда
            # прогресс остаётся неопределённым (без fraction).
            duration = info.duration
            has_duration = isinstance(duration, (int, float)) and duration > 0
            self._emit(0.0 if has_duration else None)
            segments: list[TranscriptionSegment] = []
            for segment in raw_segments:
                fraction: float | None = None
                if has_duration:
                    # Конец уже обработанного сегмента, нормированный на
                    # длительность аудио, ограничен [0; 1]. Генератор отдаёт
                    # сегменты в порядке возрастания времени — прогресс растёт.
                    fraction = max(0.0, min(1.0, segment.end / duration))
                self._emit(fraction)
                segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                        avg_logprob=segment.avg_logprob,
                    )
                )
        except Exception as exc:
            raise TranscriptionError(
                f"Ошибка при распознавании речи в файле {audio_path}: {exc}"
            ) from exc

        return segments, info.language, info.duration
