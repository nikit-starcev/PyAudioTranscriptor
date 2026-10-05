"""Гибридный ASR: «плохие» сегменты основного движка перегоняются Whisper (#57).

Основной движок (GigaAM) быстр и хорошо работает на чистой русской речи, но
может ошибаться на шумных, тихих или очень коротких фрагментах и не даёт
пунктуации в некоторых режимах. Идея гибрида: определить такие сегменты **после**
основного прохода и повторно распознать **только их** резервным движком
(faster-whisper/whisper.cpp). Это сохраняет выигрыш по скорости — дорабатываются
единицы сегментов, а не вся запись.

Признаки «плохого» сегмента (любой срабатывает):
* пустой текст;
* очень короткая длительность;
* низкая средняя логвероятность (``avg_logprob``);
* высокая вероятность отсутствия речи (``no_speech_prob``);
* низкая RMS-энергия (тишина/шум) на интервале сегмента.

Резервный движок получает **вырезку с контекстом** вокруг сегмента, а его
таймстемпы переводятся в абсолютные смещением на начало вырезки и затем
**обрезаются границами исходного сегмента**. Так исключается дрейф времени на
стыках и дублирование речи соседних сегментов: контекст помогает распознать
слова на границе, но результат не «расползается» за пределы своего окна.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_HYBRID_CONTEXT_SECONDS,
    DEFAULT_HYBRID_LOW_LOGPROB_THRESHOLD,
    DEFAULT_HYBRID_MIN_SEGMENT_SECONDS,
    DEFAULT_HYBRID_NO_SPEECH_THRESHOLD,
    DEFAULT_HYBRID_SILENCE_RMS_THRESHOLD,
)
from audio_transcriber.domain.models import TranscriptionSegment, WordTimestamp
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform, write_wav
from audio_transcriber.utils.exceptions import TranscriptionError

logger = logging.getLogger(__name__)

#: Версия реализации гибридного ASR. Участвует в ключе кэша: изменение логики
#: отбора/склейки меняет результат при тех же параметрах.
HYBRID_IMPL_VERSION = 1


@dataclass(frozen=True, slots=True)
class HybridOptions:
    """Пороги отбора «плохих» сегментов и параметры склейки.

    Значения по умолчанию подобраны консервативно (см. докстринг модуля):
    ``low_logprob_threshold=-1.0`` совпадает с порогом «низкой уверенности»
    проекта, ``min_segment_seconds=0.5`` отсекает обрывки, а
    ``silence_rms_threshold=0.003`` на шкале [-1, 1] ловит почти полную тишину,
    не задевая тихую, но разборчивую речь.
    """

    low_logprob_threshold: float = DEFAULT_HYBRID_LOW_LOGPROB_THRESHOLD
    no_speech_threshold: float = DEFAULT_HYBRID_NO_SPEECH_THRESHOLD
    silence_rms_threshold: float = DEFAULT_HYBRID_SILENCE_RMS_THRESHOLD
    min_segment_seconds: float = DEFAULT_HYBRID_MIN_SEGMENT_SECONDS
    context_seconds: float = DEFAULT_HYBRID_CONTEXT_SECONDS
    #: Ниже этой длительности вырезку для Whisper не делаем (защита от
    #: вырожденных окон); такой сегмент остаётся как есть.
    min_chunk_seconds: float = 0.1


@dataclass(frozen=True, slots=True)
class BadSegment:
    """Сегмент, признанный «плохим», и причины отбора (для лога/тестов)."""

    index: int
    start: float
    end: float
    reasons: tuple[str, ...]


def segment_rms(
    waveform: np.ndarray, start: float, end: float, *, sample_rate: int = SAMPLE_RATE
) -> float:
    """RMS-энергия среза waveform ``[start, end)`` секунд.

    Пустой или вывернутый интервал даёт ``0.0`` — это трактуется как тишина.
    """
    total = len(waveform)
    if total == 0 or end <= start:
        return 0.0
    first = max(0, min(total, round(start * sample_rate)))
    last = max(0, min(total, round(end * sample_rate)))
    if last <= first:
        return 0.0
    chunk = waveform[first:last].astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(chunk))))


def classify_segment(
    segment: TranscriptionSegment,
    *,
    waveform: np.ndarray | None,
    options: HybridOptions,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[str, ...]:
    """Возвращает список причин, по которым сегмент считается «плохим».

    Пустой кортеж — сегмент хороший. Проверки независимы: чем больше причин,
    тем увереннее сегмент отправляется на доработку.
    """
    reasons: list[str] = []
    text = segment.text.strip()
    if not text:
        reasons.append("empty")
    if segment.end - segment.start < options.min_segment_seconds:
        reasons.append("short")
    if (
        segment.avg_logprob is not None
        and segment.avg_logprob < options.low_logprob_threshold
    ):
        reasons.append("low_logprob")
    if (
        segment.no_speech_prob is not None
        and segment.no_speech_prob > options.no_speech_threshold
    ):
        reasons.append("no_speech")
    if waveform is not None and (
        segment_rms(waveform, segment.start, segment.end, sample_rate=sample_rate)
        < options.silence_rms_threshold
    ):
        reasons.append("silence")
    return tuple(reasons)


def detect_bad_segments(
    segments: list[TranscriptionSegment],
    *,
    waveform: np.ndarray | None,
    options: HybridOptions,
    sample_rate: int = SAMPLE_RATE,
) -> list[BadSegment]:
    """Отбирает «плохие» сегменты для повторного распознавания Whisper."""
    bad: list[BadSegment] = []
    for index, segment in enumerate(segments):
        reasons = classify_segment(
            segment, waveform=waveform, options=options, sample_rate=sample_rate
        )
        if reasons:
            bad.append(
                BadSegment(
                    index=index,
                    start=segment.start,
                    end=segment.end,
                    reasons=reasons,
                )
            )
    return bad


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _shift_clamped_words(
    words: list[WordTimestamp],
    offset: float,
    low: float,
    high: float,
) -> list[WordTimestamp]:
    """Переносит пословные метки в абсолютное время и обрезает окном (#45).

    ``offset`` — начало вырезки, отданной резервному движку; ``low``/``high`` —
    границы исходного сегмента. Слова, целиком лежащие вне окна, отбрасываются;
    остальные обрезаются, как и таймстемпы сегмента.
    """
    result: list[WordTimestamp] = []
    for word in words:
        start = word.start + offset
        end = word.end + offset
        if end <= low or start >= high:
            continue
        result.append(
            WordTimestamp(
                text=word.text,
                start=_clamp(start, low, high),
                end=_clamp(end, low, high),
                probability=word.probability,
            )
        )
    return result


class HybridSpeechRecognizer:
    """Основной движок + выборочная доработка «плохих» сегментов Whisper.

    Реализует протокол :class:`~audio_transcriber.transcription.base.SpeechRecognizer`
    и прозрачен для конвейера: он не знает, что внутри два движка.
    """

    def __init__(
        self,
        primary: SpeechRecognizer,
        fallback: SpeechRecognizer,
        *,
        options: HybridOptions | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._options = options or HybridOptions()
        self._on_progress = on_progress
        # Доля аудио (по длительности сегментов), которую доработал Whisper.
        self.last_refined_fraction: float = 0.0
        # Число сегментов, отправленных на доработку.
        self.last_refined_segments: int = 0
        # Суммарная длительность доработанных сегментов, секунды.
        self.last_refined_seconds: float = 0.0

    def _emit(self, detail: str) -> None:
        if self._on_progress is not None:
            self._on_progress(
                ProgressEvent("asr", "Распознавание речи", fraction=None, detail=detail)
            )

    def _refine_segment(
        self,
        waveform: np.ndarray,
        original: TranscriptionSegment,
        *,
        duration: float,
        language: str | None,
        chunk_path: Path,
    ) -> list[TranscriptionSegment]:
        """Повторно распознаёт один сегмент с контекстом; таймстемпы — абсолютные.

        Возвращает непустой список сегментов либо исходный сегмент (если
        резервный движок не дал ничего осмысленного — не теряем данные).
        """
        context = self._options.context_seconds
        region_start = max(0.0, original.start - context)
        region_end = min(duration, original.end + context)
        if region_end - region_start < self._options.min_chunk_seconds:
            return [original]

        first = max(0, min(len(waveform), round(region_start * SAMPLE_RATE)))
        last = max(0, min(len(waveform), round(region_end * SAMPLE_RATE)))
        if last <= first:
            return [original]
        write_wav(chunk_path, waveform[first:last])

        try:
            fallback_segments, _, _ = self._fallback.transcribe(
                chunk_path, language=language
            )
        except TranscriptionError as exc:
            logger.warning(
                "Гибрид ASR: доработка сегмента %.2f–%.2f с не удалась (%s) — "
                "оставляю результат основного движка",
                original.start,
                original.end,
                exc,
            )
            return [original]

        refined: list[TranscriptionSegment] = []
        for fallback_segment in fallback_segments:
            # Абсолютное смещение — без накопления ошибки: начало вырезки
            # прибавляется ровно один раз.
            abs_start = fallback_segment.start + region_start
            abs_end = fallback_segment.end + region_start
            # Оставляем только пересекающиеся с исходным окном и обрезаем по
            # его границам — так результат не залезает на соседние сегменты и
            # нет дрейфа на стыках.
            if abs_end <= original.start or abs_start >= original.end:
                continue
            start = _clamp(abs_start, original.start, original.end)
            end = _clamp(abs_end, original.start, original.end)
            if end <= start:
                continue
            text = fallback_segment.text.strip()
            if not text:
                continue
            refined.append(
                TranscriptionSegment(
                    start=start,
                    end=end,
                    text=text,
                    avg_logprob=fallback_segment.avg_logprob,
                    no_speech_prob=fallback_segment.no_speech_prob,
                    # Пословные метки резервного движка (#45) переводим в
                    # абсолютное время и обрезаем окном сегмента, как и его
                    # границы; слова вне окна отбрасываем.
                    words=_shift_clamped_words(
                        fallback_segment.words, region_start, start, end
                    ),
                )
            )

        if not refined:
            return [original]
        refined.sort(key=lambda segment: (segment.start, segment.end))
        return refined

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        segments, detected_language, duration = self._primary.transcribe(
            audio_path, language=language
        )
        self.last_refined_fraction = 0.0
        self.last_refined_segments = 0
        self.last_refined_seconds = 0.0

        if not segments:
            return segments, detected_language, duration

        try:
            waveform = load_waveform(audio_path)
        except Exception as exc:  # noqa: BLE001 — без waveform гибрид бесполезен
            logger.warning(
                "Гибрид ASR: не удалось декодировать %s для доработки (%s) — "
                "возвращаю результат основного движка",
                audio_path,
                exc,
            )
            return segments, detected_language, duration

        bad = detect_bad_segments(
            segments, waveform=waveform, options=self._options, sample_rate=SAMPLE_RATE
        )
        if not bad:
            logger.info("Гибрид ASR: «плохих» сегментов не найдено — доработка не нужна")
            return segments, detected_language, duration

        refined_seconds = sum(item.end - item.start for item in bad)
        refined_indices = {item.index for item in bad}
        logger.info(
            "Гибрид ASR: доработка Whisper %d сегмент(ов) (%.1f с из %.1f с, %.1f%%)",
            len(bad),
            refined_seconds,
            duration,
            (refined_seconds / duration * 100.0) if duration > 0 else 0.0,
        )
        self._emit(f"доработка Whisper: {len(bad)} сегмент(ов)")

        result: list[TranscriptionSegment] = []
        with tempfile.TemporaryDirectory(prefix="hybrid-asr-") as tmpdir:
            chunk_path = Path(tmpdir) / "segment.wav"
            for index, segment in enumerate(segments):
                if index not in refined_indices:
                    result.append(segment)
                    continue
                result.extend(
                    self._refine_segment(
                        waveform,
                        segment,
                        duration=duration,
                        language=language,
                        chunk_path=chunk_path,
                    )
                )

        result.sort(key=lambda segment: (segment.start, segment.end))
        self.last_refined_segments = len(bad)
        self.last_refined_seconds = refined_seconds
        self.last_refined_fraction = (
            refined_seconds / duration if duration > 0 else 0.0
        )
        return result, detected_language, duration
