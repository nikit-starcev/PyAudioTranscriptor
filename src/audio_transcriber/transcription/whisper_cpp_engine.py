"""Реализация распознавания речи через whisper.cpp (CLI).

В отличие от faster-whisper, whisper.cpp ускоряется на GPU не через
CUDA/PyTorch, а через Vulkan — поэтому работает и на AMD-картах
(например, Polaris/RX 4xx–5xx), которые ROCm/PyTorch не поддерживают.

Движок запускает собранный бинарник ``whisper-cli`` как внешний процесс
и разбирает его JSON-вывод, сохраняя интерфейс :class:`SpeechRecognizer`,
чтобы остальной конвейер (диаризация, объединение, экспорт) не менялся.

Длинные файлы режутся на куски (см. :data:`DEFAULT_CHUNK_SECONDS`): whisper.cpp
накапливает текстовый контекст между внутренними 30-с окнами одного запуска, и
на длинной записи это приводит к потере речи и галлюцинациям. Каждый кусок
распознаётся **отдельным** вызовом whisper-cli, таймкоды сдвигаются, а реплики
из области перекрытия дедуплицируются (остаётся вариант из более «центрального»
куска). Границы кусков сдвигаются к локальному минимуму энергии, чтобы не
разрезать слово.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from audio_transcriber.domain.models import TranscriptionSegment, WordTimestamp
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    AudioProbe,
    load_waveform,
    probe_audio,
    write_wav,
)
from audio_transcriber.utils.env import with_library_path
from audio_transcriber.utils.exceptions import AudioFileError, TranscriptionError
from audio_transcriber.utils.subprocess_registry import register_process, terminate_process

logger = logging.getLogger(__name__)

#: Версия реализации ASR whisper.cpp. Участвует в ключе кэша (см.
#: ``pipeline._asr_cache_params``): при изменении логики, влияющей на результат
#: при тех же параметрах (отказ от лишнего перекодирования входа, чанкинг
#: длинных файлов, пословные таймстемпы, учёт фактического денойза #83), старый
#: кэш должен инвалидироваться.
ASR_IMPL_VERSION = 5

#: Таймаут одного вызова whisper-cli (секунды). Битая входная дорожка или
#: дедлок GPU/Vulkan может повесить распознавание навсегда; по истечении
#: процесс принудительно гасится, а стадия падает внятной ошибкой. ``None``
#: (или ``<= 0``) отключает таймаут — прежнее поведение. Настраивается через
#: ``WHISPER_CPP_TIMEOUT``.
DEFAULT_WHISPER_CPP_TIMEOUT = 3600.0

#: Целевая длина куска при чанкинге длинных файлов (с). 30 с — «родное» окно
#: Whisper: кусок распознаётся за один проход, без накопления текстового
#: контекста между окнами.
DEFAULT_CHUNK_SECONDS = 30.0

#: Перекрытие соседних кусков (с). Нужно, чтобы реплика на стыке целиком попала
#: хотя бы в один кусок; дубликаты из перекрытия затем дедуплицируются.
DEFAULT_CHUNK_OVERLAP = 2.0

#: Окно и шаг расчёта энергии для поиска тихих точек разреза (с).
_ENERGY_FRAME_SECONDS = 0.02
_ENERGY_HOP_SECONDS = 0.01

#: Пороги отнесения двух реплик из области перекрытия к одной и той же.
_DEDUP_MIN_TIME_COVERAGE = 0.6
_DEDUP_MIN_TEXT_RATIO = 0.6
_DEDUP_MIN_TOKEN_JACCARD = 0.6

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+(?:\.\d+)?)%")


def _is_native_wav(probe: AudioProbe) -> bool:
    """True, если whisper-cli прочитает файл напрямую, без перекодирования.

    Подходят 16-кГц моно PCM WAV — именно такой формат отдаёт шумоподавление.
    Прочие WAV (другая частота/каналы) и не-WAV (mp4/webm/mp3/…) конвертируются
    во временный WAV, как и раньше: убирать эту конвертацию шире, чем для
    «родного» формата, рискованно для качества и совместимости.
    """

    formats = {part.strip() for part in (probe.format_name or "").split(",")}
    return (
        "wav" in formats
        and probe.sample_rate == SAMPLE_RATE
        and probe.channels == 1
        and (probe.codec or "").startswith("pcm_")
    )


# Параметры VAD по умолчанию — как в faster-whisper
# (``faster_whisper.vad.VadOptions``), чтобы оба движка отсекали тишину/не-речь
# по одинаковым условиям. whisper.cpp требует отдельную Silero-VAD-модель
# (``--vad-model``), поэтому VAD включается, только если путь задан.
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_DURATION_MS = 0
VAD_MIN_SILENCE_DURATION_MS = 2000
VAD_SPEECH_PAD_MS = 400

# Служебные токены whisper.cpp в полном JSON (``-ojf``): ``[_BEG_]``,
# ``[_TT_129]``, ``[_EOT_]`` и т.п. У них тоже есть вероятность ``p``, но она
# относится не к речи, поэтому в среднюю уверенность не входит.
_SPECIAL_TOKEN_RE = re.compile(r"^\[_.*\]$")

# Нормализация текста для сравнения реплик из области перекрытия.
_DEDUP_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_DEDUP_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _segment_avg_logprob(item: dict) -> float | None:
    """Средняя логвероятность сегмента из полного JSON whisper.cpp (``-ojf``).

    whisper.cpp отдаёт вероятность ``p`` каждого токена. Аналог
    ``avg_logprob`` faster-whisper — среднее натуральных логарифмов ``p`` по
    «речевым» токенам (служебные ``[_...]`` пропускаются). Если токенов с
    вероятностями нет (например, запуск без ``-ojf``), возвращается ``None`` —
    фича мягко деградирует и не роняет конвейер.
    """
    tokens = item.get("tokens")
    if not isinstance(tokens, list):
        return None

    logprobs: list[float] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = str(token.get("text", "")).strip()
        if _SPECIAL_TOKEN_RE.match(text):
            continue
        probability = token.get("p")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            continue
        probability = float(probability)
        if probability <= 0.0:
            continue
        logprobs.append(math.log(probability))

    if not logprobs:
        return None
    return sum(logprobs) / len(logprobs)


def _token_seconds(value: object) -> float | None:
    """Миллисекунды токена (``offsets.from``/``to``) в секунды.

    Возвращает ``None`` для отсутствующего/нечислового значения: такой токен в
    пословной разметке не участвует (мягкая деградация).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) / 1000.0


def _merge_word_probability(
    left: float | None, right: float | None
) -> float | None:
    """Наихудшая (минимальная) вероятность из двух необязательных.

    Как и ``_min_optional`` в объединителе реплик: ``None`` игнорируется; если
    оба ``None`` — ``None``.
    """
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _word_timestamps(
    tokens: object, *, chunk_offset: float = 0.0
) -> list[WordTimestamp]:
    """Собирает пословные метки из токенов полного JSON whisper.cpp (``-ojf``).

    Вход — список ``tokens`` одного сегмента: у каждого ``text``, ``offsets``
    (``from``/``to`` в мс) и ``p``. Служебные токены (``[_BEG_]``, ``[_TT_*]``)
    пропускаются. Слово начинается токеном с ведущим пробелом; последующие
    токены без пробела — его под-токены (в т.ч. пунктуация) и дописываются к
    текущему слову. ``chunk_offset`` (секунды) прибавляется ко временам — при
    чанкинге вызывающий передаёт офсет куска.

    Первое слово сегмента, начатое токеном **без** ведущего пробела, помечается
    ``continuation=True`` — это продолжение слова из предыдущего куска; сшивка
    выполняется в :func:`_stitch_chunk_words`.

    Вероятность слова — минимум ``p`` по «лексическим» токенам слова (чистая
    пунктуация не учитывается), ``None`` — вероятностей не было.
    """
    if not isinstance(tokens, list):
        return []

    words: list[WordTimestamp] = []
    parts: list[str] = []
    start: float | None = None
    end: float | None = None
    probabilities: list[float] = []
    continuation = False

    def flush() -> None:
        nonlocal parts, start, end, probabilities, continuation
        if parts and start is not None:
            resolved_end = end if end is not None else start
            if resolved_end < start:
                # Инвертированный offsets.to у некоторых токенов whisper.cpp —
                # не допускаем отрицательной длительности слова.
                resolved_end = start
            words.append(
                WordTimestamp(
                    text="".join(parts),
                    start=start + chunk_offset,
                    end=resolved_end + chunk_offset,
                    probability=min(probabilities) if probabilities else None,
                    continuation=continuation,
                )
            )
        parts, start, end, probabilities, continuation = [], None, None, [], False

    for token in tokens:
        if not isinstance(token, dict):
            continue
        raw_text = str(token.get("text", ""))
        stripped = raw_text.strip()
        if not stripped or _SPECIAL_TOKEN_RE.match(stripped):
            continue
        offsets = token.get("offsets")
        if not isinstance(offsets, dict):
            continue
        token_start = _token_seconds(offsets.get("from"))
        if token_start is None:
            continue
        token_end = _token_seconds(offsets.get("to"))
        starts_new_word = bool(raw_text) and raw_text[0].isspace()
        if starts_new_word and parts:
            flush()
        if not parts:
            start = token_start
            continuation = not starts_new_word
        parts.append(stripped)
        if token_end is not None:
            end = token_end if end is None else max(end, token_end)
        probability = token.get("p")
        # Вероятность слова считаем по «лексическим» токенам: чистая пунктуация
        # (``.``, ``,``) низкой ``p`` не должна занижать уверенность слова.
        if (
            isinstance(probability, (int, float))
            and not isinstance(probability, bool)
            and probability > 0.0
            and any(character.isalnum() for character in stripped)
        ):
            probabilities.append(float(probability))
    flush()
    return words


def _shift_words(
    words: list[WordTimestamp], offset: float
) -> list[WordTimestamp]:
    """Сдвигает пословные метки на ``offset`` секунд (офсет куска).

    Пустой список или нулевой сдвиг возвращаются как копия исходного.
    """
    if not words or offset == 0.0:
        return list(words)
    return [
        replace(word, start=word.start + offset, end=word.end + offset)
        for word in words
    ]


def _stitch_chunk_words(
    segments: list[TranscriptionSegment],
) -> list[TranscriptionSegment]:
    """Склеивает слова, разрезанные границей соседних кусков.

    Слово с ``continuation=True`` (его первый токен не имел ведущего пробела) —
    продолжение слова из предыдущего сегмента: приклеиваем его текст и время к
    последнему слову предыдущего сегмента. Если предыдущего слова нет (сегмент
    срезан в самом начале), флаг просто снимается. После сшивки флаг
    ``continuation`` в результате не остаётся.
    """
    result = list(segments)
    last_position: tuple[int, int] | None = None
    for index, segment in enumerate(result):
        words = list(segment.words)
        if not words:
            continue
        first = words[0]
        if first.continuation and last_position is not None:
            prev_segment_index, prev_word_index = last_position
            prev_words = list(result[prev_segment_index].words)
            previous = prev_words[prev_word_index]
            merged = replace(
                previous,
                text=previous.text + first.text,
                start=min(previous.start, first.start),
                end=max(previous.end, first.end),
                probability=_merge_word_probability(
                    previous.probability, first.probability
                ),
                continuation=False,
            )
            prev_words[prev_word_index] = merged
            result[prev_segment_index] = replace(
                result[prev_segment_index], words=prev_words
            )
            words = words[1:]
            result[index] = replace(segment, words=words)
            # Продолжение принадлежит предыдущему слову: если после сшивки у
            # сегмента не осталось слов, «хвост» остаётся на слитом слове.
            last_position = (
                (index, len(words) - 1) if words else (prev_segment_index, prev_word_index)
            )
            continue
        if first.continuation:
            words[0] = replace(first, continuation=False)
            result[index] = replace(segment, words=words)
        last_position = (index, len(words) - 1)
    return result


def _frame_energy(
    waveform: np.ndarray, *, sample_rate: int = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray, int]:
    """RMS-энергия waveform по кадрам.

    Возвращает ``(энергии, индексы начала кадров, длина кадра)``. Используется
    для поиска тихих точек разреза: считать полное преобразование незачем,
    достаточно локальной энергии.
    """
    frame = max(1, round(_ENERGY_FRAME_SECONDS * sample_rate))
    hop = max(1, round(_ENERGY_HOP_SECONDS * sample_rate))
    total = len(waveform)
    if total == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64), frame

    squared = np.square(waveform.astype(np.float64))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, max(1, total - frame + 1), hop, dtype=np.int64)
    ends = np.minimum(starts + frame, total)
    lengths = np.maximum(ends - starts, 1)
    energy = (cumulative[ends] - cumulative[starts]) / lengths
    return energy, starts, frame


def _min_energy_sample(
    waveform: np.ndarray, *, target: int, radius: int, sample_rate: int = SAMPLE_RATE
) -> int:
    """Ближайшая к ``target`` точка минимума энергии в пределах ``radius``.

    Разрез по тишине/паузе не разрубает слово. Если энергии посчитать нельзя или
    рядом нет кадров, возвращается ``target``.
    """
    if radius <= 0 or len(waveform) == 0:
        return target

    energy, starts, frame = _frame_energy(waveform, sample_rate=sample_rate)
    if energy.size == 0:
        return target

    centers = starts + frame // 2
    mask = (centers >= target - radius) & (centers <= target + radius)
    if not mask.any():
        return target

    candidates = np.nonzero(mask)[0].tolist()
    best = min(
        candidates,
        key=lambda index: (
            float(energy[index]),
            abs(int(centers[index]) - target),
        ),
    )
    return int(centers[best])


def _choose_chunk_bounds(
    waveform: np.ndarray,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """Разбивает waveform на куски с перекрытием; границы — по тишине.

    Возвращает список пар ``(start, end)`` в сэмплах. Первый кусок начинается с
    нуля, последний заканчивается концом файла, каждый следующий начинается на
    ``chunk_seconds - overlap_seconds`` позже предыдущего. Точка разреза
    сдвигается к локальному минимуму энергии в пределах ``overlap_seconds / 2``,
    чтобы не разрезать слово. Короткий «хвост» (< перекрытия) поглощается
    предыдущим куском — куцые вызовы whisper-cli не создаются.
    """
    total = len(waveform)
    if total <= 0:
        return [(0, 0)]

    safe_overlap = max(0.0, min(overlap_seconds, chunk_seconds / 2.0))
    chunk = max(1, round(chunk_seconds * sample_rate))
    step = max(1, round((chunk_seconds - safe_overlap) * sample_rate))
    radius = round(safe_overlap * sample_rate / 2.0)
    min_tail = round(safe_overlap * sample_rate)

    starts = [0]
    while True:
        nominal = starts[-1] + step
        if nominal >= total:
            break
        if total - nominal <= min_tail:
            # Остаток слишком мал для отдельного куска: предыдущий кусок
            # растягивается до конца файла (его длина остаётся <= chunk_seconds).
            break
        snapped = _min_energy_sample(
            waveform, target=nominal, radius=radius, sample_rate=sample_rate
        )
        starts.append(max(starts[-1] + 1, min(snapped, total - 1)))

    bounds: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = total if index == len(starts) - 1 else min(start + chunk, total)
        bounds.append((start, end))
    return bounds


@dataclass(frozen=True, slots=True)
class _ChunkedSegment:
    """Реплика куска вместе с геометрией куска (для дедупликации перекрытий)."""

    segment: TranscriptionSegment
    chunk_center: float
    chunk_index: int


def _normalize_for_dedup(text: str) -> str:
    """Нижний регистр без пунктуации и лишних пробелов — для сравнения текста."""
    folded = _DEDUP_NON_WORD_RE.sub(" ", text.casefold())
    return _DEDUP_WHITESPACE_RE.sub(" ", folded).strip()


def _segments_are_duplicates(
    left: TranscriptionSegment, right: TranscriptionSegment
) -> bool:
    """Похожи ли две реплики на одну и ту же (совпадение по времени и тексту).

    Требуется и существенное перекрытие интервалов, и близость текста — так
    осмысленные короткие повторы («да», «да») не схлопываются.
    """
    overlap = min(left.end, right.end) - max(left.start, right.start)
    if overlap <= 0:
        return False

    shorter = min(left.end - left.start, right.end - right.start)
    union = max(left.end, right.end) - min(left.start, right.start)
    if shorter <= 0 or union <= 0:
        return False
    if overlap / shorter < _DEDUP_MIN_TIME_COVERAGE and overlap / union < 0.5:
        return False

    left_text = _normalize_for_dedup(left.text)
    right_text = _normalize_for_dedup(right.text)
    if not left_text or not right_text:
        return False
    if left_text == right_text or left_text in right_text or right_text in left_text:
        return True
    if (
        difflib.SequenceMatcher(None, left_text, right_text).ratio()
        >= _DEDUP_MIN_TEXT_RATIO
    ):
        return True

    left_words = set(left_text.split())
    right_words = set(right_text.split())
    if left_words and right_words:
        jaccard = len(left_words & right_words) / len(left_words | right_words)
        if jaccard >= _DEDUP_MIN_TOKEN_JACCARD:
            return True
    return False


def _centrality_distance(item: _ChunkedSegment) -> float:
    """Насколько реплика далека от центра своего куска (меньше — центральнее)."""
    midpoint = (item.segment.start + item.segment.end) / 2.0
    return abs(midpoint - item.chunk_center)


def _deduplicate_chunk_segments(
    items: list[_ChunkedSegment],
) -> list[TranscriptionSegment]:
    """Убирает дубликаты реплик из областей перекрытия соседних кусков.

    Дубликатами считаются реплики из **разных** кусков, совпадающие по времени и
    близкие по тексту (одна и та же фраза, распознанная у края двух кусков).
    Остаётся вариант из более «центрального» куска: у края куска распознавание
    искажено границей, а в центре — надёжнее. Реплики одного куска не
    схлопываются — это законные соседние фразы.
    """
    ordered = sorted(items, key=lambda item: (item.segment.start, item.segment.end))
    kept: list[_ChunkedSegment] = []
    for candidate in ordered:
        duplicate_at: int | None = None
        for index, existing in enumerate(kept):
            if existing.chunk_index == candidate.chunk_index:
                continue
            # Дешёвая проверка перекрытия по времени до сравнения текста.
            if (
                existing.segment.end <= candidate.segment.start
                or existing.segment.start >= candidate.segment.end
            ):
                continue
            if _segments_are_duplicates(existing.segment, candidate.segment):
                duplicate_at = index
                break
        if duplicate_at is None:
            kept.append(candidate)
            continue

        existing = kept[duplicate_at]
        candidate_is_better = _centrality_distance(candidate) < _centrality_distance(
            existing
        ) or (
            _centrality_distance(candidate) == _centrality_distance(existing)
            and len(candidate.segment.text) > len(existing.segment.text)
        )
        if candidate_is_better:
            kept[duplicate_at] = candidate

    result = [item.segment for item in kept]
    result.sort(key=lambda segment: (segment.start, segment.end))
    return result


class WhisperCppRecognizer:
    """Распознаёт речь через whisper.cpp. Реализует протокол ``SpeechRecognizer``."""

    def __init__(
        self,
        model_path: Path,
        *,
        binary: str = "whisper-cli",
        library_path: str | None = None,
        threads: int | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress: ProgressCallback | None = None,
        vad_filter: bool = True,
        vad_model: Path | None = None,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        chunk_overlap: float = DEFAULT_CHUNK_OVERLAP,
        word_timestamps: bool = True,
        timeout: float | None = DEFAULT_WHISPER_CPP_TIMEOUT,
    ) -> None:
        self._model_path = Path(model_path)
        self._binary = binary
        self._library_path = library_path
        self._threads = threads
        self._initial_prompt = initial_prompt
        self._hotwords = hotwords
        self._on_progress = on_progress
        self._vad_filter = vad_filter
        self._vad_model = Path(vad_model) if vad_model is not None else None
        #: Собирать ли пословные таймстемпы из токенов ``-ojf`` (#45).
        self._word_timestamps = word_timestamps
        #: Таймаут одного вызова whisper-cli; ``None`` — без ограничения.
        self._timeout: float | None = (
            float(timeout) if timeout is not None and timeout > 0 else None
        )
        # ``chunk_seconds <= 0`` — чанкинг выключен (старое поведение: один
        # вызов whisper-cli на весь файл).
        self._chunk_seconds = max(0.0, float(chunk_seconds))
        overlap = max(0.0, float(chunk_overlap))
        if self._chunk_seconds > 0.0:
            overlap = min(overlap, self._chunk_seconds / 2.0)
        self._chunk_overlap = overlap

    @property
    def chunk_seconds(self) -> float:
        """Целевая длина куска (с); ``0`` — чанкинг выключен."""
        return self._chunk_seconds

    @property
    def chunk_overlap(self) -> float:
        """Перекрытие соседних кусков (с)."""
        return self._chunk_overlap

    @property
    def word_timestamps(self) -> bool:
        """Собираются ли пословные таймстемпы (#45)."""
        return self._word_timestamps

    @property
    def timeout(self) -> float | None:
        """Таймаут одного вызова whisper-cli (с); ``None`` — без ограничения."""
        return self._timeout

    def _emit(self, fraction: float | None = None, detail: str = "") -> None:
        if self._on_progress is not None:
            self._on_progress(
                ProgressEvent("asr", "Распознавание речи", fraction=fraction, detail=detail)
            )

    def _build_prompt(self) -> str | None:
        """Собирает подсказку для ASR из initial_prompt и hotwords.

        whisper.cpp не имеет отдельного механизма hotwords (в отличие от
        faster-whisper), поэтому оба источника объединяются в ``--prompt``.
        """
        parts: list[str] = []
        if self._initial_prompt and self._initial_prompt.strip():
            parts.append(self._initial_prompt.strip())
        if self._hotwords and self._hotwords.strip():
            parts.append(self._hotwords.strip())
        return " ".join(parts) or None

    def _vad_args(self) -> list[str]:
        """Флаги VAD для whisper-cli, выровненные с faster-whisper.

        whisper.cpp включает VAD только вместе с моделью Silero (``--vad-model``).
        Если фильтр включён, а модель не задана/не найдена, VAD мягко
        пропускается (как и прочие необязательные возможности проекта), чтобы
        не ронять распознавание.
        """
        if not self._vad_filter:
            return []
        if self._vad_model is None:
            logger.debug(
                "VAD включён, но модель whisper.cpp VAD не задана "
                "(WHISPER_CPP_VAD_MODEL) — распознавание идёт без VAD"
            )
            return []
        if not self._vad_model.is_file():
            logger.warning(
                "Модель whisper.cpp VAD не найдена: %s — распознавание идёт без VAD",
                self._vad_model,
            )
            return []
        return [
            "--vad",
            "-vm",
            str(self._vad_model),
            # Значения — как в faster_whisper.vad.VadOptions по умолчанию.
            "-vt",
            str(VAD_THRESHOLD),
            "-vspd",
            str(VAD_MIN_SPEECH_DURATION_MS),
            "-vsd",
            str(VAD_MIN_SILENCE_DURATION_MS),
            "-vp",
            str(VAD_SPEECH_PAD_MS),
        ]

    def _prepare_input(self, audio_path: Path, tmpdir_path: Path) -> tuple[Path, float]:
        """Готовит вход для whisper-cli и возвращает (путь, длительность, с).

        «Родной» для whisper-cli вход (16-кГц моно PCM WAV, например результат
        шумоподавления) отдаётся в ``-f`` как есть: лишний round-trip
        декодирование→запись вносил разницу в 1 LSB, из-за которой whisper.cpp
        терял речь (~59 с на реальном файле). Остальные форматы, которые
        miniaudio не декодирует (mp4/webm/mp3/…), а также WAV другой частоты или
        числа каналов по-прежнему перекодируются во временный 16-кГц WAV.

        Длительность берём из заголовка при passthrough — декодировать сэмплы
        для этого не нужно; иначе она равна длине декодированного waveform.
        """

        probe: AudioProbe | None
        try:
            probe = probe_audio(audio_path)
        except AudioFileError:
            probe = None

        if probe is not None and _is_native_wav(probe):
            logger.debug("whisper.cpp: вход отдаётся напрямую (%s)", audio_path)
            duration = probe.duration_seconds or 0.0
            return audio_path, duration

        wav_path = tmpdir_path / "audio.wav"
        waveform = load_waveform(audio_path)
        write_wav(wav_path, waveform)
        # Реальная длительность аудио (включая хвостовую тишину), а не конец
        # последнего сегмента — VAD отсекает тишину, из-за чего ``end``
        # последней реплики систематически занижает длительность.
        return wav_path, len(waveform) / SAMPLE_RATE

    def _run_whisper(
        self,
        *,
        input_path: Path,
        output_base: Path,
        language: str | None,
        on_fraction: Callable[[float], None] | None = None,
    ) -> tuple[list[TranscriptionSegment], str | None]:
        """Один вызов whisper-cli: прогоняет ``input_path`` и разбирает JSON.

        ``on_fraction`` получает прогресс распознавания (0..1) **внутри** этого
        вызова — используется, чтобы при чанкинге отображать общий прогресс.
        """
        cmd = [
            self._binary,
            "-m",
            str(self._model_path),
            "-f",
            str(input_path),
            "-l",
            language or "auto",
            # -ojf (полный JSON) дополнительно отдаёт вероятности токенов,
            # по которым считается средняя уверенность реплики
            # (аналог avg_logprob faster-whisper).
            "-ojf",
            "-of",
            str(output_base),
            # -pp печатает «progress = N%» в stderr — по нему TUI
            # показывает реальный прогресс распознавания.
            "-pp",
        ]
        cmd += self._vad_args()
        if self._threads:
            cmd += ["-t", str(self._threads)]
        prompt = self._build_prompt()
        if prompt:
            cmd += ["--prompt", prompt]

        env = with_library_path(os.environ, self._library_path)

        logger.debug("Запуск whisper.cpp: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except FileNotFoundError as exc:
            raise TranscriptionError(f"Бинарник whisper-cli не найден: {self._binary}") from exc

        # Регистрируем процесс в общем реестре: при SIGINT/SIGTERM или
        # аварийном выходе whisper-cli не останется висеть.
        register_process(proc)
        stderr_tail: list[str] = []

        def _drain_stderr() -> None:
            """Читает stderr в фоне: хвост для диагностики + прогресс-события."""
            assert proc.stderr is not None
            try:
                for line in proc.stderr:
                    stderr_tail.append(line)
                    if len(stderr_tail) > 40:
                        stderr_tail.pop(0)
                    match = _PROGRESS_RE.search(line)
                    if match and on_fraction is not None:
                        on_fraction(float(match.group(1)) / 100.0)
            except (OSError, ValueError):
                # Поток закрыт при принудительном завершении процесса.
                pass

        reader = threading.Thread(
            target=_drain_stderr, name="whisper-cpp-stderr", daemon=True
        )
        reader.start()
        try:
            try:
                # Ждём завершения с дедлайном: при зависании (битый вход,
                # дедлок GPU/Vulkan) процесс не должен висеть вечно (#82).
                proc.wait(timeout=self._timeout)
            except subprocess.TimeoutExpired as exc:
                terminate_process(proc)
                reader.join(timeout=5.0)
                detail = "".join(stderr_tail).strip()[-2000:]
                message = (
                    f"whisper.cpp не завершился за {self._timeout:g} с "
                    "(возможен дедлок GPU/Vulkan или битый вход) — "
                    "процесс принудительно остановлен"
                )
                if detail:
                    message = f"{message}. Последние строки stderr: {detail}"
                raise TranscriptionError(message) from exc
        finally:
            terminate_process(proc)
        reader.join(timeout=5.0)

        json_path = Path(str(output_base) + ".json")
        if proc.returncode != 0 or not json_path.exists():
            detail = "".join(stderr_tail).strip()[-2000:]
            raise TranscriptionError(
                f"whisper.cpp завершился с ошибкой (код {proc.returncode}): {detail}"
            )

        data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = [
            TranscriptionSegment(
                start=item["offsets"]["from"] / 1000.0,
                end=item["offsets"]["to"] / 1000.0,
                text=item["text"].strip(),
                avg_logprob=_segment_avg_logprob(item),
                words=(
                    _word_timestamps(item.get("tokens"))
                    if self._word_timestamps
                    else []
                ),
            )
            for item in data.get("transcription", [])
            if item.get("text", "").strip()
        ]
        detected = data.get("result", {}).get("language")
        return segments, detected if isinstance(detected, str) else None

    def _transcribe_chunked(
        self, input_path: Path, tmpdir_path: Path, language: str | None
    ) -> tuple[list[TranscriptionSegment], str | None, float]:
        """Режет аудио на куски, распознаёт каждый отдельно и склеивает."""
        waveform = load_waveform(input_path)
        decoded_duration = len(waveform) / SAMPLE_RATE
        bounds = _choose_chunk_bounds(
            waveform,
            chunk_seconds=self._chunk_seconds,
            overlap_seconds=self._chunk_overlap,
            sample_rate=SAMPLE_RATE,
        )
        total_chunks = len(bounds)
        logger.info(
            "whisper.cpp: чанкинг длинного аудио — %d кусков по %.1f с "
            "(перекрытие %.1f с, длительность %.1f с)",
            total_chunks,
            self._chunk_seconds,
            self._chunk_overlap,
            decoded_duration,
        )

        items: list[_ChunkedSegment] = []
        detected: str | None = None
        for index, (start_sample, end_sample) in enumerate(bounds):
            chunk_path = tmpdir_path / f"chunk_{index:04d}.wav"
            write_wav(chunk_path, waveform[start_sample:end_sample])
            start_seconds = start_sample / SAMPLE_RATE
            center_seconds = ((start_sample + end_sample) / 2.0) / SAMPLE_RATE
            output_base = tmpdir_path / f"result_{index:04d}"

            def on_fraction(
                fraction: float, index: int = index, total: int = total_chunks
            ) -> None:
                self._emit(
                    fraction=(index + fraction) / total,
                    detail=f"кусок {index + 1}/{total}",
                )

            segments, chunk_language = self._run_whisper(
                input_path=chunk_path,
                output_base=output_base,
                language=language,
                on_fraction=on_fraction,
            )
            if detected is None and chunk_language:
                detected = chunk_language
            for segment in segments:
                items.append(
                    _ChunkedSegment(
                        segment=replace(
                            segment,
                            start=segment.start + start_seconds,
                            end=segment.end + start_seconds,
                            # Пословные метки получают тот же офсет куска, что
                            # и границы сегмента.
                            words=_shift_words(segment.words, start_seconds),
                        ),
                        chunk_center=center_seconds,
                        chunk_index=index,
                    )
                )
            self._emit(
                fraction=(index + 1) / total_chunks,
                detail=f"кусок {index + 1}/{total_chunks}",
            )

        segments = _deduplicate_chunk_segments(items)
        if self._word_timestamps:
            # Слова, разрезанные стыком соседних кусков, склеиваем после
            # дедупликации: остаётся вариант сегмента из более центрального
            # куска, а «продолжение» слова с другого куска приклеивается к нему.
            segments = _stitch_chunk_words(segments)
        logger.info(
            "whisper.cpp: чанкинг завершён — кусков %d, реплик после дедупликации %d",
            total_chunks,
            len(segments),
        )
        return segments, detected, decoded_duration

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        if not self._model_path.is_file():
            raise TranscriptionError(f"Модель whisper.cpp не найдена: {self._model_path}")

        with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path, audio_duration = self._prepare_input(audio_path, tmpdir_path)

            # Чанкинг только для длинного аудио. Если длительность неизвестна
            # (probe не дал заголовка), решает уже декодированный waveform —
            # ``_transcribe_chunked`` корректно обрабатывает и короткий файл.
            use_chunking = self._chunk_seconds > 0 and (
                audio_duration <= 0 or audio_duration > self._chunk_seconds
            )
            if use_chunking:
                segments, detected, decoded_duration = self._transcribe_chunked(
                    input_path, tmpdir_path, language
                )
                if decoded_duration > 0:
                    audio_duration = max(audio_duration, decoded_duration)
            else:
                output_base = tmpdir_path / "result"
                on_fraction = (
                    (lambda fraction: self._emit(fraction=fraction))
                    if self._on_progress is not None
                    else None
                )
                segments, detected = self._run_whisper(
                    input_path=input_path,
                    output_base=output_base,
                    language=language,
                    on_fraction=on_fraction,
                )
                if self._word_timestamps:
                    # Даже без чанкинга whisper.cpp может разрезать слово
                    # границей сегмента — сшиваем и снимаем служебный флаг.
                    segments = _stitch_chunk_words(segments)

        # Длительность берём из декодированного аудио. Если по какой-то причине
        # она неизвестна, откатываемся к концу последней реплики.
        duration = audio_duration if audio_duration > 0 else (segments[-1].end if segments else 0.0)

        return segments, language or detected or "", duration
