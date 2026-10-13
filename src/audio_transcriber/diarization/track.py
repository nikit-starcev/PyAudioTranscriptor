"""Дорожка отдельного говорящего: заглушение речи остальных участников (#108).

По таймкодам реплик строится набор интервалов, которые нужно **заглушить**
(речь других говорящих), и по нему правится waveform исходной записи. Основной
говорящий реплики и участники наложения (``extra_speakers``) считаются «своими»
для выбранного говорящего; зоны без реплик (паузы, фоновый шум) не трогаются —
их заглушение не определено таймкодами реплик.

Разделения источников здесь нет: это «solo»-дорожка (mute остальных), а не TSE
по enrollment-эмбеддингу. Замена заглушения на извлечение целевого голоса —
следующий шаг (см. ``diarization/embeddings`` и ``diarization/enrollment``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from audio_transcriber.diarization.samples import Interval
from audio_transcriber.domain.models import TranscriptEntry
from audio_transcriber.utils.audio import SAMPLE_RATE

#: Множитель амплитуды на заглушаемых участках: ``0.0`` — полная тишина.
DEFAULT_TRACK_ATTENUATION = 0.0

#: Длительность линейного сглаживания на границах заглушения (секунды):
#: убирает щелчки от скачка амплитуды.
DEFAULT_TRACK_FADE_SECONDS = 0.015


def _entry_speaker_ids(entry: TranscriptEntry) -> tuple[str, ...]:
    """Идентификаторы всех говорящих реплики: основной и участники наложения."""
    ids: list[str] = []
    if entry.speaker is not None:
        ids.append(entry.speaker.id)
    ids.extend(speaker.id for speaker in entry.extra_speakers)
    return tuple(ids)


def _merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    """Склеивает пересекающиеся/соприкасающиеся интервалы в непересекающиеся."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def mute_intervals(
    entries: Sequence[TranscriptEntry],
    speaker_id: str,
    *,
    include_overlap: bool = True,
) -> list[Interval]:
    """Интервалы реплик, которые нужно заглушить для дорожки ``speaker_id``.

    Заглушается реплика, в которой выбранный говорящий не участвует (ни как
    основной, ни как участник наложения). При ``include_overlap=False``
    дополнительно глушатся реплики-наложения, где выбранный говорит вместе с
    другими: убрать чужие голоса из общей реплики без разделения источников
    нельзя, поэтому такой режим отдаёт «только чистые реплики» говорящего.
    Реплики без говорящего считаются чужими и глушатся.
    """
    muted: list[Interval] = []
    for entry in entries:
        if entry.end <= entry.start:
            continue
        ids = _entry_speaker_ids(entry)
        if speaker_id not in ids or (not include_overlap and len(set(ids)) > 1):
            muted.append((entry.start, entry.end))
    return _merge_intervals(muted)


def apply_track(
    waveform: np.ndarray,
    *,
    intervals: Sequence[Interval],
    sample_rate: int = SAMPLE_RATE,
    attenuation: float = DEFAULT_TRACK_ATTENUATION,
    fade_seconds: float = DEFAULT_TRACK_FADE_SECONDS,
) -> np.ndarray:
    """Возвращает копию waveform с ослабленной амплитудой на ``intervals``.

    На каждом заглушаемом участке амплитуда умножается на ``attenuation``
    (``0.0`` — тишина), а на границах участка применяется короткое линейное
    сглаживание к исходной громкости, чтобы не было щелчков. Участки берутся
    уже объединёнными (см. :func:`mute_intervals`), поэтому сглаживания соседних
    интервалов не накладываются.
    """
    samples = np.array(waveform, dtype=np.float32, copy=True)
    total = samples.shape[0]
    if total == 0 or not intervals:
        return samples
    level = float(attenuation)
    if not 0.0 <= level < 1.0:
        raise ValueError("attenuation должен быть в диапазоне [0, 1)")
    fade = max(0, round(fade_seconds * sample_rate))
    for start, end in intervals:
        first = max(0, min(total, round(start * sample_rate)))
        last = max(0, min(total, round(end * sample_rate)))
        span = last - first
        if span <= 0:
            continue
        gains = np.full(span, level, dtype=np.float32)
        edge = min(fade, span // 2)
        if edge > 0:
            if first > 0:
                gains[:edge] = np.linspace(1.0, level, edge, endpoint=False, dtype=np.float32)
            if last < total:
                gains[-edge:] = np.linspace(level, 1.0, edge, endpoint=False, dtype=np.float32)
        samples[first:last] *= gains
    return samples


def fingerprint_intervals(intervals: Sequence[Interval]) -> str:
    """Короткий устойчивый хэш интервалов — ключ кэша дорожки говорящего."""
    payload = ",".join(f"{round(start, 3):.3f}:{round(end, 3):.3f}" for start, end in intervals)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
