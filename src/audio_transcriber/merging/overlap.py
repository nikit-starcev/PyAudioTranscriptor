"""Пометка реплик, попавших в зоны наложения речи.

Диаризация помечает одного говорящего на момент, но иногда говорят
одновременно двое. Зоны такого наложения приходят отдельно (см.
:func:`audio_transcriber.diarization.overlap.compute_overlap_regions`); здесь
реплики, пересекающиеся с этими зонами, получают признак ``overlap``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from audio_transcriber.domain.models import SpeakerOverlap, TranscriptEntry

logger = logging.getLogger(__name__)


def _intersects(entry: TranscriptEntry, regions: Sequence[SpeakerOverlap]) -> bool:
    """Пересекается ли реплика хотя бы с одной зоной наложения."""
    return any(entry.start < region.end and entry.end > region.start for region in regions)


def mark_overlap_entries(
    entries: list[TranscriptEntry], regions: Sequence[SpeakerOverlap]
) -> list[TranscriptEntry]:
    """Возвращает реплики с проставленным признаком ``overlap``.

    Реплики вне зон наложения возвращаются как есть; уже помеченные не
    дублируются.
    """
    if not regions:
        return entries

    result: list[TranscriptEntry] = []
    marked = 0
    for entry in entries:
        if not entry.overlap and _intersects(entry, regions):
            result.append(replace(entry, overlap=True))
            marked += 1
        else:
            result.append(entry)

    if marked:
        logger.info("Наложение речи: помечено реплик — %d", marked)
    return result
