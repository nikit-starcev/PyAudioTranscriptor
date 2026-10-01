"""Вычисление интервалов наложения речи (>= 2 говорящих одновременно).

Эксклюзивная диаризация присваивает каждому моменту одного «доминирующего»
говорящего, поэтому зоны перекрытия в ней теряются. Если у результата pyannote
доступна обычная (не эксклюзивная) разметка ``speaker_diarization``, из неё
можно восстановить интервалы, где реально говорили двое и более человек.

Функция :func:`compute_overlap_regions` — чистый алгоритм (без pyannote): по
списку интервалов говорящих строит объединённые зоны с числом активных
говорящих >= 2.
"""

from __future__ import annotations

from collections.abc import Iterable

from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment

_MIN_OVERLAP_SPEAKERS = 2

# Версия формата/алгоритма диаризации для ключа кэша. Изменение списка
# участников зон наложения (``SpeakerOverlap.speaker_ids``) меняет результат
# при тех же входных параметрах, поэтому старый кэш должен быть пересчитан
# ровно один раз (см. ``pipeline._diarization_cache_params``).
# 3 — добавлены настраиваемые гиперпараметры диаризации (min_duration_off,
# cluster threshold/Fb) и диапазон числа говорящих.
DIARIZATION_IMPL_VERSION = 3


def compute_overlap_regions(
    speaker_segments: Iterable[SpeakerSegment],
) -> list[SpeakerOverlap]:
    """Возвращает интервалы, где одновременно активны два и более говорящих.

    Реализовано линейным заметанием: события начала (+1) и конца (-1)
    сортируются по времени, при равенстве сначала обрабатываются концы — чтобы
    стык «один закончил, другой начал» не давал ложного нулевого наложения.

    Каждая зона несёт ``speaker_ids`` — идентификаторы всех говорящих, активных
    внутри неё хотя бы в один момент (объединение), в устойчивом порядке
    (лексикографически). Это позволяет позже назвать участников наложения.
    """
    events: list[tuple[float, int, str]] = []
    for segment in speaker_segments:
        if segment.end <= segment.start:
            continue
        events.append((segment.start, 1, segment.speaker_id))
        events.append((segment.end, -1, segment.speaker_id))

    if not events:
        return []

    events.sort(key=lambda event: (event[0], event[1]))

    # ``active`` хранит только говорящих с положительным числом активных
    # сегментов: нулевые записи удаляются, иначе они бы попадали в участников
    # зоны. ``region_speakers`` — объединение активных за время зоны.
    active: dict[str, int] = {}
    distinct = 0
    region_start: float | None = None
    region_speakers: dict[str, None] = {}
    regions: list[SpeakerOverlap] = []

    for time, delta, speaker_id in events:
        if delta == 1:
            if active.get(speaker_id, 0) == 0:
                distinct += 1
            active[speaker_id] = active.get(speaker_id, 0) + 1
        else:
            remaining = active.get(speaker_id, 0) - 1
            if remaining > 0:
                active[speaker_id] = remaining
            else:
                active.pop(speaker_id, None)
                distinct -= 1

        if distinct >= _MIN_OVERLAP_SPEAKERS:
            if region_start is None:
                region_start = time
            for active_id in sorted(active):
                region_speakers.setdefault(active_id, None)
        elif region_start is not None:
            if time > region_start:
                regions.append(
                    SpeakerOverlap(
                        start=region_start,
                        end=time,
                        speaker_ids=tuple(region_speakers),
                    )
                )
            region_start = None
            region_speakers = {}

    return regions
