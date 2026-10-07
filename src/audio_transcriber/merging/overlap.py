"""Пометка реплик, попавших в зоны наложения речи, и добор сов-говорящих.

Диаризация помечает одного говорящего на момент, но иногда говорят
одновременно двое. Зоны такого наложения приходят отдельно (см.
:func:`audio_transcriber.diarization.overlap.compute_overlap_regions`); здесь
реплики, пересекающиеся с этими зонами, получают признак ``overlap``, а если
зоны несут участников (``speaker_ids``) — ещё и ``extra_speakers`` с именами
сов-говорящих.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from audio_transcriber.domain.models import Speaker, SpeakerOverlap, TranscriptEntry
from audio_transcriber.merging.aligner import DEFAULT_OVERLAP_MIN_SECONDS

logger = logging.getLogger(__name__)


def _intersects(entry: TranscriptEntry, regions: Sequence[SpeakerOverlap]) -> bool:
    """Пересекается ли реплика хотя бы с одной зоной наложения."""
    return any(entry.start < region.end and entry.end > region.start for region in regions)


def mark_overlap_entries(
    entries: list[TranscriptEntry], regions: Sequence[SpeakerOverlap]
) -> list[TranscriptEntry]:
    """Возвращает реплики с проставленным признаком ``overlap``.

    Реплики вне зон наложения возвращаются как есть; уже помеченные не
    дублируются. Это «зональная» пометка без имён: используется как запасной
    путь, когда участники зон неизвестны (например, старый кэш диаризации).
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


def apply_overlap_regions(
    entries: list[TranscriptEntry],
    speakers: list[Speaker],
    regions: Sequence[SpeakerOverlap],
    *,
    known_speakers: dict[str, str] | None = None,
    overlap_min_seconds: float = DEFAULT_OVERLAP_MIN_SECONDS,
) -> tuple[list[TranscriptEntry], list[Speaker]]:
    """Добавляет сов-говорящих из зон наложения и помечает реплики.

    Дополнительный (к заметанию по ``speaker_segments``) путь: pyannote отдаёт
    эксклюзивную разметку без пересечений, поэтому набор говорящих в зоне
    наложения приходит только в ``SpeakerOverlap.speaker_ids``. Для каждой
    реплики:

    * собираются зоны, пересекающиеся с её интервалом (пересечение > 0);
    * если пересечение зоны с репликой не меньше ``overlap_min_seconds``, все
      говорящие этой зоны (кроме основного) становятся кандидатами, а их
      суммарное пересечение с репликой накапливается;
    * кандидаты сортируются по убыванию суммарного пересечения (при равенстве —
      по идентификатору) и объединяются с уже имеющимися ``extra_speakers``
      (их мог дать заметание по перекрывающимся сегментам) без дублей;
    * ``overlap`` взводится, если есть хоть один сов-говорящий или хоть одно
      пересечение с зоной. Для старого кэша без ``speaker_ids`` сов-говорящих
      нет, но ``overlap`` по-прежнему выставляется по зонам.

    Новые говорящие регистрируются в ``speakers`` (следующим по счёту «Спикер
    N» или известным именем из ``known_speakers``), поэтому LLM и экспорт видят
    их как обычных участников.
    """
    if not regions:
        return entries, speakers

    names = known_speakers or {}
    speakers_by_id: dict[str, Speaker] = {speaker.id: speaker for speaker in speakers}

    result: list[TranscriptEntry] = []
    added = 0
    marked = 0
    for entry in entries:
        main_id = entry.speaker.id if entry.speaker is not None else None
        durations: dict[str, float] = {}
        intersects = False
        for region in regions:
            overlap_seconds = min(entry.end, region.end) - max(entry.start, region.start)
            if overlap_seconds <= 0.0:
                continue
            intersects = True
            if overlap_seconds < overlap_min_seconds:
                continue
            for speaker_id in region.speaker_ids:
                if speaker_id == main_id:
                    continue
                durations[speaker_id] = durations.get(speaker_id, 0.0) + overlap_seconds

        candidates = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
        extra_speakers = list(entry.extra_speakers)
        seen = {speaker.id for speaker in extra_speakers}
        for speaker_id, _seconds in candidates:
            if speaker_id in seen:
                continue
            speaker = speakers_by_id.get(speaker_id)
            if speaker is None:
                speaker = Speaker(
                    id=speaker_id,
                    display_name=names.get(speaker_id, f"Спикер {len(speakers_by_id) + 1}"),
                )
                speakers_by_id[speaker_id] = speaker
                added += 1
            extra_speakers.append(speaker)
            seen.add(speaker_id)

        overlap = entry.overlap or bool(extra_speakers) or intersects
        if extra_speakers != entry.extra_speakers or overlap != entry.overlap:
            if overlap and not entry.overlap:
                marked += 1
            result.append(replace(entry, extra_speakers=extra_speakers, overlap=overlap))
        else:
            result.append(entry)

    if added:
        logger.info("Наложение речи: добавлено сов-говорящих — %d", added)
    if marked:
        logger.info("Наложение речи: помечено реплик — %d", marked)
    return result, list(speakers_by_id.values())


def trim_artifact_overlaps(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
    """Убирает пересечения соседних реплик, не помеченные как наложение речи.

    На стыке ASR-чанков последняя реплика предшествующего куска может быть
    растянута до его конца, а первая реплика следующего куска начинается на
    перекрытие раньше — интервалы ``start``/``end`` пересекаются, хотя реального
    одновременного говора нет. Такое пересечение (обе реплики **без** признака
    ``overlap`` и **без** сов-говорящих) — артефакт выравнивания/склейки:
    ``end`` предыдущей реплики подрезается до ``start`` следующей. Настоящее
    наложение речи (``overlap``/``extra_speakers``) не трогается — оно несёт
    участников и должно отображаться как перекрытие.

    Пословные метки подрезанной реплики, вылезающие за новый ``end``, тоже
    ограничиваются, чтобы экспорт не показывал слова за пределами реплики.
    """
    result: list[TranscriptEntry] = []
    trimmed = 0
    for entry in entries:
        if result:
            previous = result[-1]
            artifact = (
                previous.end > entry.start > previous.start
                and not previous.overlap
                and not entry.overlap
                and not previous.extra_speakers
                and not entry.extra_speakers
            )
            if artifact:
                new_end = entry.start
                words = [
                    replace(word, end=min(word.end, new_end))
                    if word.end > new_end
                    else word
                    for word in previous.words
                ]
                words = [
                    word if word.end >= word.start else replace(word, end=word.start)
                    for word in words
                ]
                result[-1] = replace(previous, end=new_end, words=words)
                trimmed += 1
                logger.debug(
                    "Артефакт наложения: реплика подрезана %.2f → %.2f с",
                    previous.end,
                    new_end,
                )
        result.append(entry)

    if trimmed:
        logger.info("Артефакты наложения речи: подрезано реплик — %d", trimmed)
    return result
