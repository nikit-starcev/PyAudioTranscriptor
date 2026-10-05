"""Автослияние кластеров диаризации с одинаковым уверенно присвоенным именем.

Enrollment работает в режиме many-to-one: один реальный участник, раздробленный
диаризацией на несколько кластеров (разные окна/условия записи), получает одно и
то же имя сразу в нескольких кластерах. В стенограмме это выглядит как несколько
«говорящих» с одинаковым именем. Функции модуля сворачивают такие кластеры в
одного: реплики (включая дополнительных участников наложения) переводятся на один
``speaker_id``, а дубликаты убираются из списка ``speakers``.

Сливаются только говорящие с **точным** совпадением уверенно присвоенного имени
(enrollment, ручное переименование или ``--speaker-name``). Безымянные метки
(«Спикер N», ``SPEAKER_NN``, ``?``) и пустые имена не сливаются; нечёткое
совпадение имён (регистр/опечатки) намеренно не используется. Целевым становится
первый говорящий с данным именем — как правило, самый крупный кластер, идущий в
списке первым.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from audio_transcriber.domain.editing import merge_speaker_lists
from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult
from audio_transcriber.llm.chunking import human_name

logger = logging.getLogger(__name__)


def same_name_merge_pairs(speakers: Sequence[Speaker]) -> list[tuple[str, str]]:
    """Пары ``(source_id, target_id)`` для слияния говорящих с одним именем.

    Целевым становится **первый** в списке говорящий с данным уверенным именем,
    остальные — источниками. Уверенным считается имя, которое :func:`human_name`
    распознаёт как человеческое (не «Спикер N»/``SPEAKER_NN``); совпадение —
    точное. Возвращённый порядок устойчив (порядок списка говорящих).
    """
    target_by_name: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for speaker in speakers:
        name = human_name(speaker.display_name)
        if name is None:
            continue
        target_id = target_by_name.get(name)
        if target_id is None:
            target_by_name[name] = speaker.id
        elif target_id != speaker.id:
            pairs.append((speaker.id, target_id))
    return pairs


def merge_same_name_speakers(
    entries: list[TranscriptEntry], speakers: list[Speaker]
) -> tuple[list[TranscriptEntry], list[Speaker]]:
    """Сводит кластеры с одинаковым уверенным именем в одного говорящего.

    Возвращает новые списки реплик и говорящих; при отсутствии пар для слияния
    входные списки возвращаются без изменений. Дополнительные говорящие наложения
    переименовываются вместе с основными (см. :func:`merge_speaker_lists`), поэтому
    «висячих» id после слияния не остаётся. Каждое слияние логируется на INFO.
    """
    pairs = same_name_merge_pairs(speakers)
    if not pairs:
        return entries, speakers
    for source_id, target_id in pairs:
        entries, speakers = merge_speaker_lists(entries, speakers, source_id, target_id)
        logger.info(
            "Слияние кластеров с одинаковым именем говорящего: %s → %s",
            source_id,
            target_id,
        )
    return entries, speakers


def merge_result_same_name_speakers(result: TranscriptionResult) -> TranscriptionResult:
    """Результат-обёртка :func:`merge_same_name_speakers`.

    Удобна там, где стенограмма уже собрана (веб-слой, протокол): возвращает
    новый :class:`TranscriptionResult` с теми же полями. Если сливать нечего —
    возвращается исходный объект.
    """
    entries, speakers = merge_same_name_speakers(result.entries, result.speakers)
    if entries is result.entries and speakers is result.speakers:
        return result
    return replace(result, entries=entries, speakers=speakers)
