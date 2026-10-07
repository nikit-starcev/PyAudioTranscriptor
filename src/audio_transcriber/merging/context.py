"""Добор говорящего коротким репликам в «дырках» разметки диаризации (#93).

Диаризация оставляет длинные провалы (тишина, музыка, нераспознанная речь), и
короткая реплика внутри такого провала остаётся без говорящего (``None``):
``MAX_NEAREST_GAP_SECONDS`` намеренно не даёт «угадывать» далёкого говорящего.
Однако совсем короткая реплика (несколько слов), зажатая между репликами
размеченной речи, почти наверняка принадлежит ближайшему участнику — тогда
говорящий назначается по контексту, и реплик вовсе без говорящего в стенограмме
не остаётся.

Правило консервативное: назначаем только **коротким** репликам и только если
ближайшая размеченная реплика не дальше :data:`DEFAULT_CONTEXT_MAX_GAP_SECONDS`.
Из двух соседей берётся тот, что ближе по времени; при равном зазоре — сосед
слева. Если соседей нет или они слишком далеко, говорящий не выдумывается.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from audio_transcriber.domain.models import Speaker, TranscriptEntry

logger = logging.getLogger(__name__)

#: Максимум слов в реплике, которой можно добрать говорящего по контексту.
#: Короткая вставка/реплика в провале разметки — это несколько слов; длинную
#: реплику без говорящего контекстом закрывать нельзя (внутри мог быть кто
#: угодно).
DEFAULT_CONTEXT_MAX_WORDS = 4

#: Максимальный временной зазор (секунды) до ближайшей размеченной реплики, при
#: котором короткой реплике ещё можно добрать говорящего. Провалы разметки
#: диаризации на реальной записи достигали нескольких секунд; больше — уже
#: слепая догадка.
DEFAULT_CONTEXT_MAX_GAP_SECONDS = 15.0


def assign_context_speakers(
    entries: list[TranscriptEntry],
    *,
    max_words: int = DEFAULT_CONTEXT_MAX_WORDS,
    max_gap: float = DEFAULT_CONTEXT_MAX_GAP_SECONDS,
) -> list[TranscriptEntry]:
    """Назначает говорящего коротким репликам без него по ближайшему контексту.

    Реплика без говорящего получает говорящего ближайшей слева/справа реплики с
    говорящим, если сама не длиннее ``max_words`` слов, а ближайший сосед не
    дальше ``max_gap`` секунд. Иначе реплика остаётся без говорящего.
    """
    result = list(entries)
    assigned = 0
    for index, entry in enumerate(result):
        if entry.speaker is not None or _word_count(entry) > max_words:
            continue
        neighbours = _nearest_with_speaker(result, index, -1), _nearest_with_speaker(
            result, index, 1
        )
        speaker = _closest_speaker(entry, neighbours, max_gap)
        if speaker is None:
            continue
        result[index] = replace(entry, speaker=speaker)
        assigned += 1

    if assigned:
        logger.info("Добор говорящего по контексту: %d реплик", assigned)
    return result


def _word_count(entry: TranscriptEntry) -> int:
    """Число слов реплики: по пословным меткам, иначе по тексту."""
    if entry.words:
        return len(entry.words)
    return len(entry.text.split())


def _closest_speaker(
    entry: TranscriptEntry,
    neighbours: tuple[TranscriptEntry | None, TranscriptEntry | None],
    max_gap: float,
) -> Speaker | None:
    """Говорящий ближайшего соседа в пределах ``max_gap`` (или ``None``)."""
    previous, following = neighbours
    candidates: list[tuple[float, int, Speaker]] = []
    if previous is not None and previous.speaker is not None:
        candidates.append((entry.start - previous.end, 0, previous.speaker))
    if following is not None and following.speaker is not None:
        candidates.append((following.start - entry.end, 1, following.speaker))
    if not candidates:
        return None
    # Минимальный зазор; при равенстве предпочитаем соседа слева (порядок 0).
    candidates.sort(key=lambda item: (item[0], item[1]))
    gap, _order, speaker = candidates[0]
    return speaker if gap <= max_gap else None


def _nearest_with_speaker(
    entries: list[TranscriptEntry], start: int, step: int
) -> TranscriptEntry | None:
    """Ближайшая реплика с говорящим в направлении ``step`` (``-1``/``+1``)."""
    index = start + step
    while 0 <= index < len(entries):
        if entries[index].speaker is not None:
            return entries[index]
        index += step
    return None
