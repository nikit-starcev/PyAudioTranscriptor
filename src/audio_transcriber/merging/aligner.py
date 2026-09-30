"""Объединение сегментов ASR и диаризации по максимальному перекрытию во времени."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
)

_MAX_GAP_DEFAULT = 5.0

# Запись интервала диаризации для оффлайн-обходов:
# ``(sort_key, coord, original_index, score)``. ``score`` максимизируется
# (кортежное сравнение), ``coord`` — координата в BIT (end или start).
_Record = tuple[float, float, int, tuple[float, ...]]
# Запрос: ``(sort_key, coord_bound, query_id)``.
_Query = tuple[float, float, int]


class OverlapSegmentMerger:
    """Присваивает каждому сегменту речи говорящего с наибольшим перекрытием.

    Если перекрытия нет (реплика попала между интервалами диаризации),
    берётся ближайший по времени говорящий — но не далее ``max_gap`` секунд,
    иначе говорящий не назначается. Реализует протокол ``SegmentMerger``.
    """

    def __init__(self, *, max_gap: float = _MAX_GAP_DEFAULT) -> None:
        self._max_gap = max_gap

    def merge(
        self,
        transcription_segments: list[TranscriptionSegment],
        speaker_segments: list[SpeakerSegment],
        known_speakers: dict[str, str] | None = None,
    ) -> tuple[list[TranscriptEntry], list[Speaker]]:
        known_speakers = known_speakers or {}
        speakers_by_id: dict[str, Speaker] = {}
        entries: list[TranscriptEntry] = []
        resolver = _SpeakerResolver(speaker_segments, max_gap=self._max_gap)
        speaker_ids = resolver.best_speaker_ids(transcription_segments)

        for segment, speaker_id in zip(transcription_segments, speaker_ids, strict=True):
            speaker = None
            if speaker_id is not None:
                if speaker_id not in speakers_by_id:
                    index = len(speakers_by_id) + 1
                    display_name = known_speakers.get(speaker_id, f"Спикер {index}")
                    speakers_by_id[speaker_id] = Speaker(id=speaker_id, display_name=display_name)
                speaker = speakers_by_id[speaker_id]
            entries.append(
                TranscriptEntry(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=speaker,
                    avg_logprob=segment.avg_logprob,
                )
            )

        return entries, list(speakers_by_id.values())


def _better_overlap(
    current: tuple[float, float] | None, candidate: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Выбирает кандидата с максимальным перекрытием, при равенстве — с меньшим индексом."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    if candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
        return candidate
    return current


def _better_gap(
    current: tuple[float, float] | None, candidate: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Выбирает кандидата с минимальным зазором, при равенстве — с меньшим индексом."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    if candidate[0] < current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
        return candidate
    return current


def _sweep(
    records: Sequence[_Record],
    queries: Sequence[_Query],
    *,
    ascending: bool,
    select_ge: bool,
    coords: Sequence[float],
) -> dict[int, tuple[float, ...]]:
    """Оффлайн-обход с деревом Фенвика (BIT) для максимума score.

    Записи добавляются по ``sort_key`` (возрастающе — ``<=`` ключа запроса,
    убывающе — ``>=``), после чего берётся максимум ``score`` среди добавленных
    с координатой ``<= bound`` (``select_ge=False``) или ``>= bound``
    (``select_ge=True``). Сложность — ``O((n + m) log n)``.
    """
    if not records or not queries:
        return {}

    size = len(coords)
    position = {value: index for index, value in enumerate(coords)}
    tree: list[tuple[float, ...] | None] = [None] * (size + 1)

    def update(index: int, value: tuple[float, ...]) -> None:
        i = index + 1
        while i <= size:
            current = tree[i]
            if current is None or value > current:
                tree[i] = value
            i += i & (-i)

    def query(index: int) -> tuple[float, ...] | None:
        if index < 0:
            return None
        best: tuple[float, ...] | None = None
        i = index + 1
        while i > 0:
            current = tree[i]
            if current is not None and (best is None or current > best):
                best = current
            i -= i & (-i)
        return best

    ordered_records = sorted(records, key=lambda record: record[0], reverse=not ascending)
    ordered_queries = sorted(queries, key=lambda item: item[0], reverse=not ascending)

    result: dict[int, tuple[float, ...]] = {}
    pointer = 0
    for sort_key, bound, query_id in ordered_queries:
        while pointer < len(ordered_records):
            record = ordered_records[pointer]
            if ascending:
                if record[0] > sort_key:
                    break
            elif record[0] < sort_key:
                break
            bit_index = position[record[1]]
            if select_ge:
                bit_index = size - 1 - bit_index
            update(bit_index, record[3])
            pointer += 1

        if select_ge:
            bit_index = size - 1 - bisect_left(coords, bound)
        else:
            bit_index = bisect_right(coords, bound) - 1
        best = query(bit_index)
        if best is not None:
            result[query_id] = best

    return result


class _SpeakerResolver:
    """Ищет говорящего для реплики: максимальное перекрытие, затем ближайший.

    Диаризация обычно выдаёт неперекрывающиеся интервалы, но при наложении речи
    интервалы пересекаются. Чтобы не перебирать все ``M`` интервалов на каждую
    реплику (``O(N * M)``), ответы считаются пакетно оффлайн: четыре
    «квадранта» максимального перекрытия и три запроса ближайшего говорящего
    обрабатываются ленивыми обходами с BIT за ``O((N + M) log N)``. Результат
    полностью совпадает с прежним перебором «в лоб», включая тай-брейк по
    исходному порядку интервалов и произвольные (в т.ч. множественные)
    перекрытия.
    """

    def __init__(
        self, speaker_segments: Sequence[SpeakerSegment], *, max_gap: float
    ) -> None:
        self._max_gap = max_gap
        self._brute: list[SpeakerSegment] = list(speaker_segments)
        # ``(start, end, original_index)`` — исходный индекс нужен для тай-брейка.
        self._records: list[tuple[float, float, int]] = [
            (segment.start, segment.end, index)
            for index, segment in enumerate(speaker_segments)
        ]
        self._speaker_ids = [segment.speaker_id for segment in speaker_segments]
        # Обратные (end < start) интервалы нефизичны и не покрываются быстрым
        # путём — для них сохраняем прежний точный перебор, чтобы не менять
        # поведение на любых входных данных.
        self._fallback = any(end < start for start, end, _ in self._records)
        if self._records and not self._fallback:
            self._coords_e = sorted({end for _, end, _ in self._records})
            self._coords_s = sorted({start for start, _, _ in self._records})

    def best_speaker_id(self, segment: TranscriptionSegment) -> str | None:
        return self.best_speaker_ids([segment])[0]

    def best_speaker_ids(
        self, transcription_segments: Sequence[TranscriptionSegment]
    ) -> list[str | None]:
        count = len(transcription_segments)
        if not self._records:
            return [None] * count
        # Нефизичные интервалы (обратные) не покрываются быстрым путём — для
        # любых таких входных данных сохраняем прежний точный перебор.
        if self._fallback or any(
            segment.end < segment.start for segment in transcription_segments
        ):
            return [self._brute_best_speaker_id(segment) for segment in transcription_segments]

        queries: list[tuple[float, float, int]] = [
            (segment.start, segment.end, index)
            for index, segment in enumerate(transcription_segments)
        ]
        if not queries:
            return []

        resolution: list[str | None] = [None] * count
        best_overlap = self._best_overlap(queries)
        unresolved: list[tuple[float, float, int]] = []
        for query in queries:
            candidate = best_overlap.get(query[2])
            if candidate is not None and candidate[0] > 0.0:
                resolution[query[2]] = self._speaker_ids[int(candidate[1])]
            else:
                unresolved.append(query)

        if unresolved:
            nearest = self._nearest(unresolved)
            for query in unresolved:
                candidate = nearest.get(query[2])
                if candidate is None:
                    continue
                index, gap = candidate
                if gap <= self._max_gap:
                    resolution[query[2]] = self._speaker_ids[int(index)]

        return resolution

    # -- максимальное перекрытие ----------------------------------------------
    def _best_overlap(
        self, queries: Sequence[tuple[float, float, int]]
    ) -> dict[int, tuple[float, float]]:
        """Для каждой реплики — (перекрытие, исходный индекс) лучшего говорящего.

        Реплика ``[qs, qe]`` и интервал ``[s, e]`` разбиваются на четыре
        квадранта по взаимному положению границ; перекрытие в каждом имеет
        простую форму (``e - qs``, ``qe - s``, ``qe - qs``, ``e - s``), а
        максимум по каждому квадранту берётся одним обходом.
        """
        records_e: list[_Record] = []
        records_d: list[_Record] = []
        for start, end, index in self._records:
            records_e.append((start, end, index, (end, -index)))
            records_d.append((start, end, index, (end - start, -index)))
        records_b = [(s, e, i, (-s, -i)) for s, e, i in self._records]
        records_c = [(s, e, i, (-i,)) for s, e, i in self._records]

        queries_cd = list(queries)
        # A: s <= qs, e <= qe  →  overlap = e - qs
        result_a = _sweep(records_e, queries_cd, ascending=True, select_ge=False, coords=self._coords_e)
        # B: s >= qs, e >= qe  →  overlap = qe - s
        result_b = _sweep(records_b, queries_cd, ascending=False, select_ge=True, coords=self._coords_e)
        # C: s <= qs, e >= qe  →  overlap = qe - qs (полное перекрытие реплики)
        result_c = _sweep(records_c, queries_cd, ascending=True, select_ge=True, coords=self._coords_e)
        # D: s >= qs, e <= qe  →  overlap = e - s
        result_d = _sweep(records_d, queries_cd, ascending=False, select_ge=False, coords=self._coords_e)

        best: dict[int, tuple[float, float]] = {}
        for qs, qe, query_id in queries_cd:
            candidate: tuple[float, float] | None = None
            found_a = result_a.get(query_id)
            if found_a is not None:
                candidate = _better_overlap(candidate, (found_a[0] - qs, -found_a[1]))
            found_b = result_b.get(query_id)
            if found_b is not None:
                candidate = _better_overlap(candidate, (qe + found_b[0], -found_b[1]))
            found_c = result_c.get(query_id)
            if found_c is not None:
                candidate = _better_overlap(candidate, (qe - qs, -found_c[0]))
            found_d = result_d.get(query_id)
            if found_d is not None:
                candidate = _better_overlap(candidate, (found_d[0], -found_d[1]))
            if candidate is not None:
                best[query_id] = candidate
        return best

    # -- ближайший говорящий --------------------------------------------------
    def _nearest(
        self, queries: Sequence[tuple[float, float, int]]
    ) -> dict[int, tuple[float, float]]:
        """Для каждой реплики без перекрытия — (исходный индекс, зазор)."""
        records_gap0 = [(s, e, i, (-i,)) for s, e, i in self._records]
        # Зазор 0: интервал касается/накрывает реплику (``s <= qe`` и ``e >= qs``).
        queries_gap0 = [(qe, qs, query_id) for qs, qe, query_id in queries]
        result_gap0 = _sweep(
            records_gap0, queries_gap0, ascending=True, select_ge=True, coords=self._coords_e
        )

        # Интервал до реплики: ``e <= qs`` (максимизируем e, затем min индекс).
        records_before = [(e, e, i, (e, -i)) for s, e, i in self._records]
        queries_before = [(qs, qs, query_id) for qs, qe, query_id in queries]
        result_before = _sweep(
            records_before, queries_before, ascending=True, select_ge=False, coords=self._coords_e
        )

        # Интервал после реплики: ``s >= qe`` (минимизируем s, затем min индекс).
        records_after = [(s, s, i, (-s, -i)) for s, e, i in self._records]
        queries_after = [(qe, qe, query_id) for qs, qe, query_id in queries]
        result_after = _sweep(
            records_after, queries_after, ascending=False, select_ge=True, coords=self._coords_s
        )

        nearest: dict[int, tuple[float, float]] = {}
        for qs, qe, query_id in queries:
            found_gap0 = result_gap0.get(query_id)
            if found_gap0 is not None:
                nearest[query_id] = (-found_gap0[0], 0.0)
                continue
            candidate: tuple[float, float] | None = None
            found_before = result_before.get(query_id)
            if found_before is not None:
                candidate = _better_gap(candidate, (qs - found_before[0], -found_before[1]))
            found_after = result_after.get(query_id)
            if found_after is not None:
                candidate = _better_gap(candidate, (-found_after[0] - qe, -found_after[1]))
            if candidate is not None:
                nearest[query_id] = (candidate[1], candidate[0])
        return nearest

    # -- запасной путь (обратные интервалы) -----------------------------------
    def _brute_best_speaker_id(self, segment: TranscriptionSegment) -> str | None:
        best_id: str | None = None
        best_overlap = 0.0

        for speaker_segment in self._brute:
            overlap = min(segment.end, speaker_segment.end) - max(
                segment.start, speaker_segment.start
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = speaker_segment.speaker_id

        if best_id is not None:
            return best_id

        return self._brute_nearest_id(segment)

    def _brute_nearest_id(self, segment: TranscriptionSegment) -> str | None:
        nearest_id: str | None = None
        nearest_gap: float | None = None

        for speaker_segment in self._brute:
            if segment.end < speaker_segment.start:
                gap = speaker_segment.start - segment.end
            elif segment.start > speaker_segment.end:
                gap = segment.start - speaker_segment.end
            else:
                gap = 0.0
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest_id = speaker_segment.speaker_id

        if nearest_id is not None and nearest_gap is not None and nearest_gap <= self._max_gap:
            return nearest_id
        return None
