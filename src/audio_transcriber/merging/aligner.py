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

# Минимальная суммарная длительность (в секундах), в течение которой говорящий
# должен быть активен **одновременно** с кем-то ещё внутри интервала реплики,
# чтобы попасть в ``extra_speakers``. Порог намеренно небольшой: настоящие
# перекрытия речи обычно короткие (доли секунды), и их нельзя «съедать».
# Смена говорящего («один закончил — другой начал») одновременной речью не
# считается: интервалы лишь касаются границей, непустого пересечения нет.
DEFAULT_OVERLAP_MIN_SECONDS = 0.3

# Запись интервала диаризации для оффлайн-обходов:
# ``(sort_key, coord, original_index, score)``. ``score`` максимизируется
# (кортежное сравнение), ``coord`` — координата в BIT (end или start).
_Record = tuple[float, float, int, tuple[float, ...]]
# Запрос: ``(sort_key, coord_bound, query_id)``.
_Query = tuple[float, float, int]
# Интервал диаризации для индекса покрытия: ``(start, end, speaker_id)``.
_CoverageInterval = tuple[float, float, str]
# Атомарный интервал одновременной речи: ``(start, end, активные_говорящие)``.
# Существует только там, где одновременно активны >= 2 разных говорящих.
_SimultaneousInterval = tuple[float, float, frozenset[str]]


def _merged_intervals_by_speaker(
    speaker_segments: Sequence[SpeakerSegment],
) -> list[_CoverageInterval]:
    """Объединяет сегменты каждого говорящего в непересекающиеся интервалы.

    Нужно, чтобы доля покрытия реплики и перекрытие с говорящим не считались
    дважды при наложенных друг на друга сегментах одного говорящего.
    Нефизичные интервалы (``end <= start``) игнорируются.
    """
    grouped: dict[str, list[tuple[float, float]]] = {}
    for segment in speaker_segments:
        if segment.end <= segment.start:
            continue
        grouped.setdefault(segment.speaker_id, []).append((segment.start, segment.end))

    merged: list[_CoverageInterval] = []
    for speaker_id, intervals in grouped.items():
        intervals.sort()
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                if end > current_end:
                    current_end = end
            else:
                merged.append((current_start, current_end, speaker_id))
                current_start, current_end = start, end
        merged.append((current_start, current_end, speaker_id))
    return merged


def _simultaneous_intervals(
    merged: Sequence[_CoverageInterval],
) -> list[_SimultaneousInterval]:
    """Атомарные интервалы, где одновременно активны >= 2 говорящих.

    Линейное заметание по событиям начала/конца объединённых интервалов
    говорящих. На промежутке между соседними событиями набор активных
    говорящих постоянен, поэтому сохраняются только промежутки с >= 2
    активными. Концы обрабатываются раньше начал (при равном времени), чтобы
    стык «один закончил — другой начал» не считался наложением.
    """
    events: list[tuple[float, int, str]] = []
    for start, end, speaker_id in merged:
        events.append((start, 1, speaker_id))
        events.append((end, -1, speaker_id))
    if not events:
        return []

    events.sort(key=lambda event: (event[0], event[1]))

    active: dict[str, int] = {}
    intervals: list[_SimultaneousInterval] = []
    previous_time: float | None = None
    for time, delta, speaker_id in events:
        if previous_time is not None and time > previous_time and len(active) >= 2:
            intervals.append((previous_time, time, frozenset(active)))
        if delta == 1:
            active[speaker_id] = active.get(speaker_id, 0) + 1
        else:
            remaining = active.get(speaker_id, 0) - 1
            if remaining > 0:
                active[speaker_id] = remaining
            else:
                active.pop(speaker_id, None)
        previous_time = time
    return intervals


class _IntervalIndex[PayloadT]:
    """Отчёт обо всех интервалах, пересекающих запрос, за ``O(log n + k)``.

    Центрированное интервальное дерево: узлы, накрывающие ``center``,
    хранятся двумя сортированными списками (по началу и по концу), остальные
    рекурсивно уходят влево/вправо. Запрос интервала ``[qs, qe]`` возвращает
    все накрывающие его интервалы (включая касающиеся, их отсеивает
    вызывающий по фактическому перекрытию) суммарно за ``O(log n + k)``, где
    ``k`` — число найденных интервалов, без полного ``O(n)`` перебора.
    """

    __slots__ = ("_by_end", "_by_start", "_center", "_left", "_right")

    def __init__(self, intervals: Sequence[tuple[float, float, PayloadT]]) -> None:
        self._center: float | None = None
        self._by_start: list[tuple[float, float, PayloadT]] = []
        self._by_end: list[tuple[float, float, PayloadT]] = []
        self._left: _IntervalIndex[PayloadT] | None = None
        self._right: _IntervalIndex[PayloadT] | None = None
        if not intervals:
            return

        endpoints: list[float] = []
        for start, end, _payload in intervals:
            endpoints.append(start)
            endpoints.append(end)
        endpoints.sort()
        # Медиана концов всегда принадлежит какому-то интервалу как его начало
        # или конец, поэтому хотя бы один интервал накрывает центр — рекурсия
        # не может зациклиться.
        center = endpoints[len(endpoints) // 2]
        self._center = center

        left: list[tuple[float, float, PayloadT]] = []
        right: list[tuple[float, float, PayloadT]] = []
        for start, end, payload in intervals:
            if end < center:
                left.append((start, end, payload))
            elif start > center:
                right.append((start, end, payload))
            else:
                self._by_start.append((start, end, payload))

        self._by_start.sort(key=lambda item: item[0])
        self._by_end = sorted(self._by_start, key=lambda item: item[1], reverse=True)
        if left:
            self._left = _IntervalIndex(left)
        if right:
            self._right = _IntervalIndex(right)

    def query(
        self, start: float, end: float, out: list[tuple[float, float, PayloadT]]
    ) -> None:
        """Добавляет в ``out`` все интервалы, пересекающие ``[start, end]``."""
        if self._center is None:
            return
        if end < self._center:
            # Накрывающие центр интервалы начинаются не позже центра (а значит,
            # и не позже ``end``) и заканчиваются не раньше ``end >= start`` —
            # остаётся отсечь те, что начинаются уже после ``end``.
            for item in self._by_start:
                if item[0] > end:
                    break
                out.append(item)
            if self._left is not None:
                self._left.query(start, end, out)
        elif start > self._center:
            for item in self._by_end:
                if item[1] < start:
                    break
                out.append(item)
            if self._right is not None:
                self._right.query(start, end, out)
        else:
            # Центр внутри запроса: все накрывающие его интервалы пересекают
            # запрос (хотя бы в точке центра).
            out.extend(self._by_start)
            if self._left is not None:
                self._left.query(start, end, out)
            if self._right is not None:
                self._right.query(start, end, out)


class OverlapSegmentMerger:
    """Присваивает каждому сегменту речи говорящего с наибольшим перекрытием.

    Если перекрытия нет (реплика попала между интервалами диаризации),
    берётся ближайший по времени говорящий — но не далее ``max_gap`` секунд,
    иначе говорящий не назначается. Реализует протокол ``SegmentMerger``.

    Помимо основного говорящего для каждой реплики собираются
    ``extra_speakers`` — прочие говорящие, которые **одновременно** активны
    (непустое пересечение) с кем-то ещё внутри интервала реплики не меньше
    :data:`DEFAULT_OVERLAP_MIN_SECONDS` секунд суммарно, и
    ``speaker_confidence`` — доля интервала реплики, покрытая сегментами
    основного говорящего. Смена говорящего внутри реплики (один закончил —
    другой начал) одновременной речью не считается и в ``extra_speakers`` не
    попадает. При ``mark_overlap=False`` дополнительные говорящие не
    добавляются (реплика ведёт себя как раньше), а ``speaker_confidence`` всё
    равно считается.
    """

    def __init__(
        self,
        *,
        max_gap: float = _MAX_GAP_DEFAULT,
        mark_overlap: bool = True,
        overlap_min_seconds: float = DEFAULT_OVERLAP_MIN_SECONDS,
    ) -> None:
        self._max_gap = max_gap
        self._mark_overlap = mark_overlap
        self._overlap_min_seconds = overlap_min_seconds

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
        has_diarization = bool(speaker_segments)
        coverage = resolver.coverage_by_speaker(transcription_segments)
        # Собирать одновременных говорящих нужно только в режиме пометки: при
        # ``mark_overlap=False`` лишний проход по индексу не выполняется.
        simultaneous: list[dict[str, float]] | None = (
            resolver.simultaneous_by_speaker(transcription_segments)
            if self._mark_overlap and has_diarization
            else None
        )

        def ensure_speaker(speaker_id: str) -> Speaker:
            if speaker_id not in speakers_by_id:
                index = len(speakers_by_id) + 1
                display_name = known_speakers.get(speaker_id, f"Спикер {index}")
                speakers_by_id[speaker_id] = Speaker(id=speaker_id, display_name=display_name)
            return speakers_by_id[speaker_id]

        for index, (segment, speaker_id) in enumerate(
            zip(transcription_segments, speaker_ids, strict=True)
        ):
            speaker = ensure_speaker(speaker_id) if speaker_id is not None else None
            duration = segment.end - segment.start
            hits = coverage[index]
            extra_speakers: list[Speaker] = []
            if simultaneous is not None:
                candidates: list[tuple[str, float]] = []
                for other_id, simultaneous_seconds in simultaneous[index].items():
                    if other_id == speaker_id:
                        continue
                    if simultaneous_seconds < self._overlap_min_seconds:
                        continue
                    candidates.append((other_id, simultaneous_seconds))
                # По убыванию суммарной одновременной длительности; при
                # равенстве — по идентификатору, чтобы порядок был устойчивым.
                candidates.sort(key=lambda item: (-item[1], item[0]))
                extra_speakers = [ensure_speaker(other_id) for other_id, _ in candidates]

            confidence: float | None = None
            if has_diarization:
                covered = hits.get(speaker_id, 0.0) if speaker_id is not None else 0.0
                confidence = 0.0 if duration <= 0 else min(1.0, covered / duration)

            entries.append(
                TranscriptEntry(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=speaker,
                    avg_logprob=segment.avg_logprob,
                    overlap=bool(extra_speakers),
                    extra_speakers=extra_speakers,
                    speaker_confidence=confidence,
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
        # Индекс покрытия (для ``speaker_confidence``): объединённые интервалы
        # каждого говорящего. ``None`` — данных нет.
        merged = _merged_intervals_by_speaker(speaker_segments)
        self._coverage_index: _IntervalIndex[str] | None = (
            _IntervalIndex(merged) if merged else None
        )
        # Индекс одновременной речи (для ``extra_speakers``): атомарные
        # подынтервалы, где активны >= 2 говорящих. ``None`` — данных нет.
        simultaneous = _simultaneous_intervals(merged)
        self._simultaneous_index: _IntervalIndex[frozenset[str]] | None = (
            _IntervalIndex(simultaneous) if simultaneous else None
        )

    def best_speaker_id(self, segment: TranscriptionSegment) -> str | None:
        return self.best_speaker_ids([segment])[0]

    def coverage_by_speaker(
        self, transcription_segments: Sequence[TranscriptionSegment]
    ) -> list[dict[str, float]]:
        """Доля покрытия каждой реплики сегментами каждого говорящего.

        Возвращает по одному словарю ``{speaker_id: перекрытие_в_секундах}`` на
        реплику. Учитываются только говорящие с положительным перекрытием;
        интервалы говорящего объединены, поэтому одно наложение не считается
        дважды. Сложность — ``O((N + M) log M + K)`` (интервальное дерево), без
        перебора всех пар ``N * M``.
        """
        count = len(transcription_segments)
        if self._coverage_index is None:
            return [{} for _ in range(count)]

        result: list[dict[str, float]] = []
        for segment in transcription_segments:
            hits: dict[str, float] = {}
            found: list[_CoverageInterval] = []
            self._coverage_index.query(segment.start, segment.end, found)
            for start, end, speaker_id in found:
                overlap = min(end, segment.end) - max(start, segment.start)
                if overlap > 0.0:
                    hits[speaker_id] = hits.get(speaker_id, 0.0) + overlap
            result.append(hits)
        return result

    def simultaneous_by_speaker(
        self, transcription_segments: Sequence[TranscriptionSegment]
    ) -> list[dict[str, float]]:
        """Суммарная одновременная речь каждого говорящего внутри реплики.

        По каждой реплике возвращает ``{speaker_id: секунд_одновременной_речи}``
        — сколько времени внутри интервала реплики говорящий был активен
        **одновременно** хотя бы с одним другим говорящим. Смена говорящего
        (интервалы лишь касаются границей) не даёт вклада: непустого
        пересечения нет. Сложность — ``O((N + M) log M + K)`` (интервальное
        дерево по атомарным интервалам одновременной речи).
        """
        count = len(transcription_segments)
        if self._simultaneous_index is None:
            return [{} for _ in range(count)]

        result: list[dict[str, float]] = []
        for segment in transcription_segments:
            durations: dict[str, float] = {}
            found: list[_SimultaneousInterval] = []
            self._simultaneous_index.query(segment.start, segment.end, found)
            for start, end, active in found:
                overlap = min(end, segment.end) - max(start, segment.start)
                if overlap > 0.0:
                    for speaker_id in active:
                        durations[speaker_id] = durations.get(speaker_id, 0.0) + overlap
            result.append(durations)
        return result

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
