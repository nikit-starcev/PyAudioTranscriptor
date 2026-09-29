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

        for segment in transcription_segments:
            speaker_id = resolver.best_speaker_id(segment)
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


class _SpeakerResolver:
    """Ищет говорящего для реплики: максимальное перекрытие, затем ближайший.

    Диаризация (pyannote) выдаёт неперекрывающиеся интервалы: в каждый момент
    говорит один человек. Для таких данных достаточно отсортировать границы и
    отвечать на запрос за O(log M + k), где ``k`` — число интервалов, попавших
    в окно реплики, вместо перебора всех ``M`` интервалов на каждую реплику.

    Если интервалы пересекаются (или есть вырожденные нулевой длины), индекс
    неприменим — используется прежний перебор «в лоб», поэтому поведение
    сохраняется для любых входных данных.
    """

    def __init__(
        self, speaker_segments: Sequence[SpeakerSegment], *, max_gap: float
    ) -> None:
        self._max_gap = max_gap
        self._brute: list[SpeakerSegment] = list(speaker_segments)

        ordered = sorted(
            enumerate(speaker_segments), key=lambda item: (item[1].start, item[0])
        )
        self._indexable = not self._is_overlapping(ordered) and not any(
            segment.end <= segment.start for segment in speaker_segments
        )

        if self._indexable:
            self._segments = [segment for _, segment in ordered]
            self._orig_index = [index for index, _ in ordered]
            self._starts = [segment.start for segment in self._segments]
            self._ends = [segment.end for segment in self._segments]

    @staticmethod
    def _is_overlapping(ordered: Sequence[tuple[int, SpeakerSegment]]) -> bool:
        if len(ordered) < 2:
            return False
        running_end = ordered[0][1].end
        for _, segment in ordered[1:]:
            if segment.start < running_end:
                return True
            running_end = max(running_end, segment.end)
        return False

    def best_speaker_id(self, segment: TranscriptionSegment) -> str | None:
        if not self._indexable:
            return self._brute_best_speaker_id(segment)
        return self._indexed_best_speaker_id(segment)

    # -- быстрый путь (неперекрывающиеся интервалы) --------------------
    def _indexed_best_speaker_id(self, segment: TranscriptionSegment) -> str | None:
        # Отрезок [lo, hi) — интервалы, гарантированно перекрывающиеся с репликой:
        # end > start реплики (lo) и start < end реплики (hi).
        lo = bisect_right(self._ends, segment.start)
        hi = bisect_left(self._starts, segment.end)

        speaker_id = self._best_overlap_id(segment, lo, hi)
        if speaker_id is not None:
            return speaker_id
        return self._nearest_id(segment, lo)

    def _best_overlap_id(
        self, segment: TranscriptionSegment, lo: int, hi: int
    ) -> str | None:
        best_id: str | None = None
        best_key: tuple[float, int] | None = None
        for index in range(lo, hi):
            speaker_segment = self._segments[index]
            overlap = min(segment.end, speaker_segment.end) - max(
                segment.start, speaker_segment.start
            )
            # Ничью разрешаем по исходному порядку — как перебор «в лоб».
            key = (overlap, -self._orig_index[index])
            if best_key is None or key > best_key:
                best_key = key
                best_id = speaker_segment.speaker_id
        return best_id

    def _nearest_id(self, segment: TranscriptionSegment, lo: int) -> str | None:
        nearest_id: str | None = None
        nearest_key: tuple[float, int] | None = None

        def consider(gap: float, orig_index: int, speaker_id: str) -> None:
            nonlocal nearest_id, nearest_key
            key = (gap, orig_index)
            if nearest_key is None or key < nearest_key:
                nearest_key = key
                nearest_id = speaker_id

        # Ближайшими могут быть только два соседа окна: последний интервал,
        # закончившийся до реплики, и первый, начавшийся после неё.
        if lo > 0:
            before = self._segments[lo - 1]
            consider(segment.start - before.end, self._orig_index[lo - 1], before.speaker_id)
        if lo < len(self._segments):
            after = self._segments[lo]
            consider(after.start - segment.end, self._orig_index[lo], after.speaker_id)

        if nearest_key is not None and nearest_key[0] <= self._max_gap:
            return nearest_id
        return None

    # -- запасной путь (пересекающиеся интервалы) ----------------------
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
