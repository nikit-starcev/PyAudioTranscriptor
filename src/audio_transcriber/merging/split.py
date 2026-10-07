"""Автоматическое разделение реплики по смене говорящего (#93).

Движок ASR нередко объединяет несколько коротких фраз в один сегмент, а
:class:`~audio_transcriber.merging.sentence_merger.SentenceMerger` затем склеивает
соседние сегменты одного говорящего в «реплику-предложение». Если внутри такого
интервала диаризация видит **смену говорящего** (один закончил — другой начал),
реплику нужно разрезать: иначе короткая вставка другого участника целиком
приписывается соседу (запись 17-04-31: «Да, я здесь.» ~1 с приписано Степанову).

Модуль реализует автоматический аналог ручного «разделить» (#78), опираясь на
пословные метки (#45) и сегменты диаризации:

* каждый токен реплики относится к говорящему по максимальному перекрытию его
  временного интервала с сегментами диаризации; токены в «дырке» разметки
  продолжают предыдущего говорящего (та же реплика);
* последовательность токенов схлопывается в непрерывные отрезки одного
  говорящего; смена говорящего учитывается, только если новый отрезок длится не
  меньше :data:`DEFAULT_MIN_RUN_SECONDS` (или, при нулевой длительности меток,
  содержит хотя бы два слова) — мелкие/шумовые колебания разметки игнорируются,
  чтобы не дробить реплику зря;
* если после фильтрации остаётся один говорящий — реплика не меняется;
* иначе она разрезается по границам отрезков, каждой части пересчитывается
  говорящий, уверенность привязки, наложение и ``extra_speakers`` стандартным
  объединителем, а ``words`` распределяются между частями.

Ручные правки текста (#26) не трогаются: реплики с ``edited=True`` не делятся
(текст пользователя не восстановить из ``words``). Если разбить текст без потерь
нельзя (``words`` не соответствуют тексту), реплика остаётся как есть.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionSegment,
    WordTimestamp,
)
from audio_transcriber.merging.aligner import (
    DEFAULT_OVERLAP_MIN_SECONDS,
    OverlapSegmentMerger,
)

logger = logging.getLogger(__name__)

#: Минимальная длительность (секунды) отрезка другого говорящего, при которой он
#: считается настоящей сменой говорящего, а не шумовым колебанием разметки.
#: Короткая вставка (~1–1.5 с) порог уверенно проходит; односложный «мигающий»
#: сегмент на доли секунды — нет.
DEFAULT_MIN_RUN_SECONDS = 0.4

#: Порог для «точечных» слов (нулевой длительности) в метрике перекрытия.
_POINT_EPS = 1e-6


@dataclass(slots=True)
class _Run:
    """Непрерывный отрезок слов, отнесённых к одному говорящему."""

    speaker_id: str | None
    words: list[WordTimestamp]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return max(word.end for word in self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _overlap(word: WordTimestamp, segment: SpeakerSegment) -> float:
    """Перекрытие слова и сегмента диаризации (секунды, неотрицательное).

    Нулевые по длительности слова (частый случай для whisper.cpp) трактуются
    как точка: слово внутри сегмента получает ненулевой «вес», иначе короткое
    слово на границе оставалось бы без говорящего.
    """
    start = max(word.start, segment.start)
    end = min(word.end, segment.end)
    overlap = end - start
    if overlap > 0.0:
        return overlap
    if (
        word.start == word.end
        and segment.start <= word.start <= segment.end
        and segment.end > segment.start
    ):
        return _POINT_EPS
    return 0.0


def _word_speaker_id(
    word: WordTimestamp, speaker_segments: Sequence[SpeakerSegment]
) -> str | None:
    """Говорящий слова — сегмент диаризации с наибольшим перекрытием."""
    best_id: str | None = None
    best_overlap = 0.0
    for segment in speaker_segments:
        if segment.end <= segment.start:
            continue
        overlap = _overlap(word, segment)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = segment.speaker_id
    return best_id


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _runs_from_words(
    words: Sequence[WordTimestamp], speaker_segments: Sequence[SpeakerSegment]
) -> list[_Run]:
    """Схлопывает слова в непрерывные отрезки одного говорящего.

    Токены без перекрытия с разметкой (короткие паузы/дырки) продолжают
    предыдущего говорящего; ведущие токены без говорящего получают говорящего
    следующего отрезка.
    """
    assigned: list[str | None] = [
        _word_speaker_id(word, speaker_segments) for word in words
    ]
    previous: str | None = None
    for index, speaker_id in enumerate(assigned):
        if speaker_id is None:
            assigned[index] = previous
        else:
            previous = speaker_id
    following: str | None = None
    for index in range(len(assigned) - 1, -1, -1):
        if assigned[index] is None:
            assigned[index] = following
        else:
            following = assigned[index]

    runs: list[_Run] = []
    for word, speaker_id in zip(words, assigned, strict=True):
        if runs and runs[-1].speaker_id == speaker_id:
            runs[-1].words.append(word)
        else:
            runs.append(_Run(speaker_id=speaker_id, words=[word]))
    return runs


def _is_significant(run: _Run, *, min_run_seconds: float) -> bool:
    """Настоящая ли смена говорящего или шумовое колебание разметки.

    Значим отрезок, если он длится не меньше порога. У движков с нулевыми
    пословными интервалами (start == end) длительность теряется — тогда
    значимость даёт хотя бы пара слов.
    """
    if run.duration >= min_run_seconds:
        return True
    return run.duration <= 0.0 and len(run.words) >= 2


def _prune_runs(runs: list[_Run], *, min_run_seconds: float) -> list[_Run]:
    """Убирает незначимые отрезки, сливая их с ближайшим значимым соседом.

    Шумовой отрезок присоединяется к предыдущему значимому говорящему, а если
    значимых до него ещё не было — к следующему. Затем соседние отрезки с
    одинаковым говорящим склеиваются.
    """
    if not runs:
        return []
    if not any(_is_significant(run, min_run_seconds=min_run_seconds) for run in runs):
        speaker_id = next(
            (run.speaker_id for run in runs if run.speaker_id is not None), None
        )
        return [_Run(speaker_id=speaker_id, words=[w for run in runs for w in run.words])]

    assigned: list[str | None] = []
    last_significant: str | None = None
    for run in runs:
        if _is_significant(run, min_run_seconds=min_run_seconds):
            last_significant = run.speaker_id
            assigned.append(run.speaker_id)
        else:
            assigned.append(last_significant)

    following: str | None = None
    for index in range(len(assigned) - 1, -1, -1):
        if assigned[index] is None:
            assigned[index] = following
        else:
            following = assigned[index]

    kept: list[_Run] = []
    for run, speaker_id in zip(runs, assigned, strict=True):
        if kept and kept[-1].speaker_id == speaker_id:
            kept[-1].words.extend(run.words)
        else:
            kept.append(_Run(speaker_id=speaker_id, words=list(run.words)))
    return kept


def _join_words(words: Sequence[WordTimestamp]) -> str:
    return _normalize(" ".join(word.text for word in words))


def _words_monotonic(words: Sequence[WordTimestamp]) -> bool:
    """Идут ли слова в неубывающем порядке по времени.

    Стыки ASR-чанков с перекрытием иногда дают неотсортированные пословные
    метки. Резать такую реплику по словам нельзя: части получились бы с
    «прыгающими» границами и в неверном порядке. Ручной порядок слов сохраняем,
    а от разделения отказываемся.
    """
    previous = float("-inf")
    for word in words:
        if word.start < previous:
            return False
        previous = word.start
    return True


def _ensure_speaker(
    registry: dict[str, Speaker],
    speaker_id: str,
    known_speakers: dict[str, str] | None,
) -> Speaker:
    """Возвращает (создавая при необходимости) говорящего реестра."""
    existing = registry.get(speaker_id)
    if existing is not None:
        return existing
    name = (known_speakers or {}).get(speaker_id, f"Спикер {len(registry) + 1}")
    speaker = Speaker(id=speaker_id, display_name=name)
    registry[speaker_id] = speaker
    return speaker


class SpeakerChangeSplitter:
    """Разрезает реплики, внутри которых диаризация видит смену говорящего.

    :param min_run_seconds: минимальная длительность отрезка другого говорящего,
        считающегося настоящей сменой (а не шумом).
    :param overlap_min_seconds: порог суммарной одновременной речи для
        ``extra_speakers`` частей (тот же смысл, что у объединителя).
    :param mark_overlap: собирать ли ``extra_speakers``/``overlap`` у частей.
    """

    def __init__(
        self,
        *,
        min_run_seconds: float = DEFAULT_MIN_RUN_SECONDS,
        overlap_min_seconds: float = DEFAULT_OVERLAP_MIN_SECONDS,
        mark_overlap: bool = True,
    ) -> None:
        if min_run_seconds < 0.0:
            raise ValueError("min_run_seconds не может быть отрицательным")
        self._min_run_seconds = min_run_seconds
        self._mark_overlap = mark_overlap
        # Объединитель считает говорящего/уверенность/наложения частей так же,
        # как для обычных реплик, — логика выбора говорящего одна на проект.
        self._merger = OverlapSegmentMerger(
            mark_overlap=mark_overlap,
            overlap_min_seconds=overlap_min_seconds,
        )

    def split(
        self,
        entries: list[TranscriptEntry],
        speaker_segments: Sequence[SpeakerSegment],
        speakers: list[Speaker] | None = None,
        known_speakers: dict[str, str] | None = None,
    ) -> tuple[list[TranscriptEntry], list[Speaker]]:
        """Возвращает реплики с разрезанными по смене говорящего и говорящих.

        Реплики без смены (или которые нельзя разделить) возвращаются как есть.
        Порядок сохраняется; список говорящих дополняется участниками, впервые
        появившимися у новых частей (короткая вставка, которой раньше не было
        ни у одной реплики).
        """
        registry: dict[str, Speaker] = {speaker.id: speaker for speaker in (speakers or [])}
        if not speaker_segments:
            return entries, list(registry.values())

        result: list[TranscriptEntry] = []
        changed = 0
        for entry in entries:
            parts = self._split_entry(
                entry, speaker_segments, registry, known_speakers
            )
            if parts is None:
                result.append(entry)
            else:
                result.extend(parts)
                changed += 1
        if changed:
            logger.info("Разделение реплик по смене говорящего: %d", changed)
        return result, list(registry.values())

    def _split_entry(
        self,
        entry: TranscriptEntry,
        speaker_segments: Sequence[SpeakerSegment],
        registry: dict[str, Speaker],
        known_speakers: dict[str, str] | None,
    ) -> list[TranscriptEntry] | None:
        """Разбивает реплику или ``None``, если делить нечего/нельзя."""
        if entry.edited or len(entry.words) < 2:
            return None
        if not _words_monotonic(entry.words):
            return None
        runs = _runs_from_words(entry.words, speaker_segments)
        pruned = _prune_runs(runs, min_run_seconds=self._min_run_seconds)
        if len(pruned) < 2 or len({run.speaker_id for run in pruned}) < 2:
            return None
        # Текст должен восстанавливаться из слов без потерь: иначе разделение
        # «съест» часть текста (нормализация/правки); ручные правки исключены
        # флагом ``edited`` выше.
        rebuilt = _normalize(" ".join(_join_words(run.words) for run in pruned))
        if rebuilt != _normalize(entry.text):
            return None

        parts: list[TranscriptEntry] = []
        count = len(pruned)
        for index, run in enumerate(pruned):
            text = _join_words(run.words)
            if not text:
                return None
            parts.append(
                self._build_part(
                    entry,
                    run,
                    speaker_segments,
                    start=entry.start if index == 0 else run.start,
                    end=entry.end if index == count - 1 else run.end,
                    text=text,
                    registry=registry,
                    known_speakers=known_speakers,
                )
            )
        if len({part.speaker.id if part.speaker else None for part in parts}) < 2:
            return None
        return parts

    def _build_part(
        self,
        entry: TranscriptEntry,
        run: _Run,
        speaker_segments: Sequence[SpeakerSegment],
        *,
        start: float,
        end: float,
        text: str,
        registry: dict[str, Speaker],
        known_speakers: dict[str, str] | None,
    ) -> TranscriptEntry:
        """Собирает часть реплики с пересчётом говорящего/уверенности/наложений.

        Основного говорящего берём из отрезка слов (он и есть причина
        разделения), а уверенность привязки и ``extra_speakers`` пересчитывает
        общий объединитель по интервалу части.
        """
        speaker = (
            _ensure_speaker(registry, run.speaker_id, known_speakers)
            if run.speaker_id is not None
            else None
        )
        merged, _ = self._merger.merge(
            [TranscriptionSegment(start=start, end=end, text=text)],
            list(speaker_segments),
            known_speakers,
        )
        base = merged[0]
        extra_speakers: list[Speaker] = []
        if self._mark_overlap and speaker is not None:
            for extra in base.extra_speakers:
                if extra.id != speaker.id:
                    extra_speakers.append(
                        _ensure_speaker(registry, extra.id, known_speakers)
                    )
        confidence = base.speaker_confidence
        if (
            speaker is not None
            and (base.speaker is None or base.speaker.id != speaker.id)
        ):
            # Объединитель выбрал другого говорящего (часть короче / смещена):
            # уверенность привязки к нашему говорящему неизвестна — не врём.
            confidence = 0.0
        return replace(
            entry,
            start=start,
            end=end,
            text=text,
            speaker=speaker,
            words=list(run.words),
            extra_speakers=extra_speakers,
            overlap=base.overlap or bool(extra_speakers),
            speaker_confidence=confidence,
        )
