"""Правка говорящих в готовом результате без повторной расшифровки.

Функции чистые: возвращают новый :class:`TranscriptionResult`, не меняя
исходный. Используются и TUI, и веб-интерфейсом (редактор говорящих).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from audio_transcriber.domain.models import (
    Speaker,
    TranscriptEntry,
    TranscriptionResult,
    WordTimestamp,
)

#: Идентификатор «автоматического» говорящего вида ``SPEAKER_07``.
_AUTO_SPEAKER_ID = re.compile(r"SPEAKER_(\d+)")


def next_speaker_id(result: TranscriptionResult) -> str:
    """Свободный идентификатор нового говорящего вида ``SPEAKER_NN``.

    Учитывает как ``result.speakers``, так и говорящих, упомянутых только в
    репликах (основной и дополнительные участники). Номер берётся на единицу
    больше максимального существующего ``SPEAKER_N``, поэтому новый говорящий
    встаёт в конец привычного порядка. Если говорящих ``SPEAKER_*`` нет (или id
    произвольные), возвращается ``SPEAKER_00``.
    """
    used = _used_speaker_ids(result)
    numbers = [number for number in map(_auto_speaker_number, used) if number is not None]
    candidate_number = max(numbers, default=-1) + 1
    candidate = f"SPEAKER_{candidate_number:02d}"
    while candidate in used:
        candidate_number += 1
        candidate = f"SPEAKER_{candidate_number:02d}"
    return candidate


def _used_speaker_ids(result: TranscriptionResult) -> set[str]:
    """Все известные id говорящих результата (список + реплики/наложения)."""
    used = {speaker.id for speaker in result.speakers}
    for entry in result.entries:
        if entry.speaker is not None:
            used.add(entry.speaker.id)
        for extra in entry.extra_speakers:
            used.add(extra.id)
    return used


def _auto_speaker_number(speaker_id: str) -> int | None:
    """Номер из ``SPEAKER_NN`` или ``None`` для произвольного id."""
    match = _AUTO_SPEAKER_ID.fullmatch(speaker_id)
    return int(match.group(1)) if match is not None else None


@dataclass(frozen=True, slots=True)
class EntrySpeakerChange:
    """Изменение говорящего одной реплики при переназначении окна (#40/#41).

    ``index`` — позиция реплики в списке результата; ``before_*``/``after_*`` —
    основной говорящий и дополнительные участники до и после правки. Годится и
    для показа изменений пользователю, и для отката на клиенте.
    """

    index: int
    before_speaker_id: str | None
    after_speaker_id: str | None
    before_extra_ids: tuple[str, ...]
    after_extra_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API."""
        return {
            "index": self.index,
            "before_speaker_id": self.before_speaker_id,
            "after_speaker_id": self.after_speaker_id,
            "before_extra_ids": list(self.before_extra_ids),
            "after_extra_ids": list(self.after_extra_ids),
        }


def _entry_speaker_ids(entry: TranscriptEntry) -> tuple[str | None, tuple[str, ...]]:
    """``(основной, дополнительные)`` идентификаторы говорящих реплики."""
    primary = entry.speaker.id if entry.speaker is not None else None
    return primary, tuple(extra.id for extra in entry.extra_speakers)


def _set_primary_speaker(entry: TranscriptEntry, target: Speaker) -> TranscriptEntry:
    """Заменяет основного говорящего реплики целевым, убирая дубль из доп.

    Прежний основной говорящий не сохраняется: это исправление ошибки диаризации
    (#40), а не пометка наложения. Целевой говорящий, если он был в
    ``extra_speakers``, убирается оттуда, чтобы не дублироваться.
    """
    extras = [extra for extra in entry.extra_speakers if extra.id != target.id]
    return replace(entry, speaker=target, extra_speakers=extras)


def _add_co_speaker(entry: TranscriptEntry, target: Speaker) -> TranscriptEntry:
    """Добавляет целевого говорящего сов-участником наложения (#41, «разделить»).

    Реплика частично выходит за выбранное окно, поэтому целиком отдавать её
    новому говорящему нельзя: без пословных таймкодов текст не разделить, и
    реплика помечается как произнесённая обоими (основной + целевой).
    """
    if entry.speaker is not None and entry.speaker.id == target.id:
        return entry
    if any(extra.id == target.id for extra in entry.extra_speakers):
        return entry
    return replace(entry, extra_speakers=[*entry.extra_speakers, target])


def reassign_window(
    result: TranscriptionResult,
    *,
    start: float,
    end: float,
    target: Speaker,
    split: bool = False,
) -> tuple[TranscriptionResult, list[EntrySpeakerChange]]:
    """Переназначает реплики, пересекающиеся с окном ``[start, end)``, целевому.

    Пересечение учитывается частичное (достаточно ненулевого нахлёста), поэтому
    реплика, лишь краем заходящая в окно, тоже переназначается (#40).

    ``split=False`` (перенос): основной говорящий пересекающихся реплик
    заменяется целевым целиком. ``split=True`` («разделить», #41): реплики,
    целиком лежащие в окне, также переназначаются, а частично перекрывающиеся
    получают целевого **сов-говорящим** (``extra_speakers``) — без пословных
    таймкодов «разрезать» текст реплики между двумя невозможно, а сов-участие
    сохраняет и текст, и обоих говорящих.

    Целевой говорящий добавляется в ``result.speakers``, если его там ещё нет
    (создание нового, #41). Возвращает новый результат и список изменений.
    """
    speakers = list(result.speakers)
    if not any(speaker.id == target.id for speaker in speakers):
        speakers.append(target)
    entries = relink_entry_speakers(result.entries, speakers)

    changes: list[EntrySpeakerChange] = []
    updated: list[TranscriptEntry] = []
    for index, entry in enumerate(entries):
        if not (entry.end > start and entry.start < end):
            updated.append(entry)
            continue
        fully_inside = entry.start >= start and entry.end <= end
        can_split = (
            split
            and not fully_inside
            and entry.speaker is not None
            and entry.speaker.id != target.id
        )
        if can_split:
            candidate = _add_co_speaker(entry, target)
        else:
            candidate = _set_primary_speaker(entry, target)
        before_primary, before_extras = _entry_speaker_ids(entry)
        after_primary, after_extras = _entry_speaker_ids(candidate)
        if (before_primary, before_extras) != (after_primary, after_extras):
            changes.append(
                EntrySpeakerChange(
                    index=index,
                    before_speaker_id=before_primary,
                    after_speaker_id=after_primary,
                    before_extra_ids=before_extras,
                    after_extra_ids=after_extras,
                )
            )
        updated.append(candidate)
    return replace(result, speakers=speakers, entries=updated), changes


def relink_entry_speakers(
    entries: list[TranscriptEntry], speakers: list[Speaker]
) -> list[TranscriptEntry]:
    """Перепривязывает говорящих реплик (основного и доп.) к актуальным объектам.

    Имена в ``extra_speakers`` могли устареть после переименования, объединения
    или присвоения имён: здесь каждый говорящий реплики (основной и каждый
    дополнительный) заменяется на объект с тем же ``id`` из ``speakers``. Так
    дополнительные участники наложения не «застревают» со старым именем, а
    разрешаются по актуальному списку. Говорящие, которых нет в ``speakers``,
    остаются как есть, чтобы не терять данные.
    """
    by_id = {speaker.id: speaker for speaker in speakers}

    def resolve(speaker: Speaker) -> Speaker:
        return by_id.get(speaker.id, speaker)

    return [
        replace(
            entry,
            speaker=resolve(entry.speaker) if entry.speaker is not None else None,
            extra_speakers=[resolve(extra) for extra in entry.extra_speakers],
        )
        for entry in entries
    ]


def rename_speaker(
    result: TranscriptionResult, speaker_id: str, new_name: str
) -> TranscriptionResult:
    """Возвращает копию результата с новым именем говорящего у него и реплик.

    Ничего не делает, если говорящий с таким идентификатором не найден.
    """

    if not any(speaker.id == speaker_id for speaker in result.speakers):
        return result
    speakers = [
        replace(speaker, display_name=new_name) if speaker.id == speaker_id else speaker
        for speaker in result.speakers
    ]
    entries = relink_entry_speakers(result.entries, speakers)
    return replace(result, speakers=speakers, entries=entries)


def merge_speaker_lists(
    entries: list[TranscriptEntry],
    speakers: list[Speaker],
    source_id: str,
    target_id: str,
) -> tuple[list[TranscriptEntry], list[Speaker]]:
    """Сливает говорящего ``source_id`` в ``target_id`` на уровне списков.

    Низкоуровневая основа :func:`merge_speakers`: все реплики источника
    (включая дополнительных участников наложения) переназначаются целевому,
    источник удаляется из списка. Если цели нет или идентификаторы совпадают,
    входные списки возвращаются без изменений. Нужна конвейеру, где реплики и
    говорящие ещё не собраны в :class:`TranscriptionResult`.
    """

    target = next((speaker for speaker in speakers if speaker.id == target_id), None)
    if target is None or source_id == target_id:
        return entries, speakers
    new_speakers = [speaker for speaker in speakers if speaker.id != source_id]
    new_entries = _remap_speaker(entries, source_id, target)
    return new_entries, new_speakers


def merge_speakers(
    result: TranscriptionResult, source_id: str, target_id: str
) -> TranscriptionResult:
    """Сливает говорящего ``source_id`` в ``target_id``.

    Все реплики источника переназначаются целевому говорящему, источник
    удаляется из списка. Если целевого говорящего нет, результат не меняется.
    """

    entries, speakers = merge_speaker_lists(
        result.entries, result.speakers, source_id, target_id
    )
    return replace(result, speakers=speakers, entries=entries)


def _remap_speaker(
    entries: list[TranscriptEntry], source_id: str, target: Speaker
) -> list[TranscriptEntry]:
    """Заменяет источник на цель в основном и доп. говорящих реплик.

    После переназначения целевой говорящий не должен дублироваться в
    ``extra_speakers``: если он уже присутствует (или стал основным), лишнее
    упоминание убирается с сохранением порядка остальных участников.
    """

    def remap(speaker: Speaker) -> Speaker:
        return target if speaker.id == source_id else speaker

    remapped: list[TranscriptEntry] = []
    for entry in entries:
        speaker = remap(entry.speaker) if entry.speaker is not None else None
        seen = {speaker.id} if speaker is not None else set()
        extras: list[Speaker] = []
        for extra in entry.extra_speakers:
            candidate = remap(extra)
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            extras.append(candidate)
        remapped.append(replace(entry, speaker=speaker, extra_speakers=extras))
    return remapped


def add_extra_speaker(entry: TranscriptEntry, target: Speaker) -> TranscriptEntry:
    """Добавляет целевого говорящего сов-участником наложения (#78).

    Если цель уже основной говорящий или уже есть в ``extra_speakers``,
    реплика возвращается без изменений. Пометка наложения ``overlap``
    выставляется, потому что со-говорящий означает одновременную речь.
    """
    if entry.speaker is not None and entry.speaker.id == target.id:
        return entry
    if any(extra.id == target.id for extra in entry.extra_speakers):
        return entry
    return replace(entry, extra_speakers=[*entry.extra_speakers, target], overlap=True)


def remove_extra_speaker(entry: TranscriptEntry, speaker_id: str) -> TranscriptEntry:
    """Убирает говорящего ``speaker_id`` из сов-участников наложения (#78).

    Если такого участника нет, реплика возвращается без изменений. Когда после
    удаления сов-участников не осталось, снимается и пометка ``overlap``.
    """
    extras = [extra for extra in entry.extra_speakers if extra.id != speaker_id]
    if len(extras) == len(entry.extra_speakers):
        return entry
    return replace(entry, extra_speakers=extras, overlap=entry.overlap and bool(extras))


def _partition_words(
    words: list[WordTimestamp], boundary: float
) -> tuple[list[WordTimestamp], list[WordTimestamp]]:
    """Делит пословные метки на две группы по середине слова относительно границы."""
    left: list[WordTimestamp] = []
    right: list[WordTimestamp] = []
    for word in words:
        midpoint = (word.start + word.end) / 2
        (left if midpoint <= boundary else right).append(word)
    return left, right


def _words_text(words: list[WordTimestamp]) -> str:
    """Склеивает слова в текст (пробел между словами)."""
    return " ".join(word.text for word in words)


def _split_text(text: str, fraction: float) -> tuple[str, str]:
    """Делит текст по долям слов, ближайшим к ``fraction`` (0..1).

    Текст режется по границам слов (пробелам), чтобы не разрывать слово. Одна
    доля всегда непуста, если в тексте больше одного слова.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text, ""
    ratio = min(max(fraction, 0.0), 1.0)
    cut = round(ratio * len(tokens))
    cut = min(max(cut, 1), len(tokens) - 1)
    return " ".join(tokens[:cut]), " ".join(tokens[cut:])


def _dedupe_primary(entry: TranscriptEntry) -> TranscriptEntry:
    """Убирает основного говорящего из ``extra_speakers``, если он там остался."""
    if entry.speaker is None:
        return entry
    primary_id = entry.speaker.id
    extras = [extra for extra in entry.extra_speakers if extra.id != primary_id]
    if len(extras) == len(entry.extra_speakers):
        return entry
    return replace(entry, extra_speakers=extras)


def split_entry(
    entry: TranscriptEntry,
    boundary_seconds: float,
    *,
    first_speaker: Speaker,
    second_speaker: Speaker,
) -> tuple[TranscriptEntry, TranscriptEntry]:
    """Разрезает реплику по времени на две, каждой даёт своего говорящего (#78).

    Граница ``boundary_seconds`` (секунды аудио) ограничивается интервалом
    реплики. Текст делится по пословным таймкодам, когда они есть и реплика не
    правилась вручную: слова с серединой до границы уходят в первую часть,
    остальные — во вторую. Без слов (или у реплики с ручной правкой #26) текст
    делится пропорционально доле времени по границам слов, а ручная правка
    сохраняется: ``edited``/``original_text`` переносятся на обе части
    разделёнными по той же доле, поэтому общий текст не теряется.

    Основной говорящий каждой части заменяется на ``first_speaker``/
    ``second_speaker``; участники наложения сохраняются на обеих частях (без
    дублирования основного). Пометки ``overlap``, ``speaker_confidence`` и
    прочие поля наследуются.
    """
    boundary = min(max(boundary_seconds, entry.start), entry.end)
    left_words, right_words = _partition_words(entry.words, boundary)
    total = entry.end - entry.start
    fraction = (boundary - entry.start) / total if total > 0 else 0.5
    tokens = entry.text.split()
    word_aligned = (
        not entry.edited
        and bool(entry.words)
        and len(tokens) == len(entry.words)
        and 0 < len(left_words) < len(entry.words)
    )
    left_text: str
    right_text: str
    left_edited: bool
    right_edited: bool
    left_origin: str | None
    right_origin: str | None
    if word_aligned:
        cut = len(left_words)
        left_text = " ".join(tokens[:cut])
        right_text = " ".join(tokens[cut:])
        left_edited = right_edited = False
        left_origin = right_origin = None
    elif not entry.edited and entry.words and 0 < len(left_words) < len(entry.words):
        left_text = _words_text(left_words)
        right_text = _words_text(right_words)
        left_edited = right_edited = False
        left_origin = right_origin = None
    else:
        left_text, right_text = _split_text(entry.text, fraction)
        if entry.edited:
            left_edited = right_edited = True
            if isinstance(entry.original_text, str):
                left_origin, right_origin = _split_text(entry.original_text, fraction)
            else:
                left_origin = right_origin = None
        else:
            left_edited = right_edited = False
            left_origin = right_origin = None
    left = replace(
        entry,
        end=boundary,
        text=left_text,
        speaker=first_speaker,
        words=left_words,
        edited=left_edited,
        original_text=left_origin,
    )
    right = replace(
        entry,
        start=boundary,
        text=right_text,
        speaker=second_speaker,
        words=right_words,
        edited=right_edited,
        original_text=right_origin,
    )
    return _dedupe_primary(left), _dedupe_primary(right)


def split_entry_in_result(
    result: TranscriptionResult,
    index: int,
    boundary_seconds: float,
    *,
    first_speaker: Speaker,
    second_speaker: Speaker,
) -> tuple[TranscriptionResult, tuple[TranscriptEntry, TranscriptEntry]]:
    """Разрезает реплику ``index`` результата и добавляет новых говорящих (#78).

    Возвращает новый результат (реплика заменена двумя) и обе созданные части.
    Оба говорящих добавляются в ``result.speakers``, если их там ещё нет.
    """
    entries = list(result.entries)
    if index < 0 or index >= len(entries):
        raise ValueError(f"Реплика #{index} не найдена")
    left, right = split_entry(
        entries[index],
        boundary_seconds,
        first_speaker=first_speaker,
        second_speaker=second_speaker,
    )
    speakers = list(result.speakers)
    for speaker in (first_speaker, second_speaker):
        if not any(existing.id == speaker.id for existing in speakers):
            speakers.append(speaker)
    entries[index : index + 1] = [left, right]
    return replace(result, speakers=speakers, entries=entries), (left, right)
