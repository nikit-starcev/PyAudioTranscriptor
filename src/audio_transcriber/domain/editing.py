"""Правка говорящих в готовом результате без повторной расшифровки.

Функции чистые: возвращают новый :class:`TranscriptionResult`, не меняя
исходный. Используются и TUI, и веб-интерфейсом (редактор говорящих).
"""

from __future__ import annotations

from dataclasses import replace

from audio_transcriber.domain.models import Speaker, TranscriptEntry, TranscriptionResult


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


def merge_speakers(
    result: TranscriptionResult, source_id: str, target_id: str
) -> TranscriptionResult:
    """Сливает говорящего ``source_id`` в ``target_id``.

    Все реплики источника переназначаются целевому говорящему, источник
    удаляется из списка. Если целевого говорящего нет, результат не меняется.
    """

    target = next((speaker for speaker in result.speakers if speaker.id == target_id), None)
    if target is None or source_id == target_id:
        return result
    speakers = [speaker for speaker in result.speakers if speaker.id != source_id]
    entries = _remap_speaker(result.entries, source_id, target)
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
