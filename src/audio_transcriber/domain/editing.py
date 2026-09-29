"""Правка говорящих в готовом результате без повторной расшифровки.

Функции чистые: возвращают новый :class:`TranscriptionResult`, не меняя
исходный. Используются и TUI, и веб-интерфейсом (редактор говорящих).
"""

from __future__ import annotations

from dataclasses import replace

from audio_transcriber.domain.models import TranscriptionResult


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
    renamed = next(speaker for speaker in speakers if speaker.id == speaker_id)
    entries = [
        replace(entry, speaker=renamed)
        if entry.speaker is not None and entry.speaker.id == speaker_id
        else entry
        for entry in result.entries
    ]
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
    entries = [
        replace(entry, speaker=target)
        if entry.speaker is not None and entry.speaker.id == source_id
        else entry
        for entry in result.entries
    ]
    return replace(result, speakers=speakers, entries=entries)
