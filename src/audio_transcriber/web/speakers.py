"""Правка говорящих в JSON-результате задачи и применение имён по голосу.

Модуль — «клей» между плоским форматом результата (контракт API) и доменными
моделями: восстанавливает :class:`TranscriptionResult` из JSON, применяет
переименования/объединения (переиспользуя :mod:`audio_transcriber.domain.editing`),
синхронизирует файлы образцов голоса и снова сериализует результат.

Флаги ``low_confidence`` у реплик вычисляются на этапе сериализации исходного
результата и в JSON уже «зашиты»; правка говорящих не меняет порядок и текст
реплик, поэтому после пересборки они восстанавливаются по индексу.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from audio_transcriber.diarization.enrollment import EnrollmentOutcome, enroll_speakers
from audio_transcriber.diarization.voices import merge_references
from audio_transcriber.domain.editing import (
    EntrySpeakerChange,
    add_extra_speaker,
    merge_speakers,
    next_speaker_id,
    reassign_window,
    remove_extra_speaker,
    rename_speaker,
    split_entry_in_result,
)
from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionResult,
    WordTimestamp,
)
from audio_transcriber.merging.same_name import (
    merge_result_same_name_speakers,
    same_name_merge_pairs,
)
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.text import sanitize_filename
from audio_transcriber.web.results import serialize_result


def result_from_payload(
    payload: Mapping[str, object], *, source_path: Path
) -> TranscriptionResult:
    """Восстанавливает :class:`TranscriptionResult` из плоского JSON результата."""
    speakers: list[Speaker] = []
    for item in _as_list(payload.get("speakers")):
        if not isinstance(item, Mapping):
            continue
        speaker_id = item.get("id")
        if not isinstance(speaker_id, str) or not speaker_id:
            continue
        display = item.get("display_name")
        speakers.append(
            Speaker(id=speaker_id, display_name=str(display) if display else speaker_id)
        )
    by_id = {speaker.id: speaker for speaker in speakers}

    entries: list[TranscriptEntry] = []
    for item in _as_list(payload.get("entries")):
        if not isinstance(item, Mapping):
            continue
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"))
        if start is None or end is None:
            continue
        speaker_id = item.get("speaker_id")
        speaker: Speaker | None = None
        if isinstance(speaker_id, str) and speaker_id:
            speaker = by_id.get(speaker_id) or Speaker(id=speaker_id, display_name=speaker_id)
        extra_speakers: list[Speaker] = []
        for extra_id in _as_list(item.get("extra_speaker_ids")):
            if not isinstance(extra_id, str) or not extra_id:
                continue
            if speaker is not None and extra_id == speaker.id:
                continue
            extra_speakers.append(
                by_id.get(extra_id) or Speaker(id=extra_id, display_name=extra_id)
            )
        text = item.get("text")
        original_text = item.get("original_text")
        words: list[WordTimestamp] = []
        for raw_word in _as_list(item.get("words")):
            if not isinstance(raw_word, Mapping):
                continue
            word_text = raw_word.get("text")
            word_start = _as_float(raw_word.get("start"))
            word_end = _as_float(raw_word.get("end"))
            if not isinstance(word_text, str) or word_start is None or word_end is None:
                continue
            words.append(
                WordTimestamp(
                    text=word_text,
                    start=word_start,
                    end=word_end,
                    probability=_as_float(raw_word.get("probability")),
                )
            )
        entries.append(
            TranscriptEntry(
                start=start,
                end=end,
                text=str(text) if text is not None else "",
                speaker=speaker,
                overlap=bool(item.get("overlap")),
                extra_speakers=extra_speakers,
                speaker_confidence=_as_float(item.get("speaker_confidence")),
                edited=bool(item.get("edited")),
                original_text=str(original_text) if isinstance(original_text, str) else None,
                # Пословные таймстемпы сохраняются при правке говорящих (#45).
                words=words,
            )
        )

    language = payload.get("language")
    summary = payload.get("summary")
    return TranscriptionResult(
        source_path=source_path,
        language=str(language) if isinstance(language, str) else None,
        duration=_as_float(payload.get("duration")) or 0.0,
        entries=entries,
        speakers=speakers,
        summary=str(summary) if isinstance(summary, str) else None,
    )


def apply_speaker_changes(
    payload: Mapping[str, object],
    *,
    source_path: Path,
    renames: Mapping[str, str],
    merges: Sequence[tuple[str, str]],
    samples: Mapping[str, str],
    data_dir: Path,
) -> dict[str, object]:
    """Применяет переименования/объединения и возвращает новый JSON результата.

    Переименование также переименовывает файл образца голоса говорящего в
    каталоге образцов задачи; объединение переносит образец источника целевому
    (если у цели своего нет) или удаляет лишний. Все файловые операции
    ограничены каталогом данных задачи.
    """
    result = result_from_payload(payload, source_path=source_path)
    for speaker_id, name in renames.items():
        result = rename_speaker(result, speaker_id, str(name))
    for source_id, target_id in merges:
        result = merge_speakers(result, source_id, target_id)

    # Автослияние кластеров с одинаковым итоговым именем (#13): enrollment
    # много-к-одному даёт одно имя нескольким кластерам. Пары считаем после
    # переименований/ручных слияний, чтобы сливать уже итоговые имена; образцы
    # источников переносятся/удаляются тем же механизмом, что и у #40/#41.
    auto_merges = same_name_merge_pairs(result.speakers)
    result = merge_result_same_name_speakers(result)

    displays = {speaker.id: speaker.display_name for speaker in result.speakers}
    updated_samples: dict[str, str] = {
        key: value for key, value in samples.items() if isinstance(value, str)
    }
    _sync_renamed_files(updated_samples, renames, data_dir)
    _sync_merged_files(updated_samples, [*merges, *auto_merges], displays, data_dir)

    return _reserialize(payload, result, updated_samples)


def _reserialize(
    payload: Mapping[str, object],
    result: TranscriptionResult,
    samples: Mapping[str, str],
) -> dict[str, object]:
    """Сериализует результат, сохраняя вычисленные флаги исходных реплик.

    ``low_confidence`` не восстанавливается из ``result`` (у доменной модели нет
    порога), поэтому переносится из исходного JSON по индексу реплики. Порядок и
    число реплик при правке говорящих не меняются, поэтому индекс устойчив.
    """
    cleaned = {key: value for key, value in samples.items() if isinstance(value, str)}
    new_payload = serialize_result(result, samples=cleaned)
    new_payload["samples"] = cleaned
    _preserve_entry_flags(payload, new_payload)
    return new_payload


def find_speaker(result: TranscriptionResult, speaker_id: str) -> Speaker | None:
    """Находит говорящего по id среди списка и участников реплик (включая доп.)."""
    for speaker in result.speakers:
        if speaker.id == speaker_id:
            return speaker
    for entry in result.entries:
        if entry.speaker is not None and entry.speaker.id == speaker_id:
            return entry.speaker
        for extra in entry.extra_speakers:
            if extra.id == speaker_id:
                return extra
    return None


def find_speaker_by_name(result: TranscriptionResult, name: str) -> Speaker | None:
    """Находит говорящего с таким отображаемым именем (без учёта регистра)."""
    needle = name.casefold()
    seen: set[str] = set()
    candidates = list(result.speakers)
    for entry in result.entries:
        if entry.speaker is not None:
            candidates.append(entry.speaker)
        candidates.extend(entry.extra_speakers)
    for speaker in candidates:
        if speaker.id in seen:
            continue
        seen.add(speaker.id)
        if speaker.display_name.casefold() == needle:
            return speaker
    return None


def make_speaker(result: TranscriptionResult, name: str) -> Speaker:
    """Создаёт нового говорящего с именем и свободным id (``SPEAKER_NN``, #41)."""
    return Speaker(id=next_speaker_id(result), display_name=name)


def apply_window_reassign(
    payload: Mapping[str, object],
    *,
    source_path: Path,
    start: float,
    end: float,
    target: Speaker,
    split: bool,
    samples: Mapping[str, str],
) -> tuple[dict[str, object], list[EntrySpeakerChange]]:
    """Переназначает реплики окна ``[start, end)`` целевому говорящему (#40/#41).

    Возвращает новый JSON результата (с сохранением ручных правок текста #26 и
    вычисленных флагов) и список изменённых реплик. Файлы образцов не трогаются:
    перенос реплик не меняет уже сохранённые образцы говорящих.
    """
    result = result_from_payload(payload, source_path=source_path)
    updated, changes = reassign_window(
        result, start=start, end=end, target=target, split=split
    )
    return _reserialize(payload, updated, samples), changes


def apply_entry_speaker_assign(
    payload: Mapping[str, object],
    *,
    indexes: Sequence[int],
    target: Speaker,
    co_speaker: bool = False,
) -> tuple[dict[str, object], list[EntrySpeakerChange]]:
    """Принудительно назначает говорящего выбранным репликам (#59).

    В отличие от :func:`apply_window_reassign` (перенос окна варианта) реплики
    задаются явными индексами из таблицы стенограммы. Основной говорящий
    заменяется целевым, а целевой убирается из ``extra_speaker_ids`` — так он не
    дублируется. ``co_speaker=True`` добавляет целевого участником наложения, не
    трогая основного (учёт перекрытий без «разрезания» текста).

    Правится только состав говорящих: текст, таймкоды, пометки ручной правки
    (#26) и вычисленные флаги реплик остаются как есть. Новый говорящий
    добавляется в ``speakers``, если его там ещё нет; возвращается новый JSON
    результата и список фактически изменённых реплик.
    """
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("В результате нет реплик")
    selected = set(indexes)
    speaker_items = _payload_speakers(payload)
    if not any(item.get("id") == target.id for item in speaker_items):
        speaker_items.append(
            {"id": target.id, "display_name": target.display_name, "has_sample": False}
        )

    changes: list[EntrySpeakerChange] = []
    updated_entries: list[object] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            updated_entries.append(raw)
            continue
        item = dict(raw)
        if index in selected:
            before_primary = _str_or_none(item.get("speaker_id"))
            before_extras = _str_list(item.get("extra_speaker_ids"))
            if co_speaker:
                if before_primary != target.id and target.id not in before_extras:
                    after_extras = [*before_extras, target.id]
                else:
                    after_extras = list(before_extras)
                item["extra_speaker_ids"] = after_extras
            else:
                item["speaker_id"] = target.id
                item["extra_speaker_ids"] = [
                    extra for extra in before_extras if extra != target.id
                ]
            after_primary = _str_or_none(item.get("speaker_id"))
            after_extras = _str_list(item.get("extra_speaker_ids"))
            if (before_primary, tuple(before_extras)) != (after_primary, tuple(after_extras)):
                changes.append(
                    EntrySpeakerChange(
                        index=index,
                        before_speaker_id=before_primary,
                        after_speaker_id=after_primary,
                        before_extra_ids=tuple(before_extras),
                        after_extra_ids=tuple(after_extras),
                    )
                )
        updated_entries.append(item)

    updated = dict(payload)
    updated["entries"] = updated_entries
    updated["speakers"] = speaker_items
    return updated, changes


def apply_entry_extra_speaker(
    payload: Mapping[str, object],
    *,
    source_path: Path,
    indexes: Sequence[int],
    target: Speaker,
    remove: bool,
    samples: Mapping[str, str],
) -> tuple[dict[str, object], list[EntrySpeakerChange]]:
    """Добавляет или убирает второго говорящего у выбранных реплик (#78).

    Второй говорящий — участник наложения (``extra_speakers``): основной
    говорящий не меняется, а метка реплики становится вида «Имя1 + Имя2».
    ``remove=True`` убирает целевого из участников наложения. Ручные правки
    текста (#26), таймкоды, пословные метки и вычисленные флаги сохраняются,
    потому что число и порядок реплик не меняются.
    """
    result = result_from_payload(payload, source_path=source_path)
    entries = list(result.entries)
    speakers = list(result.speakers)
    if not remove and not any(speaker.id == target.id for speaker in speakers):
        speakers.append(target)
    changes: list[EntrySpeakerChange] = []
    for index in indexes:
        if index < 0 or index >= len(entries):
            raise ValueError(f"Реплика #{index} не найдена")
        entry = entries[index]
        before_primary, before_extras = _entry_speaker_ids(entry)
        candidate = (
            remove_extra_speaker(entry, target.id)
            if remove
            else add_extra_speaker(entry, target)
        )
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
        entries[index] = candidate
    updated = replace(result, speakers=speakers, entries=entries)
    return _reserialize(payload, updated, samples), changes


def apply_split_entry(
    payload: Mapping[str, object],
    *,
    source_path: Path,
    index: int,
    boundary: float,
    first_speaker: Speaker,
    second_speaker: Speaker,
    samples: Mapping[str, str],
) -> tuple[dict[str, object], tuple[TranscriptEntry, TranscriptEntry]]:
    """Разрезает реплику ``index`` по ``boundary`` и сохраняет правку (#78).

    Возвращает новый JSON результата и обе созданные части. Пословные таймкоды
    и ручные правки текста (#26) распределяются доменной операцией
    :func:`split_entry_in_result`; вычисленные флаги реплик (``low_confidence``,
    ``low_speaker_confidence``) переносятся на части по их временному интервалу,
    потому что после разреза индексы реплик сдвигаются.
    """
    result = result_from_payload(payload, source_path=source_path)
    if index < 0 or index >= len(result.entries):
        raise ValueError(f"Реплика #{index} не найдена")
    entry = result.entries[index]
    if not entry.start < boundary < entry.end:
        raise ValueError("Граница должна быть внутри реплики")
    updated, parts = split_entry_in_result(
        result,
        index,
        boundary,
        first_speaker=first_speaker,
        second_speaker=second_speaker,
    )
    cleaned = {key: value for key, value in samples.items() if isinstance(value, str)}
    new_payload = serialize_result(updated, samples=cleaned)
    new_payload["samples"] = cleaned
    _preserve_flags_by_span(payload, new_payload)
    return new_payload, parts


def _entry_speaker_ids(entry: TranscriptEntry) -> tuple[str | None, tuple[str, ...]]:
    """``(основной, дополнительные)`` идентификаторы говорящих реплики."""
    primary = entry.speaker.id if entry.speaker is not None else None
    return primary, tuple(extra.id for extra in entry.extra_speakers)


def _payload_speakers(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Копия списка говорящих результата (с сохранением ``has_sample``)."""
    raw = payload.get("speakers")
    items: list[dict[str, object]] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, Mapping):
                items.append(dict(entry))
    return items


def _str_or_none(value: object) -> str | None:
    """Непустая строка иначе ``None``."""
    return value if isinstance(value, str) and value else None


def _str_list(value: object) -> list[str]:
    """Список непустых строк из значения (иначе пустой список)."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]



def build_speaker_segments(payload: Mapping[str, object]) -> list[SpeakerSegment]:
    """Собирает сегменты говорящих из реплик результата (для enrollment)."""
    segments: list[SpeakerSegment] = []
    for item in _as_list(payload.get("entries")):
        if not isinstance(item, Mapping):
            continue
        speaker_id = item.get("speaker_id")
        start = _as_float(item.get("start"))
        end = _as_float(item.get("end"))
        if not isinstance(speaker_id, str) or not speaker_id:
            continue
        if start is None or end is None or end <= start:
            continue
        segments.append(SpeakerSegment(start=start, end=end, speaker_id=speaker_id))
    return segments


def apply_names(
    payload: Mapping[str, object],
    *,
    source_path: Path,
    explicit_references: Mapping[str, Sequence[Path]],
    library_references: Mapping[str, Sequence[Path]],
    samples: Mapping[str, str],
    data_dir: Path,
    min_similarity: float,
    local_model_path: Path | str | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[dict[str, object], EnrollmentOutcome]:
    """Сопоставляет говорящих с именами по образцам и применяет совпадения.

    Образцы: явные (из запроса) + библиотека ``voices/``. Возвращает новый JSON
    результата и подробный итог enrollment (совпадения и лучших недобранных).

    ``on_progress`` — необязательный колбэк этапов (образцы → эмбеддинги →
    сопоставление → применение имён) для индикатора в веб-интерфейсе.
    """
    emit = on_progress or (lambda _event: None)
    references = merge_references(explicit_references, library_references)
    segments = build_speaker_segments(payload)
    outcome = enroll_speakers(
        speaker_segments=segments,
        references=references,
        audio_path=source_path,
        min_similarity=min_similarity,
        local_model_path=local_model_path,
        on_progress=on_progress,
    )
    if not outcome.mapping:
        return dict(payload), outcome
    emit(ProgressEvent("apply", "Применение имён говорящих", None))
    updated = apply_speaker_changes(
        payload,
        source_path=source_path,
        renames=outcome.mapping,
        merges=(),
        samples=samples,
        data_dir=data_dir,
    )
    emit(ProgressEvent("apply", "Имена применены", 1.0))
    return updated, outcome


def _sync_renamed_files(
    samples: dict[str, str], renames: Mapping[str, str], data_dir: Path
) -> None:
    """Переименовывает файлы образцов под новые имена говорящих."""
    for speaker_id, name in renames.items():
        relative = samples.get(speaker_id)
        if not relative:
            continue
        path = _resolve_sample(relative, data_dir)
        if path is None or not path.is_file():
            continue
        desired = f"{sanitize_filename(str(name))}.wav"
        if path.name == desired:
            continue
        target = _unique_path(path.parent, Path(desired).stem)
        if _rename(path, target):
            updated = _relative_to(target, data_dir)
            if updated is not None:
                samples[speaker_id] = updated


def _sync_merged_files(
    samples: dict[str, str],
    merges: Sequence[tuple[str, str]],
    displays: Mapping[str, str],
    data_dir: Path,
) -> None:
    """Переносит образец источника целевому говорящему или удаляет лишний."""
    for source_id, target_id in merges:
        relative = samples.pop(source_id, None)
        if not relative:
            continue
        target_relative = samples.get(target_id)
        if target_relative is not None:
            # У целевого говорящего уже есть свой образец — лишний удаляем.
            # Но если источник ссылается на тот же файл (дублирующееся
            # отображение в старом результате), удалять нельзя: пропал бы
            # образец самого целевого говорящего.
            source_path = _resolve_sample(relative, data_dir)
            target_path = _resolve_sample(target_relative, data_dir)
            if source_path is not None and source_path == target_path:
                continue
            _delete_file(source_path)
            continue
        path = _resolve_sample(relative, data_dir)
        if path is None or not path.is_file():
            continue
        stem = sanitize_filename(displays.get(target_id, target_id))
        target = _unique_path(path.parent, stem)
        if _rename(path, target):
            updated = _relative_to(target, data_dir)
            if updated is not None:
                samples[target_id] = updated


def _preserve_entry_flags(
    original: Mapping[str, object], updated: dict[str, object]
) -> None:
    """Возвращает вычисленные флаги реплик из исходного JSON (по индексу)."""
    old_entries = _as_list(original.get("entries"))
    new_entries = updated.get("entries")
    if not isinstance(new_entries, list):
        return
    for index, entry in enumerate(new_entries):
        if index >= len(old_entries) or not isinstance(entry, dict):
            break
        source = old_entries[index]
        if isinstance(source, Mapping) and "low_confidence" in source:
            entry["low_confidence"] = bool(source["low_confidence"])


def _preserve_flags_by_span(
    original: Mapping[str, object], updated: dict[str, object]
) -> None:
    """Переносит ``low_confidence`` по временному интервалу, а не по индексу.

    Нужно после разреза реплики: число реплик растёт, индексы сдвигаются, и
    позиционное сопоставление :func:`_preserve_entry_flags` дало бы неверные
    пометки. Каждая новая реплика привязывается к исходной, в интервал которой
    попадает её середина.
    """
    old_entries = [
        item for item in _as_list(original.get("entries")) if isinstance(item, Mapping)
    ]
    new_entries = updated.get("entries")
    if not isinstance(new_entries, list):
        return
    for entry in new_entries:
        if not isinstance(entry, dict):
            continue
        start = _as_float(entry.get("start"))
        end = _as_float(entry.get("end"))
        if start is None or end is None:
            continue
        source = _containing_entry(old_entries, (start + end) / 2)
        if source is not None and "low_confidence" in source:
            entry["low_confidence"] = bool(source["low_confidence"])


def _containing_entry(
    entries: Sequence[Mapping[str, object]], point: float
) -> Mapping[str, object] | None:
    """Исходная реплика, в интервал ``[start, end]`` которой попадает ``point``."""
    for entry in entries:
        start = _as_float(entry.get("start"))
        end = _as_float(entry.get("end"))
        if start is None or end is None:
            continue
        if start <= point <= end:
            return entry
    return None


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_sample(relative: str, data_dir: Path) -> Path | None:
    """Разрешает относительный путь образца внутри каталога данных (без выхода)."""
    try:
        root = Path(data_dir).resolve()
        resolved = (root / relative).resolve()
    except OSError:
        return None
    return resolved if resolved.is_relative_to(root) else None


def _relative_to(path: Path, data_dir: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(Path(data_dir).resolve()))
    except (OSError, ValueError):
        return None


def _unique_path(directory: Path, stem: str) -> Path:
    candidate = directory / f"{stem}.wav"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem} {counter}.wav"
        counter += 1
    return candidate


def _rename(source: Path, target: Path) -> bool:
    try:
        source.rename(target)
    except OSError:
        return False
    return True


def _delete_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        return
