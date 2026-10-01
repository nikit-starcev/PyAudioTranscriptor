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
from pathlib import Path

from audio_transcriber.diarization.enrollment import EnrollmentOutcome, enroll_speakers
from audio_transcriber.diarization.voices import merge_references
from audio_transcriber.domain.editing import merge_speakers, rename_speaker
from audio_transcriber.domain.models import (
    Speaker,
    SpeakerSegment,
    TranscriptEntry,
    TranscriptionResult,
)
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

    displays = {speaker.id: speaker.display_name for speaker in result.speakers}
    updated_samples: dict[str, str] = {
        key: value for key, value in samples.items() if isinstance(value, str)
    }
    _sync_renamed_files(updated_samples, renames, data_dir)
    _sync_merged_files(updated_samples, merges, displays, data_dir)

    new_payload = serialize_result(result, samples=updated_samples)
    new_payload["samples"] = updated_samples
    _preserve_entry_flags(payload, new_payload)
    return new_payload


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
) -> tuple[dict[str, object], EnrollmentOutcome]:
    """Сопоставляет говорящих с именами по образцам и применяет совпадения.

    Образцы: явные (из запроса) + библиотека ``voices/``. Возвращает новый JSON
    результата и подробный итог enrollment (совпадения и лучших недобранных).
    """
    references = merge_references(explicit_references, library_references)
    segments = build_speaker_segments(payload)
    outcome = enroll_speakers(
        speaker_segments=segments,
        references=references,
        audio_path=source_path,
        min_similarity=min_similarity,
        local_model_path=local_model_path,
    )
    if not outcome.mapping:
        return dict(payload), outcome
    updated = apply_speaker_changes(
        payload,
        source_path=source_path,
        renames=outcome.mapping,
        merges=(),
        samples=samples,
        data_dir=data_dir,
    )
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
        path = _resolve_sample(relative, data_dir)
        if target_id in samples:
            # У целевого говорящего уже есть свой образец — лишний удаляем.
            _delete_file(path)
            continue
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
