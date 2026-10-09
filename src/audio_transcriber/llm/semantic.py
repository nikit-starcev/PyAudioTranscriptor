"""Опциональный LLM-проход семантической правки стенограммы (#75).

Наивная автоматическая правка опасна: LLM склонна «переписывать» текст и
додумывать фрагменты (галлюцинированные вставки). Поэтому проход только
**предлагает** минимальные спановые замены явно невозможных/нелогичных мест
(следствие ошибки ASR). Ни одна правка не применяется автоматически: список
``Suggestion(kind="semantic")`` сохраняется рядом с результатом, а применяет
их пользователь через редактор (accept/reject).

Позициям, которые вернула LLM, не доверяем: ``before`` ищется в тексте реплик
детерминированно. Ответ фильтруется по уверенности и лимитам, несовпавшие
фрагменты и замены ``after == before`` отбрасываются.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from audio_transcriber.config.defaults import (
    DEFAULT_LLM_SEMANTIC_MAX_EDIT_CHARS,
    DEFAULT_LLM_SEMANTIC_MAX_EDITS,
    DEFAULT_LLM_SEMANTIC_MIN_CONFIDENCE,
)
from audio_transcriber.correction.editorial import KIND_SEMANTIC, Suggestion
from audio_transcriber.domain.models import TranscriptEntry
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.chunking import (
    iter_entry_chunks,
    speaker_labels,
    unique_speakers,
)
from audio_transcriber.llm.json_utils import extract_json_object

logger = logging.getLogger(__name__)

#: Суффикс файла с сохранёнными семантическими предложениями.
SEMANTIC_SUGGESTIONS_SUFFIX = ".semantic.json"

#: Резервный текст причины, если модель не объяснила правку.
_DEFAULT_REASON = "семантическая правка (LLM)"

_SYSTEM_PROMPT = (
    "Ты — редактор стенограммы, распознанной автоматически (ASR). "
    "Найди ТОЛЬКО явно невозможные или нелогичные фрагменты, возникшие из-за "
    "ошибки распознавания (бессмысленный набор слов, нарушенное согласование, "
    "слипшиеся или разорванные слова). Для каждого предложи МИНИМАЛЬНУЮ замену: "
    "правь только то, что невозможно. НЕ перефразируй, не меняй порядок слов, "
    "не добавляй и не удаляй слова сверх необходимого. НЕ трогай имена, числа, "
    "даты и отрицания. Если фрагмент логичен или ты сомневаешься — не включай "
    "его. Верни ТОЛЬКО строгий JSON без пояснений и текста вокруг."
)


def _user_prompt(chunk: str) -> str:
    """Пользовательский промпт семантической правки для одного фрагмента."""
    return (
        f"Стенограмма (фрагмент):\n{chunk}\n\n"
        "Найди только явно невозможные/нелогичные фрагменты и верни строгий JSON:\n"
        '{"edits": [{"before": "точная цитата из текста", "after": "минимальная замена", '
        '"reason": "кратко, почему", "confidence": 0.9}]}\n'
        "confidence — твоя уверенность в правке (число 0.0..1.0), ставь своё значение, "
        "не копируй пример; reason — короткая причина.\n"
        'Если таких фрагментов нет, верни {"edits": []}.'
    )


@dataclass(frozen=True, slots=True)
class SemanticEdit:
    """Одна предложенная LLM минимальная замена.

    ``index``/``start``/``end`` заполняются детерминированно при локализации
    ``before`` в тексте реплик (``-1`` — ещё не локализована).
    """

    before: str
    after: str
    reason: str
    confidence: float
    index: int = -1
    start: int = -1
    end: int = -1


def semantic_suggestions_path(output_dir: Path, stem: str) -> Path:
    """Путь к файлу семантических предложений рядом с результатом."""
    return output_dir / f"{stem}{SEMANTIC_SUGGESTIONS_SUFFIX}"


def _confidence(item: Mapping[str, object]) -> float:
    """Уверенность из ответа LLM; нераспознанное значение — ``0.0`` (отбросится)."""
    raw = item.get("confidence")
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return 0.0
    else:
        return 0.0
    return min(1.0, max(0.0, value))


def parse_semantic_edits(raw: str) -> list[SemanticEdit]:
    """Разбирает ответ LLM ``{"edits": [...]}`` в список валидных правок.

    Отбрасываются пустые ``before``/``after`` и замены ``after == before``.
    """
    try:
        payload = extract_json_object(raw)
    except ValueError:
        return []

    edits = payload.get("edits")
    if not isinstance(edits, list):
        return []

    result: list[SemanticEdit] = []
    for item in edits:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before", "")).strip()
        after = str(item.get("after", "")).strip()
        if not before or not after or before == after:
            continue
        reason = str(item.get("reason", "")).strip()
        result.append(
            SemanticEdit(
                before=before,
                after=after,
                reason=reason,
                confidence=_confidence(item),
            )
        )
    return result


def locate_before(text: str, before: str) -> tuple[int, int] | None:
    """Детерминированно находит ``before`` в тексте (первое вхождение)."""
    if not before:
        return None
    start = text.find(before)
    if start < 0:
        return None
    return start, start + len(before)


def _locate_in_chunk(
    entries: list[TranscriptEntry], chunk: Sequence[tuple[int, str]], before: str
) -> tuple[int, int, int] | None:
    """Ищет ``before`` в тексте реплик фрагмента; возвращает ``(index, start, end)``."""
    for index, _line in chunk:
        located = locate_before(entries[index].text, before)
        if located is not None:
            return index, located[0], located[1]
    return None


def _drop_overlaps(edits: Sequence[SemanticEdit]) -> list[SemanticEdit]:
    """Убирает пересекающиеся правки внутри одной реплики, сохраняя порядок."""
    ordered = sorted(edits, key=lambda edit: (edit.index, edit.start, -(edit.end - edit.start)))
    kept: list[SemanticEdit] = []
    last_end: dict[int, int] = {}
    for edit in ordered:
        if edit.start < last_end.get(edit.index, -1):
            continue
        kept.append(edit)
        last_end[edit.index] = edit.end
    return kept


def collect_semantic_edits(
    entries: list[TranscriptEntry],
    *,
    llm: LlmClient,
    max_chunk_chars: int,
    min_confidence: float = DEFAULT_LLM_SEMANTIC_MIN_CONFIDENCE,
    max_edits: int = DEFAULT_LLM_SEMANTIC_MAX_EDITS,
    max_edit_chars: int = DEFAULT_LLM_SEMANTIC_MAX_EDIT_CHARS,
) -> list[SemanticEdit]:
    """Собирает предложения семантической правки от LLM (без применения).

    Стенограмма режется на фрагменты; для каждого запрашивается список
    минимальных замен. Правка принимается, только если её ``before`` найден в
    тексте реплики детерминированно, уверенность не ниже ``min_confidence`` и
    длины фрагментов не превышают ``max_edit_chars``. Сбой отдельного фрагмента
    не роняет этап — пропускается с предупреждением.
    """
    if max_edits <= 0:
        return []

    labels = speaker_labels(unique_speakers(entries))
    chunks = iter_entry_chunks(entries, labels, max_chars=max_chunk_chars)

    accepted: list[SemanticEdit] = []
    seen: set[tuple[int, int, int]] = set()
    for chunk in chunks:
        user_prompt = _user_prompt("\n".join(line for _index, line in chunk))
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:  # noqa: BLE001 — мягкий пропуск фрагмента
            logger.warning("Семантические правки (фрагмент) не удались: %s", exc)
            continue

        for edit in parse_semantic_edits(raw):
            if edit.confidence < min_confidence:
                continue
            if len(edit.before) > max_edit_chars or len(edit.after) > max_edit_chars:
                continue
            located = _locate_in_chunk(entries, chunk, edit.before)
            if located is None:
                continue
            index, start, end = located
            key = (index, start, end)
            if key in seen:
                continue
            seen.add(key)
            accepted.append(replace(edit, index=index, start=start, end=end))
            if len(accepted) >= max_edits:
                break
        if len(accepted) >= max_edits:
            break

    return _drop_overlaps(accepted)


def to_suggestions(edits: Iterable[SemanticEdit]) -> list[Suggestion]:
    """Превращает локализованные правки в контракт ``Suggestion`` редактора."""
    return [
        Suggestion(
            index=edit.index,
            start=edit.start,
            end=edit.end,
            before=edit.before,
            after=edit.after,
            kind=KIND_SEMANTIC,
            reason=edit.reason or _DEFAULT_REASON,
        )
        for edit in edits
        if edit.index >= 0 and edit.start >= 0
    ]


def edit_to_dict(edit: SemanticEdit) -> dict[str, object]:
    """Сериализует правку (совместимо с ``Suggestion.as_dict`` плюс confidence)."""
    return {
        "index": edit.index,
        "start": edit.start,
        "end": edit.end,
        "before": edit.before,
        "after": edit.after,
        "kind": KIND_SEMANTIC,
        "reason": edit.reason or _DEFAULT_REASON,
        "confidence": edit.confidence,
    }


def write_semantic_edits(path: Path, edits: Sequence[SemanticEdit]) -> Path | None:
    """Сохраняет предложения рядом с результатом; ``None`` — нечего/сбой."""
    if not edits:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "edits": [edit_to_dict(edit) for edit in edits]}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Не удалось сохранить семантические правки %s: %s", path, exc)
        return None
    logger.info("Семантические правки сохранены: %s", path)
    return path


def load_stored_edits(path: Path) -> list[dict[str, object]]:
    """Читает сохранённые правки; отсутствие/битый файл — пустой список."""
    try:
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    edits = payload.get("edits")
    if not isinstance(edits, list):
        return []
    return [item for item in edits if isinstance(item, dict)]


def suggestions_from_stored_edits(
    texts: Sequence[str],
    stored: Sequence[Mapping[str, object]],
    *,
    min_confidence: float = DEFAULT_LLM_SEMANTIC_MIN_CONFIDENCE,
    max_edits: int = DEFAULT_LLM_SEMANTIC_MAX_EDITS,
    max_edit_chars: int = DEFAULT_LLM_SEMANTIC_MAX_EDIT_CHARS,
) -> list[Suggestion]:
    """Пересобирает предложения из сохранённых правок под текущий текст.

    ``start``/``end`` из файла игнорируются: ``before`` повторно locates в
    текущем тексте реплики. Так правки остаются корректными после ручных или
    автоматических изменений текста. Ненайденные/невалидные отбрасываются.
    """
    suggestions: list[Suggestion] = []
    seen: set[tuple[int, int, int]] = set()
    for item in stored:
        index_raw = item.get("index")
        if isinstance(index_raw, bool) or not isinstance(index_raw, int):
            continue
        if index_raw < 0 or index_raw >= len(texts):
            continue
        before = str(item.get("before", ""))
        after = str(item.get("after", ""))
        if not before or not after or before == after:
            continue
        if len(before) > max_edit_chars or len(after) > max_edit_chars:
            continue
        if _confidence(item) < min_confidence:
            continue
        located = locate_before(texts[index_raw], before)
        if located is None:
            continue
        start, end = located
        key = (index_raw, start, end)
        if key in seen:
            continue
        seen.add(key)
        reason = str(item.get("reason", "")).strip()
        suggestions.append(
            Suggestion(
                index=index_raw,
                start=start,
                end=end,
                before=before,
                after=after,
                kind=KIND_SEMANTIC,
                reason=reason or _DEFAULT_REASON,
            )
        )
        if len(suggestions) >= max_edits:
            break
    return suggestions
