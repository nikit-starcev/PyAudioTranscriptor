"""Общие примитивы нарезки стенограммы на фрагменты под контекст LLM.

Длинная стенограмма не влезает в контекст локальной модели, поэтому все
LLM-этапы (извлечение имён, правка терминов, резюме) режут её одним и тем же
механизмом на фрагменты по ~``max_chars`` символов. Вынесено в отдельный
модуль, чтобы этапы не дублировали логику и не зависели друг от друга.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from audio_transcriber.config.defaults import DEFAULT_CONTEXT_SIZE
from audio_transcriber.domain.models import Speaker, TranscriptEntry

# Разделитель между машинной меткой и подставленным именем в ``display_name``
# («Спикер 1 — Максим»). Совпадает с форматом ``apply_participant_names``.
_NAME_SEPARATOR = " — "

# Метка-заглушка говорящего без имени: «Спикер 1», «SPEAKER_00», «speaker-2».
_GENERIC_SPEAKER_LABEL = re.compile(
    r"^(?:спикер|speaker)\s*[_\-]?\s*\d*$", re.IGNORECASE
)

# Прочие безликие метки, которые не являются именем участника.
_PLACEHOLDER_LABELS = frozenset({"", "?", "??", "???"})


def unique_speakers(entries: list[TranscriptEntry]) -> list[Speaker]:
    """Возвращает уникальных говорящих в порядке первого появления."""
    seen: dict[str, Speaker] = {}
    for entry in entries:
        if entry.speaker is not None and entry.speaker.id not in seen:
            seen[entry.speaker.id] = entry.speaker
    return list(seen.values())


def speaker_labels(speakers: list[Speaker]) -> dict[str, str]:
    """Метки говорящих для LLM: ``speaker_id -> «Спикер N»`` (1-indexed)."""
    return {
        speaker.id: f"Спикер {index}" for index, speaker in enumerate(speakers, start=1)
    }


def human_name(display_name: str) -> str | None:
    """Человеческое имя говорящего из ``display_name`` (или ``None``).

    Возвращает ``None``, если у говорящего только машинная метка («Спикер N»,
    ``SPEAKER_00``, ``?``). Имя, подставленное enrollment/LLM, хранится в
    ``display_name`` как «Спикер N — Имя» — тогда возвращается часть после
    разделителя.
    """
    text = display_name.strip()
    if text in _PLACEHOLDER_LABELS:
        return None
    if _NAME_SEPARATOR in text:
        candidate = text.rsplit(_NAME_SEPARATOR, 1)[-1].strip()
        return candidate or None
    if _GENERIC_SPEAKER_LABEL.match(text):
        return None
    return text or None


def named_speaker_labels(speakers: list[Speaker]) -> dict[str, str]:
    """Метки говорящих с человеческими именами (для резюме).

    Использует имя из ``display_name`` (enrollment, ручное переименование,
    подстановка LLM), иначе — «Спикер N» по порядку. Нужно, чтобы резюме
    составляло список участников по актуальным именам, а не по машинным меткам.
    """
    return {
        speaker.id: human_name(speaker.display_name) or f"Спикер {index}"
        for index, speaker in enumerate(speakers, start=1)
    }


def transcript_lines(
    entries: list[TranscriptEntry], labels: dict[str, str]
) -> Iterator[str]:
    """Строки стенограммы с метками «Спикер N: реплика» (пустые пропускаются)."""
    for entry in entries:
        if not entry.text.strip():
            continue
        label = labels.get(entry.speaker.id, "?") if entry.speaker else "?"
        yield f"{label}: {entry.text.strip()}"


def iter_transcript_chunks(
    entries: list[TranscriptEntry], labels: dict[str, str], *, max_chars: int
) -> list[str]:
    """Режет стенограмму на фрагменты по ~``max_chars`` символов.

    Нужно, потому что длинная стенограмма превышает контекст LLM — тогда
    результат считается по фрагментам и объединяется.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in transcript_lines(entries, labels):
        if current and size + len(line) > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# Запас под контекст LLM: на 4096 токенов безопасно ~6000 символов русского
# текста вместе с системным промптом, инструкцией и ответом модели. Коэффициент
# привязывает размер чанка к фактическому ``llm_context_size``, а не к
# магическому числу «под 4096».
CHUNK_CHARS_PER_CONTEXT_TOKEN = 6000 / 4096
# Нижняя граница: крошечный контекст не должен давать неработоспособный чанк.
MIN_CHUNK_CHARS = 1000


def chunk_chars_for_context(context_size: int) -> int:
    """Размер фрагмента стенограммы под контекст LLM в токенах.

    Пропорционален ``context_size`` с запасом на промпт и ответ модели.
    """
    return max(MIN_CHUNK_CHARS, int(context_size * CHUNK_CHARS_PER_CONTEXT_TOKEN))


# Резерв символов на ответ модели (JSON со списком правок терминов).
TERM_CHECK_RESPONSE_RESERVE_CHARS = 600


def transcript_chunk_chars_for_prompt(
    total_budget_chars: int,
    *,
    overhead_chars: int,
) -> int:
    """Сколько символов стенограммы можно добавить, не выходя за бюджет запроса.

    ``total_budget_chars`` — безопасный объём всего запроса (системный промпт +
    пользовательский промпт + ответ модели) в символах, ``overhead_chars`` —
    объём фиксированной части промпта без текста стенограммы (инструкции,
    список терминов, резерв на ответ). Так накладные расходы промпта учитываются
    в размере фрагмента, и запрос не превышает контекст модели (иначе — HTTP 400).
    """
    return max(1, total_budget_chars - max(0, overhead_chars))



# Значение по умолчанию — под стандартный контекст 4096 токенов.
DEFAULT_CHUNK_CHARS = chunk_chars_for_context(DEFAULT_CONTEXT_SIZE)
