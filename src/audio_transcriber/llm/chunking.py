"""Общие примитивы нарезки стенограммы на фрагменты под контекст LLM.

Длинная стенограмма не влезает в контекст локальной модели, поэтому все
LLM-этапы (извлечение имён, правка терминов, резюме) режут её одним и тем же
механизмом на фрагменты по ~``max_chars`` символов. Вынесено в отдельный
модуль, чтобы этапы не дублировали логику и не зависели друг от друга.
"""

from __future__ import annotations

from collections.abc import Iterator

from audio_transcriber.config.defaults import DEFAULT_CONTEXT_SIZE
from audio_transcriber.domain.models import Speaker, TranscriptEntry


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
