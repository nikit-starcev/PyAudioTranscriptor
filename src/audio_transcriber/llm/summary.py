"""Резюме встречи локальной LLM (map-reduce по фрагментам).

Длинная стенограмма не влезает в контекст модели, поэтому резюме считается
в два приёма: короткая стенограмма суммируется одним запросом, длинная —
сначала по фрагментам (map), затем тезисы сводятся в единое резюме (reduce).

Промпт строгий: модель не должна выдумывать факты и включает только те
разделы, для которых в тексте есть данные. При сбое LLM этап мягко
пропускается (возвращается ``None``) — конвейер продолжает работу без резюме.
"""

from __future__ import annotations

import logging
import re

from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.chunking import (
    DEFAULT_CHUNK_CHARS,
    iter_transcript_chunks,
    named_speaker_labels,
    unique_speakers,
)

logger = logging.getLogger(__name__)

# Единый формат резюме, которого придерживаются все промпты этапа.
_SUMMARY_FORMAT = (
    "Тема: <основная тема встречи>\n"
    "Участники: <имена, если они есть в тексте>\n"
    "Решения и договорённости:\n- <пункт>\n"
    "Открытые вопросы:\n- <пункт>\n"
    "Что сделать:\n- <действие> — <кому, если сказано>\n"
    "Сроки и системы: <упомянутые сроки/даты и названия систем>"
)

_SUMMARY_RULES = (
    "Пиши только на русском языке и только по фактам из стенограммы. СТРОГО "
    "запрещено выдумывать имена, решения, договорённости, сроки, действия и "
    "названия систем, которых нет в тексте. Не пересказывай стенограмму — "
    "только выводы. Если данных для раздела нет, не включай этот раздел. "
    "Верни ТОЛЬКО резюме в строгом формате, без пояснений и без JSON."
)

_SUMMARY_SYSTEM_PROMPT = (
    "Ты — ассистент, который составляет краткое резюме деловой встречи по её "
    "стенограмме с метками говорящих. " + _SUMMARY_RULES + "\n" + _SUMMARY_FORMAT
)

_PARTIAL_SYSTEM_PROMPT = (
    "Ты — ассистент, который выделяет из ФРАГМЕНТА стенограммы деловой встречи "
    "тезисы для последующего общего резюме. Пиши только на русском языке и "
    "только по фактам из текста; не выдумывай имена, решения, сроки и системы. "
    "Верни кратко, по разделам (только те, для которых есть данные), в формате:\n"
    + _SUMMARY_FORMAT
)

_REDUCE_SYSTEM_PROMPT = (
    "Ты — ассистент, который сводит разрозненные тезисы по одной встрече в "
    "единое краткое резюме. Убери повторы, объедини близкие пункты и сохрани "
    "только факты из тезисов; ничего не выдумывай. " + _SUMMARY_RULES + "\n"
    + _SUMMARY_FORMAT
)

# Ответы-заглушки, которыми LLM сообщает, что данных нет: резюме не создаём.
_EMPTY_SUMMARY_MARKERS = frozenset(
    {
        "нет данных",
        "нет информации",
        "недостаточно данных",
        "нет содержания",
        "не удалось составить резюме",
        "отсутствует",
    }
)


def _is_empty_summary(text: str) -> bool:
    """Пустой/заглушечный ли ответ LLM (не считаем его резюме)."""
    folded = re.sub(r"[\s.,!?:;—–-]+", " ", text).strip()
    return not folded or folded in _EMPTY_SUMMARY_MARKERS


def _normalize_summary(raw: str) -> str | None:
    """Приводит ответ модели к резюме; пустой/заглушечный ответ → ``None``."""
    text = raw.strip()
    if not text:
        return None
    text = re.sub(r"\n{3,}", "\n\n", text)
    if _is_empty_summary(text):
        logger.info("LLM не дала содержательного резюме — пропускаю")
        return None
    return text


def _chat(llm: LlmClient, system: str, user: str) -> str | None:
    """Один запрос к LLM; при сбое — предупреждение и ``None`` (мягко)."""
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
    except Exception as exc:  # noqa: BLE001 — сбой резюме не роняет конвейер
        logger.warning("Резюме (фрагмент) не удалось: %s", exc)
        return None
    return _normalize_summary(raw)


def _single_user_prompt(chunk: str) -> str:
    return (
        f"Стенограмма встречи:\n{chunk}\n\n"
        "Составь краткое резюме по правилам. Если данных для раздела нет — "
        "не включай этот раздел."
    )


def _partial_user_prompt(chunk: str) -> str:
    return (
        f"Фрагмент стенограммы встречи:\n{chunk}\n\n"
        "Выдели краткие тезисы по разделам. Если данных для раздела нет — "
        "не включай этот раздел."
    )


def _reduce_user_prompt(partials: str) -> str:
    return (
        f"Тезисы по фрагментам встречи:\n{partials}\n\n"
        "Сведи их в единое краткое резюме по правилам. Если данных для раздела "
        "нет — не включай этот раздел."
    )


def summarize_meeting(
    entries: list[TranscriptEntry],
    speakers: list[Speaker] | None = None,
    *,
    llm: LlmClient,
    max_chunk_chars: int | None = None,
) -> str | None:
    """Строит резюме встречи через локальную LLM.

    Короткая стенограмма суммируется одним запросом; длинная режется на
    фрагменты, которые суммируются по отдельности (map), а затем сводятся в
    единое резюме (reduce). Возвращает текст резюме или ``None``, если данных
    нет либо LLM недоступна (этап мягко пропускается).

    Метки говорящих для модели берутся из ``display_name``: если имя известно
    (enrollment, ручное переименование, подстановка LLM) — в стенограмме и
    разделе «Участники» резюме будут актуальные имена, а не «Спикер N».
    """
    speakers = speakers or unique_speakers(entries)
    labels = named_speaker_labels(speakers)
    max_chars = max_chunk_chars if max_chunk_chars is not None else DEFAULT_CHUNK_CHARS
    chunks = iter_transcript_chunks(entries, labels, max_chars=max_chars)
    if not chunks:
        return None

    if len(chunks) == 1:
        return _chat(llm, _SUMMARY_SYSTEM_PROMPT, _single_user_prompt(chunks[0]))

    partials: list[str] = []
    for chunk in chunks:
        partial = _chat(llm, _PARTIAL_SYSTEM_PROMPT, _partial_user_prompt(chunk))
        if partial:
            partials.append(partial)

    if not partials:
        return None
    if len(partials) == 1:
        return partials[0]

    combined = "\n\n".join(partials)
    reduced = _chat(llm, _REDUCE_SYSTEM_PROMPT, _reduce_user_prompt(combined))
    # Если свести не удалось — отдаём собранные тезисы, а не теряем их.
    return reduced if reduced is not None else combined
