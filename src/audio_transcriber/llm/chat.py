"""Чат по стенограмме: сборка контекста, промпта и цитат (issue #54/#96).

В отличие от резюме (:mod:`audio_transcriber.llm.summary`), чат — диалог:
пользователь задаёт вопрос, а локальная LLM отвечает по тексту активной
записи. Опорный контекст — реплики стенограммы с говорящими и таймкодами.

Реплики нумеруются (``[N] [M:SS] Говорящий: текст``), и модель просим ссылаться
на источники этими номерами. После ответа номера разбираются в структурированные
цитаты (:class:`ChatCitation`) — по ним UI переходит к реплике и включает
воспроизведение. Модуль не зависит от HTTP/БД и переиспользует разметку имён и
нарезку контекста из :mod:`audio_transcriber.llm.chunking`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.llm.chunking import named_speaker_labels

#: Системная инструкция чата: отвечать только по стенограмме и ссылаться на реплики.
CHAT_SYSTEM_PROMPT = (
    "Ты — ассистент, который отвечает на вопросы по стенограмме деловой встречи. "
    "Отвечай только на русском языке и строго по фактам из стенограммы. "
    "Ничего не выдумывай: если ответа в тексте нет — честно скажи, что данных нет. "
    "Каждое утверждение подкрепляй номером реплики-источника в квадратных скобках, "
    "например [3] или [3, 7]. Номера бери из пронумерованной стенограммы ниже. "
    "Верни только ответ, без пояснений о процессе."
)

#: Бюджет контекста стенограммы по умолчанию (символы). С запасом под
#: системный промпт, историю диалога и ответ модели при контексте 4096 токенов.
DEFAULT_CONTEXT_CHARS = 4000

#: Сколько последних сообщений истории включать в запрос (пары вопрос/ответ).
DEFAULT_HISTORY_MESSAGES = 12

#: Максимальная длина текста цитаты в пейлоаде API.
MAX_CITATION_TEXT_CHARS = 400

_EMPTY_CONTEXT = "(стенограмма пуста)"

#: Номера реплик-источников в ответе: ``[3]``, ``[3, 7]``, ``[3,7]``.
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


@dataclass(frozen=True, slots=True)
class ChatCitation:
    """Ссылка на реплику стенограммы, упомянутую в ответе LLM."""

    index: int
    start: float
    end: float
    speaker: str
    text: str

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для JSON-ответа API."""
        text = self.text.strip()
        if len(text) > MAX_CITATION_TEXT_CHARS:
            text = text[: MAX_CITATION_TEXT_CHARS - 1].rstrip() + "…"
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": text,
        }


def format_timecode(seconds: float) -> str:
    """Таймкод ``M:SS`` (для часа и больше — ``H:MM:SS``)."""
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _speaker_label(labels: Mapping[str, str], entry: TranscriptEntry) -> str:
    if entry.speaker is None:
        return "?"
    return labels.get(entry.speaker.id, entry.speaker.display_name or "?")


def build_transcript_context(
    entries: Sequence[TranscriptEntry],
    speakers: Sequence[Speaker] | None = None,
    *,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
) -> str:
    """Нумерованная стенограмма для промпта чата.

    Каждая реплика — ``[индекс] [M:SS] Говорящий: текст``; индекс — позиция в
    исходном списке, по ней затем разрешаются цитаты. Если текст не влезает в
    ``max_chars``, хвост опускается, и об этом явно сообщается в конце контекста.
    """
    labels = named_speaker_labels(list(speakers) if speakers is not None else _unique_speakers(entries))
    lines: list[str] = []
    size = 0
    omitted = 0
    for index, entry in enumerate(entries):
        text = entry.text.strip()
        if not text:
            continue
        line = f"[{index}] [{format_timecode(entry.start)}] {_speaker_label(labels, entry)}: {text}"
        if lines and size + len(line) + 1 > max_chars:
            omitted = len(entries) - index
            break
        lines.append(line)
        size += len(line) + 1
    if not lines:
        return _EMPTY_CONTEXT
    if omitted > 0:
        lines.append(f"[…] Ещё {omitted} реплик(и) опущено из-за ограничения контекста.")
    return "\n".join(lines)


def parse_citations(
    answer: str,
    entries: Sequence[TranscriptEntry],
    speakers: Sequence[Speaker] | None = None,
) -> list[ChatCitation]:
    """Разбирает номера реплик из ответа в структурированные цитаты.

    Учитываются ссылки вида ``[3]`` и ``[3, 7]``. Номера вне диапазона
    игнорируются; дубликаты схлопываются (порядок — первого упоминания).
    """
    labels = named_speaker_labels(list(speakers) if speakers is not None else _unique_speakers(entries))
    seen: set[int] = set()
    result: list[ChatCitation] = []
    for match in _CITATION_RE.finditer(answer):
        for raw in match.group(1).split(","):
            token = raw.strip()
            if not token.isdigit():
                continue
            index = int(token)
            if index in seen or index < 0 or index >= len(entries):
                continue
            seen.add(index)
            entry = entries[index]
            result.append(
                ChatCitation(
                    index=index,
                    start=entry.start,
                    end=entry.end,
                    speaker=_speaker_label(labels, entry),
                    text=entry.text,
                )
            )
    return result


def build_chat_messages(
    entries: Sequence[TranscriptEntry],
    speakers: Sequence[Speaker] | None,
    history: Iterable[Mapping[str, str]],
    question: str,
    *,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    max_history: int = DEFAULT_HISTORY_MESSAGES,
) -> list[dict[str, str]]:
    """Сообщения OpenAI Chat Completions для одного хода чата.

    Системное сообщение содержит инструкцию и нумерованную стенограмму; далее
    идут последние ``max_history`` реплик диалога, затем текущий вопрос.
    """
    context = build_transcript_context(
        entries, speakers, max_chars=max_context_chars
    )
    system = f"{CHAT_SYSTEM_PROMPT}\n\nСтенограмма (реплики пронумерованы):\n{context}"

    normalized: list[dict[str, str]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content})
    if max_history > 0:
        normalized = normalized[-max_history:]

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(normalized)
    messages.append({"role": "user", "content": question.strip()})
    return messages


def _unique_speakers(entries: Sequence[TranscriptEntry]) -> list[Speaker]:
    """Уникальные говорящие в порядке первого появления (fallback без списка)."""
    seen: dict[str, Speaker] = {}
    for entry in entries:
        if entry.speaker is not None and entry.speaker.id not in seen:
            seen[entry.speaker.id] = entry.speaker
    return list(seen.values())
