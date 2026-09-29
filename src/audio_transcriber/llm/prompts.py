"""Прозрачность LLM-промптов и пользовательские доп. инструкции.

Два независимых механизма, общих для всех LLM-этапов (резюме, термины,
имена):

- **Доп. инструкции пользователя.** Текст из конфига (``LLM_PROMPT_EXTRA``)
  и/или файла (``LLM_PROMPT_FILE``) единообразно подмешивается в системный
  промпт каждого запроса. Длина ограничена, чтобы не вытеснить стенограмму
  из контекста модели.
- **Сохранение фактических промптов.** Обёртка :class:`PromptRecordingClient`
  перехватывает сообщения прямо перед отправкой в LLM, и
  :class:`PromptRecorder` записывает их (system + user по каждому этапу) в
  файл ``<output_stem>.llm_prompt.txt`` — пользователь может посмотреть и
  скопировать то, что реально ушло в модель.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from audio_transcriber.llm.base import LlmClient

logger = logging.getLogger(__name__)

# Доп. инструкции добавляются к системному промпту каждого этапа, поэтому не
# должны вытеснять саму стенограмму из контекста. Предел пропорционален
# контексту и ограничен разумным диапазоном.
_MIN_EXTRA_CHARS = 500
_MAX_EXTRA_CHARS = 4000

# Колбэк, получающий этап и фактические сообщения, отправленные в LLM.
PromptSink = Callable[[str, list[dict[str, str]]], None]


def max_extra_chars_for_context(context_size: int) -> int:
    """Предел длины доп. инструкций под контекст LLM (в символах)."""
    return max(_MIN_EXTRA_CHARS, min(_MAX_EXTRA_CHARS, context_size // 4))


def load_extra_instructions(
    extra_text: str | None, extra_file: Path | None
) -> str | None:
    """Собирает доп. инструкции из текста конфига и файла.

    Сначала добавляется инлайн-текст, затем содержимое файла. Пустые значения
    ничего не меняют: если оба источника пусты, возвращается ``None``.
    Недоступный файл не роняет конвейер — только предупреждение в лог.
    """
    parts: list[str] = []
    if extra_text and extra_text.strip():
        parts.append(extra_text.strip())
    if extra_file is not None:
        try:
            content = Path(extra_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Не удалось прочитать файл доп. инструкций %s: %s", extra_file, exc)
            content = ""
        if content:
            parts.append(content)
    if not parts:
        return None
    return "\n".join(parts)


def append_extra_instructions(
    system_prompt: str, extra: str | None, *, max_chars: int
) -> str:
    """Добавляет доп. инструкции к системному промпту с ограничением длины."""
    if not extra or not extra.strip():
        return system_prompt
    text = extra.strip()
    if len(text) > max_chars:
        logger.warning(
            "Доп. инструкции обрезаны до %d символов (лимит под контекст LLM)", max_chars
        )
        text = text[:max_chars].rstrip() + " …"
    return f"{system_prompt}\n\nДополнительные инструкции пользователя:\n{text}"


def apply_extra_to_messages(
    messages: list[dict[str, str]], extra: str | None, *, max_chars: int
) -> list[dict[str, str]]:
    """Возвращает копию сообщений с доп. инструкциями в системном сообщении.

    Доп. инструкции добавляются к первому system-сообщению; если его нет —
    создаётся новое системное сообщение. Пустые инструкции возвращают исходный
    список без изменений.
    """
    if not extra or not extra.strip():
        return messages
    result = [dict(message) for message in messages]
    for message in result:
        if message.get("role") == "system":
            merged = append_extra_instructions(
                message.get("content", ""), extra, max_chars=max_chars
            )
            message["content"] = merged
            return result
    result.insert(
        0,
        {
            "role": "system",
            "content": append_extra_instructions("", extra, max_chars=max_chars).lstrip(),
        },
    )
    return result


class PromptRecordingClient:
    """Обёртка :class:`LlmClient`: доп. инструкции + запись фактических промптов.

    Подмешивает доп. инструкции в системный промпт каждого запроса и передаёт
    фактические сообщения в ``on_prompt`` перед отправкой в модель. Это
    позволяет применять одну и ту же логику ко всем этапам без изменения их
    сигнатур.
    """

    def __init__(
        self,
        client: LlmClient,
        *,
        stage: str,
        extra_instructions: str | None = None,
        max_extra_chars: int = _MAX_EXTRA_CHARS,
        on_prompt: PromptSink | None = None,
    ) -> None:
        self._client = client
        self._stage = stage
        self._extra_instructions = extra_instructions
        self._max_extra_chars = max_extra_chars
        self._on_prompt = on_prompt

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Добавляет доп. инструкции, записывает промпт и делегирует в LLM."""
        effective = apply_extra_to_messages(
            messages, self._extra_instructions, max_chars=self._max_extra_chars
        )
        if self._on_prompt is not None:
            self._on_prompt(self._stage, effective)
        return self._client.chat(effective)

    def close(self) -> None:
        """Ничего не делает: обёртка не владеет исходным клиентом.

        Жизненным циклом ``llama-server`` управляет оркестратор
        (:func:`audio_transcriber.llm.postprocess.run_llm_postprocess`), поэтому
        закрывать клиент через обёртку нельзя — он общий для всех этапов.
        """


class PromptRecorder:
    """Накапливает фактические промпты и сохраняет их в один файл.

    Промпты записываются по этапам; файл создаётся один раз в конце
    LLM-постобработки. Если промптов не было или путь не задан, файл не
    создаётся.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._entries: list[tuple[str, list[dict[str, str]]]] = []

    @property
    def entries(self) -> list[tuple[str, list[dict[str, str]]]]:
        """Записанные промпты ``(этап, сообщения)`` — для тестов и логов."""
        return list(self._entries)

    def record(self, stage: str, messages: list[dict[str, str]]) -> None:
        """Сохраняет фактический промпт этапа в памяти."""
        self._entries.append((stage, [dict(message) for message in messages]))

    def write(self) -> Path | None:
        """Записывает все промпты в файл; возвращает путь или ``None``."""
        if self._path is None or not self._entries:
            return None

        blocks: list[str] = []
        for stage, messages in self._entries:
            lines = [f"=== {stage} ==="]
            for message in messages:
                lines.append(f"--- {message.get('role', '')} ---")
                lines.append(message.get("content", ""))
            blocks.append("\n".join(lines))
        text = "\n\n".join(blocks) + "\n"

        try:
            self._path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.warning("Не удалось сохранить файл промптов %s: %s", self._path, exc)
            return None
        logger.info("Фактические промпты LLM сохранены: %s", self._path)
        return self._path
