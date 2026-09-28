"""Интерфейс LLM-клиента для постобработки стенограммы."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LlmClient(Protocol):
    """Контракт клиента к локальной LLM.

    Реализации должны уметь принимать список сообщений в формате
    OpenAI Chat Completions (``[{"role": ..., "content": ...}]``) и
    возвращать текст ответа. Тяжёлые импорты и запуск сервера происходят
    лениво — внутри реализации.
    """

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Отправляет сообщения модели и возвращает текст ответа."""
        ...

    def close(self) -> None:
        """Освобождает ресурсы (останавливает сервер, если был запущен)."""
        ...
