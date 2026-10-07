"""Интерфейс LLM-клиента для постобработки стенограммы."""

from __future__ import annotations

from collections.abc import Iterator
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


@runtime_checkable
class StreamingLlmClient(Protocol):
    """Клиент LLM с потоковой выдачей ответа (для чата по стенограмме).

    Отдельный протокол, а не расширение :class:`LlmClient`: не все реализации
    и тестовые заглушки умеют стриминг, а ``@runtime_checkable`` проверяет
    наличие метода. Чат использует этот протокол, а при его отсутствии —
    мягко деградирует к обычному :meth:`LlmClient.chat`.
    """

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Отдаёт ответ модели по мере генерации (последовательность фрагментов)."""
        ...
