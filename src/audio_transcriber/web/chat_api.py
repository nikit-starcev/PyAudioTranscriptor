"""REST API чата по стенограмме (``/api/jobs/{job_id}/chat``, issue #54/#96).

Диалог с локальной LLM (или внешним провайдером, #21) по тексту активной
записи. Опорный контекст — реплики стенограммы с говорящими и таймкодами;
ответ стримится по SSE, а ссылки на реплики возвращаются структурировано, чтобы
UI переходил к нужному таймкоду и включал воспроизведение.

Эндпоинты:

* ``POST /api/jobs/{job_id}/chat`` — вопрос пользователя; ответ — поток SSE
  (``start`` → ``token``* → ``done`` либо ``error``);
* ``GET /api/jobs/{job_id}/chat`` — сохранённая история диалога;
* ``DELETE /api/jobs/{job_id}/chat`` — очистка истории задачи.

Модуль не знает про ``JobsDB``/HTTP-слой приложения: пути к БД, конфигурация и
клиент LLM передаются колбэками из :func:`audio_transcriber.web.app.register_api`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import TranscriptEntry, TranscriptionResult
from audio_transcriber.llm.base import LlmClient, StreamingLlmClient
from audio_transcriber.llm.chat import (
    DEFAULT_HISTORY_MESSAGES,
    build_chat_messages,
    parse_citations,
)
from audio_transcriber.llm.chunking import chunk_chars_for_context
from audio_transcriber.storage.chat_db import MAX_MESSAGE_LENGTH, ChatDB, ChatMessage

#: Заголовки SSE: без кэша и без буферизации прокси (как у остальных SSE-роутов).
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: Резерв символов промпта под инструкцию, историю диалога и ответ модели.
_PROMPT_RESERVE_CHARS = 2000

#: Фабрика LLM-клиента по конфигурации (подменяется в тестах).
LlmFactory = Callable[[AppConfig], LlmClient | None]

#: Восстановление стенограммы задачи (404, если результат не готов).
TranscriptResolver = Callable[[str], TranscriptionResult]

#: Конфигурация LLM задачи (с подставленным API-ключом из секретов).
ConfigResolver = Callable[[str], AppConfig]


class ChatRequest(BaseModel):
    """Тело ``POST /api/jobs/{job_id}/chat`` — вопрос пользователя."""

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


def _sse(event: Mapping[str, object]) -> str:
    """Кадр SSE с JSON-пейлоадом события чата."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _open(db_path: Callable[[], Path]) -> ChatDB:
    try:
        return ChatDB(db_path())
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Не удалось открыть БД истории чата: {exc}"
        ) from exc


def _history_pairs(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    """Приводит сохранённые сообщения к формату ``role/content`` для промпта."""
    return [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role in {"user", "assistant"} and message.content.strip()
    ]


def _transcript_max_chars(config: AppConfig) -> int:
    """Сколько символов стенограммы дать в промпт под текущий контекст LLM."""
    return max(1000, chunk_chars_for_context(config.llm_context_size) - _PROMPT_RESERVE_CHARS)


def _resolve_llm(
    config_for: ConfigResolver, llm_factory: LlmFactory, job_id: str
) -> tuple[AppConfig, LlmClient]:
    """Готовит конфигурацию и LLM-клиент или поднимает понятную HTTP-ошибку."""
    try:
        config = config_for(job_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Не удалось собрать конфигурацию LLM: {exc}"
        ) from exc

    if not config.llm_enabled:
        raise HTTPException(
            status_code=409,
            detail="LLM отключена. Включите «LLM-постобработку» в настройках, "
            "чтобы задавать вопросы по стенограмме.",
        )
    client = llm_factory(config)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="LLM включена, но не заданы модель и/или базовый URL. "
            "Укажите их в настройках (LLM_MODEL для локальной модели или "
            "LLM_BASE_URL и LLM_MODEL_NAME для внешнего провайдера).",
        )
    return config, client


def register_chat_routes(
    router: APIRouter,
    *,
    db_path: Callable[[], Path],
    resolve_transcript: TranscriptResolver,
    config_for: ConfigResolver,
    llm_factory: LlmFactory,
) -> None:
    """Регистрирует маршруты чата по стенограмме на роутере."""

    @router.get("/jobs/{job_id}/chat")
    def get_chat_history(job_id: str) -> dict[str, object]:
        resolve_transcript(job_id)
        with _open(db_path) as db:
            messages = db.list_messages(job_id)
            return {"messages": [message.as_dict() for message in messages]}

    @router.delete("/jobs/{job_id}/chat")
    def clear_chat_history(job_id: str) -> dict[str, object]:
        resolve_transcript(job_id)
        with _open(db_path) as db:
            cleared = db.clear(job_id)
        return {"cleared": cleared}

    @router.post("/jobs/{job_id}/chat")
    def chat(job_id: str, payload: ChatRequest) -> StreamingResponse:
        question = payload.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Пустой вопрос")

        result = resolve_transcript(job_id)
        entries: list[TranscriptEntry] = list(result.entries)
        if not entries:
            raise HTTPException(
                status_code=409, detail="В стенограмме нет реплик — нечего спрашивать"
            )
        config, client = _resolve_llm(config_for, llm_factory, job_id)
        max_chars = _transcript_max_chars(config)

        try:
            with _open(db_path) as db:
                history = db.list_messages(job_id)
            messages = build_chat_messages(
                entries,
                result.speakers,
                _history_pairs(history),
                question,
                max_context_chars=max_chars,
                max_history=DEFAULT_HISTORY_MESSAGES,
            )
            with _open(db_path) as db:
                db.add_message(job_id, "user", question)
        except Exception:
            client.close()
            raise

        def stream() -> Iterator[str]:
            yield _sse({"type": "start"})
            parts: list[str] = []
            try:
                if isinstance(client, StreamingLlmClient):
                    for chunk in client.chat_stream(messages):
                        parts.append(chunk)
                        yield _sse({"type": "token", "text": chunk})
                else:
                    text = client.chat(messages)
                    parts.append(text)
                    yield _sse({"type": "token", "text": text})
            except Exception as exc:  # noqa: BLE001 — ошибка LLM не должна ронять сервер
                yield _sse({"type": "error", "message": f"LLM не ответила: {exc}"})
                return
            finally:
                client.close()

            answer = "".join(parts).strip()
            if not answer:
                yield _sse({"type": "error", "message": "LLM вернула пустой ответ"})
                return
            citations = [item.as_dict() for item in parse_citations(answer, entries, result.speakers)]
            try:
                with _open(db_path) as db:
                    message_id = db.add_message(
                        job_id, "assistant", answer, citations=citations
                    )
                    stored = db.list_messages(job_id)
            except HTTPException as exc:
                yield _sse({"type": "error", "message": str(exc.detail)})
                return
            saved = next((m for m in stored if m.id == message_id), None)
            event: dict[str, object] = {
                "type": "done",
                "content": answer,
                "citations": citations,
            }
            if saved is not None:
                event["message"] = saved.as_dict()
            yield _sse(event)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=dict(SSE_HEADERS)
        )
