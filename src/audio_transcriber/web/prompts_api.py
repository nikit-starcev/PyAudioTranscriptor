"""REST API пользовательских шаблонов промпта резюме (``/api/summary-prompts``, #97).

Тонкая обёртка над :class:`~audio_transcriber.storage.prompts_db.PromptsDB`:
список шаблонов, создание/редактирование/удаление, выбор активного. Активный
шаблон используется при постановке задачи (``build_job_config``) и может быть
переопределён на кнопку «Сформировать протокол». Ошибки sqlite превращаются в
понятные ответы 4xx/5xx, сервер не падает.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from audio_transcriber.storage.prompts_db import Prompt, PromptsDB


class PromptCreate(BaseModel):
    """Тело ``POST /api/summary-prompts``."""

    name: str
    body: str


class PromptUpdate(BaseModel):
    """Тело ``PATCH /api/summary-prompts/{id}`` (все поля опциональны)."""

    name: str | None = None
    body: str | None = None


def register_prompts_routes(router: APIRouter, *, db_path: Callable[[], Path]) -> None:
    """Регистрирует маршруты ``/summary-prompts`` на роутере."""

    def _open() -> PromptsDB:
        try:
            return PromptsDB(db_path())
        except (sqlite3.Error, OSError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось открыть БД шаблонов: {exc}"
            ) from exc

    @router.get("/summary-prompts")
    def list_prompts() -> dict[str, object]:
        with _open() as db:
            prompts = db.list_prompts()
            return {
                "prompts": [_prompt_to_dict(prompt) for prompt in prompts],
                "active_id": db.active_id(),
            }

    @router.post("/summary-prompts", status_code=201)
    def create_prompt(payload: PromptCreate) -> dict[str, object]:
        with _open() as db:
            try:
                prompt_id = db.add_prompt(payload.name, payload.body)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except sqlite3.IntegrityError as exc:
                raise HTTPException(
                    status_code=409, detail="Шаблон с таким именем уже существует"
                ) from exc
            prompt = db.get_prompt(prompt_id)
        if prompt is None:
            raise HTTPException(status_code=500, detail="Шаблон не сохранился")
        return _prompt_to_dict(prompt)

    @router.patch("/summary-prompts/{prompt_id}")
    def patch_prompt(prompt_id: int, payload: PromptUpdate) -> dict[str, object]:
        if payload.name is None and payload.body is None:
            raise HTTPException(status_code=400, detail="Не указано ни одного поля")
        with _open() as db:
            try:
                updated = db.update_prompt(
                    prompt_id, name=payload.name, body=payload.body
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except sqlite3.IntegrityError as exc:
                raise HTTPException(
                    status_code=409, detail="Шаблон с таким именем уже существует"
                ) from exc
            if not updated:
                raise HTTPException(status_code=404, detail="Шаблон не найден")
            prompt = db.get_prompt(prompt_id)
        if prompt is None:
            raise HTTPException(status_code=404, detail="Шаблон не найден")
        return _prompt_to_dict(prompt)

    @router.delete("/summary-prompts/{prompt_id}")
    def delete_prompt(prompt_id: int) -> dict[str, object]:
        with _open() as db:
            if not db.delete_prompt(prompt_id):
                raise HTTPException(status_code=404, detail="Шаблон не найден")
            active_id = db.active_id()
        return {"deleted": prompt_id, "active_id": active_id}

    @router.post("/summary-prompts/{prompt_id}/activate")
    def activate_prompt(prompt_id: int) -> dict[str, object]:
        with _open() as db:
            if not db.set_active(prompt_id):
                raise HTTPException(status_code=404, detail="Шаблон не найден")
            prompt = db.get_prompt(prompt_id)
            active_id = db.active_id()
        if prompt is None:
            raise HTTPException(status_code=404, detail="Шаблон не найден")
        return {"active_id": active_id, "prompt": _prompt_to_dict(prompt)}


def _prompt_to_dict(prompt: Prompt) -> dict[str, object]:
    return {
        "id": prompt.id,
        "name": prompt.name,
        "body": prompt.body,
        "builtin": prompt.builtin,
        "created_at": prompt.created_at,
        "updated_at": prompt.updated_at,
    }
