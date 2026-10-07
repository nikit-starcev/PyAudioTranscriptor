"""REST API глоссария веб-интерфейса (``/api/glossary``).

Тонкая обёртка над :class:`~audio_transcriber.storage.glossary_db.GlossaryDB`:
список источников и записей, включение/отключение, CRUD и импорт ``.txt``/
``.csv``. БД берётся из настроек (``resolved_glossary_db``); ошибки sqlite
превращаются в понятные ответы 4xx/5xx, сервер не падает.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from audio_transcriber.storage.glossary_db import Entry, GlossaryDB, Source


class EntryCreate(BaseModel):
    """Тело ``POST /api/glossary/entries``."""

    canonical: str
    variant: str | None = None
    note: str | None = None
    source: str | None = None


class EntryUpdate(BaseModel):
    """Тело ``PATCH /api/glossary/entries/{id}`` (все поля опциональны)."""

    enabled: bool | None = None
    canonical: str | None = None
    variant: str | None = None
    note: str | None = None


class EntryQuickCreate(BaseModel):
    """Тело ``POST /api/glossary/quick`` — добавление термина из выделения.

    ``term`` — выделенный в стенограмме фрагмент; он попадает в «ошибочную
    форму» (``variant``), а правильное написание вводит пользователь в
    ``canonical``. Пустой канон отклоняется — сохранять термин без канона
    нельзя. Если ``variant`` не задан отдельно, но канон отличается от
    выделения, выделение сохраняется как ошибочная форма. ``source`` — имя
    записи/встречи, к которой привязывается термин.
    """

    term: str
    canonical: str | None = None
    variant: str | None = None
    note: str | None = None
    source: str | None = None


class SourceUpdate(BaseModel):
    """Тело ``PATCH /api/glossary/sources/{name}``."""

    enabled: bool


def register_glossary_routes(router: APIRouter, *, db_path: Callable[[], Path]) -> None:
    """Регистрирует маршруты ``/glossary`` на роутере."""

    def _open() -> GlossaryDB:
        try:
            return GlossaryDB(db_path())
        except (sqlite3.Error, OSError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось открыть БД глоссария: {exc}"
            ) from exc

    def _source_names() -> dict[int, str]:
        with _open() as db:
            return {source.id: source.name for source in db.list_sources()}

    @router.get("/glossary/sources")
    def list_sources() -> list[dict[str, object]]:
        with _open() as db:
            counts = db.entry_counts()
            return [
                _source_to_dict(source, counts.get(source.name, 0))
                for source in db.list_sources()
            ]

    @router.patch("/glossary/sources/{name}")
    def patch_source(name: str, payload: SourceUpdate) -> dict[str, object]:
        with _open() as db:
            if not db.set_source_enabled(name, payload.enabled):
                raise HTTPException(status_code=404, detail="Источник не найден")
            counts = db.entry_counts()
            source = next((item for item in db.list_sources() if item.name == name), None)
            if source is None:
                raise HTTPException(status_code=404, detail="Источник не найден")
            return _source_to_dict(source, counts.get(name, 0))

    @router.delete("/glossary/sources/{name}")
    def delete_source_route(name: str) -> dict[str, object]:
        with _open() as db:
            exists = any(source.name == name for source in db.list_sources())
            if not exists:
                raise HTTPException(status_code=404, detail="Источник не найден")
            affected = db.delete_source(name)
        return {"deleted": name, "entries": affected}

    @router.get("/glossary/entries")
    def list_entries(
        source: str | None = None,
        search: str | None = None,
        enabled_only: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, object]:
        with _open() as db:
            names = {item.id: item.name for item in db.list_sources()}
            needle = (search or "").strip()
            if needle:
                # Поиск по подстроке выполняется на стороне Python
                # (регистронезависимо, включая кириллицу) — пагинируем после него.
                matches = db.list_entries(
                    source=source or None,
                    search=needle,
                    enabled_only=enabled_only,
                )
                total = len(matches)
                start = max(offset, 0)
                page = matches[start:] if start else matches
                if limit is not None:
                    page = page[: max(limit, 0)]
            else:
                # Без поиска пагинация уходит в SQL: страница не требует полной
                # выгрузки таблицы (issue #89).
                total = db.count_entries(source=source or None, enabled_only=enabled_only)
                page = db.list_entries(
                    source=source or None,
                    enabled_only=enabled_only,
                    limit=limit,
                    offset=offset,
                )
        return {"entries": [_entry_to_dict(entry, names) for entry in page], "total": total}

    @router.post("/glossary/entries", status_code=201)
    def create_entry(payload: EntryCreate) -> dict[str, object]:
        with _open() as db:
            try:
                entry_id = db.add_entry(
                    payload.canonical,
                    variant=payload.variant,
                    note=payload.note,
                    source=payload.source or "manual",
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            names = {item.id: item.name for item in db.list_sources()}
            entry = db.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=500, detail="Запись не сохранилась")
        return _entry_to_dict(entry, names)

    @router.post("/glossary/quick", status_code=201)
    def create_entry_quick(payload: EntryQuickCreate) -> dict[str, object]:
        """Добавляет термин из выделения стенограммы (#17, #31).

        Выделение (``term``) сохраняется как «ошибочная форма», а канон
        (правильное написание) вводит пользователь: пустой ``canonical``
        отклоняется с 400. Пробелы и окружающая пунктуация обрезаются.
        """
        term = _clean_term(payload.term)
        if not term:
            raise HTTPException(status_code=400, detail="Термин не может быть пустым")
        canonical = _clean_term(payload.canonical)
        if not canonical:
            raise HTTPException(
                status_code=400, detail="Укажите канон (правильное написание)"
            )
        variant = _clean_term(payload.variant) or None
        if variant is None and canonical != term:
            variant = term
        note = payload.note.strip() if isinstance(payload.note, str) and payload.note.strip() else None
        with _open() as db:
            try:
                entry_id = db.add_entry(
                    canonical,
                    variant=variant,
                    note=note,
                    source=payload.source or "manual",
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            names = {item.id: item.name for item in db.list_sources()}
            entry = db.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=500, detail="Запись не сохранилась")
        return _entry_to_dict(entry, names)

    @router.patch("/glossary/entries/{entry_id}")
    def patch_entry(entry_id: int, payload: EntryUpdate) -> dict[str, object]:
        if payload.enabled is None and payload.canonical is None and payload.variant is None and payload.note is None:
            raise HTTPException(status_code=400, detail="Не указано ни одного поля для изменения")
        with _open() as db:
            try:
                updated = db.update_entry(
                    entry_id,
                    canonical=payload.canonical,
                    variant=payload.variant,
                    note=payload.note,
                    enabled=payload.enabled,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except sqlite3.IntegrityError as exc:
                raise HTTPException(
                    status_code=409, detail="Такая запись уже существует в источнике"
                ) from exc
            if not updated:
                raise HTTPException(status_code=404, detail="Запись не найдена")
            names = {item.id: item.name for item in db.list_sources()}
            entry = db.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Запись не найдена")
        return _entry_to_dict(entry, names)

    @router.delete("/glossary/entries/{entry_id}")
    def delete_entry_route(entry_id: int) -> dict[str, object]:
        with _open() as db:
            if not db.delete_entry(entry_id):
                raise HTTPException(status_code=404, detail="Запись не найдена")
        return {"deleted": entry_id}

    @router.post("/glossary/import")
    async def import_glossary(
        file: Annotated[UploadFile, File()],
        source: Annotated[str | None, Form()] = None,
        kind: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Пустой файл")
        original_name = Path(file.filename or "glossary.txt").name
        resolved_kind = _resolve_kind(kind, original_name)
        source_name = (source or "").strip() or original_name
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resolved_kind}") as handle:
                handle.write(raw)
                temp_path = Path(handle.name)
            with _open() as db:
                if resolved_kind == "csv":
                    report = db.import_csv(temp_path, source=source_name)
                else:
                    report = db.import_txt(temp_path, source=source_name)
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Не удалось импортировать: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return asdict(report)

    @router.get("/glossary/stats")
    def glossary_stats() -> dict[str, int]:
        with _open() as db:
            return {
                "sources": len(db.list_sources()),
                "entries": db.count(),
                "enabled": db.count_enabled(),
            }


#: Окружающая пунктуация, которую снимаем с выделенного термина.
_TERM_PUNCTUATION = "«»\"'`“”„‘’()[]{}<>,;:!?—–-….,"


def _clean_term(value: str | None) -> str:
    """Обрезает пробелы и окружающую пунктуацию, схлопывает пробелы."""
    if not value:
        return ""
    cleaned = value.strip().strip(_TERM_PUNCTUATION).strip()
    cleaned = cleaned.strip(_TERM_PUNCTUATION).strip()
    return " ".join(cleaned.split())


def _resolve_kind(kind: str | None, filename: str) -> str:
    value = (kind or "").strip().casefold()
    if value:
        if value not in {"txt", "csv"}:
            raise HTTPException(status_code=400, detail="kind должен быть txt или csv")
        return value
    return "csv" if Path(filename).suffix.casefold() == ".csv" else "txt"


def _source_to_dict(source: Source, count: int) -> dict[str, object]:
    return {
        "name": source.name,
        "kind": source.kind,
        "enabled": source.enabled,
        "count": count,
        "path": source.path,
    }


def _entry_to_dict(entry: Entry, source_names: Mapping[int, str]) -> dict[str, object]:
    source_name = source_names.get(entry.source_id) if entry.source_id is not None else None
    return {
        "id": entry.id,
        "canonical": entry.canonical,
        "variant": entry.variant,
        "category": entry.category,
        "note": entry.note,
        "source": source_name,
        "enabled": entry.enabled,
    }
