"""REST API архива стенограмм (``/api/archive``, issue #99).

Архив привязывает готовые стенограммы к датам/встречам: заголовок,
редактируемая дата, участники/теги и заметки. Отдельный эндпоинт выполняет
полнотекстовый поиск по текстам стенограмм с фильтрами по говорящему и дате.

Эндпоинты:

* ``GET /api/archive`` — записи архива (задачи с готовым результатом) с
  фильтром по диапазону дат, списком говорящих для фильтра;
* ``GET /api/archive/search`` — полнотекстовый поиск по текстам (``q``),
  фильтры ``speaker``/``date_from``/``date_to``, переход к задаче и реплике;
* ``GET /api/archive/{job_id}`` — запись одной задачи (создаётся с датой по
  умолчанию, если ещё нет);
* ``PATCH /api/archive/{job_id}`` — правка заголовка/даты/участников/заметок.

Индекс текстов строится лениво: при первом обращении к архиву/поиску задачи с
готовым результатом индексируются и переиндексируются только при изменении
файла результата. Модуль не знает про ``JobsDB``/пути: доступ к задачам и
результатам передаётся колбэками из :func:`audio_transcriber.web.app.register_api`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from audio_transcriber.storage.archive_db import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    ArchiveDB,
    ArchiveRecord,
    SearchHit,
)
from audio_transcriber.web.storage.jobs_db import Job

#: Задача и сведения о её JSON-результате (путь и время изменения).
ResultInfo = tuple[Path, int]
#: Чтение JSON-результата по пути (``None`` — нет/битый файл).
ResultLoader = Callable[[Path], dict[str, object] | None]
#: Сведения о результате задачи или ``None``, если результат не готов.
ResultInfoFn = Callable[[Job], ResultInfo | None]
#: Все задачи для обхода при ленивой индексации.
JobsFn = Callable[[], Iterable[Job]]
#: Задача по id или ``None``.
JobGetter = Callable[[str], Job | None]


class RecordUpdate(BaseModel):
    """Тело ``PATCH /api/archive/{job_id}`` (все поля опциональны).

    Пустая строка у ``title``/``meeting_date``/``notes`` и пустой список
    ``participants`` очищают соответствующее поле. Дату присылают в формате
    ``ГГГГ-ММ-ДД``.
    """

    title: str | None = None
    meeting_date: str | None = None
    participants: list[str] | None = None
    notes: str | None = None


def register_archive_routes(
    router: APIRouter,
    *,
    db_path: Callable[[], Path],
    iter_jobs: JobsFn,
    result_info: ResultInfoFn,
    load_result: ResultLoader,
    get_job: JobGetter,
) -> None:
    """Регистрирует маршруты ``/archive`` на роутере."""

    def _open() -> ArchiveDB:
        try:
            return ArchiveDB(db_path())
        except (sqlite3.Error, OSError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось открыть БД архива: {exc}"
            ) from exc

    def _sync(db: ArchiveDB) -> None:
        """Лениво индексирует стенограммы готовых задач, обновляя устаревшие."""
        for job in iter_jobs():
            info = result_info(job)
            if info is None:
                continue
            path, mtime = info
            if not db.is_current(job.id, mtime):
                payload = load_result(path)
                if payload is None:
                    continue
                db.index_job(
                    job.id, _extract_entries(payload), result_mtime=mtime
                )
            db.ensure_record(job.id, default_date=_job_date(job))

    @router.get("/archive")
    def list_archive(
        date_from: str | None = Query(default=None),
        date_to: str | None = Query(default=None),
    ) -> dict[str, object]:
        with _open() as db:
            _sync(db)
            records = _filter_by_date(db.list_records(), date_from, date_to)
            jobs = {job.id: job for job in iter_jobs()}
            payload = [_record_payload(record, jobs.get(record.job_id)) for record in records]
            return {"records": payload, "speakers": db.speakers()}

    @router.get("/archive/search")
    def search_archive(
        q: str = Query(default=""),
        speaker: str | None = Query(default=None),
        date_from: str | None = Query(default=None),
        date_to: str | None = Query(default=None),
        limit: int = Query(default=DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, object]:
        text = (q or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Пустой поисковый запрос")
        with _open() as db:
            _sync(db)
            job_ids = _job_ids_in_range(db.list_records(), date_from, date_to)
            hits, total = db.search(
                text, job_ids=job_ids, speaker=speaker, limit=limit, offset=offset
            )
            jobs = {job.id: job for job in iter_jobs()}
            result = [
                _hit_payload(hit, jobs.get(hit.job_id), db.get_record(hit.job_id))
                for hit in hits
            ]
            return {
                "query": text,
                "hits": result,
                "total": total,
                "limit": limit,
                "offset": offset,
                "speakers": db.speakers(),
            }

    @router.get("/archive/{job_id}")
    def get_archive_record(job_id: str) -> dict[str, object]:
        job = get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        with _open() as db:
            record = db.ensure_record(job_id, default_date=_job_date(job))
        return _record_payload(record, job)

    @router.patch("/archive/{job_id}")
    def patch_archive_record(job_id: str, payload: RecordUpdate) -> dict[str, object]:
        job = get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        updates = payload.model_dump(exclude_unset=True)
        with _open() as db:
            try:
                record = db.upsert_record(job_id, **updates)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except (sqlite3.Error, OSError) as exc:
                raise HTTPException(
                    status_code=500, detail=f"Не удалось сохранить запись архива: {exc}"
                ) from exc
        return _record_payload(record, job)


def _extract_entries(payload: Mapping[str, object]) -> list[tuple[int, str | None, str]]:
    """Реплики результата как ``(индекс, говорящий, текст)`` для индекса."""
    speakers: dict[str, str] = {}
    raw_speakers = payload.get("speakers")
    if isinstance(raw_speakers, list):
        for raw in raw_speakers:
            if isinstance(raw, Mapping) and isinstance(raw.get("id"), str):
                name = raw.get("display_name")
                speakers[raw["id"]] = name if isinstance(name, str) else raw["id"]
    entries: list[tuple[int, str | None, str]] = []
    raw_entries = payload.get("entries")
    if isinstance(raw_entries, list):
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, Mapping):
                continue
            text = raw.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            speaker_id = raw.get("speaker_id")
            speaker = speakers.get(speaker_id) if isinstance(speaker_id, str) else None
            entries.append((index, speaker, text))
    return entries


def _job_date(job: Job) -> str | None:
    """Дата по умолчанию для записи архива.

    Сначала дата файла-источника (обычно совпадает со встречей), затем дата
    завершения задачи, затем дата постановки.
    """
    try:
        mtime = Path(job.source_path).stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=UTC).date().isoformat()
    except OSError:
        pass
    for raw in (job.finished_at, job.created_at):
        if not raw:
            continue
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            continue
        return moment.date().isoformat()
    return None


def _record_payload(record: ArchiveRecord, job: Job | None) -> dict[str, object]:
    """Запись архива с краткой сводкой задачи для списка/карточки."""
    payload = record.as_dict()
    if job is not None:
        payload.update(
            {
                "name": job.name,
                "source_path": job.source_path,
                "status": job.status,
                "language": job.language,
                "duration": job.duration,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
            }
        )
    return payload


def _hit_payload(
    hit: SearchHit, job: Job | None, record: ArchiveRecord | None
) -> dict[str, object]:
    """Результат поиска с заголовком/датой задачи для перехода и подписи."""
    payload = hit.as_dict()
    if job is not None:
        payload["name"] = job.name
        payload["status"] = job.status
    if record is not None:
        payload["title"] = record.title
        payload["meeting_date"] = record.meeting_date
    return payload


def _filter_by_date(
    records: Sequence[ArchiveRecord], date_from: str | None, date_to: str | None
) -> list[ArchiveRecord]:
    """Оставляет записи, чья дата попадает в диапазон (границы включительно)."""
    start = _parse_bound(date_from)
    end = _parse_bound(date_to)
    if start is None and end is None:
        return list(records)
    result: list[ArchiveRecord] = []
    for record in records:
        value = _parse_bound(record.meeting_date)
        if value is None:
            continue
        if start is not None and value < start:
            continue
        if end is not None and value > end:
            continue
        result.append(record)
    return result


def _job_ids_in_range(
    records: Sequence[ArchiveRecord], date_from: str | None, date_to: str | None
) -> list[str] | None:
    """id задач в диапазоне дат или ``None`` — фильтр по дате не задан."""
    if _parse_bound(date_from) is None and _parse_bound(date_to) is None:
        return None
    return [record.job_id for record in _filter_by_date(records, date_from, date_to)]


def _parse_bound(value: str | None) -> str | None:
    """Нормализует границу диапазона дат; ``None`` при пустом/некорректном."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return None
