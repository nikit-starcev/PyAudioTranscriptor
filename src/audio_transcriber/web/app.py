"""FastAPI-приложение локального веб-интерфейса.

Сервер поднимается только на ``127.0.0.1`` (см. ``audio-transcriber web``) и
отдаёт:

* REST API под ``/api`` — файлы, очередь задач, прогресс (SSE), результат,
  образцы голоса и исходное аудио с поддержкой Range;
* собранное SPA из ``web/static`` (или страницу-заглушку, если фронт не собран).

CORS не нужен: фронт и API раздаёт один и тот же origin.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import uuid
import webbrowser
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import av
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel

from audio_transcriber import __version__
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.utils.text import sanitize_filename
from audio_transcriber.web.config import build_job_config, public_config
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.paths import STATIC_DIR, WebPaths
from audio_transcriber.web.results import load_result_file, result_summary
from audio_transcriber.web.runner import ConfigBuilder, JobRunner, PipelineFn
from audio_transcriber.web.storage.jobs_db import (
    STATUS_DONE,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
    JobsDB,
)

#: Заголовки SSE: без кэша и без буферизации прокси.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: Аудио-расширения, показываемые в списке файлов.
MEDIA_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".amr",
    ".mp2",
    ".mpga",
    ".wv",
    ".mp4",
    ".m4v",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".ts",
    ".3gp",
}

_PLACEHOLDER_HTML = """<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>AudioTranscriber — веб-интерфейс</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 44rem; margin: 4rem auto; line-height: 1.5">
  <h1>Фронтенд не собран</h1>
  <p>Сервер запущен, API доступен по адресу <code>/api/health</code>.</p>
  <p>Чтобы собрать веб-интерфейс, выполните в каталоге проекта:</p>
  <pre style="background:#f4f4f4;padding:1rem;border-radius:6px">cd webui
npm install
npm run build</pre>
  <p>Готовые файлы попадут в <code>src/audio_transcriber/web/static/</code>.</p>
</body>
</html>
"""


class CreateJobRequest(BaseModel):
    """Тело ``POST /api/jobs``."""

    path: str


def create_app(
    *,
    paths: WebPaths | None = None,
    pipeline_fn: PipelineFn | None = None,
    config_builder: ConfigBuilder | None = None,
    heartbeat: float = 15.0,
) -> FastAPI:
    """Собирает приложение FastAPI с изолированным окружением данных.

    ``pipeline_fn`` и ``config_builder`` подменяются в тестах, чтобы не
    требовать GPU/моделей и реального ``config.env``.
    """
    resolved_paths = paths or WebPaths.default()
    resolved_paths.ensure()
    store = JobsDB(resolved_paths.jobs_db)
    store.initialize()
    bus = JobEventBus(heartbeat=heartbeat)

    def default_config_builder(job_id: str, source_path: Path) -> AppConfig:
        return build_job_config(
            source_path,
            output_dir=resolved_paths.results_dir / job_id,
            data_dir=resolved_paths.data_dir,
        )

    runner = JobRunner(
        store,
        bus,
        resolved_paths,
        config_builder or default_config_builder,
        pipeline_fn=pipeline_fn,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        title="AudioTranscriber Web",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.paths = resolved_paths
    app.state.store = store
    app.state.bus = bus
    app.state.runner = runner
    router = APIRouter(prefix="/api")
    register_api(router, store=store, bus=bus, runner=runner, paths=resolved_paths)
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        return _index_response()

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> Response:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Не найдено")
        candidate = STATIC_DIR / full_path
        try:
            if candidate.is_file():
                return FileResponse(candidate)
        except OSError:
            pass
        return _index_response()

    return app


def register_api(
    router: APIRouter,
    *,
    store: JobsDB,
    bus: JobEventBus,
    runner: JobRunner,
    paths: WebPaths,
) -> None:
    """Регистрирует все маршруты API v1 на переданном роутере."""

    @router.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__}

    @router.get("/config")
    def get_config() -> dict[str, object]:
        return public_config(input_dir=paths.input_dir, output_dir=paths.results_dir).as_dict()

    @router.get("/files")
    def list_files() -> list[dict[str, object]]:
        return _list_files(paths.input_dir)

    @router.post("/files/upload", status_code=201)
    async def upload_file(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        return await _save_upload(file, paths.input_dir)

    @router.get("/jobs")
    def list_jobs() -> list[dict[str, object]]:
        return [job.as_dict() for job in store.list()]

    @router.post("/jobs", status_code=201)
    def create_job(payload: CreateJobRequest) -> dict[str, object]:
        source = _resolve_input_path(paths, payload.path)
        job = store.create(uuid.uuid4().hex, source)
        return job.as_dict()

    @router.get("/jobs/{job_id}")
    def job_details(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        payload = job.as_dict()
        payload["summary"] = None
        result = load_result_file(_result_path(paths, job))
        if result is not None:
            payload["summary"] = result_summary(result)
        return payload

    @router.post("/jobs/{job_id}/run")
    def run_job(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        if job.status == STATUS_RUNNING:
            raise HTTPException(status_code=409, detail="Задача уже выполняется")
        source = Path(job.source_path)
        if not source.is_file():
            raise HTTPException(status_code=400, detail="Исходный файл не найден")
        store.update(job_id, status=STATUS_QUEUED, stage=STATUS_QUEUED, fraction=0.0, error=None)
        if not runner.submit(job_id, source):
            raise HTTPException(status_code=409, detail="Задача уже в очереди")
        updated = store.get(job_id)
        return updated.as_dict() if updated is not None else job.as_dict()

    @router.delete("/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        if job.status == STATUS_RUNNING:
            raise HTTPException(status_code=409, detail="Нельзя удалить выполняющуюся задачу")
        _remove_job_artifacts(paths, job)
        store.delete(job_id)
        bus.clear(job_id)
        return {"deleted": job_id}

    @router.get("/jobs/{job_id}/result")
    def job_result(job_id: str) -> Response:
        job = _require_job(store, job_id)
        result = load_result_file(_result_path(paths, job))
        if result is None:
            raise HTTPException(status_code=404, detail="Результат ещё не готов")
        return JSONResponse(result)

    @router.get("/jobs/{job_id}/samples/{speaker_id}")
    def job_sample(job_id: str, speaker_id: str) -> Response:
        job = _require_job(store, job_id)
        result = load_result_file(_result_path(paths, job))
        if result is None:
            raise HTTPException(status_code=404, detail="Результат ещё не готов")
        samples = result.get("samples")
        relative = samples.get(speaker_id) if isinstance(samples, dict) else None
        if not isinstance(relative, str) or not relative:
            raise HTTPException(status_code=404, detail="Образец не найден")
        sample_path = (paths.data_dir / relative).resolve()
        if not sample_path.is_relative_to(paths.data_dir.resolve()) or not sample_path.is_file():
            raise HTTPException(status_code=404, detail="Образец не найден")
        return FileResponse(sample_path, media_type="audio/wav", filename=sample_path.name)

    @router.get("/jobs/{job_id}/audio")
    def job_audio(job_id: str, request: Request) -> Response:
        job = _require_job(store, job_id)
        source = Path(job.source_path)
        if not source.is_file():
            raise HTTPException(status_code=404, detail="Исходное аудио не найдено")
        media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        return _range_response(source, request.headers.get("range"), media_type=media_type)

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = _require_job(store, job_id)
        if job.is_terminal:
            event = {
                "stage": job.stage or job.status,
                "fraction": job.fraction,
                "message": _terminal_message(job),
                "status": job.status,
            }

            async def immediate() -> AsyncIterator[str]:
                yield _sse(event)

            return StreamingResponse(
                immediate(), media_type="text/event-stream", headers=dict(SSE_HEADERS)
            )

        async def stream() -> AsyncIterator[str]:
            yield _sse(
                {
                    "stage": job.stage or STATUS_QUEUED,
                    "fraction": job.fraction,
                    "message": "Подключено",
                    "status": job.status,
                }
            )
            async for event in bus.subscribe(job_id):
                yield ": ping\n\n" if event is None else _sse(event)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=dict(SSE_HEADERS)
        )


def _terminal_message(job: Job) -> str:
    if job.status == STATUS_DONE:
        return "Готово"
    return job.error or "Обработка завершена"


def _sse(event: Mapping[str, object]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _index_response() -> Response:
    index_path = STATIC_DIR / "index.html"
    try:
        if index_path.is_file():
            return FileResponse(index_path, media_type="text/html")
    except OSError:
        pass
    return HTMLResponse(_PLACEHOLDER_HTML)


def _list_files(input_dir: Path) -> list[dict[str, object]]:
    if not input_dir.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(input_dir.iterdir(), key=lambda item: item.name.casefold()):
        try:
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "duration": _probe_duration(path),
                }
            )
        except OSError:
            continue
    return items


async def _save_upload(file: UploadFile, input_dir: Path) -> dict[str, object]:
    original = Path(file.filename or "audio")
    safe_stem = sanitize_filename(original.stem, fallback="audio")
    suffix = original.suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        suffix = ""
    target = _unique_path(input_dir, f"{safe_stem}{suffix}")
    data = await file.read()
    try:
        target.write_bytes(data)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить файл: {exc}") from exc
    return {
        "name": target.name,
        "path": str(target),
        "size": target.stat().st_size,
        "duration": None,
    }


def _unique_path(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    path = Path(name)
    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _resolve_input_path(paths: WebPaths, raw: str) -> Path:
    value = raw.strip()
    if not value:
        raise HTTPException(status_code=400, detail="Не указан путь к файлу")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = paths.input_dir / candidate
    try:
        candidate = candidate.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Некорректный путь: {exc}") from exc
    if not candidate.is_relative_to(paths.input_dir.resolve()):
        raise HTTPException(
            status_code=400, detail="Файл должен находиться в каталоге загрузок"
        )
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return candidate


def _require_job(store: JobsDB, job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


def _result_path(paths: WebPaths, job: Job) -> Path:
    """Путь к JSON-результату задачи (только внутри каталога результатов)."""
    path = Path(job.result_path) if job.result_path else paths.results_dir / f"{job.id}.json"
    return path


def _remove_job_artifacts(paths: WebPaths, job: Job) -> None:
    """Удаляет JSON-результат и постадийный вывод задачи (безопасно по путям)."""
    results_root = paths.results_dir.resolve()
    for candidate in (_result_path(paths, job), paths.results_dir / job.id):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(results_root):
            continue
        if resolved.is_file():
            resolved.unlink(missing_ok=True)
        elif resolved.is_dir():
            for child in sorted(resolved.rglob("*"), reverse=True):
                try:
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                except OSError:
                    continue
            try:
                resolved.rmdir()
            except OSError:
                continue


def _probe_duration(path: Path) -> float | None:
    """Длительность аудио из контейнера без декодирования (``None`` при ошибке)."""
    try:
        with av.open(str(path)) as container:
            if container.duration is None:
                return None
            return round(container.duration / av.time_base, 2)
    except Exception:  # noqa: BLE001 — длительность не критична для списка
        return None


def _range_response(path: Path, range_header: str | None, *, media_type: str) -> Response:
    """Ответ с поддержкой HTTP Range для прослушивания исходного аудио."""
    size = path.stat().st_size
    start, end = 0, max(size - 1, 0)
    status = 200
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(size)}
    if range_header and range_header.startswith("bytes=") and size > 0:
        spec = range_header[len("bytes=") :].split(",", 1)[0].strip()
        first, _, last = spec.partition("-")
        resolved = _parse_range(first, last, size)
        if resolved is None:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
            )
        start, end = resolved
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
    length = end - start + 1 if size > 0 else 0
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(length)
    return Response(content=data, status_code=status, media_type=media_type, headers=headers)


def _parse_range(first: str, last: str, size: int) -> tuple[int, int] | None:
    """Разбирает ``bytes=first-last``; ``None`` — недопустимый диапазон."""
    try:
        if first:
            start = int(first)
            end = int(last) if last else size - 1
        elif last:
            start = max(0, size - int(last))
            end = size - 1
        else:
            return None
    except ValueError:
        return None
    start = max(0, start)
    end = min(end, size - 1)
    if start > end or start >= size:
        return None
    return start, end


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    reload: bool = False,
) -> None:
    """Поднимает uvicorn; по умолчанию открывает браузер на ``http://host:port/``."""
    import uvicorn

    if reload:
        uvicorn.run(
            "audio_transcriber.web.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    if open_browser:
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{host}:{port}/")
        ).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
