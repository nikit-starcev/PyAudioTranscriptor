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
import logging
import mimetypes
import shutil
import tempfile
import threading
import uuid
import webbrowser
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from audio_transcriber import __version__
from audio_transcriber.config.defaults import DEFAULT_ENROLLMENT_MIN_SIMILARITY
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.samples import (
    normalize_sample,
    select_sample_variants,
    slice_waveform,
)
from audio_transcriber.diarization.voices import (
    base_sample_name,
    collect_voice_library,
    delete_voice_sample,
    delete_voice_samples,
    save_speaker_sample,
    unique_sample_path,
)
from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.export.factory import create_exporter
from audio_transcriber.protocol import ProtocolArtifacts, generate_protocol
from audio_transcriber.utils.audio import load_waveform, write_wav
from audio_transcriber.utils.playback import amplitude_envelope, read_duration
from audio_transcriber.utils.text import sanitize_filename
from audio_transcriber.web.config import build_job_config, env_defaults, public_config
from audio_transcriber.web.doctor_api import (
    build_doctor_env,
    check_hf_access,
    doctor_report,
)
from audio_transcriber.web.estimates import StageEstimator, probe_duration
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.glossary_api import register_glossary_routes
from audio_transcriber.web.models import (
    MODEL_CATALOG,
    DownloadBus,
    Downloader,
    HfDownloader,
    ModelDownloadManager,
    ModelEntry,
    delete_model_files,
    find_model,
    free_space,
    local_status,
    model_payload,
    resolve_target,
)
from audio_transcriber.web.paths import STATIC_DIR, WebPaths
from audio_transcriber.web.processed import clear_processed, is_processed
from audio_transcriber.web.results import load_result_file, result_summary
from audio_transcriber.web.runner import ConfigBuilder, JobRunner, PipelineFn
from audio_transcriber.web.secrets import (
    SecretsError,
    SecretsStore,
    effective_hf_token,
    mask_hf_token,
)
from audio_transcriber.web.settings import (
    SettingsError,
    SettingsStore,
    settings_from_mapping,
    validate_settings,
)
from audio_transcriber.web.setup import build_setup_steps
from audio_transcriber.web.speakers import (
    apply_names,
    apply_speaker_changes,
    build_speaker_segments,
    result_from_payload,
)
from audio_transcriber.web.storage.jobs_db import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
    JobsDB,
)
from audio_transcriber.web.voices import (
    VoiceSample,
    find_voice_sample,
    find_voice_sample_file,
    list_voice_groups,
)

#: Функция формирования протокола (совместима с ``generate_protocol``).
ProtocolFn = Callable[..., ProtocolArtifacts]
#: Провайдер каталога библиотеки голосов (читает актуальные настройки).
VoicesResolver = Callable[[], Path]

logger = logging.getLogger(__name__)

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

#: MIME-типы прямой выгрузки стенограммы по форматам экспорта.
_EXPORT_MEDIA_TYPES: dict[str, str] = {
    "txt": "text/plain; charset=utf-8",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "json": "application/json",
    "srt": "application/x-subrip",
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
    """Тело ``POST /api/jobs``.

    ``num_speakers`` — необязательное ожидаемое число говорящих; ``None``
    (по умолчанию) — автоопределение диаризатором. ``min_speakers``/
    ``max_speakers`` задают диапазон (``None`` — без ограничения); при
    заданном ``num_speakers`` диапазон игнорируется.
    """

    path: str
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)


class UpdateJobRequest(BaseModel):
    """Тело ``PATCH /api/jobs/{id}``: правка числа говорящих до/после запуска."""

    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)


class MergeSpec(BaseModel):
    """Одно объединение говорящих: ``source`` сливается в ``target``."""

    source: str
    target: str


class SpeakerEditsRequest(BaseModel):
    """Тело ``PATCH /api/jobs/{id}/speakers``."""

    renames: dict[str, str] = Field(default_factory=dict)
    merges: list[MergeSpec] = Field(default_factory=list)


class ToLibraryRequest(BaseModel):
    """Тело ``POST /api/jobs/{id}/speakers/{sid}/to-library``.

    ``start``/``end`` — необязательное окно исходного аудио: если заданы, в
    библиотеку сохраняется именно выбранный вариант говорящего (#25), а не
    готовый образец задачи.
    """

    name: str
    start: float | None = None
    end: float | None = None


class ApplyNamesRequest(BaseModel):
    """Тело ``POST /api/jobs/{id}/apply-names``."""

    min_similarity: float | None = None
    references: dict[str, str] = Field(default_factory=dict)


class SettingsUpdate(BaseModel):
    """Тело ``PUT /api/settings``: частичное обновление (``None`` — не менять).

    ``hf_token`` — отдельный секрет: он сохраняется не в ``settings.json``, а в
    ``web-data/secrets.json`` с правами ``0600``. Пустая строка удаляет токен.
    """

    glossary_enabled: bool | None = None
    glossary_db: str | None = None
    voices_dir: str | None = None
    export_formats: list[str] | None = None
    llm_enabled: bool | None = None
    llm_summary: bool | None = None
    denoise: bool | None = None
    mark_overlap: bool | None = None
    normalize_text: bool | None = None
    clean_artifacts: bool | None = None
    protocol_auto: bool | None = None
    hf_token: str | None = None
    asr_backend: str | None = None
    device: str | None = None
    whisper_cpp_model: str | None = None
    whisper_cpp_binary: str | None = None
    llm_model: str | None = None
    llm_binary: str | None = None
    pyannote_local_model: str | None = None


class HfCheckRequest(BaseModel):
    """Тело ``POST /api/doctor/hf-check``: необязательный токен для проверки.

    Если токен не передан, проверяется сохранённый (или из ``config.env``).
    """

    token: str | None = None


def create_app(
    *,
    paths: WebPaths | None = None,
    pipeline_fn: PipelineFn | None = None,
    config_builder: ConfigBuilder | None = None,
    protocol_fn: ProtocolFn | None = None,
    voices_dir: Path | None = None,
    downloader: Downloader | None = None,
    heartbeat: float = 15.0,
) -> FastAPI:
    """Собирает приложение FastAPI с изолированным окружением данных.

    ``pipeline_fn``, ``config_builder``, ``protocol_fn`` и ``downloader``
    подменяются в тестах, чтобы не требовать GPU/моделей, сети и реального
    ``config.env``. ``voices_dir`` позволяет подменить каталог библиотеки
    голосов (в тестах) вместо ``VOICES_DIR``; в остальных случаях он берётся
    из сохранённых настроек.
    """
    resolved_paths = paths or WebPaths.default()
    resolved_paths.ensure()
    settings_store = SettingsStore(resolved_paths.settings_json)
    secrets_store = SecretsStore(resolved_paths.secrets_json)
    store = JobsDB(resolved_paths.jobs_db)
    store.initialize()
    bus = JobEventBus(heartbeat=heartbeat)
    download_bus = DownloadBus(heartbeat=heartbeat)
    # Оценки прогресса/ETA/здоровья по истории завершённых задач (#15/#24).
    estimator = StageEstimator(store)

    def resolve_model_target(entry: ModelEntry) -> Path:
        return resolve_target(
            entry,
            models_root=resolved_paths.models_dir,
            settings=settings_store.load(),
        )

    downloads = ModelDownloadManager(
        models_root=resolved_paths.models_dir,
        downloader=downloader or HfDownloader(),
        resolve_target=resolve_model_target,
        bus=download_bus,
    )

    def resolve_voices() -> Path:
        if voices_dir is not None:
            return Path(voices_dir)
        return settings_store.load().resolved_voices_dir()

    def default_config_builder(job_id: str, source_path: Path) -> AppConfig:
        overrides = settings_store.load().env_overrides()
        token = effective_hf_token(secrets_store, env_defaults())
        if token:
            overrides["HF_TOKEN"] = token
        return build_job_config(
            source_path,
            output_dir=resolved_paths.results_dir / job_id,
            data_dir=resolved_paths.data_dir,
            overrides=overrides,
        )

    effective_builder = config_builder or default_config_builder
    runner = JobRunner(
        store,
        bus,
        resolved_paths,
        effective_builder,
        pipeline_fn=pipeline_fn,
        estimator=estimator,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Подвешенные задачи прошлого процесса (running/queued вне активного
        # набора воркера) помечаем как error ещё до приёма запросов.
        runner.reconcile_orphans()
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
    app.state.estimator = estimator
    app.state.settings_store = settings_store
    app.state.secrets_store = secrets_store
    app.state.voices_dir = resolve_voices()
    app.state.downloads = downloads
    app.state.download_bus = download_bus
    app.state.models_dir = resolved_paths.models_dir
    router = APIRouter(prefix="/api")
    register_api(
        router,
        store=store,
        bus=bus,
        runner=runner,
        estimator=estimator,
        paths=resolved_paths,
        settings_store=settings_store,
        secrets_store=secrets_store,
        resolve_voices=resolve_voices,
        config_builder=effective_builder,
        protocol_fn=protocol_fn or generate_protocol,
        downloads=downloads,
        download_bus=download_bus,
        models_root=resolved_paths.models_dir,
    )
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
    estimator: StageEstimator,
    paths: WebPaths,
    settings_store: SettingsStore,
    secrets_store: SecretsStore,
    resolve_voices: VoicesResolver,
    config_builder: ConfigBuilder,
    protocol_fn: ProtocolFn,
    downloads: ModelDownloadManager,
    download_bus: DownloadBus,
    models_root: Path,
) -> None:
    """Регистрирует все маршруты API v1 на переданном роутере."""

    def _glossary_db_path() -> Path:
        return settings_store.load().resolved_glossary_db()

    def _effective_token() -> str | None:
        return effective_hf_token(secrets_store, env_defaults())

    def job_payload(job: Job) -> dict[str, object]:
        """Представление задачи с признаком «обрабатывается этим воркером».

        ``active=False`` у ``running``-задачи означает, что её никто не ведёт
        (осиротевшая): клиенту не стоит показывать «живой» прогресс и можно
        предлагать удаление/перезапуск. Сюда же добавляются оценки прогресса
        (``progress_percent``), ETA (``eta_seconds``/``eta_by_stage``) и
        «здоровье» задачи (``health``) — см. #15/#24.
        """
        payload = job.as_dict()
        active = runner.is_active(job.id)
        payload["active"] = active
        payload.update(estimator.snapshot(job, active=active))
        return payload

    register_glossary_routes(router, db_path=_glossary_db_path)

    @router.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__}

    @router.get("/config")
    def get_config() -> dict[str, object]:
        return public_config(input_dir=paths.input_dir, output_dir=paths.results_dir).as_dict()

    @router.get("/settings")
    def get_settings() -> dict[str, object]:
        settings = settings_store.load()
        payload = settings.as_dict()
        payload["input_dir"] = str(paths.input_dir)
        payload["output_dir"] = str(paths.results_dir)
        payload["glossary_db_path"] = str(settings.resolved_glossary_db())
        payload["voices_dir_resolved"] = str(resolve_voices())
        token = _effective_token()
        payload["hf_token_set"] = token is not None
        payload["hf_token_masked"] = mask_hf_token(token)
        return payload

    @router.put("/settings")
    def put_settings(payload: SettingsUpdate) -> dict[str, object]:
        current = settings_store.load()
        updates = payload.model_dump(exclude_none=True)
        token_requested = "hf_token" in updates
        raw_token = updates.pop("hf_token", None)
        try:
            merged = settings_from_mapping(updates, base=current)
            validate_settings(merged)
            saved = settings_store.save(merged)
            if token_requested:
                secrets_store.set_hf_token(raw_token)
        except (SettingsError, SecretsError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = saved.as_dict()
        result["input_dir"] = str(paths.input_dir)
        result["output_dir"] = str(paths.results_dir)
        result["glossary_db_path"] = str(saved.resolved_glossary_db())
        result["voices_dir_resolved"] = str(resolve_voices())
        token = _effective_token()
        result["hf_token_set"] = token is not None
        result["hf_token_masked"] = mask_hf_token(token)
        return result

    @router.get("/doctor")
    def get_doctor() -> dict[str, object]:
        """Отчёт о готовности (те же проверки, что у CLI ``doctor``)."""
        config_path, env = build_doctor_env(settings_store, secrets_store, paths)
        return doctor_report(config_path, env)

    @router.post("/doctor/recheck")
    def recheck_doctor() -> dict[str, object]:
        """Повторная диагностика готовности («Проверить снова»)."""
        config_path, env = build_doctor_env(settings_store, secrets_store, paths)
        return doctor_report(config_path, env)

    @router.post("/doctor/hf-check")
    def hf_check(payload: HfCheckRequest | None = None) -> dict[str, object]:
        """Мягкая проверка HF-токена: ``ok`` / ``no_access`` / ``no_token``.

        Если в теле передан токен, проверяется именно он (для проверки до
        сохранения); иначе — сохранённый секрет или значение из ``config.env``.
        """
        requested = payload.token if payload is not None else None
        token = requested.strip() if isinstance(requested, str) and requested.strip() else None
        if token is None:
            token = _effective_token()
        return check_hf_access(token).as_dict()

    @router.get("/setup")
    def get_setup() -> dict[str, object]:
        """План мастера первого запуска: шаги, варианты железа, нужные модели."""
        settings = settings_store.load()
        config_path, env = build_doctor_env(settings_store, secrets_store, paths)
        report = doctor_report(config_path, env)
        models = [
            model_payload(
                entry,
                models_root=models_root,
                settings=settings,
                state=downloads.state(entry.id),
            )
            for entry in MODEL_CATALOG
        ]
        return build_setup_steps(
            settings=settings,
            report=report,
            models=models,
            hf_token_set=_effective_token() is not None,
        )

    @router.get("/models")
    def list_models() -> dict[str, object]:
        """Каталог моделей с локальным статусом, прогрессом и свободным местом."""
        settings = settings_store.load()
        entries = [
            model_payload(
                entry,
                models_root=models_root,
                settings=settings,
                state=downloads.state(entry.id),
            )
            for entry in MODEL_CATALOG
        ]
        return {
            "models": entries,
            "disk": {"free": free_space(models_root), "models_dir": str(models_root)},
        }

    @router.get("/models/events")
    async def models_events() -> StreamingResponse:
        """SSE-поток прогресса скачивания (сначала история, затем живой поток)."""

        async def stream() -> AsyncIterator[str]:
            for event in download_bus.history():
                yield _sse(event)
            async for update in download_bus.subscribe():
                yield ": ping\n\n" if update is None else _sse(update)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=dict(SSE_HEADERS)
        )

    @router.post("/models/{model_id}/download", status_code=202)
    def start_model_download(model_id: str) -> dict[str, object]:
        """Запускает фоновое скачивание модели (прогресс — ``/api/models/events``)."""
        entry = find_model(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Модель не найдена")
        if downloads.is_running(model_id):
            raise HTTPException(status_code=409, detail="Модель уже скачивается")
        token = _effective_token()
        if entry.gated and not token:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Для этой модели нужен токен Hugging Face — получите его и "
                    "сохраните в настройках."
                ),
            )
        target = resolve_target(
            entry, models_root=models_root, settings=settings_store.load()
        )
        remaining = max(entry.approx_size - local_status(entry, target).size, 0)
        available = free_space(models_root)
        if available and remaining and available < remaining:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Недостаточно места на диске: нужно ещё ~{_format_size(remaining)}, "
                    f"свободно ~{_format_size(available)}."
                ),
            )
        if not downloads.start(entry, token=token):
            raise HTTPException(status_code=409, detail="Модель уже скачивается")
        return {"id": model_id, "status": "downloading", "fraction": 0.0}

    @router.post("/models/{model_id}/cancel")
    def cancel_model_download(model_id: str) -> dict[str, object]:
        """Просит прервать активное скачивание модели."""
        if find_model(model_id) is None:
            raise HTTPException(status_code=404, detail="Модель не найдена")
        if not downloads.cancel(model_id):
            raise HTTPException(status_code=409, detail="Модель не скачивается")
        return {"id": model_id, "status": "cancelled"}

    @router.delete("/models/{model_id}")
    def remove_model(model_id: str) -> dict[str, object]:
        """Удаляет локальные файлы модели (нельзя во время скачивания)."""
        entry = find_model(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Модель не найдена")
        if downloads.is_running(model_id):
            raise HTTPException(
                status_code=409, detail="Нельзя удалить модель во время скачивания"
            )
        target = resolve_target(
            entry, models_root=models_root, settings=settings_store.load()
        )
        removed = delete_model_files(entry, target)
        return {"deleted": model_id, "removed": removed}

    @router.get("/files")
    def list_files(include_processed: bool = False) -> list[dict[str, object]]:
        """Список загруженных файлов.

        По умолчанию обработанные файлы (issue #16) скрыты; ``include_processed=
        true`` показывает и их — с полем ``processed``, чтобы UI мог выделить
        группу «Обработанные» и предложить возврат.
        """
        return _list_files(paths.input_dir, include_processed=include_processed)

    @router.post("/files/upload", status_code=201)
    async def upload_file(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        return await _save_upload(file, paths.input_dir)

    @router.delete("/files/{name}")
    def delete_file(name: str) -> dict[str, object]:
        """Удаляет загруженный файл внутри каталога загрузок."""
        target = _resolve_upload_file(paths, name)
        for job in store.list():
            if job.status == STATUS_RUNNING and _same_file(
                Path(job.source_path), target
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Файл используется выполняющейся задачей",
                )
        try:
            target.unlink()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Файл не найден") from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось удалить файл: {exc}"
            ) from exc
        # Удаляем и sidecar-маркер, иначе он останется «висеть» для нового файла
        # с тем же именем (issue #16).
        clear_processed(target)
        return {"deleted": target.name}

    @router.post("/files/{name}/restore")
    def restore_file(name: str) -> dict[str, object]:
        """Снимает метку «обработан» — возвращает файл в основной список (#16).

        Идемпотентно: повторный вызов для невыделенного файла тоже успешен.
        """
        target = _resolve_upload_file(paths, name)
        clear_processed(target)
        return {"restored": target.name, "processed": False}

    @router.get("/jobs")
    def list_jobs() -> list[dict[str, object]]:
        return [job_payload(job) for job in store.list()]

    @router.post("/jobs", status_code=201)
    def create_job(payload: CreateJobRequest) -> dict[str, object]:
        source = _resolve_input_path(paths, payload.path)
        _validate_speaker_range(payload.min_speakers, payload.max_speakers)
        job = store.create(
            uuid.uuid4().hex,
            source,
            num_speakers=payload.num_speakers,
            min_speakers=payload.min_speakers,
            max_speakers=payload.max_speakers,
        )
        return job_payload(job)

    @router.patch("/jobs/{job_id}")
    def update_job(job_id: str, payload: UpdateJobRequest) -> dict[str, object]:
        """Меняет число говорящих задачи (``null`` — авто) до/после запуска."""
        job = _require_job(store, job_id)
        if job.status == STATUS_RUNNING and runner.is_active(job_id):
            raise HTTPException(status_code=409, detail="Задача уже выполняется")
        speech_fields = {"num_speakers", "min_speakers", "max_speakers"}
        provided = payload.model_fields_set & speech_fields
        if not provided:
            raise HTTPException(status_code=400, detail="Нет полей для обновления")
        # Диапазон проверяем по итоговым (слитым со строкой) значениям.
        effective_min = (
            payload.min_speakers if "min_speakers" in provided else job.min_speakers
        )
        effective_max = (
            payload.max_speakers if "max_speakers" in provided else job.max_speakers
        )
        _validate_speaker_range(effective_min, effective_max)
        updates = {field_name: getattr(payload, field_name) for field_name in provided}
        updated = store.update(job_id, **updates)
        return job_payload(updated) if updated is not None else job_payload(job)

    @router.get("/jobs/{job_id}")
    def job_details(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        payload = job_payload(job)
        payload["summary"] = None
        result = load_result_file(_result_path(paths, job))
        if result is not None:
            payload["summary"] = result_summary(result)
        return payload

    @router.post("/jobs/{job_id}/run")
    def run_job(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        if job.status == STATUS_RUNNING and runner.is_active(job_id):
            raise HTTPException(status_code=409, detail="Задача уже выполняется")
        source = Path(job.source_path)
        if not source.is_file():
            raise HTTPException(status_code=400, detail="Исходный файл не найден")
        store.update(
            job_id,
            status=STATUS_QUEUED,
            stage=STATUS_QUEUED,
            fraction=0.0,
            error=None,
            started_at=None,
            finished_at=None,
            stage_started_at=None,
            stage_times=[],
        )
        if not runner.submit(job_id, source):
            raise HTTPException(status_code=409, detail="Задача уже в очереди")
        updated = store.get(job_id)
        return job_payload(updated) if updated is not None else job_payload(job)

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, object]:
        """Останавливает активную задачу.

        Возвращает ``cancelled`` сразу (флаг отмены выставлен, дочерние
        процессы погашены); терминальный статус ``cancelled`` воркер проставит
        по завершении потока задачи. Для неактивной/уже завершённой задачи —
        409, чтобы клиент не считал отменённой то, что не выполняется.
        """
        _require_job(store, job_id)
        if not runner.is_active(job_id) or not runner.cancel(job_id):
            raise HTTPException(status_code=409, detail="Задача не выполняется")
        return {"id": job_id, "status": STATUS_CANCELLED}

    @router.delete("/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        # 409 — только для реально активного прогона. Осиротевшую ``running``-
        # задачу (воркер её не ведёт) удаляем вместе с артефактами.
        if job.status == STATUS_RUNNING and runner.is_active(job_id):
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

    @router.post("/jobs/{job_id}/protocol")
    def job_protocol(job_id: str) -> dict[str, object]:
        """Формирует протокол по текущему результату (резюме + экспорт).

        Синхронный вызов: FastAPI выполняет его в рабочем потоке, поэтому
        остальные запросы не блокируются. LLM-резюме считается по актуальной
        стенограмме — уже с применёнными именами говорящих.
        """
        job = _require_job(store, job_id)
        if job.status != STATUS_DONE:
            raise HTTPException(status_code=409, detail="Результат ещё не готов")
        source = Path(job.source_path)
        if not source.is_file():
            raise HTTPException(status_code=400, detail="Исходный файл не найден")
        payload = _require_result(paths, job)
        try:
            config = config_builder(job_id, source)
            result = result_from_payload(payload, source_path=source)
            artifacts = protocol_fn(config, result)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось сформировать протокол: {exc}"
            ) from exc
        updated = _store_protocol(paths, job, payload, artifacts)
        return {
            "paths": [str(path) for path in artifacts.paths],
            "summary": artifacts.summary,
            "protocol": updated,
        }

    @router.get("/jobs/{job_id}/summary")
    def job_summary(job_id: str) -> dict[str, object]:
        job = _require_job(store, job_id)
        payload = _require_result(paths, job)
        summary = payload.get("summary")
        return {"summary": summary if isinstance(summary, str) else None}

    @router.get("/jobs/{job_id}/protocol/download")
    def protocol_download(job_id: str, fmt: str = "txt") -> Response:
        job = _require_job(store, job_id)
        value = fmt.strip().casefold()
        if value not in {"txt", "docx"}:
            raise HTTPException(status_code=400, detail="fmt должен быть txt или docx")
        target = _protocol_file(paths, job, value)
        if target is None:
            raise HTTPException(status_code=404, detail="Протокол ещё не сформирован")
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if value == "docx"
            else "text/plain"
        )
        return FileResponse(target, media_type=media_type, filename=target.name)

    @router.get("/jobs/{job_id}/export")
    def job_export(job_id: str, fmt: str = "txt") -> Response:
        """Прямая выгрузка стенограммы в выбранном формате (без протокола).

        Переиспользует текущий JSON результата (уже с применёнными именами
        говорящих) и существующие экспортёры ``export/*``. Резюме **не**
        вычисляется: отдаётся именно стенограмма, а не протокол. Файл
        собирается во временном каталоге — результаты задачи не перезаписываются.
        """
        job = _require_job(store, job_id)
        value = fmt.strip().casefold()
        try:
            export_format = ExportFormat(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="fmt должен быть txt, docx, json или srt"
            ) from exc
        payload = _require_result(paths, job)
        source = Path(job.source_path)
        result = result_from_payload(payload, source_path=source)
        # Это выгрузка стенограммы, а не протокол: резюме не добавляем.
        result.summary = None
        tmp_dir = Path(tempfile.mkdtemp(prefix="audio-transcriber-export-"))
        target = tmp_dir / f"{source.stem}.{value}"
        try:
            create_exporter(export_format).export(result, target)
        except Exception as exc:  # сбой экспорта не должен ронять сервер
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500, detail=f"Не удалось экспортировать стенограмму: {exc}"
            ) from exc
        return FileResponse(
            target,
            media_type=_EXPORT_MEDIA_TYPES[value],
            filename=target.name,
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )

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

    @router.get("/jobs/{job_id}/samples")
    def job_samples(job_id: str) -> list[dict[str, object]]:
        """Метаданные образцов голоса говорящих задачи (длительность, размер)."""
        job = _require_job(store, job_id)
        result = load_result_file(_result_path(paths, job))
        if result is None:
            raise HTTPException(status_code=404, detail="Результат ещё не готов")
        items: list[dict[str, object]] = []
        for speaker_id, relative in _result_samples(result).items():
            sample_path = _sample_path(paths, relative)
            if sample_path is None or not sample_path.is_file():
                items.append({"speaker_id": speaker_id, "duration": 0.0, "size": 0})
                continue
            items.append(
                {
                    "speaker_id": speaker_id,
                    "duration": round(read_duration(sample_path), 2),
                    "size": sample_path.stat().st_size,
                }
            )
        return items

    @router.patch("/jobs/{job_id}/speakers")
    def edit_speakers(job_id: str, payload: SpeakerEditsRequest) -> Response:
        """Переименовать/объединить говорящих и перезаписать JSON результата."""
        job = _require_job(store, job_id)
        result = _require_result(paths, job)
        renames = {
            speaker_id: name.strip()
            for speaker_id, name in payload.renames.items()
            if speaker_id and name.strip()
        }
        merges = [
            (merge.source, merge.target)
            for merge in payload.merges
            if merge.source and merge.target and merge.source != merge.target
        ]
        try:
            new_result = apply_speaker_changes(
                result,
                source_path=Path(job.source_path),
                renames=renames,
                merges=merges,
                samples=_result_samples(result),
                data_dir=paths.data_dir,
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Не удалось применить правки: {exc}") from exc
        _write_result(paths, job, new_result)
        return JSONResponse(new_result)

    @router.post("/jobs/{job_id}/apply-names")
    def apply_names_route(job_id: str, payload: ApplyNamesRequest | None = None) -> Response:
        """Сопоставить говорящих с именами по образцам (библиотека + явные)."""
        job = _require_job(store, job_id)
        result = _require_result(paths, job)
        threshold = (
            payload.min_similarity
            if payload is not None and payload.min_similarity is not None
            else DEFAULT_ENROLLMENT_MIN_SIMILARITY
        )
        source = Path(job.source_path)
        explicit = _explicit_references(payload.references if payload is not None else {})
        library = collect_voice_library(resolve_voices())
        references_total = len(explicit) + len(library)
        segments = build_speaker_segments(result)
        if not source.is_file():
            return _apply_names_error(result, threshold, "Исходное аудио не найдено")
        if references_total == 0:
            return _apply_names_error(
                result, threshold, "Нет образцов голоса: библиотека пуста и явные не заданы"
            )
        if not segments:
            return _apply_names_error(
                result, threshold, "Нет сегментов говорящих для сопоставления"
            )
        try:
            updated, outcome = apply_names(
                result,
                source_path=source,
                explicit_references=explicit,
                library_references=library,
                samples=_result_samples(result),
                data_dir=paths.data_dir,
                min_similarity=threshold,
                local_model_path=_local_model_path(),
            )
        except Exception as exc:  # noqa: BLE001 — модель/аудио недоступны: мягкая деградация
            return _apply_names_error(result, threshold, f"Сопоставление недоступно: {exc}")
        if outcome.mapping:
            _write_result(paths, job, updated)
        best = {
            speaker_id: {"name": name, "score": round(float(score), 3)}
            for speaker_id, (name, score) in outcome.best_candidates.items()
        }
        return JSONResponse(
            {
                "result": updated,
                "matched": dict(outcome.mapping),
                "best_candidates": best,
                "threshold": threshold,
                "error": None,
            }
        )

    @router.get("/jobs/{job_id}/speakers/{speaker_id}/variants")
    def speaker_variants(job_id: str, speaker_id: str, count: int = 5) -> dict[str, object]:
        """Варианты прослушивания говорящего — неперекрывающиеся окна аудио.

        Окна ``[start, end)`` исходного аудио, отсортированные по убыванию
        «полезности» (энергия речи; без аудио — длительность чистых реплик).
        Звук отдаётся существующим ``GET /jobs/{id}/audio`` с Range, поэтому
        фронтенд проигрывает окно, выставляя ``currentTime = start``.
        """
        job = _require_job(store, job_id)
        payload = _require_result(paths, job)
        source = Path(job.source_path)
        result = result_from_payload(payload, source_path=source)
        if speaker_id not in {speaker.id for speaker in result.speakers}:
            raise HTTPException(status_code=404, detail="Говорящий не найден")
        limit = max(1, min(count, 20))
        waveform = None
        if source.is_file():
            try:
                waveform = load_waveform(source)
            except Exception as exc:  # noqa: BLE001 — без аудио деградируем к репликам
                logger.warning(
                    "Варианты говорящего: не удалось прочитать аудио %s: %s", source, exc
                )
        variants = select_sample_variants(
            result.entries, speaker_id, waveform=waveform, count=limit
        )
        return {"speaker_id": speaker_id, "variants": [variant.as_dict() for variant in variants]}

    @router.post("/jobs/{job_id}/speakers/{speaker_id}/to-library", status_code=201)
    def speaker_to_library(
        job_id: str, speaker_id: str, payload: ToLibraryRequest
    ) -> dict[str, object]:
        """Сохраняет образец говорящего в библиотеку голосов под именем.

        Без окна копируется готовый образец задачи; при заданных ``start``/``end``
        в библиотеку вырезается выбранный вариант прослушивания (#25).
        """
        job = _require_job(store, job_id)
        result = _require_result(paths, job)
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Не указано имя образца")
        directory = resolve_voices()
        if payload.start is not None or payload.end is not None:
            window = _variant_window(payload.start, payload.end)
            source = Path(job.source_path)
            if not source.is_file():
                raise HTTPException(status_code=404, detail="Исходное аудио не найдено")
            try:
                target = _save_window_to_library(source, window, directory, name)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Не удалось сохранить образец: {exc}"
                ) from exc
            return VoiceSample(name=base_sample_name(target.stem), path=target).as_dict()

        relative = _result_samples(result).get(speaker_id)
        sample_path = _sample_path(paths, relative) if relative else None
        if sample_path is None or not sample_path.is_file():
            raise HTTPException(status_code=404, detail="Образец говорящего не найден")
        try:
            target = save_speaker_sample(sample_path, directory, name)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось сохранить образец: {exc}"
            ) from exc
        return VoiceSample(name=base_sample_name(target.stem), path=target).as_dict()

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
            event: dict[str, object] = {
                "stage": job.stage or job.status,
                "fraction": job.fraction,
                "message": _terminal_message(job),
                "status": job.status,
                "stage_times": [timing.as_dict() for timing in job.stage_times],
            }
            event.update(estimator.snapshot(job, active=False))

            async def immediate() -> AsyncIterator[str]:
                yield _sse(event)

            return StreamingResponse(
                immediate(), media_type="text/event-stream", headers=dict(SSE_HEADERS)
            )

        async def stream() -> AsyncIterator[str]:
            active = runner.is_active(job_id)
            initial: dict[str, object] = {
                "stage": job.stage or STATUS_QUEUED,
                "fraction": job.fraction,
                "message": "Подключено",
                "status": job.status,
                "active": active,
                "elapsed": job.total_seconds if active else None,
                "stage_elapsed": job.stage_elapsed if active else None,
                "stage_times": [timing.as_dict() for timing in job.stage_times],
            }
            initial.update(estimator.snapshot(job, active=active))
            yield _sse(initial)
            async for event in bus.subscribe(job_id):
                yield ": ping\n\n" if event is None else _sse(event)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers=dict(SSE_HEADERS)
        )

    @router.get("/voices")
    def voices_list() -> list[dict[str, object]]:
        """Библиотека, сгруппированная по человеку: имя → список образцов."""
        return [group.as_dict() for group in list_voice_groups(resolve_voices())]

    @router.post("/voices", status_code=201)
    async def voices_upload(
        file: Annotated[UploadFile, File()],
        name: Annotated[str, Form()],
    ) -> dict[str, object]:
        """Добавить образец библиотеки: ``<имя>.wav`` или ``<имя> (N).wav``.

        Существующие образцы не перезаписываются — новый файл получает свободный
        номер в группе имени.
        """
        raw_name = name.strip()
        clean = sanitize_filename(raw_name)
        if not raw_name or not clean:
            raise HTTPException(status_code=400, detail="Не указано имя образца")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Пустой файл")
        directory = resolve_voices()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=500, detail=f"Не удалось создать библиотеку: {exc}"
            ) from exc
        target = unique_sample_path(directory, raw_name)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix == ".wav":
            try:
                target.write_bytes(data)
            except OSError as exc:
                raise HTTPException(
                    status_code=500, detail=f"Не удалось сохранить образец: {exc}"
                ) from exc
        else:
            _write_converted_wav(data, suffix, target)
        return VoiceSample(name=base_sample_name(target.stem), path=target).as_dict()

    @router.get("/voices/samples/{filename}/audio")
    def voice_sample_audio(filename: str) -> Response:
        """Звук конкретного образца по имени файла (поддерживает дубликаты)."""
        sample = find_voice_sample_file(resolve_voices(), filename)
        if sample is None:
            raise HTTPException(status_code=404, detail="Образец не найден")
        return FileResponse(sample.path, media_type="audio/wav", filename=sample.path.name)

    @router.get("/voices/samples/{filename}/envelope")
    def voice_sample_envelope(filename: str, columns: int = 120) -> dict[str, object]:
        sample = find_voice_sample_file(resolve_voices(), filename)
        if sample is None:
            raise HTTPException(status_code=404, detail="Образец не найден")
        width = max(1, min(columns, 2000))
        return {
            "name": sample.name,
            "filename": sample.filename,
            "duration": round(read_duration(sample.path), 2),
            "columns": width,
            "envelope": amplitude_envelope(sample.path, width),
        }

    @router.delete("/voices/samples/{filename}")
    def voice_sample_delete(filename: str) -> dict[str, object]:
        sample = find_voice_sample_file(resolve_voices(), filename)
        if sample is None:
            raise HTTPException(status_code=404, detail="Образец не найден")
        if not delete_voice_sample(sample.path, resolve_voices()):
            raise HTTPException(status_code=404, detail="Образец не найден")
        return {"deleted": sample.filename}

    @router.delete("/voices/people/{name}")
    def voice_person_delete(name: str) -> dict[str, object]:
        """Удаляет **все** образцы человека (всю группу имени)."""
        deleted = delete_voice_samples(resolve_voices(), name)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Образцы не найдены")
        return {"deleted": name, "count": deleted}

    @router.get("/voices/{name}/audio")
    def voice_audio(name: str) -> Response:
        sample = find_voice_sample(resolve_voices(), name)
        if sample is None:
            raise HTTPException(status_code=404, detail="Образец не найден")
        return FileResponse(sample.path, media_type="audio/wav", filename=sample.path.name)

    @router.get("/voices/{name}/envelope")
    def voice_envelope(name: str, columns: int = 120) -> dict[str, object]:
        sample = find_voice_sample(resolve_voices(), name)
        if sample is None:
            raise HTTPException(status_code=404, detail="Образец не найден")
        width = max(1, min(columns, 2000))
        return {
            "name": sample.name,
            "duration": round(read_duration(sample.path), 2),
            "columns": width,
            "envelope": amplitude_envelope(sample.path, width),
        }

    @router.delete("/voices/{name}")
    def voice_delete(name: str) -> dict[str, object]:
        sample = find_voice_sample(resolve_voices(), name)
        if sample is None:
            raise HTTPException(status_code=404, detail="Образец не найден")
        if not delete_voice_sample(sample.path, resolve_voices()):
            raise HTTPException(status_code=404, detail="Образец не найден")
        return {"deleted": sample.filename}


def _format_size(num_bytes: int) -> str:
    """Человекочитаемый размер (Б/КБ/МБ/ГБ) для текстов ошибок."""
    value = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024
    return f"{int(num_bytes)} Б"


def _terminal_message(job: Job) -> str:
    if job.status == STATUS_DONE:
        return "Готово"
    if job.status == STATUS_CANCELLED:
        return "Остановлено пользователем"
    return job.error or "Обработка завершена"


def _sse(event: Mapping[str, object]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _require_result(paths: WebPaths, job: Job) -> dict[str, object]:
    """Читает JSON-результат задачи или отвечает 404, если он ещё не готов."""
    result = load_result_file(_result_path(paths, job))
    if result is None:
        raise HTTPException(status_code=404, detail="Результат ещё не готов")
    return result


def _result_samples(result: Mapping[str, object]) -> dict[str, str]:
    """Отображение ``speaker_id -> относительный путь`` из JSON результата."""
    samples = result.get("samples")
    if not isinstance(samples, Mapping):
        return {}
    return {
        str(key): value
        for key, value in samples.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _sample_path(paths: WebPaths, relative: str) -> Path | None:
    """Разрешает относительный путь образца внутри каталога данных (без выхода)."""
    try:
        root = paths.data_dir.resolve()
        candidate = (root / relative).resolve()
    except OSError:
        return None
    return candidate if candidate.is_relative_to(root) else None


def _write_result(paths: WebPaths, job: Job, payload: Mapping[str, object]) -> None:
    """Перезаписывает JSON результата задачи (как в воркере — с отступами)."""
    path = _result_path(paths, job)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Не удалось сохранить результат: {exc}"
        ) from exc


def _store_protocol(
    paths: WebPaths,
    job: Job,
    payload: Mapping[str, object],
    artifacts: ProtocolArtifacts,
) -> dict[str, str]:
    """Сохраняет пути протокола и резюме в JSON результата; возвращает карту путей."""
    protocol_map: dict[str, str] = {}
    for path in artifacts.paths:
        suffix = path.suffix.lstrip(".").casefold()
        if suffix:
            protocol_map[suffix] = str(path)
    updated: dict[str, object] = dict(payload)
    if artifacts.summary is not None:
        updated["summary"] = artifacts.summary
    updated["protocol"] = protocol_map
    _write_result(paths, job, updated)
    return protocol_map


def _protocol_file(paths: WebPaths, job: Job, fmt: str) -> Path | None:
    """Путь к файлу протокола: из JSON результата, иначе — по умолчанию."""
    payload = load_result_file(_result_path(paths, job))
    candidates: list[Path] = []
    if payload is not None:
        protocol_map = payload.get("protocol")
        if isinstance(protocol_map, Mapping):
            raw = protocol_map.get(fmt)
            if isinstance(raw, str) and raw:
                candidates.append(Path(raw))
    source = Path(job.source_path)
    candidates.append(paths.results_dir / job.id / f"{source.stem}.{fmt}")
    root = paths.data_dir.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_relative_to(root) and resolved.is_file():
            return resolved
    return None


def _explicit_references(raw: Mapping[str, str]) -> dict[str, list[Path]]:
    """Существующие файлы из тела запроса: ``имя -> [путь]``."""
    references: dict[str, list[Path]] = {}
    for name, value in raw.items():
        clean = name.strip()
        if not clean or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if path.is_file():
            references.setdefault(clean, []).append(path)
    return references


def _local_model_path() -> Path | None:
    """Локальный каталог модели эмбеддингов из настроек (``PYANNOTE_LOCAL_MODEL``)."""
    raw = env_defaults().get("PYANNOTE_LOCAL_MODEL", "").strip()
    return Path(raw) if raw else None


def _apply_names_error(
    result: Mapping[str, object], threshold: float, message: str
) -> Response:
    """Мягкий ответ ``apply-names``: 200 без изменений и с текстом причины."""
    return JSONResponse(
        {
            "result": dict(result),
            "matched": {},
            "best_candidates": {},
            "threshold": threshold,
            "error": message,
        }
    )


def _write_converted_wav(data: bytes, suffix: str, target: Path) -> None:
    """Декодирует загруженный не-WAV файл и сохраняет его как 16 кГц моно WAV."""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as handle:
            handle.write(data)
            temp_path = handle.name
        waveform = load_waveform(Path(temp_path))
        write_wav(target, waveform)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Не удалось декодировать аудио: {exc}"
        ) from exc
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def _variant_window(start: float | None, end: float | None) -> tuple[float, float]:
    """Проверяет и нормализует окно варианта ``[start, end)`` (оба конца заданы)."""
    if start is None or end is None:
        raise HTTPException(status_code=400, detail="Нужны обе границы окна: start и end")
    if start < 0 or end <= start:
        raise HTTPException(status_code=400, detail="Некорректное окно: end должен быть больше start")
    return float(start), float(end)


def _save_window_to_library(
    source: Path, window: tuple[float, float], directory: Path, name: str
) -> Path:
    """Вырезает окно исходного аудио и сохраняет его как новый образец библиотеки."""
    start, end = window
    window_wave = normalize_sample(slice_waveform(load_waveform(source), start, end))
    if window_wave.size == 0:
        raise ValueError("пустое окно образца")
    tmp_dir = Path(tempfile.mkdtemp(prefix="audio-transcriber-window-"))
    try:
        tmp = tmp_dir / "window.wav"
        write_wav(tmp, window_wave)
        return save_speaker_sample(tmp, directory, name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _index_response() -> Response:
    index_path = STATIC_DIR / "index.html"
    try:
        if index_path.is_file():
            return FileResponse(index_path, media_type="text/html")
    except OSError:
        pass
    return HTMLResponse(_PLACEHOLDER_HTML)


def _list_files(
    input_dir: Path, *, include_processed: bool = False
) -> list[dict[str, object]]:
    if not input_dir.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(input_dir.iterdir(), key=lambda item: item.name.casefold()):
        try:
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            processed = is_processed(path)
            if processed and not include_processed:
                continue
            # Абсолютный путь: клиент шлёт его обратно в ``POST /api/jobs``,
            # и он должен приниматься независимо от текущего рабочего каталога.
            items.append(
                {
                    "name": path.name,
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "duration": _probe_duration(path),
                    "processed": processed,
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
    # Новый файл не может быть «обработан»: снимаем возможный устаревший маркер
    # от прежде удалённого файла с тем же именем (issue #16).
    clear_processed(target)
    return {
        "name": target.name,
        "path": str(target.resolve()),
        "size": target.stat().st_size,
        "duration": None,
        "processed": False,
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
    """Приводит присланный клиентом путь к файлу внутри каталога загрузок.

    Принимает несколько форм, чтобы круговой сценарий «``GET /api/files`` →
    ``POST /api/jobs``» был надёжным:

    * абсолютный путь (``/data/web-data/uploads/x.mp4``);
    * простое имя файла (``x.mp4``);
    * относительный путь, уже содержащий каталог загрузок
      (``web-data/uploads/x.mp4``) — например, от старых версий SPA.

    Любой вариант обязан после разрешения лежать внутри каталога загрузок;
    иначе — ``400``. Существующий путь вне каталога не принимается никогда.
    """
    value = raw.strip()
    if not value:
        raise HTTPException(status_code=400, detail="Не указан путь к файлу")
    try:
        input_root = paths.input_dir.resolve()
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Каталог загрузок недоступен: {exc}"
        ) from exc
    candidate = Path(value).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(paths.input_dir / candidate)
        prefix = paths.input_dir.parts
        parts = candidate.parts
        if len(parts) > len(prefix) and parts[: len(prefix)] == prefix:
            candidates.append(paths.input_dir / Path(*parts[len(prefix) :]))
    inside = False
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(input_root):
            continue
        inside = True
        if resolved.is_file():
            return resolved
    if inside:
        raise HTTPException(status_code=404, detail="Файл не найден")
    raise HTTPException(
        status_code=400, detail="Файл должен находиться в каталоге загрузок"
    )


def _resolve_upload_file(paths: WebPaths, name: str) -> Path:
    """Разрешает имя файла загрузки для удаления (безопасно по путям)."""
    value = name.strip()
    if not value:
        raise HTTPException(status_code=400, detail="Не указано имя файла")
    if Path(value).name != value:
        raise HTTPException(status_code=400, detail="Некорректное имя файла")
    try:
        input_root = paths.input_dir.resolve()
        candidate = (paths.input_dir / value).resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Некорректный путь: {exc}") from exc
    if not candidate.is_relative_to(input_root):
        raise HTTPException(
            status_code=400, detail="Файл должен находиться в каталоге загрузок"
        )
    if candidate.suffix.lower() not in MEDIA_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Можно удалять только медиафайлы")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return candidate


def _same_file(left: Path, right: Path) -> bool:
    """Сравнивает два пути по разрешённому виду (устойчиво к относительным)."""
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _require_job(store: JobsDB, job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return job


def _validate_speaker_range(
    min_speakers: int | None, max_speakers: int | None
) -> None:
    """Проверяет, что нижняя граница числа говорящих не превышает верхнюю.

    Отдельные значения (>= 1) уже проверены Pydantic; здесь — только
    согласованность диапазона.
    """
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise HTTPException(
            status_code=422,
            detail="min_speakers не может быть больше max_speakers",
        )


def _result_path(paths: WebPaths, job: Job) -> Path:
    """Путь к JSON-результату задачи (только внутри каталога результатов)."""
    path = Path(job.result_path) if job.result_path else paths.results_dir / f"{job.id}.json"
    return path


def _remove_job_artifacts(paths: WebPaths, job: Job) -> None:
    """Удаляет JSON-результат и постадийный вывод задачи (безопасно по путям).

    Отсутствие файлов и любые ошибки файловой системы не считаются сбоем
    удаления задачи: сам факт удаления записи важнее очистки артефактов.
    """
    try:
        results_root = paths.results_dir.resolve()
    except OSError:
        return
    for candidate in (_result_path(paths, job), paths.results_dir / job.id):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(results_root):
            continue
        try:
            if resolved.is_file() or resolved.is_symlink():
                resolved.unlink(missing_ok=True)
            elif resolved.is_dir():
                shutil.rmtree(resolved, ignore_errors=True)
        except OSError:
            continue


def _probe_duration(path: Path) -> float | None:
    """Длительность аудио из контейнера без декодирования (``None`` при ошибке)."""
    return probe_duration(path)


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
        opener = threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/"))
        opener.daemon = True
        opener.start()
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
