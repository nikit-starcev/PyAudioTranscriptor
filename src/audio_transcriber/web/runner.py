"""Фоновый воркер очереди задач.

Задачи обрабатываются строго по одной: распознавание и диаризация конкурируют
за GPU/CPU, поэтому параллельный запуск только замедлил бы всех. Воркер —
daemon-поток, который при старте задачи вызывает переданную ``pipeline_fn``
(по умолчанию :func:`audio_transcriber.pipeline.run_pipeline`) с колбэком
прогресса и сохраняет результат в JSON рядом с БД.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.samples import find_speaker_samples, samples_directory
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.results import serialize_result
from audio_transcriber.web.storage.jobs_db import (
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RUNNING,
    JobsDB,
    utc_now_iso,
)
from audio_transcriber.web.timings import StageTimer

logger = logging.getLogger(__name__)

#: Вызываемое построение конфигурации для задачи: ``(job_id, source_path) -> AppConfig``.
ConfigBuilder = Callable[[str, Path], AppConfig]

#: Функция конвейера (совместима с ``run_pipeline``); подменяется в тестах.
PipelineFn = Callable[..., TranscriptionResult]

#: Сообщение для «подвешенной» задачи, осиротевшей после перезапуска сервера.
ORPHAN_ERROR_MESSAGE = "Прервано: сервер был перезапущен"


@dataclass(slots=True)
class JobRequest:
    """Задача, поставленная воркеру на обработку."""

    job_id: str
    source_path: Path


class JobRunner:
    """Один фоновый воркер для последовательной обработки задач."""

    def __init__(
        self,
        store: JobsDB,
        bus: JobEventBus,
        paths: WebPaths,
        config_builder: ConfigBuilder,
        *,
        pipeline_fn: PipelineFn | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._paths = paths
        self._config_builder = config_builder
        self._pipeline_fn = pipeline_fn or run_pipeline
        self._queue: queue.Queue[JobRequest | None] = queue.Queue()
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Запускает поток воркера (идемпотентно)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker, name="audio-transcriber-web-worker", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Останавливает воркер, дожидаясь завершения текущей задачи."""
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._queue.put(None)
        thread.join(timeout=timeout)

    def submit(self, job_id: str, source_path: Path) -> bool:
        """Ставит задачу в очередь; ``False``, если она уже в работе или очереди."""
        with self._lock:
            if job_id in self._active:
                return False
            self._active.add(job_id)
        self._queue.put(JobRequest(job_id=job_id, source_path=source_path))
        return True

    def active_job_ids(self) -> set[str]:
        """Снимок id задач, реально взятых воркером (в очереди или в работе)."""
        with self._lock:
            return set(self._active)

    def is_active(self, job_id: str) -> bool:
        """Обрабатывается ли задача этим воркером прямо сейчас (или ждёт в очереди)."""
        with self._lock:
            return job_id in self._active

    def reconcile_orphans(self, *, message: str = ORPHAN_ERROR_MESSAGE) -> list[str]:
        """Помечает «подвешенные» задачи как ``error`` и возвращает их id.

        Источник истины — :attr:`_active`: задачи, взятые этим воркером. Всё,
        что лежит в БД в ``running``/``queued`` вне этого набора, осиротело
        (например, процесс обработки был убит при перезапуске сервера) и
        никогда не завершится само. Такие записи переводим в ``error``, чтобы
        их можно было удалить или запустить заново, и сбрасываем прогресс.
        Активные задачи не трогаем.
        """
        active = self.active_job_ids()
        reconciled: list[str] = []
        for job in self._store.list():
            if job.status not in (STATUS_RUNNING, STATUS_QUEUED):
                continue
            if job.id in active:
                continue
            self._store.update(
                job.id,
                status=STATUS_ERROR,
                error=message,
                finished_at=utc_now_iso(),
                stage=None,
                fraction=None,
                stage_started_at=None,
                stage_times=[],
            )
            self._bus.publish(
                job.id,
                {
                    "stage": "error",
                    "fraction": None,
                    "message": message,
                    "status": STATUS_ERROR,
                },
            )
            reconciled.append(job.id)
        if reconciled:
            logger.warning(
                "Осиротевшие задачи помечены как error: %s", ", ".join(reconciled)
            )
        return reconciled

    def _worker(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                break
            try:
                self._process(request)
            except Exception:
                logger.exception("Непредвиденная ошибка обработки задачи %s", request.job_id)
            finally:
                with self._lock:
                    self._active.discard(request.job_id)

    def _process(self, request: JobRequest) -> None:
        job_id = request.job_id
        # Монотонный таймер стадий живёт ровно один прогон задачи.
        timer = StageTimer()
        self._store.update(
            job_id,
            status=STATUS_RUNNING,
            started_at=utc_now_iso(),
            finished_at=None,
            error=None,
            stage="queued",
            fraction=0.0,
            stage_started_at=None,
            stage_times=[],
        )
        # Новая попытка — чистим историю SSE, чтобы клиентам не реигрались
        # события прошлого прогона (например, «Прервано: сервер был перезапущен»).
        self._bus.clear(job_id)
        self._bus.publish(
            job_id,
            {
                "stage": "queued",
                "fraction": 0.0,
                "message": "Запуск",
                "status": STATUS_RUNNING,
            },
        )

        try:
            config = self._config_builder(job_id, request.source_path)
            # Число говорящих задаётся на уровне задачи и переопределяет дефолт
            # из настроек: ``None`` — автоопределение (pyannote сам решает).
            job = self._store.get(job_id)
            if job is not None:
                config.num_speakers = job.num_speakers
            result = self._pipeline_fn(
                config, on_progress=self._progress_callback(job_id, timer)
            )
        except Exception as exc:
            logger.exception("Задача %s завершилась ошибкой", job_id)
            timer.close()
            self._store.update(
                job_id,
                status=STATUS_ERROR,
                finished_at=utc_now_iso(),
                error=str(exc),
                stage_times=timer.timings(),
            )
            self._bus.publish(
                job_id,
                {
                    "stage": "error",
                    "fraction": None,
                    "message": str(exc),
                    "status": STATUS_ERROR,
                    "stage_times": timer.snapshot(),
                },
            )
            return

        # Стадия ``export`` закрывается здесь: ``done`` от конвейера воркер
        # намеренно игнорирует, чтобы не засчитывать запись результата.
        timer.close()
        self._finish_success(job_id, config, result, timer)

    def _progress_callback(
        self, job_id: str, timer: StageTimer
    ) -> Callable[[ProgressEvent], None]:
        current_stage: str | None = None

        def callback(event: ProgressEvent) -> None:
            nonlocal current_stage
            if event.stage == "done":
                # Финальное событие отправляет сам воркер после записи результата.
                return
            before = len(timer.timings())
            timer.observe(event)
            fields: dict[str, object] = {
                "stage": event.stage,
                "fraction": event.fraction,
            }
            if event.stage != current_stage:
                # Начало новой стадии — фиксируем её старт для живого таймера.
                current_stage = event.stage
                fields["stage_started_at"] = utc_now_iso()
            if len(timer.timings()) != before:
                # Стадия закрылась — сохраняем накопленные тайминги.
                fields["stage_times"] = timer.timings()
            self._store.update(job_id, **fields)
            self._bus.publish(
                job_id,
                {
                    "stage": event.stage,
                    "fraction": event.fraction,
                    "message": event.message,
                    "status": STATUS_RUNNING,
                    "elapsed": round(timer.elapsed(), 3),
                    "stage_elapsed": round(timer.current_elapsed(), 3),
                    "stage_times": timer.snapshot(),
                },
            )

        return callback

    def _finish_success(
        self,
        job_id: str,
        config: AppConfig,
        result: TranscriptionResult,
        timer: StageTimer,
    ) -> None:
        samples = self._collect_samples(config, result)
        payload = serialize_result(result, samples=samples)
        payload["samples"] = samples
        result_path = self._paths.results_dir / f"{job_id}.json"
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.exception("Не удалось сохранить результат задачи %s", job_id)
            self._store.update(
                job_id,
                status=STATUS_ERROR,
                finished_at=utc_now_iso(),
                error=f"Не удалось сохранить результат: {exc}",
                stage_times=timer.timings(),
            )
            self._bus.publish(
                job_id,
                {
                    "stage": "error",
                    "fraction": None,
                    "message": "Не удалось сохранить результат",
                    "status": STATUS_ERROR,
                    "stage_times": timer.snapshot(),
                },
            )
            return

        self._store.update(
            job_id,
            status=STATUS_DONE,
            finished_at=utc_now_iso(),
            stage="done",
            fraction=1.0,
            language=result.language,
            duration=result.duration,
            result_path=str(result_path),
            stage_times=timer.timings(),
        )
        self._bus.publish(
            job_id,
            {
                "stage": "done",
                "fraction": 1.0,
                "message": "Готово",
                "status": STATUS_DONE,
                "stage_times": timer.snapshot(),
            },
        )

    def _collect_samples(
        self, config: AppConfig, result: TranscriptionResult
    ) -> dict[str, str]:
        """Относительные пути сохранённых конвейером образцов голоса."""
        directory = samples_directory(config.output_dir, result.source_path)
        found = find_speaker_samples(result, directory)
        samples: dict[str, str] = {}
        for speaker_id, path in found.items():
            try:
                relative = path.resolve().relative_to(self._paths.data_dir.resolve())
            except (OSError, ValueError):
                continue
            samples[speaker_id] = str(relative)
        return samples
