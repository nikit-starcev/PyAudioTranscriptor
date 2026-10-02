"""Фоновый воркер очереди задач.

Задачи обрабатываются строго по одной: распознавание и диаризация конкурируют
за GPU/CPU, поэтому параллельный запуск только замедлил бы всех. Воркер —
daemon-поток, который при старте задачи вызывает переданную ``pipeline_fn``
(по умолчанию :func:`audio_transcriber.pipeline.run_pipeline`) с колбэком
прогресса и сохраняет результат в JSON рядом с БД.
"""

from __future__ import annotations

import inspect
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
from audio_transcriber.utils.exceptions import ProcessingCancelled
from audio_transcriber.utils.notifications import notify
from audio_transcriber.utils.subprocess_registry import terminate_all_processes
from audio_transcriber.web.estimates import StageEstimator, probe_duration
from audio_transcriber.web.events import JobEventBus
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.processed import mark_processed
from audio_transcriber.web.results import load_result_file, merge_transcript_edits, serialize_result
from audio_transcriber.web.storage.jobs_db import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_RUNNING,
    Job,
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

#: Сообщение для задачи, остановленной пользователем.
CANCELLED_MESSAGE = "Остановлено пользователем"

#: Заголовки десктоп-уведомлений по терминальному статусу задачи.
_NOTIFY_TITLES: dict[str, str] = {
    STATUS_DONE: "Транскрибация завершена",
    STATUS_ERROR: "Транскрибация не удалась",
    STATUS_CANCELLED: "Транскрибация отменена",
}

#: Сколько символов текста ошибки попадает в уведомление.
_NOTIFY_ERROR_LIMIT = 200


def _notifications_enabled(config: AppConfig | None) -> bool:
    """Включены ли уведомления для задачи (конфигурация неизвестна — выключены)."""
    return config is not None and config.notifications


def _safe_notify(title: str, message: str) -> None:
    """Отправляет уведомление, не позволяя ошибке доставки уронить воркер."""
    try:
        notify(title, message)
    except Exception:
        logger.warning("Не удалось отправить уведомление: %s", title, exc_info=True)


def _format_duration(seconds: float) -> str:
    """Человекочитаемая длительность обработки (без лишних нулей)."""
    total = max(round(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} ч {minutes} мин"
    if minutes:
        return f"{minutes} мин {secs} с"
    return f"{secs} с"


def _notification_message(job: Job, status: str, *, detail: str | None = None) -> str:
    """Текст уведомления: имя задачи, статус и (при успехе) длительность."""
    if status == STATUS_DONE:
        seconds = job.total_seconds
        if seconds:
            return f"{job.name}: готово за {_format_duration(seconds)}"
        return f"{job.name}: готово"
    if status == STATUS_ERROR:
        if detail:
            return f"{job.name}: ошибка — {detail[:_NOTIFY_ERROR_LIMIT]}"
        return f"{job.name}: ошибка"
    return f"{job.name}: отменено"



def _accepts_cancel_event(pipeline_fn: Callable[..., TranscriptionResult]) -> bool:
    """Умеет ли ``pipeline_fn`` принимать ``cancel_event`` (kwarg или ``**kwargs``).

    Штатный :func:`run_pipeline` умеет; тестовые заглушки без него — нет. От
    этого зависит, передавать ли воркеру флаг отмены, не ломая существующие
    подмены конвейера.
    """
    try:
        parameters = inspect.signature(pipeline_fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "cancel_event"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


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
        estimator: StageEstimator | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._paths = paths
        self._config_builder = config_builder
        self._pipeline_fn = pipeline_fn or run_pipeline
        #: Умеет ли конвейер принимать ``cancel_event`` (у тестовых заглушек — нет).
        self._pipeline_accepts_cancel = _accepts_cancel_event(self._pipeline_fn)
        #: Оценки прогресса/ETA/здоровья; ``None`` — отключены (тесты, минимальный режим).
        self._estimator = estimator
        self._queue: queue.Queue[JobRequest | None] = queue.Queue()
        self._lock = threading.Lock()
        self._active: set[str] = set()
        #: Флаги отмены активных задач (id → событие). Создаются в :meth:`submit`,
        #: снимаются по завершении задачи.
        self._cancel_events: dict[str, threading.Event] = {}
        #: Задача, которую воркер обрабатывает прямо сейчас (не в очереди).
        #: Нужна, чтобы отмена ждущей задачи не гасила процессы текущей.
        self._current: str | None = None
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
        """Останавливает воркер, дожидаясь завершения текущей задачи.

        При остановке сервера активные задачи получают сигнал отмены, а их
        дочерние процессы (``whisper-cli``/``llama-server``) гасятся — Ctrl+C
        не оставляет висящих процессов и не блокирует выход надолго.
        """
        with self._lock:
            thread = self._thread
            self._thread = None
            events = list(self._cancel_events.values())
        for event in events:
            event.set()
        if events:
            # Страховка поверх atexit/обработчиков сигналов реестра.
            terminate_all_processes()
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
            # Свежий флаг отмены на каждый прогон: осиротевший с прошлого раза
            # не должен мгновенно гасить новую задачу.
            self._cancel_events[job_id] = threading.Event()
        self._queue.put(JobRequest(job_id=job_id, source_path=source_path))
        return True

    def cancel(self, job_id: str) -> bool:
        """Запрашивает отмену активной задачи; ``False``, если она не активна.

        Выставляет флаг отмены (конвейер поднимет :class:`ProcessingCancelled`
        в ближайшей контрольной точке) и гасит зарегистрированные дочерние
        процессы, чтобы текущая долгая стадия завершилась быстро. Итоговый
        статус ``cancelled`` проставит воркер по завершении потока задачи.
        """
        with self._lock:
            event = self._cancel_events.get(job_id)
            if job_id not in self._active or event is None:
                return False
            event.set()
            # Гасим процессы только у обрабатываемой сейчас задачи: отмена
            # ждущей в очереди не должна трогать чужую стадию.
            is_current = self._current == job_id
        if is_current:
            terminate_all_processes()
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
        Активные задачи не трогаем. Удалённые тоже проверяем: осиротевшая
        удалённая задача не должна «зависнуть» в ``running`` навсегда (#30).
        """
        active = self.active_job_ids()
        reconciled: list[str] = []
        for job in self._store.list(include_deleted=True):
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
                    self._cancel_events.pop(request.job_id, None)
                    if self._current == request.job_id:
                        self._current = None

    def _process(self, request: JobRequest) -> None:
        job_id = request.job_id
        # Мягко удалённую задачу не запускаем: если её удалили, пока она ждала
        # в очереди, воркер должен пропустить её (восстановление вернёт обычное
        # поведение — задача снова запускаема вручную, #30).
        job = self._store.get(job_id)
        if job is None or job.deleted:
            logger.info("Задача %s удалена — пропускаю запуск", job_id)
            return
        with self._lock:
            self._current = job_id
            cancel_event = self._cancel_events.get(job_id)
        # Монотонный таймер стадий живёт ровно один прогон задачи.
        timer = StageTimer()
        # Длительность аудио нужна для ETA уже во время прогона: результат
        # сообщит её только в конце, поэтому пробуем контейнер заранее.
        duration = probe_duration(request.source_path)
        self._store.update(
            job_id,
            status=STATUS_RUNNING,
            started_at=utc_now_iso(),
            finished_at=None,
            error=None,
            stage="queued",
            fraction=0.0,
            duration=duration,
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

        config: AppConfig | None = None
        try:
            config = self._config_builder(job_id, request.source_path)
            # Число говорящих задаётся на уровне задачи и переопределяет дефолт
            # из настроек: ``None`` — автоопределение (pyannote сам решает).
            job = self._store.get(job_id)
            if job is not None:
                config.num_speakers = job.num_speakers
                config.min_speakers = job.min_speakers
                config.max_speakers = job.max_speakers
            progress_callback = self._progress_callback(job_id, timer)
            if self._pipeline_accepts_cancel:
                result = self._pipeline_fn(
                    config, on_progress=progress_callback, cancel_event=cancel_event
                )
            else:
                result = self._pipeline_fn(config, on_progress=progress_callback)
        except ProcessingCancelled as exc:
            # Штатное прерывание по запросу пользователя — не ошибка.
            logger.info("Задача %s отменена: %s", job_id, exc)
            self._finish_cancelled(
                job_id, timer, notify_enabled=_notifications_enabled(config)
            )
            return
        except Exception as exc:
            # Конвейер мог упасть из-за того, что мы погасили процесс при
            # отмене (или отмена пришла во время стадии, не конвертировавшей
            # исключение сама). Тогда это тоже отмена, а не сбой.
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Задача %s отменена во время стадии", job_id)
                self._finish_cancelled(
                    job_id, timer, notify_enabled=_notifications_enabled(config)
                )
                return
            logger.exception("Задача %s завершилась ошибкой", job_id)
            timer.close()
            failed = self._store.update(
                job_id,
                status=STATUS_ERROR,
                finished_at=utc_now_iso(),
                error=str(exc),
                stage_times=timer.timings(),
            )
            failed_payload: dict[str, object] = {
                "stage": "error",
                "fraction": None,
                "message": str(exc),
                "status": STATUS_ERROR,
                "stage_times": timer.snapshot(),
            }
            self._merge_estimate(failed_payload, failed, active=False)
            self._bus.publish(job_id, failed_payload)
            self._invalidate_estimates()
            self._notify_finished(
                failed,
                STATUS_ERROR,
                enabled=_notifications_enabled(config),
                detail=str(exc),
            )
            return

        # ``config_builder`` либо вернул конфигурацию, либо бросил исключение
        # выше — здесь ``config`` гарантированно задан.
        assert config is not None
        # Стадия ``export`` закрывается здесь: ``done`` от конвейера воркер
        # намеренно игнорирует, чтобы не засчитывать запись результата.
        timer.close()
        self._finish_success(job_id, config, result, timer)

    def _finish_cancelled(
        self, job_id: str, timer: StageTimer, *, notify_enabled: bool
    ) -> None:
        """Переводит задачу в терминальный статус ``cancelled`` и шлёт событие.

        Отменённая задача удаляема и перезапускаема; завершённые до отмены
        стадии остаются в стадийном кэше — повторный запуск возобновит работу
        с них. Событие терминальное, поэтому SSE-подписчики закрывают поток, а
        живые таймеры в UI останавливаются. ``notify_enabled`` — слать ли
        десктоп-уведомление (из конфигурации задачи).
        """
        timer.close()
        # Стадию не перезаписываем: она показывает, где именно остановились.
        cancelled = self._store.update(
            job_id,
            status=STATUS_CANCELLED,
            finished_at=utc_now_iso(),
            error=None,
            stage_times=timer.timings(),
        )
        payload: dict[str, object] = {
            "stage": "cancelled",
            "fraction": None,
            "message": CANCELLED_MESSAGE,
            "status": STATUS_CANCELLED,
            "stage_times": timer.snapshot(),
        }
        self._merge_estimate(payload, cancelled, active=False)
        self._bus.publish(job_id, payload)
        self._invalidate_estimates()
        self._notify_finished(cancelled, STATUS_CANCELLED, enabled=notify_enabled)

    def _notify_finished(
        self,
        job: Job | None,
        status: str,
        *,
        enabled: bool,
        detail: str | None = None,
    ) -> None:
        """Десктоп-уведомление о терминальном статусе задачи (best-effort).

        Отправка идёт в отдельном daemon-потоке: медленный ``notify-send`` не
        должен задерживать следующую задачу очереди. Ошибки доставки гасятся в
        :func:`_safe_notify`, поэтому завершение задачи не зависит от наличия
        уведомителя. ``enabled`` берётся из ``AppConfig.notifications``
        (``NOTIFICATIONS`` в ``config.env``).
        """
        if not enabled or job is None:
            return
        title = _NOTIFY_TITLES.get(status)
        if title is None:
            return
        message = _notification_message(job, status, detail=detail)
        threading.Thread(
            target=_safe_notify,
            args=(title, message),
            name="audio-transcriber-notify",
            daemon=True,
        ).start()

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
            updated = self._store.update(job_id, **fields)
            payload: dict[str, object] = {
                "stage": event.stage,
                "fraction": event.fraction,
                "message": event.message,
                "status": STATUS_RUNNING,
                "elapsed": round(timer.elapsed(), 3),
                "stage_elapsed": round(timer.current_elapsed(), 3),
                "stage_times": timer.snapshot(),
            }
            self._merge_estimate(payload, updated, active=True)
            self._bus.publish(job_id, payload)

        return callback

    def _merge_estimate(
        self, payload: dict[str, object], job: Job | None, *, active: bool
    ) -> None:
        """Добавляет в событие/ответ оценки прогресса, ETA и здоровья (#15/#24)."""
        if self._estimator is None or job is None:
            return
        payload.update(self._estimator.snapshot(job, active=active))

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
        # Повторный прогон (в том числе из ASR-кэша) не должен затирать ручные
        # правки текста (#26): переносим их из предыдущего результата по
        # неизменным таймкодам.
        payload = merge_transcript_edits(payload, load_result_file(result_path))
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.exception("Не удалось сохранить результат задачи %s", job_id)
            failed = self._store.update(
                job_id,
                status=STATUS_ERROR,
                finished_at=utc_now_iso(),
                error=f"Не удалось сохранить результат: {exc}",
                stage_times=timer.timings(),
            )
            failed_payload: dict[str, object] = {
                "stage": "error",
                "fraction": None,
                "message": "Не удалось сохранить результат",
                "status": STATUS_ERROR,
                "stage_times": timer.snapshot(),
            }
            self._merge_estimate(failed_payload, failed, active=False)
            self._bus.publish(job_id, failed_payload)
            self._invalidate_estimates()
            self._notify_finished(
                failed,
                STATUS_ERROR,
                enabled=config.notifications,
                detail=f"Не удалось сохранить результат: {exc}",
            )
            return

        finished = self._store.update(
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
        # Успешный прогон убирает исходный файл из основного списка ``/api/files``
        # (issue #16), не перемещая его: аудио, повторный запуск и стадийный кэш
        # продолжают работать. Помечаем до публикации ``done``, чтобы клиент,
        # перечитав список по событию, уже не увидел файл.
        self._mark_source_processed(config.input_file)
        done_payload: dict[str, object] = {
            "stage": "done",
            "fraction": 1.0,
            "message": "Готово",
            "status": STATUS_DONE,
            "stage_times": timer.snapshot(),
        }
        self._merge_estimate(done_payload, finished, active=False)
        self._bus.publish(job_id, done_payload)
        self._invalidate_estimates()
        self._notify_finished(finished, STATUS_DONE, enabled=config.notifications)

    def _invalidate_estimates(self) -> None:
        """Сбрасывает кэш статистики: завершённый прогон учтётся сразу."""
        if self._estimator is not None:
            self._estimator.invalidate()

    def _mark_source_processed(self, source: Path) -> None:
        """Помечает исходный файл обработанным после успешного прогона (#16).

        Только для файлов внутри каталога загрузок: он «уходит» из основного
        списка ``/api/files``, оставаясь на месте для прослушивания и повторного
        запуска. Ошибка файловой системы не должна ронять успешную задачу —
        достаточно предупреждения в логе.
        """
        try:
            input_root = self._paths.input_dir.resolve()
            resolved = source.resolve()
        except OSError:
            return
        if not resolved.is_relative_to(input_root) or not resolved.is_file():
            return
        if not mark_processed(resolved):
            logger.warning("Не удалось пометить файл обработанным: %s", resolved)

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
