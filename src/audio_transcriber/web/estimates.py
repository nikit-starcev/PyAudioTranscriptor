"""Оценки прогресса, ETA и «здоровья» задачи для веб-интерфейса.

Модуль реализует две связанные задачи:

* **#15 — прогноз длительности стадий (ETA).** По завершённым задачам
  собирается статистика «стоимости» каждой стадии: отношение времени стадии к
  длительности аудио (RTF, real-time factor). Для кэшированных и
  не-кэшированных стадий статистика считается раздельно, берётся медиана.
  По ней оцениваются ожидаемое время текущей стадии и остаток всего прогона.
* **#24 — сводный процент и «здоровье».** Общий прогресс =
  (сумма весов пройденных стадий + вес текущей × её ``fraction``) / сумма
  весов, где веса — те же RTF-оценки из #15 (fallback — равные). «Здоровье»
  классифицируется по признаку ``active``, давности последнего обновления и
  темпу относительно ожидаемого.

Статистика не хранится отдельно: она пересчитывается по завершённым задачам и
кэшируется на короткий TTL (:class:`StageEstimator`), поэтому новые прогоны
учитываются автоматически.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from audio_transcriber.web.storage.jobs_db import STATUS_DONE, STATUS_RUNNING, Job, JobsDB

#: Стадии конвейера в порядке выполнения (совпадает с ``progress.py``).
STAGES: tuple[str, ...] = (
    "denoise",
    "asr",
    "diarization",
    "merge",
    "clean",
    "correction",
    "llm",
    "export",
)

#: Минимальный вес стадии — защита от деления на ноль и нулевых весов.
MIN_WEIGHT = 1e-6

#: Замедление: текущая стадия идёт дольше ожидаемого в это число раз.
SLOW_FACTOR = 2.0
#: Нижняя граница «медленно», чтобы короткие стадии не считались тормозом.
SLOW_MIN_SECONDS = 30.0
#: «Зависание»: нет обновлений дольше ожидаемого в это число раз.
STALL_FACTOR = 3.0
#: Нижняя граница «нет активности» (секунды).
STALL_MIN_SECONDS = 60.0

HEALTH_OK = "ok"
HEALTH_SLOW = "slow"
HEALTH_STALLED = "stalled"

#: TTL кэша статистики по умолчанию (секунды).
DEFAULT_PROFILE_TTL = 30.0


def _parse_iso(value: str | None) -> datetime | None:
    """Разбирает ISO-8601 из БД; ``None`` при пустом/некорректном значении."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _clamp(value: float, low: float, high: float) -> float:
    """Ограничивает значение диапазоном ``[low, high]``."""
    return max(low, min(high, value))


@dataclass(slots=True)
class StageProfile:
    """Статистика «стоимости» стадий: секунды на секунду аудио (RTF)."""

    #: RTF по каждой стадии (:data:`STAGES`); во всех ключах ненулевое значение.
    weights: dict[str, float] = field(default_factory=dict)
    #: Число не-кэшированных наблюдений, по которым построена статистика.
    samples: int = 0
    #: Сколько завершённых задач с длительностью дали данные.
    jobs: int = 0

    @property
    def has_history(self) -> bool:
        """Есть ли хоть какие-то не-кэшированные наблюдения."""
        return self.samples > 0

    def weight(self, stage: str) -> float:
        """Вес стадии (``0.0`` для неизвестной стадии)."""
        return self.weights.get(stage, 0.0)


def build_profile(jobs: Iterable[Job]) -> StageProfile:
    """Собирает RTF-статистику по завершённым задачам.

    Учитываются только ``done``-задачи с известной длительностью аудио и
    непустыми ``stage_times``. Кэшированные и не-кэшированные наблюдения
    копятся раздельно; берётся медиана. Если не-кэшированной истории нет,
    возвращаются равные веса (``samples == 0``) — тогда ETA не считается.
    """
    fresh: dict[str, list[float]] = {stage: [] for stage in STAGES}
    cached: dict[str, list[float]] = {stage: [] for stage in STAGES}
    all_fresh: list[float] = []
    completed = 0

    for job in jobs:
        if job.status != STATUS_DONE:
            continue
        duration = job.duration
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        if not job.stage_times:
            continue
        completed += 1
        for timing in job.stage_times:
            rtf = max(float(timing.seconds), 0.0) / float(duration)
            bucket = cached if timing.cached else fresh
            bucket.setdefault(timing.stage, []).append(rtf)
            if not timing.cached:
                all_fresh.append(rtf)

    if not all_fresh:
        # Нет истории реальных прогонов — равные веса (процент по числу стадий).
        return StageProfile(weights=dict.fromkeys(STAGES, 1.0), samples=0, jobs=completed)

    fresh_median = {
        stage: statistics.median(values) for stage, values in fresh.items() if values
    }
    cached_median = {
        stage: statistics.median(values) for stage, values in cached.items() if values
    }
    fallback = max(statistics.median(all_fresh), MIN_WEIGHT)

    weights: dict[str, float] = {}
    for stage in STAGES:
        if stage in fresh_median:
            value = fresh_median[stage]
        elif stage in cached_median:
            # Стадия в прошлом всегда шла из кэша — она практически мгновенна.
            value = cached_median[stage]
        else:
            value = fallback
        weights[stage] = max(value, MIN_WEIGHT)

    return StageProfile(weights=weights, samples=len(all_fresh), jobs=completed)


def expected_seconds(profile: StageProfile, stage: str, duration: float | None) -> float | None:
    """Ожидаемое время стадии для аудио длительности ``duration`` (или ``None``)."""
    if not profile.has_history or duration is None or duration <= 0:
        return None
    weight = profile.weight(stage)
    if weight <= 0:
        return None
    return weight * duration


def _current_fraction(job: Job, profile: StageProfile) -> float:
    """Доля текущей стадии: ``fraction``, иначе — по прошедшему времени.

    Для стадий без собственного ``fraction`` (диаризация, экспорт и т.п.)
    используем интерполяцию по ``stage_elapsed`` относительно ожидаемой
    длительности, чтобы общий процент рос плавно, а не ступенями.
    """
    if job.fraction is not None:
        return _clamp(float(job.fraction), 0.0, 1.0)
    if job.stage is None:
        return 0.0
    expected = expected_seconds(profile, job.stage, job.duration)
    elapsed = job.stage_elapsed
    if expected is not None and expected > 0 and elapsed is not None:
        return _clamp(elapsed / expected, 0.0, 0.99)
    return 0.0


def _stage_plan(job: Job, profile: StageProfile) -> list[tuple[str, float, float]]:
    """План стадий: ``(стадия, ожидаемые секунды, доля остатка)``.

    Пройденные (по ``stage_times`` или по позиции текущей стадии) пропускаются.
    Для текущей стадии доля остатка меньше единицы.
    """
    duration = job.duration
    completed = {timing.stage for timing in job.stage_times}
    current = job.stage
    current_index = STAGES.index(current) if current in STAGES else -1
    fraction = _current_fraction(job, profile) if current in STAGES else 0.0

    plan: list[tuple[str, float, float]] = []
    for index, stage in enumerate(STAGES):
        if stage in completed or (current_index >= 0 and index < current_index):
            continue
        expected = expected_seconds(profile, stage, duration)
        if expected is None:
            continue
        if stage == current and not job.is_terminal:
            plan.append((stage, expected, max(1.0 - fraction, 0.0)))
        else:
            plan.append((stage, expected, 1.0))
    return plan


def progress_percent(job: Job, profile: StageProfile) -> float:
    """Сводный процент прогона с учётом весов стадий (#24).

    Пройденные стадии дают полный вес, текущая — вес × ``fraction``. Для
    завершённой задачи всегда 100%. Кэш-стадии, попавшие в ``stage_times``,
    считаются пройденными независимо от их длительности.
    """
    if job.status == STATUS_DONE:
        return 100.0

    total = sum(profile.weight(stage) for stage in STAGES)
    if total <= 0:
        return 0.0

    completed = {timing.stage for timing in job.stage_times}
    current = job.stage
    current_index = STAGES.index(current) if current in STAGES else -1
    fraction = _current_fraction(job, profile) if current in STAGES else 0.0

    done = 0.0
    for index, stage in enumerate(STAGES):
        weight = profile.weight(stage)
        if stage in completed or (current_index >= 0 and index < current_index):
            done += weight
        elif stage == current and not job.is_terminal:
            done += weight * fraction
    return round(_clamp(done / total * 100.0, 0.0, 100.0), 1)


def eta_seconds(job: Job, profile: StageProfile) -> float | None:
    """Остаток всего прогона в секундах или ``None`` (нет данных/истории).

    Для завершённой задачи — ``0.0``; для незапущенной — ``None``.
    """
    if job.status == STATUS_DONE:
        return 0.0
    if job.status != STATUS_RUNNING or not profile.has_history:
        return None
    if job.duration is None or job.duration <= 0:
        return None
    plan = _stage_plan(job, profile)
    if not plan:
        return 0.0
    remaining = sum(expected * left for _, expected, left in plan)
    return round(remaining, 1)


def eta_by_stage(job: Job, profile: StageProfile) -> dict[str, float] | None:
    """Остаток по стадиям (секунды) или ``None``, если посчитать нельзя."""
    if job.status not in (STATUS_RUNNING, STATUS_DONE):
        return None
    if job.status == STATUS_DONE:
        return {}
    if not profile.has_history:
        return None
    if job.duration is None or job.duration <= 0:
        return None
    plan = _stage_plan(job, profile)
    if not plan:
        return {}
    return {stage: round(expected * left, 1) for stage, expected, left in plan}


@dataclass(slots=True)
class Health:
    """«Здоровье» задачи: статус и давность последнего обновления."""

    status: str
    last_update_seconds: float | None


def _seconds_since(value: str | None, now: datetime) -> float | None:
    moment = _parse_iso(value)
    if moment is None:
        return None
    return max((now - moment).total_seconds(), 0.0)


def classify_health(
    job: Job,
    profile: StageProfile,
    *,
    active: bool,
    now: datetime | None = None,
) -> Health:
    """Классифицирует состояние задачи: ``ok`` / ``slow`` / ``stalled``.

    * ``stalled`` — задача не активна (осиротевшая ``running``) либо давно не
      приходило обновлений (дольше ``STALL_FACTOR`` ожидаемой стадии, но не
      меньше :data:`STALL_MIN_SECONDS`);
    * ``slow`` — текущая стадия идёт заметно дольше ожидаемого;
    * ``ok`` — всё в норме.
    """
    if now is None:
        now = datetime.now(UTC)

    last_update = _seconds_since(job.updated_at, now)
    if last_update is None:
        # Старые записи без ``updated_at``: берём возраст текущей стадии.
        last_update = job.stage_elapsed
    last_update_rounded = round(last_update, 1) if last_update is not None else None

    if job.status == STATUS_RUNNING and not active:
        return Health(HEALTH_STALLED, last_update_rounded)

    expected = (
        expected_seconds(profile, job.stage, job.duration) if job.stage else None
    )
    elapsed = job.stage_elapsed

    stall_threshold = STALL_MIN_SECONDS
    slow_threshold: float | None = None
    if expected is not None and expected > 0:
        stall_threshold = max(STALL_MIN_SECONDS, STALL_FACTOR * expected)
        slow_threshold = max(SLOW_MIN_SECONDS, SLOW_FACTOR * expected)

    if last_update is not None and last_update >= stall_threshold:
        return Health(HEALTH_STALLED, last_update_rounded)

    slow = False
    if slow_threshold is not None and elapsed is not None and elapsed >= slow_threshold:
        slow = True
    if (
        not slow
        and expected is not None
        and expected > 0
        and job.fraction is not None
        and elapsed is not None
    ):
        # Доля выполнена мала, а времени прошло заметно больше, чем должно
        # было уйти на эту долю, — прогресс буксует.
        expected_elapsed = float(job.fraction) * expected
        if expected_elapsed > 0 and elapsed >= max(
            SLOW_MIN_SECONDS, SLOW_FACTOR * expected_elapsed
        ):
            slow = True

    return Health(HEALTH_SLOW if slow else HEALTH_OK, last_update_rounded)


def health_payload(
    job: Job, profile: StageProfile, *, active: bool
) -> dict[str, object] | None:
    """JSON-представление здоровья (``None`` для не-running задач)."""
    if job.status != STATUS_RUNNING:
        return None
    health = classify_health(job, profile, active=active)
    return {
        "status": health.status,
        "last_update_seconds": health.last_update_seconds,
    }


def probe_duration(path: Path) -> float | None:
    """Длительность аудио из контейнера без декодирования (``None`` при ошибке)."""
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration is None:
                return None
            return round(container.duration / av.time_base, 2)
    except Exception:  # noqa: BLE001 — длительность не критична для оценок
        return None


class StageEstimator:
    """Кэширующий фасад: статистика стадий + оценка конкретной задачи."""

    def __init__(
        self,
        store: JobsDB,
        *,
        ttl: float = DEFAULT_PROFILE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._profile: StageProfile | None = None
        self._loaded_at = -math.inf

    def profile(self) -> StageProfile:
        """Текущая статистика (пересчитывается не чаще, чем раз в ``ttl``)."""
        now = self._clock()
        with self._lock:
            if self._profile is None or now - self._loaded_at >= self._ttl:
                self._profile = build_profile(self._store.list())
                self._loaded_at = now
            return self._profile

    def invalidate(self) -> None:
        """Сбрасывает кэш статистики (после завершения прогона)."""
        with self._lock:
            self._profile = None
            self._loaded_at = -math.inf

    def snapshot(self, job: Job, *, active: bool) -> dict[str, object]:
        """Оценки задачи для API/SSE: процент, ETA и здоровье."""
        profile = self.profile()
        return {
            "progress_percent": progress_percent(job, profile),
            "eta_seconds": eta_seconds(job, profile),
            "eta_by_stage": eta_by_stage(job, profile),
            "health": health_payload(job, profile, active=active),
        }
