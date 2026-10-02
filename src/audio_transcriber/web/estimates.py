"""Оценки прогресса, ETA и «здоровья» задачи для веб-интерфейса.

Модуль реализует две связанные задачи:

* **#15 — прогноз длительности стадий (ETA).** По завершённым задачам
  собирается статистика «стоимости» каждой стадии: отношение времени стадии к
  длительности аудио (RTF, real-time factor). В веса идут **только
  не-кэшированные** наблюдения (``cached=False``), берётся медиана; кэшированные
  стадии почти мгновенны и дали бы ложно-заниженную оценку (#33). Для стадий,
  которых в истории ещё не было, применяется консервативный запасной вес —
  медиана известных стадий. Если же стадия в истории **была только из кэша**,
  её реальная стоимость неизвестна: пока такая стадия ещё впереди, ETA не
  показывается вовсе (:data:`None`) — честнее без оценки, чем заниженная.
* **#24 — сводный процент и «здоровье».** Общий прогресс =
  (сумма весов пройденных стадий + вес текущей × её ``fraction``) / сумма
  весов. Когда свежая история не покрывает все наблюдённые стадии (типично при
  кэшированных прогонах), вместо искажённых весов берутся равные — процент
  растёт плавно по стадиям и не «перескакивает» на ранних этапах.
  «Здоровье» классифицируется по признаку ``active``, давности последнего
  обновления и темпу относительно ожидаемого.

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
from typing import TYPE_CHECKING

from audio_transcriber.web.storage.jobs_db import STATUS_DONE, STATUS_RUNNING, Job, JobsDB

if TYPE_CHECKING:
    from audio_transcriber.config.settings import AppConfig

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

#: Флаг :class:`AppConfig`, включающий стадию; ``None`` — стадия безусловная.
#: Порядок и состав повторяют условия :func:`audio_transcriber.pipeline.run_pipeline`.
_STAGE_FLAGS: dict[str, str | None] = {
    "denoise": "denoise",
    "asr": None,
    "diarization": "diarization_enabled",
    "merge": None,
    "clean": "clean_artifacts",
    "correction": "enable_correction",
    "llm": "llm_enabled",
    "export": None,
}


def planned_stages(config: AppConfig) -> list[str]:
    """Планируемые стадии конвейера в порядке выполнения для конфигурации задачи.

    Состав повторяет условия :func:`audio_transcriber.pipeline.run_pipeline`:
    ``denoise`` включается флагом ``denoise``, ``diarization`` —
    ``diarization_enabled``, ``clean`` — ``clean_artifacts``, ``correction`` —
    ``enable_correction``, ``llm`` — ``llm_enabled``; ``asr``/``merge``/``export``
    выполняются всегда. Порядок задаёт :data:`STAGES`.

    Веб-интерфейс показывает этот план сразу и целиком (а не только пройденные
    стадии), поэтому список отдаётся сервером вместе со статусом задачи.
    """
    stages: list[str] = []
    for stage in STAGES:
        flag = _STAGE_FLAGS.get(stage)
        if flag is None or bool(getattr(config, flag, False)):
            stages.append(stage)
    return stages

#: Минимальный вес стадии — защита от деления на ноль и нулевых весов.
#: Свежая стадия с нулевым временем тоже получает его, чтобы не пропасть из
#: расчёта (иначе «нулевой» вес выглядел бы как отсутствие истории).
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

#: Человекочитаемые названия стадий — для причины замедления (#36).
STAGE_TITLES: dict[str, str] = {
    "denoise": "шумоподавление",
    "asr": "распознавание речи",
    "diarization": "определение говорящих",
    "merge": "объединение сегментов",
    "clean": "очистка артефактов",
    "correction": "автоисправление",
    "llm": "LLM-постобработка",
    "export": "экспорт",
}

#: Ресурсоёмкие стадии: диаризация и ASR чаще всего упираются в GPU/CPU (#36).
HEAVY_STAGES = frozenset({"diarization", "asr"})

#: Длительность записи (секунды), начиная с которой прогон считаем «длинным»
#: и упоминаем это в причине замедления (#36).
LONG_AUDIO_SECONDS = 1800.0

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


def _equal_weight(_stage: str) -> float:
    """Равный вес для всех стадий (нейтральная шкала без свежей истории)."""
    return 1.0


@dataclass(slots=True)
class StageProfile:
    """Статистика «стоимости» стадий: секунды на секунду аудио (RTF)."""

    #: Оценочный RTF для всех стадий :data:`STAGES`: свежая медиана, а для
    #: стадий без свежих данных — консервативный запасной вес.
    weights: dict[str, float] = field(default_factory=dict)
    #: Число не-кэшированных наблюдений, по которым построена статистика.
    samples: int = 0
    #: Сколько завершённых задач с длительностью дали данные.
    jobs: int = 0
    #: Стадии, встречавшиеся в истории, включая попавшие только в кэш.
    observed: frozenset[str] = frozenset()
    #: Стадии, у которых есть хотя бы одно не-кэшированное наблюдение.
    fresh_stages: frozenset[str] = frozenset()

    @property
    def has_history(self) -> bool:
        """Есть ли хоть какие-то не-кэшированные наблюдения."""
        return self.samples > 0

    @property
    def has_fresh_coverage(self) -> bool:
        """Покрывает ли свежая история все наблюдённые стадии.

        Если какая-то стадия встречалась в прогонах только из кэша, её RTF
        недостоверен (≈ 0), и опираться на него нельзя — для процента тогда
        берутся равные веса, а ETA отдаёт ``None``, пока стадия впереди.
        """
        if not self.observed:
            # Профиль без метаданных (собран вручную/фасадом) считаем покрытым,
            # если есть наблюдения: его ``weights`` — источник истины.
            return self.samples > 0
        return all(stage in self.fresh_stages for stage in self.observed)

    def weight(self, stage: str) -> float:
        """Вес стадии (``0.0`` для неизвестной стадии)."""
        return self.weights.get(stage, 0.0)


def build_profile(jobs: Iterable[Job]) -> StageProfile:
    """Собирает RTF-статистику по завершённым задачам.

    Учитываются только ``done``-задачи с известной длительностью аудио и
    непустыми ``stage_times``. Веса строятся **исключительно по
    не-кэшированным** наблюдениям (``cached=False``): кэшированная стадия
    занимает ≈ 0 с, и её учёт занижал бы ETA (#33). Наблюдённые стадии
    запоминаются отдельно, чтобы отличить «стадия есть в конвейере, но
    стоимость неизвестна» от «стадии в конвейере нет».

    Если свежих наблюдений нет вовсе, возвращаются равные веса
    (``samples == 0``) — тогда ETA не считается. Для стадий, которых нет в
    свежей истории, берётся консервативный запасной вес — медиана свежих
    медиан стадий (а не медиана отдельных замеров, которую тянут вниз дешёвые
    стадии вроде ``merge``/``export``).
    """
    fresh: dict[str, list[float]] = {}
    observed: set[str] = set()
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
            if timing.stage not in STAGES:
                continue
            observed.add(timing.stage)
            if timing.cached:
                # Кэш не несёт информации о реальной стоимости стадии.
                continue
            fresh.setdefault(timing.stage, []).append(
                max(float(timing.seconds), 0.0) / float(duration)
            )

    samples = sum(len(values) for values in fresh.values())
    if not fresh:
        # Нет истории реальных прогонов — равные веса (процент по числу стадий).
        return StageProfile(
            weights=dict.fromkeys(STAGES, 1.0),
            samples=0,
            jobs=completed,
            observed=frozenset(observed),
        )

    fresh_median = {
        stage: max(statistics.median(values), MIN_WEIGHT)
        for stage, values in fresh.items()
    }
    # Запасной вес — типичная (медианная) стоимость известной стадии, а не
    # медиана всех замеров: иначе дешёвые стадии задают оптимистичный ориентир.
    fallback = max(statistics.median(fresh_median.values()), MIN_WEIGHT)
    weights = {stage: fresh_median.get(stage, fallback) for stage in STAGES}

    return StageProfile(
        weights=weights,
        samples=samples,
        jobs=completed,
        observed=frozenset(observed),
        fresh_stages=frozenset(fresh_median),
    )


def expected_seconds(profile: StageProfile, stage: str, duration: float | None) -> float | None:
    """Ожидаемое время стадии для аудио длительности ``duration`` (или ``None``)."""
    if not profile.has_history or duration is None or duration <= 0:
        return None
    weight = profile.weight(stage)
    if weight <= 0:
        return None
    return weight * duration


def _current_fraction(
    job: Job, weight: Callable[[str], float], duration: float | None
) -> float:
    """Доля текущей стадии: ``fraction``, иначе — по прошедшему времени.

    Для стадий без собственного ``fraction`` (диаризация, экспорт и т.п.)
    используем интерполяцию по ``stage_elapsed`` относительно ожидаемой
    длительности, чтобы общий процент рос плавно, а не ступенями. ``weight`` —
    выбранный источник весов (свежий профиль или, при непокрытой истории,
    равные веса), чтобы доля считалась по той же шкале, что и процент.
    """
    if job.fraction is not None:
        return _clamp(float(job.fraction), 0.0, 1.0)
    if job.stage is None or duration is None or duration <= 0:
        return 0.0
    stage_weight = weight(job.stage)
    if stage_weight <= 0:
        return 0.0
    elapsed = job.stage_elapsed
    if elapsed is None:
        return 0.0
    return _clamp(elapsed / (stage_weight * duration), 0.0, 0.99)


def _stage_plan(job: Job, profile: StageProfile) -> list[tuple[str, float, float]] | None:
    """План стадий: ``(стадия, ожидаемые секунды, доля остатка)``.

    Пройденные (по ``stage_times`` или по позиции текущей стадии) пропускаются.
    Для текущей стадии доля остатка меньше единицы.

    Возвращает ``None``, если среди оставшихся стадий есть наблюдённая ранее
    только из кэша: её реальная стоимость неизвестна, и любая цифра была бы
    заведомо заниженной (#33).
    """
    duration = job.duration
    completed = {timing.stage for timing in job.stage_times}
    current = job.stage
    current_index = STAGES.index(current) if current in STAGES else -1
    fraction = _current_fraction(job, profile.weight, duration) if current in STAGES else 0.0

    plan: list[tuple[str, float, float]] = []
    for index, stage in enumerate(STAGES):
        if stage in completed or (current_index >= 0 and index < current_index):
            continue
        if stage in profile.observed and stage not in profile.fresh_stages:
            # Стадия есть в конвейере, но известна лишь по кэшу — не оцениваем.
            return None
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

    Если свежая история не покрывает наблюдённые стадии (был прогон, где часть
    стадий взята из кэша), искажённые веса не используются: берутся равные —
    так процент растёт плавно по стадиям и не «перескакивает» на ранних этапах.
    """
    if job.status == STATUS_DONE:
        return 100.0

    # Непокрытая история: равные веса — нейтральная, не вводящая в заблуждение
    # шкала (ETA в таком случае не показывается).
    weight: Callable[[str], float] = (
        profile.weight if profile.has_fresh_coverage else _equal_weight
    )

    total = sum(weight(stage) for stage in STAGES)
    if total <= 0:
        return 0.0

    completed = {timing.stage for timing in job.stage_times}
    current = job.stage
    current_index = STAGES.index(current) if current in STAGES else -1
    fraction = (
        _current_fraction(job, weight, job.duration) if current in STAGES else 0.0
    )

    done = 0.0
    for index, stage in enumerate(STAGES):
        if stage in completed or (current_index >= 0 and index < current_index):
            done += weight(stage)
        elif stage == current and not job.is_terminal:
            done += weight(stage) * fraction
    return round(_clamp(done / total * 100.0, 0.0, 100.0), 1)


def eta_seconds(job: Job, profile: StageProfile) -> float | None:
    """Остаток всего прогона в секундах или ``None`` (нет данных/истории).

    Для завершённой задачи — ``0.0``; для незапущенной — ``None``. Возвращает
    ``None`` и когда оставшаяся стадия известна лишь по кэшу — оценка была бы
    заведомо оптимистичной (#33).
    """
    if job.status == STATUS_DONE:
        return 0.0
    if job.status != STATUS_RUNNING or not profile.has_history:
        return None
    if job.duration is None or job.duration <= 0:
        return None
    plan = _stage_plan(job, profile)
    if plan is None:
        return None
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
    if plan is None:
        return None
    if not plan:
        return {}
    return {stage: round(expected * left, 1) for stage, expected, left in plan}


@dataclass(slots=True)
class Health:
    """«Здоровье» задачи: статус, давность обновления и причина (#36)."""

    status: str
    last_update_seconds: float | None
    #: Короткое объяснение по-русски: что отстаёт и почему (пусто для ``ok``).
    reason: str = ""


def _seconds_since(value: str | None, now: datetime) -> float | None:
    moment = _parse_iso(value)
    if moment is None:
        return None
    return max((now - moment).total_seconds(), 0.0)


def _format_clock(seconds: float) -> str:
    """Длительность в формате ``M:SS`` / ``H:MM:SS`` (для причины, #36/#43)."""
    total = max(round(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_seconds(seconds: float) -> str:
    """Округлённые секунды по-русски («45 с»)."""
    return f"{max(seconds, 0.0):.0f} с"


def _stage_note(stage: str | None) -> str:
    """Человекочитаемое «Стадия „…“» для причины замедления."""
    if not stage:
        return "Текущая стадия"
    return f"Стадия «{STAGE_TITLES.get(stage, stage)}»"


def explain_health(
    job: Job,
    profile: StageProfile,
    *,
    active: bool,
    now: datetime | None = None,
) -> str:
    """Причина замедления/зависания задачи для tooltip (#36).

    Возвращает пустую строку, если задача в норме. Никогда не показывает
    заведомо ложные числа: если свежей истории нет, прямо говорит об этом, а
    не подставляет выдуманное ожидаемое время.
    """
    if now is None:
        now = datetime.now(UTC)

    def why(stage: str | None) -> str:
        """Почему стадия может идти дольше: тяжёлая, длинная запись, нет истории."""
        parts: list[str] = []
        if stage in HEAVY_STAGES:
            parts.append("ресурсоёмкая стадия")
        if not profile.has_history:
            parts.append("нет свежей истории — оценка приблизительная")
        elif stage is not None and stage not in profile.fresh_stages:
            parts.append("нет свежих замеров этой стадии — оценка приблизительная")
        if job.duration is not None and job.duration >= LONG_AUDIO_SECONDS:
            parts.append(f"длинная запись ({_format_clock(job.duration)})")
        return "; ".join(parts)

    if job.status == STATUS_RUNNING and not active:
        return (
            "Задачу не ведёт воркер: обработка прервана "
            "(возможно, перезапуск сервера)"
        )

    expected = (
        expected_seconds(profile, job.stage, job.duration) if job.stage else None
    )
    elapsed = job.stage_elapsed
    last_update = _seconds_since(job.updated_at, now)
    if last_update is None:
        last_update = job.stage_elapsed

    stall_threshold = STALL_MIN_SECONDS
    slow_threshold: float | None = None
    if expected is not None and expected > 0:
        stall_threshold = max(STALL_MIN_SECONDS, STALL_FACTOR * expected)
        slow_threshold = max(SLOW_MIN_SECONDS, SLOW_FACTOR * expected)

    if last_update is not None and last_update >= stall_threshold:
        detail = (
            f"{_stage_note(job.stage)}: нет обновлений "
            f"{_format_seconds(last_update)} (порог {_format_seconds(stall_threshold)})"
        )
        extra = why(job.stage)
        return f"{detail} · {extra}" if extra else detail

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
        expected_elapsed = float(job.fraction) * expected
        if expected_elapsed > 0 and elapsed >= max(
            SLOW_MIN_SECONDS, SLOW_FACTOR * expected_elapsed
        ):
            slow = True

    if slow and expected is not None and expected > 0 and elapsed is not None:
        stage_slow_threshold = max(SLOW_MIN_SECONDS, SLOW_FACTOR * expected)
        if elapsed >= stage_slow_threshold:
            ratio = elapsed / expected
            detail = (
                f"{_stage_note(job.stage)}: идёт {_format_seconds(elapsed)} "
                f"против ожидаемых {_format_seconds(expected)} (×{ratio:.1f})"
            )
        elif job.fraction is not None:
            # Прогресс буксует: сравниваем с ожидаемым временем на пройденную
            # долю, а не с полной стадией — иначе «×» выглядела бы < 1.
            expected_elapsed = float(job.fraction) * expected
            if expected_elapsed > 0:
                ratio = elapsed / expected_elapsed
                detail = (
                    f"{_stage_note(job.stage)}: {float(job.fraction) * 100:.0f}% за "
                    f"{_format_seconds(elapsed)} вместо ≈{_format_seconds(expected_elapsed)} "
                    f"(×{ratio:.1f})"
                )
            else:
                detail = f"{_stage_note(job.stage)}: прогресс буксует ({_format_seconds(elapsed)})"
        else:
            detail = f"{_stage_note(job.stage)}: идёт {_format_seconds(elapsed)}"
        extra = why(job.stage)
        return f"{detail} · {extra}" if extra else detail

    return ""


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
        return Health(
            HEALTH_STALLED,
            last_update_rounded,
            explain_health(job, profile, active=False, now=now),
        )

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
        return Health(
            HEALTH_STALLED,
            last_update_rounded,
            explain_health(job, profile, active=active, now=now),
        )

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

    status = HEALTH_SLOW if slow else HEALTH_OK
    reason = "" if status == HEALTH_OK else explain_health(job, profile, active=active, now=now)
    return Health(status, last_update_rounded, reason)


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
        "reason": health.reason,
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
