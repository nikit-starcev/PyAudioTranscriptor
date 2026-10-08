"""Тесты оценок прогресса, ETA и «здоровья» задачи (#15/#24)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_transcriber.web.estimates import (
    DEFAULT_STAGE_RTF,
    HEALTH_OK,
    HEALTH_SLOW,
    HEALTH_STALLED,
    STAGES,
    StageEstimator,
    StageProfile,
    build_profile,
    classify_health,
    default_weights,
    estimate_run,
    eta_by_stage,
    eta_seconds,
    planned_stages,
    probe_duration,
    progress_percent,
)
from audio_transcriber.web.storage.jobs_db import STATUS_DONE, STATUS_RUNNING, Job
from audio_transcriber.web.timings import StageTiming


def _job(
    *,
    status: str = STATUS_RUNNING,
    stage: str | None = "asr",
    fraction: float | None = None,
    duration: float | None = None,
    stage_elapsed: float | None = None,
    stage_times: list[StageTiming] | None = None,
    updated_at: str | None = None,
) -> Job:
    """Собирает задачу; ``stage_elapsed`` задаётся через ``stage_started_at``."""
    started = None
    if stage_elapsed is not None:
        started = (datetime.now(UTC) - timedelta(seconds=stage_elapsed)).isoformat()
    return Job(
        id="job",
        source_path="/tmp/a.mp3",
        status=status,
        created_at=datetime.now(UTC).isoformat(),
        started_at=datetime.now(UTC).isoformat(),
        stage=stage,
        fraction=fraction,
        duration=duration,
        stage_started_at=started,
        stage_times=stage_times or [],
        updated_at=updated_at,
    )


def _profile(weights: dict[str, float], samples: int = 1) -> StageProfile:
    return StageProfile(weights=weights, samples=samples, jobs=1)


# --- план стадий по конфигурации задачи ----------------------------------


def _config(**flags: bool) -> SimpleNamespace:
    """Конфигурация с флагами стадий (значения по умолчанию — выключено)."""
    defaults = {
        "denoise": False,
        "diarization_enabled": False,
        "clean_artifacts": False,
        "enable_correction": False,
        "llm_enabled": False,
    }
    defaults.update(flags)
    return SimpleNamespace(**defaults)


def test_planned_stages_all_flags_in_pipeline_order() -> None:
    """Все флаги включены — полный список ровно в порядке выполнения."""
    config = _config(
        denoise=True,
        diarization_enabled=True,
        clean_artifacts=True,
        enable_correction=True,
        llm_enabled=True,
    )

    assert planned_stages(config) == list(STAGES)


def test_planned_stages_only_unconditional_when_flags_off() -> None:
    """``asr``/``merge``/``export`` выполняются всегда, остальные — по флагам."""
    assert planned_stages(_config()) == ["asr", "merge", "export"]


def test_planned_stages_respects_partial_flags_and_order() -> None:
    config = _config(denoise=True, diarization_enabled=True, clean_artifacts=True)

    assert planned_stages(config) == [
        "denoise",
        "asr",
        "diarization",
        "merge",
        "clean",
        "export",
    ]


# --- #15: статистика по прошлым прогонам ---------------------------------


def test_build_profile_uses_only_fresh_observations() -> None:
    first = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[
            StageTiming("asr", 50.0),
            StageTiming("diarization", 20.0, cached=True),
            StageTiming("merge", 10.0),
        ],
    )
    second = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[
            StageTiming("asr", 30.0),
            StageTiming("diarization", 10.0, cached=True),
        ],
    )

    profile = build_profile([first, second])

    assert profile.has_history is True
    assert profile.samples == 3  # два asr и один merge (не-кэшированные)
    assert profile.jobs == 2
    assert profile.observed == {"asr", "diarization", "merge"}
    assert profile.fresh_stages == {"asr", "merge"}
    # Свежая история не покрывает diarization (была только из кэша).
    assert profile.has_fresh_coverage is False
    # asr: медиана RTF (0.5, 0.3) = 0.4.
    assert profile.weight("asr") == pytest.approx(0.4)
    # merge: единственное наблюдение 0.1.
    assert profile.weight("merge") == pytest.approx(0.1)
    # diarization был только из кэша — его вес (_cached_ 0.15) НЕ используется;
    # берётся запасной вес: медиана известных стадий median(0.4, 0.1) = 0.25.
    assert profile.weight("diarization") == pytest.approx(0.25)
    # Никогда не наблюдённые стадии — тот же консервативный запасной вес.
    assert profile.weight("denoise") == pytest.approx(0.25)
    assert profile.weight("llm") == pytest.approx(0.25)


def test_build_profile_fallback_is_median_of_stage_medians() -> None:
    """Запасной вес — медиана медиан стадий, а не медиана всех замеров."""
    first = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[StageTiming("asr", 50.0), StageTiming("merge", 0.01)],
    )
    second = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[StageTiming("asr", 30.0), StageTiming("merge", 0.01)],
    )

    profile = build_profile([first, second])

    # Свежие медианы стадий: asr 0.4, merge 0.0001 → запасной вес
    # median(0.4, 0.0001) = 0.20005. Медиана всех замеров дала бы
    # median(0.5, 0.3, 0.0001, 0.0001) = 0.15005 — дешевле и оптимистичнее.
    assert profile.has_fresh_coverage is True
    assert profile.weight("clean") == pytest.approx(0.20005)
    assert profile.weight("correction") == pytest.approx(0.20005)


def test_build_profile_without_history_uses_equal_weights() -> None:
    running = _job(status=STATUS_RUNNING, duration=100.0)

    profile = build_profile([running])

    assert profile.has_history is False
    assert profile.samples == 0
    assert all(profile.weight(stage) == pytest.approx(1.0) for stage in STAGES)


def test_build_profile_ignores_jobs_without_duration_or_timings() -> None:
    no_duration = _job(status=STATUS_DONE, duration=None, stage_times=[StageTiming("asr", 1.0)])
    no_timings = _job(status=STATUS_DONE, duration=10.0, stage_times=[])

    profile = build_profile([no_duration, no_timings])

    assert profile.samples == 0
    assert profile.jobs == 0


# --- #24: сводный процент -------------------------------------------------


def test_progress_percent_weights_and_fraction() -> None:
    weights = dict.fromkeys(STAGES, 1.0)
    weights["asr"] = 3.0
    profile = _profile(weights)
    job = _job(stage="asr", fraction=0.5, stage_times=[])

    # denoise (индекс < asr) — пройдена: 1.0; asr: 3.0 * 0.5 = 1.5.
    # Итого 2.5 из 10 → 25%.
    assert progress_percent(job, profile) == pytest.approx(25.0)


def test_progress_percent_counts_cached_stage_as_passed() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))
    job = _job(
        stage="merge",
        fraction=None,
        stage_times=[
            StageTiming("denoise", 10.0),
            StageTiming("asr", 0.01, cached=True),
            StageTiming("diarization", 0.01, cached=True),
        ],
    )

    # Три стадии из восьми пройдены (независимо от того, что были в кэше).
    assert progress_percent(job, profile) == pytest.approx(37.5)


def test_progress_percent_interpolates_current_stage_without_fraction() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))
    job = _job(
        stage="merge",
        fraction=None,
        duration=10.0,
        stage_elapsed=5.0,
        stage_times=[StageTiming("denoise", 1.0), StageTiming("asr", 2.0)],
    )

    # merge: elapsed/duration = 0.5. denoise, asr, diarization пройдены.
    # (3 + 0.5) / 8 = 43.75 → 43.8.
    assert progress_percent(job, profile) == pytest.approx(43.8)


def test_progress_percent_done_is_hundred() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))
    job = _job(status=STATUS_DONE, stage="done", fraction=1.0)

    assert progress_percent(job, profile) == 100.0


def test_progress_percent_queued_is_zero() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))
    job = _job(status="queued", stage="queued", fraction=0.0)

    assert progress_percent(job, profile) == 0.0


# --- #15: ETA -------------------------------------------------------------


def test_eta_seconds_and_by_stage() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))
    job = _job(
        stage="merge",
        fraction=None,
        duration=10.0,
        stage_elapsed=5.0,
        stage_times=[
            StageTiming("denoise", 1.0),
            StageTiming("asr", 4.0),
            StageTiming("diarization", 2.0),
        ],
    )

    # merge (текущая, пройдено 0.5) + clean + correction + llm + export.
    assert eta_seconds(job, profile) == pytest.approx(45.0)
    assert eta_by_stage(job, profile) == {
        "merge": 5.0,
        "clean": 10.0,
        "correction": 10.0,
        "llm": 10.0,
        "export": 10.0,
    }


def test_eta_uses_history_weights() -> None:
    history = _job(
        status=STATUS_DONE,
        duration=10.0,
        stage_times=[
            StageTiming("denoise", 1.0),
            StageTiming("asr", 5.0),
            StageTiming("merge", 4.0),
        ],
    )
    profile = build_profile([history])
    # asr — 0.5 RTF, denoise — 0.1, merge — 0.4, остальные — запасной вес
    # (медиана известных = 0.4).
    job = _job(status=STATUS_RUNNING, stage="asr", duration=20.0, stage_elapsed=0.0)

    eta = eta_seconds(job, profile)

    assert eta is not None
    # asr: 0.5*20 = 10; остальные шесть стадий по 0.4*20 = 8 каждая.
    assert eta == pytest.approx(10.0 + 6 * 8.0)


def test_eta_none_without_history_or_duration() -> None:
    no_history = _profile(dict.fromkeys(STAGES, 1.0), samples=0)
    job = _job(stage="asr", duration=10.0, stage_elapsed=0.0)
    assert eta_seconds(job, no_history) is None

    profile = _profile(dict.fromkeys(STAGES, 1.0))
    no_duration = _job(stage="asr", duration=None)
    assert eta_seconds(no_duration, profile) is None


def test_eta_done_is_zero_and_queued_is_none() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))
    assert eta_seconds(_job(status=STATUS_DONE, stage="done"), profile) == 0.0
    assert eta_seconds(_job(status="queued", stage="queued"), profile) is None


def test_eta_none_when_remaining_stage_known_only_from_cache() -> None:
    """#33: кэшированный денойз/ASR не должны давать заниженный «~1 минуту»."""
    cached_run = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[
            StageTiming("denoise", 0.05, cached=True),
            StageTiming("asr", 0.004, cached=True),
            StageTiming("diarization", 5.0, cached=True),
            StageTiming("merge", 0.01),
            StageTiming("clean", 0.02),
            StageTiming("llm", 50.0),
            StageTiming("export", 2.0),
        ],
    )
    profile = build_profile([cached_run])
    assert profile.has_history is True
    assert profile.has_fresh_coverage is False

    job = _job(stage="denoise", fraction=0.05, duration=100.0)

    # denoise/asr/diarization известны только из кэша и ещё впереди — оценку
    # не показываем вовсе, а не заниженные секунды.
    assert eta_seconds(job, profile) is None
    assert eta_by_stage(job, profile) is None
    # Процент не перескакивает: равные веса, денойз только начался (0.05 / 8).
    assert progress_percent(job, profile) == pytest.approx(0.6)


def test_eta_available_once_cached_only_stages_are_passed() -> None:
    cached_run = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[
            StageTiming("denoise", 0.05, cached=True),
            StageTiming("asr", 0.004, cached=True),
            StageTiming("diarization", 5.0, cached=True),
            StageTiming("merge", 0.01),
            StageTiming("clean", 0.02),
            StageTiming("llm", 50.0),
            StageTiming("export", 2.0),
        ],
    )
    profile = build_profile([cached_run])
    job = _job(
        stage="llm",
        fraction=None,
        duration=100.0,
        stage_elapsed=10.0,
        stage_times=[
            StageTiming("denoise", 0.05, cached=True),
            StageTiming("asr", 0.004, cached=True),
            StageTiming("diarization", 5.0, cached=True),
            StageTiming("merge", 0.01),
            StageTiming("clean", 0.02),
        ],
    )

    # Кэш-стадии уже позади; оставшиеся llm/export имеют свежие веса — ETA есть.
    eta = eta_seconds(job, profile)
    by_stage = eta_by_stage(job, profile)
    assert eta is not None
    assert by_stage is not None
    assert "llm" in by_stage


# --- #24: здоровье --------------------------------------------------------


def test_health_ok_when_running_within_expected() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=5.0,
        updated_at=(now - timedelta(seconds=1)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_OK
    assert health.last_update_seconds == pytest.approx(1.0, abs=1.0)


def test_health_slow_when_stage_overruns() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    # Ожидание asr = 50 с, порог «медленно» = max(30, 100) = 100 с.
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=250.0,
        updated_at=(now - timedelta(seconds=1)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_SLOW


def test_health_stalled_when_no_recent_updates() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.001))
    now = datetime.now(UTC)
    # Ожидание asr = 0.1 с, порог «зависание» = max(60, 0.3) = 60 с.
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=5.0,
        updated_at=(now - timedelta(seconds=300)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_STALLED
    assert health.last_update_seconds == pytest.approx(300.0, abs=1.0)


def test_health_stalled_for_orphaned_running_job() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=1.0,
        updated_at=now.isoformat(),
    )

    health = classify_health(job, profile, active=False, now=now)

    assert health.status == HEALTH_STALLED


def test_health_falls_back_to_stage_elapsed_without_updated_at() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.001))
    job = _job(stage="asr", duration=100.0, stage_elapsed=120.0, updated_at=None)

    health = classify_health(job, profile, active=True)

    # Нет ``updated_at`` — возраст берётся из ``stage_elapsed`` (> порога 60 с).
    assert health.status == HEALTH_STALLED


# --- #36: причина замедления ---------------------------------------------


def test_health_slow_reason_names_stage_and_overrun() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    # Ожидание asr = 50 с, идёт 250 с (×5) — причина должна назвать стадию,
    # фактическое и ожидаемое время и ресурсоёмкость.
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=250.0,
        updated_at=(now - timedelta(seconds=1)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_SLOW
    assert "распознавание речи" in health.reason
    assert "250 с" in health.reason
    assert "50 с" in health.reason
    assert "×5.0" in health.reason
    assert "ресурсоёмкая стадия" in health.reason


def test_health_slow_reason_mentions_long_audio() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    # Запись длиннее порога (1 ч) — причина упоминает длинную запись.
    job = _job(
        stage="asr",
        duration=3600.0,
        stage_elapsed=4000.0,
        updated_at=(now - timedelta(seconds=1)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_SLOW
    assert "длинная запись (1:00:00)" in health.reason


def test_health_slow_reason_fraction_lag_not_misleading() -> None:
    """#36: при отставании по доле сравнение идёт с долей, а не со всей стадией."""
    profile = _profile(dict.fromkeys(STAGES, 1.0))  # ожидание asr = 100 с
    now = datetime.now(UTC)
    job = _job(
        stage="asr",
        duration=100.0,
        fraction=0.1,
        stage_elapsed=45.0,
        updated_at=(now - timedelta(seconds=1)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_SLOW
    assert "10%" in health.reason
    assert "вместо" in health.reason
    # Нельзя показывать «против ожидаемых 100 с» — это выглядело бы быстрее плана.
    assert "против ожидаемых" not in health.reason


def test_health_stalled_reason_is_honest_without_history() -> None:
    """Нет свежей истории — не подставляем выдуманное ожидаемое время (#36)."""
    profile = _profile(dict.fromkeys(STAGES, 1.0), samples=0)
    now = datetime.now(UTC)
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=120.0,
        updated_at=(now - timedelta(seconds=120)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_STALLED
    assert "нет свежей истории" in health.reason
    # Никаких «против ожидаемых … с» — истории для оценки нет.
    assert "против ожидаемых" not in health.reason


def test_health_reason_for_orphaned_job() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    job = _job(stage="asr", duration=100.0, stage_elapsed=1.0, updated_at=now.isoformat())

    health = classify_health(job, profile, active=False, now=now)

    assert health.status == HEALTH_STALLED
    assert "не ведёт воркер" in health.reason


def test_health_ok_has_empty_reason() -> None:
    profile = _profile(dict.fromkeys(STAGES, 0.5))
    now = datetime.now(UTC)
    job = _job(
        stage="asr",
        duration=100.0,
        stage_elapsed=5.0,
        updated_at=(now - timedelta(seconds=1)).isoformat(),
    )

    health = classify_health(job, profile, active=True, now=now)

    assert health.status == HEALTH_OK
    assert health.reason == ""


# --- StageEstimator: кэш статистики --------------------------------------


class _FakeStore:
    """Минимальная замена ``JobsDB`` со счётчиком обращений к ``list``."""

    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = jobs
        self.calls = 0

    def list(self) -> list[Job]:
        self.calls += 1
        return list(self.jobs)


def test_stage_estimator_caches_profile_and_invalidates() -> None:
    clock = {"now": 0.0}
    store = _FakeStore([])
    estimator = StageEstimator(store, ttl=10.0, clock=lambda: clock["now"])

    first = estimator.profile()
    clock["now"] = 5.0
    second = estimator.profile()
    assert first is second
    assert store.calls == 1

    clock["now"] = 11.0
    estimator.profile()
    assert store.calls == 2

    estimator.invalidate()
    estimator.profile()
    assert store.calls == 3


def test_stage_estimator_snapshot_fields() -> None:
    store = _FakeStore(
        [
            _job(
                status=STATUS_DONE,
                stage="done",
                duration=10.0,
                stage_times=[StageTiming("asr", 5.0)],
                updated_at=datetime.now(UTC).isoformat(),
            )
        ]
    )
    estimator = StageEstimator(store, ttl=0.0)
    running = _job(
        stage="asr",
        duration=20.0,
        fraction=0.25,
        stage_elapsed=1.0,
        updated_at=datetime.now(UTC).isoformat(),
    )

    snapshot = estimator.snapshot(running, active=True)

    assert set(snapshot) == {"progress_percent", "eta_seconds", "eta_by_stage", "health"}
    assert isinstance(snapshot["progress_percent"], float)
    assert snapshot["eta_seconds"] is not None
    health = snapshot["health"]
    assert isinstance(health, dict)
    assert health["status"] in {HEALTH_OK, HEALTH_SLOW, HEALTH_STALLED}
    # #36: причина замедления идёт в API/SSE, фронт показывает её в tooltip.
    assert "reason" in health
    assert isinstance(health["reason"], str)


# --- #107: примерная оценка до запуска ------------------------------------


def test_default_weights_follow_planned_stages_and_device() -> None:
    """Дефолтные веса — только планируемые стадии; GPU ускоряет ASR."""
    cpu = _config(denoise=True, diarization_enabled=True)
    weights = default_weights(cpu)

    assert set(weights) == set(planned_stages(cpu))
    assert weights["asr"] == pytest.approx(DEFAULT_STAGE_RTF["asr"])
    assert weights["denoise"] == pytest.approx(DEFAULT_STAGE_RTF["denoise"])

    gpu = _config(denoise=True, diarization_enabled=True)
    gpu.device = "cuda"
    assert default_weights(gpu)["asr"] == pytest.approx(
        DEFAULT_STAGE_RTF["asr"] * 0.5
    )
    # Денойз всегда на CPU — множитель GPU к нему не применяется.
    assert default_weights(gpu)["denoise"] == pytest.approx(
        DEFAULT_STAGE_RTF["denoise"]
    )


def test_default_weights_without_config_covers_all_stages() -> None:
    assert set(default_weights(None)) == set(STAGES)


def test_estimate_run_uses_history_when_fresh_coverage_is_full() -> None:
    history = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[
            StageTiming("asr", 30.0),
            StageTiming("merge", 1.0),
            StageTiming("export", 1.0),
        ],
    )
    profile = build_profile([history])

    estimate = estimate_run(profile, 200.0, ["asr", "merge", "export"])

    assert estimate.exact is True
    assert estimate.has_history is True
    assert estimate.by_stage == {"asr": 60.0, "merge": 2.0, "export": 2.0}
    assert estimate.seconds == pytest.approx(64.0)


def test_estimate_run_partial_history_is_approximate() -> None:
    """Стадия без свежих замеров делает оценку приблизительной, но не пустой."""
    history = _job(
        status=STATUS_DONE,
        duration=100.0,
        stage_times=[StageTiming("asr", 30.0)],
    )
    profile = build_profile([history])

    estimate = estimate_run(profile, 100.0, ["asr", "diarization"])

    assert estimate.exact is False
    assert estimate.has_history is True
    # diarization получает консервативный запасной вес профиля (медиана известных = 0.3).
    assert estimate.by_stage is not None
    assert estimate.by_stage["asr"] == pytest.approx(30.0)
    assert estimate.by_stage["diarization"] == pytest.approx(30.0)


def test_estimate_run_without_history_uses_defaults() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0), samples=0)

    estimate = estimate_run(
        profile, 100.0, ["asr", "merge", "export"], defaults=DEFAULT_STAGE_RTF
    )

    assert estimate.has_history is False
    assert estimate.exact is False
    assert estimate.seconds == pytest.approx(32.0)
    assert estimate.by_stage == {"asr": 30.0, "merge": 1.0, "export": 1.0}

    # Без явных defaults поведение не меняется.
    assert estimate_run(profile, 100.0, ["asr"]).seconds == pytest.approx(30.0)


def test_estimate_run_none_without_duration_or_stages() -> None:
    profile = _profile(dict.fromkeys(STAGES, 1.0))

    assert estimate_run(profile, None, ["asr"]).seconds is None
    assert estimate_run(profile, 0.0, ["asr"]).seconds is None
    assert estimate_run(profile, 10.0, []).seconds is None
    # Неизвестные стадии отбрасываются — остаётся пустой план.
    assert estimate_run(profile, 10.0, ["nope"]).seconds is None


def test_stage_estimator_estimate_returns_payload() -> None:
    store = _FakeStore([])
    estimator = StageEstimator(store, ttl=0.0)

    payload = estimator.estimate(120.0, ["asr", "merge", "export"], defaults=DEFAULT_STAGE_RTF)

    assert set(payload) == {"seconds", "by_stage", "exact", "has_history"}
    assert payload["has_history"] is False
    assert payload["exact"] is False
    assert isinstance(payload["seconds"], float)


# --- прочее ---------------------------------------------------------------


def test_probe_duration_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert probe_duration(tmp_path / "missing.mp3") is None


def test_estimator_ttl_real_clock_smoke() -> None:
    store = _FakeStore([])
    estimator = StageEstimator(store, ttl=1000.0)
    before = time.monotonic()
    estimator.profile()
    assert time.monotonic() >= before
