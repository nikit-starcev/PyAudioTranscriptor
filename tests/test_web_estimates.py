"""Тесты оценок прогресса, ETA и «здоровья» задачи (#15/#24)."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from audio_transcriber.web.estimates import (
    HEALTH_OK,
    HEALTH_SLOW,
    HEALTH_STALLED,
    STAGES,
    StageEstimator,
    StageProfile,
    build_profile,
    classify_health,
    eta_by_stage,
    eta_seconds,
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


# --- прочее ---------------------------------------------------------------


def test_probe_duration_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert probe_duration(tmp_path / "missing.mp3") is None


def test_estimator_ttl_real_clock_smoke() -> None:
    store = _FakeStore([])
    estimator = StageEstimator(store, ttl=1000.0)
    before = time.monotonic()
    estimator.profile()
    assert time.monotonic() >= before
