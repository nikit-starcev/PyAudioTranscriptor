"""Отчёт о готовности и проверка HF-токена для веб-интерфейса.

Тонкая обвязка над :mod:`audio_transcriber.doctor`: те же быстрые проверки
(без загрузки моделей) отдаются в JSON, а также реализована мягкая проверка
доступа к gated-репозиторию pyannote по сохранённому токену (без тяжёлой
загрузки — только Hugging Face API).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber import doctor as doctor_module
from audio_transcriber.doctor import DoctorCheck
from audio_transcriber.utils.config_env import load_config_env
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.secrets import SecretsStore, effective_hf_token
from audio_transcriber.web.settings import SettingsStore

logger = logging.getLogger(__name__)

#: Gated-репозиторий модели диаризации, доступ к которому проверяет ``hf-check``.
PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"

#: Статусы пунктов отчёта.
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

#: Статусы проверки HF-токена.
HF_OK = "ok"
HF_NO_TOKEN = "no_token"
HF_NO_ACCESS = "no_access"
HF_ERROR = "error"

#: Время жизни кэша отчёта доктора (секунды). Проверки включают subprocess, а
#: ``/api/setup`` и ``/api/doctor`` вызываются при отрисовке шагов мастера,
#: поэтому короткий TTL гасит всплески, не показывая устаревшую картину.
DOCTOR_CACHE_TTL = 20.0


def check_status(check: DoctorCheck) -> str:
    """Статус пункта: ``ok`` / ``warn`` (некритично) / ``fail`` (критично)."""
    if check.ok:
        return STATUS_OK
    return STATUS_FAIL if check.critical else STATUS_WARN


def check_payload(check: DoctorCheck) -> dict[str, object]:
    """Пункт отчёта в виде JSON-объекта для API."""
    return {
        "id": check.key,
        "label": check.label,
        "status": check_status(check),
        "critical": check.critical,
        "detail": check.detail,
        "hint": check.hint,
        "links": list(check.links),
    }


def summarize(checks: list[DoctorCheck]) -> dict[str, int]:
    """Сводка: сколько проверок в порядке, предупреждений и провалов."""
    ok = sum(1 for check in checks if check.ok)
    warn = sum(1 for check in checks if not check.ok and not check.critical)
    fail = sum(1 for check in checks if not check.ok and check.critical)
    return {"ok": ok, "warn": warn, "fail": fail, "critical_failures": fail}


def build_doctor_env(
    settings_store: SettingsStore,
    secrets_store: SecretsStore,
    paths: WebPaths,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, dict[str, str]]:
    """Окружение для проверок: ``config.env`` + настройки веба + сохранённый токен.

    Приоритет тот же, что при сборке задачи: переменные окружения процесса, затем
    ``config.env``, затем сохранённые настройки веб-интерфейса. Токен из секретов
    перекрывает всё; каталог результатов — веб-каталог.
    """
    config_path, file_env = load_config_env()
    env = dict(os.environ if environ is None else environ)
    env.update(file_env)
    overrides = settings_store.load().env_overrides()
    env.update({key: value for key, value in overrides.items() if value})
    token = effective_hf_token(secrets_store, file_env)
    if token:
        env["HF_TOKEN"] = token
    env["OUTPUT_DIR"] = str(paths.results_dir)
    return config_path, env


def doctor_report(config_path: Path | None, env: Mapping[str, str]) -> dict[str, object]:
    """Собирает JSON-отчёт: список проверок и сводку."""
    checks = doctor_module.run_doctor(config_path, env)
    return {
        "checks": [check_payload(check) for check in checks],
        "summary": summarize(checks),
    }


class DoctorReportCache:
    """Кратковременный кэш отчёта доктора с TTL.

    Тяжёлые проверки (subprocess, файловая система) выполняются один раз на
    TTL; ``/api/doctor`` и ``/api/setup`` делят этот кэш. ``refresh`` всегда
    пересчитывает отчёт (для ``/api/doctor/recheck``), ``invalidate`` сбрасывает
    его при изменении настроек/секретов.
    """

    def __init__(
        self,
        *,
        ttl: float = DOCTOR_CACHE_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: dict[str, object] | None = None
        self._cached_at = 0.0

    def get(
        self, config_path: Path | None, env: Mapping[str, str]
    ) -> dict[str, object]:
        """Отчёт из кэша, если он свежий, иначе — новый расчёт."""
        now = self._clock()
        with self._lock:
            if self._cached is not None and (now - self._cached_at) < self._ttl:
                return self._cached
        report = doctor_report(config_path, env)
        with self._lock:
            self._cached = report
            self._cached_at = self._clock()
        return report

    def refresh(
        self, config_path: Path | None, env: Mapping[str, str]
    ) -> dict[str, object]:
        """Принудительный пересчёт отчёта и обновление кэша."""
        report = doctor_report(config_path, env)
        with self._lock:
            self._cached = report
            self._cached_at = self._clock()
        return report

    def invalidate(self) -> None:
        """Сбрасывает кэш (например, после сохранения настроек/секретов)."""
        with self._lock:
            self._cached = None
            self._cached_at = 0.0


@dataclass(frozen=True, slots=True)
class HfAccessResult:
    """Результат мягкой проверки HF-токена."""

    status: str
    message: str
    account: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "message": self.message, "account": self.account}


def _redact(text: str, token: str | None) -> str:
    """Убирает токен из текста ошибки (на случай, если он туда попал)."""
    if token and token in text:
        return text.replace(token, "***")
    return text


def _shorten(text: str, limit: int = 300) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _hf_whoami(token: str) -> dict[str, object]:
    """Запрос ``whoami`` к Hugging Face API (токен не логируется)."""
    from huggingface_hub import HfApi

    return HfApi().whoami(token=token)


def _hf_repo_accessible(token: str, repo: str) -> None:
    """Проверяет доступ к репозиторию; бросает исключение при отказе."""
    from huggingface_hub import HfApi

    HfApi().model_info(repo, token=token)


def _is_access_denied(exc: Exception) -> bool:
    name = type(exc).__name__
    if "Gated" in name or "Forbidden" in name or "Unauthorized" in name:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def check_hf_access(token: str | None, *, repo: str = PYANNOTE_REPO) -> HfAccessResult:
    """Мягко проверяет токен: нет токена / нет доступа / ок (без загрузки модели)."""
    if not token:
        return HfAccessResult(
            HF_NO_TOKEN,
            "Токен Hugging Face не задан. Получите токен и сохраните его в настройках.",
        )
    try:
        info = _hf_whoami(token)
    except Exception as exc:  # noqa: BLE001 — внешний сервис: возвращаем мягкий статус
        logger.warning("Проверка HF-токена не удалась: %s", _shorten(_redact(str(exc), token)))
        return HfAccessResult(
            HF_ERROR,
            f"Токен недействителен или сеть недоступна: {_shorten(_redact(str(exc), token))}",
        )
    raw_account = info.get("name") or info.get("user") or info.get("fullname")
    account = raw_account if isinstance(raw_account, str) and raw_account else None
    try:
        _hf_repo_accessible(token, repo)
    except Exception as exc:  # noqa: BLE001 — внешний сервис: возвращаем мягкий статус
        if _is_access_denied(exc):
            return HfAccessResult(
                HF_NO_ACCESS,
                f"Токен действителен, но нет доступа к '{repo}'. "
                "Откройте страницу модели и примите условия использования.",
                account,
            )
        return HfAccessResult(
            HF_ERROR,
            f"Не удалось проверить доступ к '{repo}': {_shorten(_redact(str(exc), token))}",
            account,
        )
    suffix = f" (аккаунт {account})" if account else ""
    return HfAccessResult(HF_OK, f"Доступ к '{repo}' подтверждён{suffix}.", account)
