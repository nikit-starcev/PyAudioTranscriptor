"""Тесты диагностики готовности и HF-токена в веб-интерфейсе.

Проверки ``run_doctor`` и обращение к Hugging Face подменяются, поэтому тесты
быстрые и не зависят от окружения. Отдельно проверяется, что токен хранится
как секрет (файл ``secrets.json`` с правами ``0600``) и никогда не попадает
в ответы API.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.doctor import DoctorCheck
from audio_transcriber.web import doctor_api
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.secrets import (
    SecretsStore,
    effective_hf_token,
    effective_llm_api_key,
    mask_hf_token,
    mask_secret,
)
from audio_transcriber.web.settings import SettingsStore


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture(autouse=True)
def isolate_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует тесты от реального ``config.env`` (в нём может быть токен)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    monkeypatch.setattr(doctor_api, "load_config_env", lambda: (None, {}))


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


def _sample_checks() -> list[DoctorCheck]:
    return [
        DoctorCheck(
            key="python",
            label="Python 3.14+",
            ok=True,
            critical=True,
            detail="текущая 3.14.7",
        ),
        DoctorCheck(
            key="bin:deep-filter",
            label="Бинарник deep-filter (денойз)",
            ok=False,
            critical=False,
            detail="не найден",
            hint="Задайте DEEP_FILTER_BINARY.",
        ),
        DoctorCheck(
            key="hf_token",
            label="Токен Hugging Face",
            ok=False,
            critical=True,
            detail="не задан",
            hint="Задайте HF_TOKEN.",
            links=("https://huggingface.co/settings/tokens",),
        ),
    ]


def test_doctor_report_shape_and_summary(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_api.doctor_module, "run_doctor", lambda _path, _env: _sample_checks())

    payload = client.get("/api/doctor").json()

    assert set(payload) == {"checks", "summary"}
    assert payload["summary"] == {"ok": 1, "warn": 1, "fail": 1, "critical_failures": 1}
    by_id = {check["id"]: check for check in payload["checks"]}
    assert set(by_id) == {"python", "bin:deep-filter", "hf_token"}
    assert by_id["python"]["status"] == "ok"
    assert by_id["bin:deep-filter"]["status"] == "warn"
    assert by_id["bin:deep-filter"]["critical"] is False
    assert by_id["hf_token"]["status"] == "fail"
    assert by_id["hf_token"]["critical"] is True
    assert by_id["hf_token"]["links"] == ["https://huggingface.co/settings/tokens"]
    for check in payload["checks"]:
        assert set(check) == {
            "id",
            "label",
            "status",
            "critical",
            "detail",
            "hint",
            "links",
        }


def test_doctor_recheck_calls_checks_again(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), _sample_checks())[1],
    )

    assert client.get("/api/doctor").status_code == 200
    assert client.post("/api/doctor/recheck").status_code == 200
    assert client.post("/api/doctor/recheck").status_code == 200

    assert len(calls) == 3


def test_doctor_cache_reuses_report_within_ttl(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), _sample_checks())[1],
    )

    client.get("/api/doctor")
    client.get("/api/doctor")
    # ``/api/setup`` делит тот же кэш, что и ``/api/doctor``.
    client.get("/api/setup")

    assert len(calls) == 1

    # Recheck всегда пересчитывает и обновляет кэш.
    client.post("/api/doctor/recheck")
    assert len(calls) == 2


def test_doctor_cache_invalidated_on_settings_change(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), _sample_checks())[1],
    )

    client.get("/api/doctor")
    assert len(calls) == 1

    assert client.put("/api/settings", json={"asr_backend": "whisper-cpp"}).status_code == 200
    client.get("/api/doctor")
    assert len(calls) == 2


def test_doctor_report_cache_expires_and_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), _sample_checks())[1],
    )
    now = [0.0]
    cache = doctor_api.DoctorReportCache(ttl=20.0, clock=lambda: now[0])

    cache.get(None, {})
    cache.get(None, {})
    assert len(calls) == 1

    now[0] = 25.0
    cache.get(None, {})
    assert len(calls) == 2

    cache.refresh(None, {})
    assert len(calls) == 3

    cache.invalidate()
    cache.get(None, {})
    assert len(calls) == 4


def test_doctor_cache_recomputes_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Параллельные запросы при истёкшем TTL считают отчёт один раз (stampede)."""
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow_run(_path: object, _env: object) -> list[DoctorCheck]:
        calls.append(1)
        started.set()
        release.wait(timeout=5.0)
        return _sample_checks()

    monkeypatch.setattr(doctor_api.doctor_module, "run_doctor", slow_run)
    cache = doctor_api.DoctorReportCache()
    results: list[dict[str, object]] = []

    def worker() -> None:
        results.append(cache.get(None, {}))

    first = threading.Thread(target=worker)
    first.start()
    assert started.wait(timeout=5.0)
    # Второй поток застаёт пустой кэш, но ждёт лок пересчёта, а не считает сам.
    second = threading.Thread(target=worker)
    second.start()
    release.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert len(calls) == 1
    assert len(results) == 2


def test_doctor_env_uses_secret_token_and_results_dir(
    client: TestClient, web_paths: WebPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "hf_secrettoken_value_123"
    client.put("/api/settings", json={"hf_token": secret})
    captured: dict[str, str] = {}

    def fake_run(_path: Path | None, env: dict[str, str]) -> list[DoctorCheck]:
        captured.update(env)
        return _sample_checks()

    monkeypatch.setattr(doctor_api.doctor_module, "run_doctor", fake_run)

    client.get("/api/doctor")

    assert captured["HF_TOKEN"] == secret
    assert captured["OUTPUT_DIR"] == str(web_paths.results_dir)


def test_hf_token_saved_as_secret_file_with_mode_0600(
    client: TestClient, web_paths: WebPaths
) -> None:
    secret = "hf_supersecrettoken999"

    response = client.put("/api/settings", json={"hf_token": secret})

    assert response.status_code == 200
    assert secret not in response.text
    body = response.json()
    assert body["hf_token_set"] is True
    assert body["hf_token_masked"] is not None
    assert secret not in str(body["hf_token_masked"])

    # Токен лежит в отдельном файле секретов, а не в settings.json.
    data = json.loads(web_paths.secrets_json.read_text(encoding="utf-8"))
    assert data["hf_token"] == secret
    mode = web_paths.secrets_json.stat().st_mode & 0o777
    assert mode == 0o600
    assert secret not in web_paths.settings_json.read_text(encoding="utf-8")

    # GET никогда не отдаёт сам токен — только флаг и маску.
    got = client.get("/api/settings")
    assert got.status_code == 200
    assert secret not in got.text
    payload = got.json()
    assert "hf_token" not in payload
    assert payload["hf_token_set"] is True


def test_hf_token_can_be_cleared(client: TestClient, web_paths: WebPaths) -> None:
    client.put("/api/settings", json={"hf_token": "hf_abc12345"})
    assert client.get("/api/settings").json()["hf_token_set"] is True

    response = client.put("/api/settings", json={"hf_token": ""})

    assert response.json()["hf_token_set"] is False
    assert SecretsStore(web_paths.secrets_json).get_hf_token() is None


def test_hf_check_no_token(client: TestClient) -> None:
    payload = client.post("/api/doctor/hf-check", json={}).json()

    assert payload["status"] == "no_token"
    assert payload["account"] is None


def test_hf_check_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    client.put("/api/settings", json={"hf_token": "hf_token_for_check"})
    monkeypatch.setattr(doctor_api, "_hf_whoami", lambda _token: {"name": "alice"})
    monkeypatch.setattr(doctor_api, "_hf_repo_accessible", lambda _token, _repo: None)

    payload = client.post("/api/doctor/hf-check", json={}).json()

    assert payload["status"] == "ok"
    assert payload["account"] == "alice"


def test_hf_check_no_access_to_gated_repo(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.put("/api/settings", json={"hf_token": "hf_token_for_check"})
    monkeypatch.setattr(doctor_api, "_hf_whoami", lambda _token: {"name": "bob"})

    class GatedRepoError(Exception):
        pass

    def deny(_token: str, _repo: str) -> None:
        raise GatedRepoError("access denied")

    monkeypatch.setattr(doctor_api, "_hf_repo_accessible", deny)

    payload = client.post("/api/doctor/hf-check", json={}).json()

    assert payload["status"] == "no_access"
    assert payload["account"] == "bob"


def test_hf_check_invalid_token_redacts_value(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "hf_badtoken_shouldnotleak"
    client.put("/api/settings", json={"hf_token": secret})

    def boom(_token: str) -> dict[str, object]:
        raise RuntimeError(f"401 Unauthorized for token {secret}")

    monkeypatch.setattr(doctor_api, "_hf_whoami", boom)

    payload = client.post("/api/doctor/hf-check", json={}).json()

    assert payload["status"] == "error"
    assert secret not in payload["message"]
    assert "***" in payload["message"]


def test_hf_check_accepts_token_from_request_body(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    def fake_whoami(token: str) -> dict[str, object]:
        seen["token"] = token
        return {"name": "carol"}

    monkeypatch.setattr(doctor_api, "_hf_whoami", fake_whoami)
    monkeypatch.setattr(doctor_api, "_hf_repo_accessible", lambda _token, _repo: None)

    payload = client.post("/api/doctor/hf-check", json={"token": "hf_from_body"}).json()

    assert payload["status"] == "ok"
    assert seen["token"] == "hf_from_body"


def test_mask_and_effective_token_helpers(web_paths: WebPaths) -> None:
    assert mask_hf_token(None) is None
    assert mask_hf_token("short") == "…"
    masked = mask_hf_token("hf_abcdefghijklmnop")
    assert masked is not None
    assert masked.startswith("hf_")
    assert masked.endswith("mnop")
    assert "abcdefghijkl" not in masked

    store = SecretsStore(web_paths.secrets_json)
    assert effective_hf_token(store, {}) is None
    assert effective_hf_token(store, {"HF_TOKEN": "hf_fromenv123"}) == "hf_fromenv123"
    store.set_hf_token("hf_secret123")
    assert effective_hf_token(store, {"HF_TOKEN": "hf_fromenv123"}) == "hf_secret123"


def test_build_doctor_env_prefers_secret(
    web_paths: WebPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = SecretsStore(web_paths.secrets_json)
    secrets.set_hf_token("hf_secret")
    settings_store = SettingsStore(web_paths.settings_json)

    config_path, env = doctor_api.build_doctor_env(
        settings_store, secrets, web_paths, environ={"HF_TOKEN": "hf_env"}
    )

    assert config_path is None
    assert env["HF_TOKEN"] == "hf_secret"
    assert env["OUTPUT_DIR"] == str(web_paths.results_dir)


def test_llm_api_key_saved_as_secret_file_with_mode_0600(
    client: TestClient, web_paths: WebPaths
) -> None:
    secret = "sk-supersecretllmkey9999"

    response = client.put("/api/settings", json={"llm_api_key": secret})

    assert response.status_code == 200
    assert secret not in response.text
    body = response.json()
    assert body["llm_api_key_set"] is True
    assert body["llm_api_key_masked"] is not None
    assert secret not in str(body["llm_api_key_masked"])

    # Ключ лежит в отдельном файле секретов, а не в settings.json.
    data = json.loads(web_paths.secrets_json.read_text(encoding="utf-8"))
    assert data["llm_api_key"] == secret
    mode = web_paths.secrets_json.stat().st_mode & 0o777
    assert mode == 0o600
    assert secret not in web_paths.settings_json.read_text(encoding="utf-8")

    # GET никогда не отдаёт сам ключ — только флаг и маску.
    got = client.get("/api/settings")
    assert got.status_code == 200
    assert secret not in got.text
    payload = got.json()
    assert "llm_api_key" not in payload
    assert payload["llm_api_key_set"] is True


def test_llm_api_key_can_be_cleared(client: TestClient, web_paths: WebPaths) -> None:
    client.put("/api/settings", json={"llm_api_key": "sk_abc12345"})
    assert client.get("/api/settings").json()["llm_api_key_set"] is True

    response = client.put("/api/settings", json={"llm_api_key": ""})

    assert response.json()["llm_api_key_set"] is False
    assert SecretsStore(web_paths.secrets_json).get_llm_api_key() is None


def test_mask_and_effective_llm_api_key_helpers(web_paths: WebPaths) -> None:
    assert mask_secret(None) is None
    assert mask_secret("short") == "…"
    masked = mask_secret("sk-abcdefghijklmnop")
    assert masked is not None
    assert masked.startswith("sk-")
    assert masked.endswith("mnop")

    store = SecretsStore(web_paths.secrets_json)
    assert effective_llm_api_key(store, {}) is None
    assert effective_llm_api_key(store, {"LLM_API_KEY": "sk_fromenv"}) == "sk_fromenv"
    store.set_llm_api_key("sk_secret")
    assert effective_llm_api_key(store, {"LLM_API_KEY": "sk_fromenv"}) == "sk_secret"
