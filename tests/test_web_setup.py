"""Тесты плана мастера первого запуска (``/api/setup``).

Мастер — тонкая надстройка над ``doctor`` и каталогом моделей, поэтому проверки
идут без браузера: у отчёта диагностики и каталога моделей подменяются данные,
а шаги плана сверяются по статусам.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.doctor import DoctorCheck
from audio_transcriber.web import doctor_api
from audio_transcriber.web import setup as web_setup
from audio_transcriber.web.app import create_app
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.settings import WebSettings


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    monkeypatch.setattr(
        "audio_transcriber.web.doctor_api.load_config_env", lambda: (None, {})
    )


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


def _check(check_id: str, status: str) -> dict[str, object]:
    return {
        "id": check_id,
        "label": check_id,
        "status": status,
        "critical": status == "fail",
        "detail": "",
        "hint": "",
        "links": [],
    }


def _report(checks: list[dict[str, object]], critical_failures: int = 0) -> dict[str, object]:
    return {"checks": checks, "summary": {"critical_failures": critical_failures}}


def _models(*, present: set[str], required_ids: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {"id": model_id, "status": {"present": model_id in present}}
        for model_id in required_ids
    ]


def test_hardware_mapping() -> None:
    assert web_setup.current_hardware(WebSettings(asr_backend="whisper-cpp")) == "amd"
    assert web_setup.current_hardware(WebSettings(device="cuda")) == "nvidia"
    assert web_setup.current_hardware(WebSettings(device="cpu")) == "cpu"
    assert web_setup.current_hardware(WebSettings()) == "cpu"
    assert len(web_setup.HARDWARE_OPTIONS) == 3
    assert web_setup.hardware_option("nvidia") is not None
    assert web_setup.hardware_option("unknown") is None


def test_required_models_depend_on_backend_and_llm() -> None:
    assert web_setup.required_model_ids(WebSettings()) == ["pyannote-community-1"]
    settings = WebSettings(asr_backend="whisper-cpp", llm_enabled=True)
    assert web_setup.required_model_ids(settings) == [
        "whisper-large-v3-turbo",
        "qwen2.5-7b-instruct-q4_k_m",
        "pyannote-community-1",
    ]


def test_binary_requirements() -> None:
    assert web_setup.binary_requirements(WebSettings()) == []
    requirements = web_setup.binary_requirements(
        WebSettings(asr_backend="whisper-cpp", llm_enabled=True)
    )
    assert [item["key"] for item in requirements] == ["whisper-cli", "llama-server"]
    assert all(item["needed"] for item in requirements)


def test_required_models_include_gigaam_only_for_gigaam_backend() -> None:
    for backend in ("faster-whisper", "whisper-cpp"):
        ids = web_setup.required_model_ids(WebSettings(asr_backend=backend))
        assert web_setup.GIGAAM_MODEL_ID not in ids

    ids = web_setup.required_model_ids(WebSettings(asr_backend="gigaam"))
    assert web_setup.GIGAAM_MODEL_ID in ids
    # Для gigaam не нужна модель whisper.cpp, а диаризация остаётся.
    assert "whisper-large-v3-turbo" not in ids
    assert "pyannote-community-1" in ids


def test_binary_requirements_include_onnx_asr_for_gigaam() -> None:
    assert web_setup.binary_requirements(WebSettings()) == []
    assert [
        item["key"] for item in web_setup.binary_requirements(WebSettings(asr_backend="gigaam"))
    ] == ["onnx-asr"]

    gigaam = web_setup.binary_requirements(WebSettings(asr_backend="gigaam"))[0]
    assert gigaam["needed"] is True
    assert gigaam["check_id"] == "dep:onnx_asr"
    assert "onnx-asr[cpu,hub]" in str(gigaam["instructions"])

    # Бинарные пункты не сломаны: у них check_id по-прежнему не задан вручную.
    whisper = web_setup.binary_requirements(WebSettings(asr_backend="whisper-cpp"))
    assert [item["key"] for item in whisper] == ["whisper-cli"]
    assert "check_id" not in whisper[0]


def test_setup_steps_gigaam_check_package_and_model() -> None:
    settings = WebSettings(asr_backend="gigaam")
    required = tuple(web_setup.required_model_ids(settings))
    report = _report([_check("hf_token", "ok"), _check("dep:onnx_asr", "fail")])

    plan = web_setup.build_setup_steps(
        settings=settings,
        report=report,
        models=_models(present=set(), required_ids=required),
        hf_token_set=False,
    )

    by_id = {step["id"]: step for step in plan["steps"]}
    assert web_setup.GIGAAM_MODEL_ID in plan["required_models"]
    assert web_setup.GIGAAM_MODEL_ID in plan["missing_models"]
    assert by_id["models"]["status"] == "todo"
    onnx = next(item for item in plan["binaries"] if item["key"] == "onnx-asr")
    assert onnx["available"] is False
    assert onnx["status"] == "fail"
    assert by_id["binaries"]["status"] == "todo"

    # Проверка доктора ``ok`` — пакет найден (check_id резолвится не в bin:).
    ready = web_setup.build_setup_steps(
        settings=settings,
        report=_report([_check("dep:onnx_asr", "ok")]),
        models=_models(present=set(required), required_ids=required),
        hf_token_set=False,
    )
    onnx_ok = next(item for item in ready["binaries"] if item["key"] == "onnx-asr")
    assert onnx_ok["available"] is True
    assert ready["missing_models"] == []


def test_steps_all_ok_when_ready() -> None:
    settings = WebSettings()
    required = tuple(web_setup.required_model_ids(settings))
    report = _report([_check("hf_token", "ok")], critical_failures=0)

    plan = web_setup.build_setup_steps(
        settings=settings,
        report=report,
        models=_models(present=set(required), required_ids=required),
        hf_token_set=False,
    )

    assert [step["id"] for step in plan["steps"]] == [
        "hardware",
        "hf_token",
        "models",
        "binaries",
        "readiness",
    ]
    assert all(step["status"] == "ok" for step in plan["steps"])
    assert plan["missing_models"] == []
    assert plan["hardware"]["current"] == "cpu"
    assert len(plan["hardware"]["options"]) == 3


def test_steps_report_missing_models_and_token() -> None:
    settings = WebSettings(asr_backend="whisper-cpp")
    required = tuple(web_setup.required_model_ids(settings))
    report = _report(
        [_check("hf_token", "fail"), _check("bin:whisper-cli", "fail")],
        critical_failures=1,
    )

    plan = web_setup.build_setup_steps(
        settings=settings,
        report=report,
        models=_models(present=set(), required_ids=required),
        hf_token_set=False,
    )

    by_id = {step["id"]: step for step in plan["steps"]}
    assert by_id["hf_token"]["status"] == "todo"
    assert by_id["models"]["status"] == "todo"
    assert by_id["binaries"]["status"] == "todo"
    assert by_id["readiness"]["status"] == "todo"
    assert plan["missing_models"] == list(required)
    assert plan["hf_token"]["required"] is True
    assert plan["binaries"][0]["available"] is False


def test_get_setup_endpoint_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    checks = [
        DoctorCheck(
            key="hf_token",
            label="Токен Hugging Face",
            ok=True,
            critical=False,
            detail="не требуется",
        ),
        DoctorCheck(
            key="bin:whisper-cli",
            label="Бинарник whisper-cli",
            ok=True,
            critical=False,
            detail="не требуется",
        ),
    ]
    monkeypatch.setattr(doctor_api.doctor_module, "run_doctor", lambda _path, _env: checks)

    payload = client.get("/api/setup").json()

    assert set(payload) >= {
        "hardware",
        "steps",
        "required_models",
        "missing_models",
        "binaries",
        "hf_token",
    }
    assert [step["id"] for step in payload["steps"]] == [
        "hardware",
        "hf_token",
        "models",
        "binaries",
        "readiness",
    ]
    assert len(payload["hardware"]["options"]) == 3
    assert payload["required_models"] == ["pyannote-community-1"]


def test_setup_shares_doctor_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    checks = [
        DoctorCheck(
            key="hf_token",
            label="Токен Hugging Face",
            ok=True,
            critical=False,
            detail="ok",
        )
    ]
    monkeypatch.setattr(
        doctor_api.doctor_module,
        "run_doctor",
        lambda _path, _env: (calls.append(1), checks)[1],
    )

    client.get("/api/setup")
    client.get("/api/setup")
    client.get("/api/doctor")

    assert len(calls) == 1
