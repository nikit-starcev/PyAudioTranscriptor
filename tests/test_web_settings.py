"""Тесты REST API настроек веб-интерфейса (``/api/settings``).

Проверяются чтение, частичное сохранение и валидация форматов/путей, а также
наложение настроек на ``config.env`` при сборке конфигурации задачи.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
    DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_LINKAGE,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_GIGAAM_MODEL,
    DEFAULT_NEMO_SPEECH_MODEL,
)
from audio_transcriber.domain.enums import AsrBackend
from audio_transcriber.storage.glossary_builder import build_glossary
from audio_transcriber.storage.glossary_db import GlossaryDB
from audio_transcriber.web.app import create_app
from audio_transcriber.web.config import build_job_config
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.settings import default_settings, settings_from_mapping


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture
def client(web_paths: WebPaths) -> Iterator[TestClient]:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as test_client:
        yield test_client


def test_get_settings_shape(client: TestClient, web_paths: WebPaths) -> None:
    payload = client.get("/api/settings").json()

    assert {
        "glossary_enabled",
        "glossary_db",
        "voices_dir",
        "export_formats",
        "llm_enabled",
        "llm_summary",
        "denoise",
        "mark_overlap",
        "normalize_text",
        "clean_artifacts",
        "enable_correction",
        "protocol_auto",
        "word_timestamps",
        "input_dir",
        "output_dir",
        "glossary_db_path",
        "voices_dir_resolved",
    } <= set(payload)
    assert payload["input_dir"] == str(web_paths.input_dir)
    assert payload["output_dir"] == str(web_paths.results_dir)


def test_put_settings_persists(client: TestClient, web_paths: WebPaths, tmp_path: Path) -> None:
    response = client.put(
        "/api/settings",
        json={
            "glossary_enabled": False,
            "glossary_db": str(tmp_path / "glossary.db"),
            "export_formats": ["txt", "docx"],
            "llm_summary": False,
            "denoise": False,
            "protocol_auto": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["glossary_enabled"] is False
    assert body["export_formats"] == ["txt", "docx"]
    assert body["llm_summary"] is False
    assert body["denoise"] is False
    assert body["protocol_auto"] is True

    saved = client.get("/api/settings").json()
    assert saved["glossary_enabled"] is False
    assert saved["export_formats"] == ["txt", "docx"]
    assert (web_paths.settings_json).is_file()


def test_put_settings_partial_update(client: TestClient) -> None:
    before = client.get("/api/settings").json()
    client.put("/api/settings", json={"denoise": False})
    after = client.get("/api/settings").json()

    assert after["denoise"] is False
    # Незаданные поля не сбрасываются.
    assert after["llm_summary"] == before["llm_summary"]
    assert after["glossary_db"] == before["glossary_db"]


def test_settings_notifications_toggle_persists(client: TestClient) -> None:
    """#35: отдельный тумблер уведомлений читается и сохраняется."""
    assert client.get("/api/settings").json()["notifications"] is True

    response = client.put("/api/settings", json={"notifications": False})

    assert response.status_code == 200
    assert response.json()["notifications"] is False
    assert client.get("/api/settings").json()["notifications"] is False


def test_settings_deep_filter_binary_persists(client: TestClient) -> None:
    """#50: путь к бинарнику deep-filter настраивается в вебе."""
    assert client.get("/api/settings").json()["deep_filter_binary"] == "deep-filter"

    response = client.put(
        "/api/settings", json={"deep_filter_binary": "/opt/deep-filter"}
    )

    assert response.status_code == 200
    assert response.json()["deep_filter_binary"] == "/opt/deep-filter"
    assert client.get("/api/settings").json()["deep_filter_binary"] == "/opt/deep-filter"


def test_settings_word_timestamps_toggle_persists(client: TestClient) -> None:
    """#45: тумблер пословных таймстемпов по умолчанию включён и сохраняется."""
    assert client.get("/api/settings").json()["word_timestamps"] is True

    response = client.put("/api/settings", json={"word_timestamps": False})

    assert response.status_code == 200
    assert response.json()["word_timestamps"] is False
    assert client.get("/api/settings").json()["word_timestamps"] is False


def test_build_job_config_word_timestamps(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Изолируемся от реального config.env: проверяем именно overrides.
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    enabled = build_job_config(
        audio_file, output_dir=tmp_path / "out", data_dir=tmp_path / "data"
    )
    disabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"WORD_TIMESTAMPS": "false"},
    )

    assert enabled.word_timestamps is True
    assert disabled.word_timestamps is False


def test_put_settings_rejects_invalid_formats(client: TestClient) -> None:
    response = client.put("/api/settings", json={"export_formats": ["txt", "xyz"]})

    assert response.status_code == 400
    assert "формат" in response.json()["detail"]


def test_put_settings_rejects_empty_formats(client: TestClient) -> None:
    response = client.put("/api/settings", json={"export_formats": []})

    assert response.status_code == 400


def test_put_settings_rejects_missing_directory(
    client: TestClient, tmp_path: Path
) -> None:
    missing = tmp_path / "nope" / "glossary.db"

    response = client.put("/api/settings", json={"glossary_db": str(missing)})

    assert response.status_code == 400
    assert "не существует" in response.json()["detail"]


def test_disabled_glossary_not_applied(audio_file: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "g.db"
    with GlossaryDB(db_path) as db:
        db.add_entry("КИСУСС", variant="кисус", source="test")

    disabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"GLOSSARY_ENABLED": "false", "GLOSSARY_DB": str(db_path)},
    )
    enabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"GLOSSARY_ENABLED": "true", "GLOSSARY_DB": str(db_path)},
    )

    assert disabled.glossary_enabled is False
    assert build_glossary(disabled) is None
    assert enabled.glossary_enabled is True
    assert build_glossary(enabled) is not None


def test_build_job_config_reads_diarization_hyperparameters(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Изолируемся от реального config.env, чтобы проверить именно overrides.
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "DIARIZATION_MIN_DURATION_OFF": "0.9",
            "DIARIZATION_CLUSTERING_THRESHOLD": "0.6",
            "DIARIZATION_CLUSTERING_FB": "1.5",
        },
    )

    assert config.diarization_min_duration_off == pytest.approx(0.9)
    assert config.diarization_clustering_threshold == pytest.approx(0.6)
    assert config.diarization_clustering_fb == pytest.approx(1.5)


def test_build_job_config_diarization_defaults(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
    )

    assert config.diarization_min_duration_off == pytest.approx(0.5)
    assert config.diarization_clustering_threshold is None
    assert config.diarization_clustering_fb is None


def test_settings_include_llm_provider_fields(client: TestClient) -> None:
    payload = client.get("/api/settings").json()

    assert payload["llm_provider"] == "llama"
    assert payload["llm_base_url"] == ""
    assert payload["llm_model_name"] == ""
    assert payload["llm_api_key_set"] is False
    assert payload["llm_api_key_masked"] is None
    # Именно ключ, а не только флаг: значение секрета наружу не отдаётся.
    assert "llm_api_key" not in payload


def test_put_settings_persists_llm_provider(client: TestClient) -> None:
    response = client.put(
        "/api/settings",
        json={
            "llm_provider": "openai",
            "llm_base_url": "http://localhost:11434/v1",
            "llm_model_name": "llama3.1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["llm_provider"] == "openai"
    assert body["llm_base_url"] == "http://localhost:11434/v1"
    assert body["llm_model_name"] == "llama3.1"

    saved = client.get("/api/settings").json()
    assert saved["llm_provider"] == "openai"


def test_put_settings_rejects_unknown_provider(client: TestClient) -> None:
    response = client.put("/api/settings", json={"llm_provider": "anthropic"})

    assert response.status_code == 400
    assert "провайдер" in response.json()["detail"]


def test_put_settings_rejects_bad_base_url(client: TestClient) -> None:
    response = client.put("/api/settings", json={"llm_base_url": "not-a-url"})

    assert response.status_code == 400
    assert "LLM_BASE_URL" in response.json()["detail"]


def test_build_job_config_maps_external_llm(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "LLM_PROVIDER": "openai",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL_NAME": "llama3.1",
            "LLM_API_KEY": "sk-injected",
        },
    )

    assert config.llm_provider == "openai"
    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_model_name == "llama3.1"
    assert config.llm_api_key == "sk-injected"


def test_build_job_config_llm_defaults_to_llama(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(audio_file, output_dir=tmp_path / "out", data_dir=tmp_path / "data")

    assert config.llm_provider == "llama"
    assert config.llm_base_url is None
    assert config.llm_model_name is None
    assert config.llm_api_key is None


def test_llm_check_without_url(client: TestClient) -> None:
    payload = client.post("/api/llm/check", json={}).json()

    assert payload["status"] == "no_url"
    assert payload["models"] == []


def test_llm_check_ok(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_probe(base_url: str, *, api_key: str | None = None):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return True, "Доступно. Моделей: 1 (llama3.1)", ["llama3.1"]

    monkeypatch.setattr("audio_transcriber.web.app.probe_openai_server", fake_probe)
    client.put("/api/settings", json={"llm_base_url": "http://localhost:11434/v1"})

    payload = client.post(
        "/api/llm/check", json={"api_key": "sk-typed"}
    ).json()

    assert payload["status"] == "ok"
    assert payload["models"] == ["llama3.1"]
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"] == "sk-typed"


def test_llm_check_reports_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "audio_transcriber.web.app.probe_openai_server",
        lambda *_args, **_kwargs: (False, "Не удалось подключиться", []),
    )

    payload = client.post(
        "/api/llm/check", json={"base_url": "http://localhost:9999/v1"}
    ).json()

    assert payload["status"] == "error"
    assert "Не удалось" in payload["message"]


# --- Подготовка эталона голоса (#29) ------------------------------------------


def test_reference_prepare_options_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from audio_transcriber.web import config as web_config

    monkeypatch.setattr(
        web_config,
        "env_defaults",
        lambda: {
            "REFERENCE_PREPARE": "false",
            "ENROLLMENT_MIN_SAMPLE_SECONDS": "4",
            "ENROLLMENT_MAX_SAMPLE_SECONDS": "8",
            "REFERENCE_TARGET_DBFS": "-35",
        },
    )

    options = web_config.reference_prepare_options()

    assert options.enabled is False
    assert options.min_speech_seconds == pytest.approx(4.0)
    assert options.max_seconds == pytest.approx(8.0)
    assert options.target_dbfs == pytest.approx(-35.0)


def test_reference_prepare_options_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from audio_transcriber.web import config as web_config

    monkeypatch.setattr(web_config, "env_defaults", lambda: {})

    options = web_config.reference_prepare_options()

    assert options.enabled is True
    assert options.min_speech_seconds == pytest.approx(3.0)
    assert options.max_seconds == pytest.approx(10.0)
    assert options.target_dbfs == pytest.approx(-30.0)


# --- GigaAM v3 (onnx-asr) -----------------------------------------------------


def test_gigaam_settings_roundtrip() -> None:
    base = default_settings({})
    # Значение по умолчанию согласовано с config.defaults.
    assert base.gigaam_model == DEFAULT_GIGAAM_MODEL
    assert base.gigaam_model_path == ""
    assert base.gigaam_quantization == ""
    assert base.gigaam_vad is True

    merged = settings_from_mapping(
        {
            "gigaam_model": "gigaam-v3-ctc",
            "gigaam_model_path": "/models/gigaam",
            "gigaam_quantization": "int8",
            "gigaam_vad": False,
        },
        base=base,
    )
    assert merged.gigaam_model == "gigaam-v3-ctc"
    assert merged.gigaam_vad is False

    env = merged.env_overrides()
    assert env["GIGAAM_MODEL"] == "gigaam-v3-ctc"
    assert env["GIGAAM_MODEL_PATH"] == "/models/gigaam"
    assert env["GIGAAM_QUANTIZATION"] == "int8"
    assert env["GIGAAM_VAD"] == "false"


def test_gigaam_settings_from_env() -> None:
    settings = default_settings(
        {
            "GIGAAM_MODEL": "gigaam-v3-e2e-ctc",
            "GIGAAM_MODEL_PATH": "/models/gigaam",
            "GIGAAM_QUANTIZATION": "int8",
            "GIGAAM_VAD": "false",
        }
    )

    assert settings.gigaam_model == "gigaam-v3-e2e-ctc"
    assert settings.gigaam_model_path == "/models/gigaam"
    assert settings.gigaam_quantization == "int8"
    assert settings.gigaam_vad is False


def test_put_settings_persists_gigaam(client: TestClient, tmp_path: Path) -> None:
    model_dir = tmp_path / "gigaam"
    response = client.put(
        "/api/settings",
        json={
            "gigaam_model": "gigaam-v3-ctc",
            "gigaam_model_path": str(model_dir),
            "gigaam_quantization": "int8",
            "gigaam_vad": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["gigaam_model"] == "gigaam-v3-ctc"
    assert body["gigaam_model_path"] == str(model_dir)
    assert body["gigaam_quantization"] == "int8"
    assert body["gigaam_vad"] is False

    saved = client.get("/api/settings").json()
    assert saved["gigaam_model"] == "gigaam-v3-ctc"


def test_put_settings_accepts_web_settings_gigaam_defaults(client: TestClient) -> None:
    payload = client.get("/api/settings").json()

    assert payload["gigaam_model"] == DEFAULT_GIGAAM_MODEL
    assert payload["gigaam_model_path"] == ""
    assert payload["gigaam_vad"] is True


def test_build_job_config_maps_gigaam(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    model_dir = tmp_path / "gigaam"
    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "ASR_BACKEND": "gigaam",
            "GIGAAM_MODEL": "gigaam-v3-ctc",
            "GIGAAM_MODEL_PATH": str(model_dir),
            "GIGAAM_QUANTIZATION": "int8",
            "GIGAAM_VAD": "false",
        },
    )

    assert config.asr_backend == AsrBackend.GIGAAM
    assert config.gigaam_model == "gigaam-v3-ctc"
    assert config.gigaam_model_path == model_dir
    assert config.gigaam_quantization == "int8"
    assert config.gigaam_vad is False


# --- Движок диаризации / NeMo-Speech.cpp (#62) ------------------------------


def test_diarization_engine_settings_roundtrip() -> None:
    base = default_settings({})
    assert base.diarization_engine == "auto"
    assert base.nemo_speech_binary == "nemo-speech"
    assert base.nemo_speech_lib_path == ""
    assert base.nemo_speech_model == DEFAULT_NEMO_SPEECH_MODEL
    assert base.nemo_speech_device == "auto"

    merged = settings_from_mapping(
        {
            "diarization_engine": "nemo-speech",
            "nemo_speech_binary": "/opt/nemo/bin/nemo-speech",
            "nemo_speech_lib_path": "/opt/nemo/lib",
            "nemo_speech_model": "sortformer",
            "nemo_speech_device": "vulkan",
        },
        base=base,
    )

    assert merged.diarization_engine == "nemo-speech"
    assert merged.nemo_speech_device == "vulkan"

    env = merged.env_overrides()
    assert env["DIARIZATION_ENGINE"] == "nemo-speech"
    assert env["NEMO_SPEECH_BINARY"] == "/opt/nemo/bin/nemo-speech"
    assert env["NEMO_SPEECH_LIB_PATH"] == "/opt/nemo/lib"
    assert env["NEMO_SPEECH_MODEL"] == "sortformer"
    assert env["NEMO_SPEECH_DEVICE"] == "vulkan"


def test_diarization_engine_settings_from_env() -> None:
    settings = default_settings(
        {
            "DIARIZATION_ENGINE": "nemo-speech",
            "NEMO_SPEECH_BINARY": "/opt/nemo/bin/nemo-speech",
            "NEMO_SPEECH_LIB_PATH": "/opt/nemo/lib",
            "NEMO_SPEECH_MODEL": "sortformer",
            "NEMO_SPEECH_DEVICE": "VULKAN",
        }
    )

    assert settings.diarization_engine == "nemo-speech"
    assert settings.nemo_speech_binary == "/opt/nemo/bin/nemo-speech"
    assert settings.nemo_speech_lib_path == "/opt/nemo/lib"
    assert settings.nemo_speech_model == "sortformer"
    assert settings.nemo_speech_device == "vulkan"


def test_validate_settings_rejects_unknown_diarization_engine() -> None:
    from audio_transcriber.web.settings import SettingsError, validate_settings

    settings = default_settings({})
    settings.diarization_engine = "unknown"

    with pytest.raises(SettingsError):
        validate_settings(settings)


def test_validate_settings_rejects_unknown_nemo_device() -> None:
    from audio_transcriber.web.settings import SettingsError, validate_settings

    settings = default_settings({})
    settings.nemo_speech_device = "cuda"

    with pytest.raises(SettingsError):
        validate_settings(settings)


def test_build_job_config_maps_diarization_engine(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "DIARIZATION_ENGINE": "nemo-speech",
            "NEMO_SPEECH_BINARY": "/opt/nemo/bin/nemo-speech",
            "NEMO_SPEECH_LIB_PATH": str(lib_dir),
            "NEMO_SPEECH_MODEL": "sortformer",
            "NEMO_SPEECH_DEVICE": "vulkan",
        },
    )

    assert config.diarization_engine == "nemo-speech"
    assert config.nemo_speech_binary == "/opt/nemo/bin/nemo-speech"
    assert config.nemo_speech_lib_path == str(lib_dir)
    assert config.nemo_speech_model == "sortformer"
    assert config.nemo_speech_device == "vulkan"


# --- Оценщик числа говорящих / маршрутизация auto (#64) ---------------------


def test_diarization_estimate_settings_roundtrip() -> None:
    base = default_settings({})
    assert base.diarization_estimate_enabled is True
    assert base.diarization_estimate_seconds == pytest.approx(
        DEFAULT_DIARIZATION_ESTIMATE_SECONDS
    )
    assert base.diarization_estimate_threshold == pytest.approx(
        DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD
    )
    assert base.diarization_estimate_model == DEFAULT_DIARIZATION_ESTIMATE_MODEL
    assert base.diarization_route_max_speakers == 4

    merged = settings_from_mapping(
        {
            "diarization_estimate_enabled": False,
            "diarization_estimate_seconds": 45,
            "diarization_estimate_threshold": "0.55",
            "diarization_estimate_model": "/models/campplus.onnx",
            "diarization_route_max_speakers": 8,
        },
        base=base,
    )

    assert merged.diarization_estimate_enabled is False
    assert merged.diarization_estimate_seconds == pytest.approx(45.0)
    assert merged.diarization_estimate_threshold == pytest.approx(0.55)
    assert merged.diarization_estimate_model == "/models/campplus.onnx"
    assert merged.diarization_route_max_speakers == 8

    env = merged.env_overrides()
    assert env["DIARIZATION_ESTIMATE_ENABLED"] == "false"
    assert env["DIARIZATION_ESTIMATE_SECONDS"] == "45.0"
    assert env["DIARIZATION_ESTIMATE_THRESHOLD"] == "0.55"
    assert env["DIARIZATION_ESTIMATE_MODEL"] == "/models/campplus.onnx"
    assert env["DIARIZATION_ROUTE_MAX_SPEAKERS"] == "8"


def test_diarization_estimate_settings_from_env() -> None:
    settings = default_settings(
        {
            "DIARIZATION_ESTIMATE_ENABLED": "false",
            "DIARIZATION_ESTIMATE_SECONDS": "20",
            "DIARIZATION_ESTIMATE_THRESHOLD": "0.65",
            "DIARIZATION_ESTIMATE_MODEL": "custom.onnx",
            "DIARIZATION_ROUTE_MAX_SPEAKERS": "6",
        }
    )

    assert settings.diarization_estimate_enabled is False
    assert settings.diarization_estimate_seconds == pytest.approx(20.0)
    assert settings.diarization_estimate_threshold == pytest.approx(0.65)
    assert settings.diarization_estimate_model == "custom.onnx"
    assert settings.diarization_route_max_speakers == 6


def test_validate_settings_rejects_bad_estimate_threshold() -> None:
    from audio_transcriber.web.settings import SettingsError, validate_settings

    settings = default_settings({})
    settings.diarization_estimate_threshold = 3.0

    with pytest.raises(SettingsError):
        validate_settings(settings)


def test_validate_settings_rejects_bad_route_max_speakers() -> None:
    from audio_transcriber.web.settings import SettingsError, validate_settings

    settings = default_settings({})
    settings.diarization_route_max_speakers = 0

    with pytest.raises(SettingsError):
        validate_settings(settings)


def test_validate_settings_rejects_bad_hybrid_linkage() -> None:
    from audio_transcriber.web.settings import SettingsError, validate_settings

    settings = default_settings({})
    settings.diarization_hybrid_linkage = "banana"

    with pytest.raises(SettingsError):
        validate_settings(settings)


def test_validate_settings_rejects_bad_hybrid_threshold() -> None:
    from audio_transcriber.web.settings import SettingsError, validate_settings

    settings = default_settings({})
    settings.diarization_hybrid_threshold = 2.5

    with pytest.raises(SettingsError):
        validate_settings(settings)


def test_build_job_config_maps_diarization_estimate(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "DIARIZATION_ESTIMATE_ENABLED": "false",
            "DIARIZATION_ESTIMATE_SECONDS": "25",
            "DIARIZATION_ESTIMATE_THRESHOLD": "0.6",
            "DIARIZATION_ESTIMATE_MODEL": "custom.onnx",
            "DIARIZATION_ROUTE_MAX_SPEAKERS": "8",
        },
    )

    assert config.diarization_estimate_enabled is False
    assert config.diarization_estimate_seconds == pytest.approx(25.0)
    assert config.diarization_estimate_threshold == pytest.approx(0.6)
    assert config.diarization_estimate_model == "custom.onnx"
    assert config.diarization_route_max_speakers == 8


# --- Гибридная диаризация (обход лимита 4 говорящих) -------------------------


def test_diarization_hybrid_settings_roundtrip() -> None:
    base = default_settings({})
    assert base.diarization_hybrid_enabled is True
    assert base.diarization_hybrid_window_seconds == pytest.approx(
        DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS
    )
    assert base.diarization_hybrid_overlap_seconds == pytest.approx(
        DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS
    )
    assert base.diarization_hybrid_min_speaker_seconds == pytest.approx(
        DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS
    )
    assert base.diarization_hybrid_linkage == DEFAULT_DIARIZATION_HYBRID_LINKAGE
    assert base.diarization_hybrid_threshold == pytest.approx(
        DEFAULT_DIARIZATION_HYBRID_THRESHOLD
    )

    merged = settings_from_mapping(
        {
            "diarization_hybrid_enabled": False,
            "diarization_hybrid_window_seconds": 60,
            "diarization_hybrid_overlap_seconds": "1.5",
            "diarization_hybrid_min_speaker_seconds": 2.0,
            "diarization_hybrid_linkage": "COMPLETE",
            "diarization_hybrid_threshold": "1.1",
        },
        base=base,
    )

    assert merged.diarization_hybrid_enabled is False
    assert merged.diarization_hybrid_window_seconds == pytest.approx(60.0)
    assert merged.diarization_hybrid_overlap_seconds == pytest.approx(1.5)
    assert merged.diarization_hybrid_min_speaker_seconds == pytest.approx(2.0)
    assert merged.diarization_hybrid_linkage == "complete"
    assert merged.diarization_hybrid_threshold == pytest.approx(1.1)

    env = merged.env_overrides()
    assert env["DIARIZATION_HYBRID_ENABLED"] == "false"
    assert env["DIARIZATION_HYBRID_WINDOW_SECONDS"] == "60.0"
    assert env["DIARIZATION_HYBRID_OVERLAP_SECONDS"] == "1.5"
    assert env["DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS"] == "2.0"
    assert env["DIARIZATION_HYBRID_LINKAGE"] == "complete"
    assert env["DIARIZATION_HYBRID_THRESHOLD"] == "1.1"


def test_build_job_config_maps_hybrid(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    config = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={
            "DIARIZATION_HYBRID_ENABLED": "false",
            "DIARIZATION_HYBRID_WINDOW_SECONDS": "75",
            "DIARIZATION_HYBRID_OVERLAP_SECONDS": "3",
        },
    )

    assert config.diarization_hybrid_enabled is False
    assert config.diarization_hybrid_window_seconds == pytest.approx(75.0)
    assert config.diarization_hybrid_overlap_seconds == pytest.approx(3.0)


# --- Сохранение настроек диаризации через PUT /api/settings -------------------


def test_put_settings_persists_diarization(client: TestClient, tmp_path: Path) -> None:
    """Все поля диаризации (#62/#64) редактируются и сохраняются через API."""
    binary = tmp_path / "nemo-speech"
    lib_path = tmp_path / "nemo-speech-lib"

    response = client.put(
        "/api/settings",
        json={
            "diarization_engine": "hybrid",
            "nemo_speech_binary": str(binary),
            "nemo_speech_lib_path": str(lib_path),
            "nemo_speech_model": "nvidia/diar_streaming_sortformer_4spk-v2",
            "nemo_speech_device": "vulkan",
            "diarization_estimate_enabled": False,
            "diarization_estimate_seconds": 45,
            "diarization_estimate_threshold": 0.55,
            "diarization_estimate_model": "/models/campplus.onnx",
            "diarization_route_max_speakers": 8,
            "diarization_hybrid_enabled": False,
            "diarization_hybrid_window_seconds": 60,
            "diarization_hybrid_overlap_seconds": 1.5,
            "diarization_hybrid_min_speaker_seconds": 2.0,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["diarization_engine"] == "hybrid"
    assert body["nemo_speech_binary"] == str(binary)
    assert body["nemo_speech_lib_path"] == str(lib_path)
    assert body["nemo_speech_model"] == "nvidia/diar_streaming_sortformer_4spk-v2"
    assert body["nemo_speech_device"] == "vulkan"
    assert body["diarization_estimate_enabled"] is False
    assert body["diarization_estimate_seconds"] == pytest.approx(45.0)
    assert body["diarization_estimate_threshold"] == pytest.approx(0.55)
    assert body["diarization_estimate_model"] == "/models/campplus.onnx"
    assert body["diarization_route_max_speakers"] == 8
    assert body["diarization_hybrid_enabled"] is False
    assert body["diarization_hybrid_window_seconds"] == pytest.approx(60.0)
    assert body["diarization_hybrid_overlap_seconds"] == pytest.approx(1.5)
    assert body["diarization_hybrid_min_speaker_seconds"] == pytest.approx(2.0)

    saved = client.get("/api/settings").json()
    assert saved["diarization_engine"] == "hybrid"
    assert saved["nemo_speech_binary"] == str(binary)
    assert saved["diarization_estimate_seconds"] == pytest.approx(45.0)
    assert saved["diarization_route_max_speakers"] == 8
    assert saved["diarization_hybrid_min_speaker_seconds"] == pytest.approx(2.0)


def test_put_settings_rejects_bad_hybrid_overlap(client: TestClient) -> None:
    """Перекрытие больше окна — 400 из валидации (поле не «проглатывается»)."""
    response = client.put(
        "/api/settings",
        json={
            "diarization_hybrid_window_seconds": 10,
            "diarization_hybrid_overlap_seconds": 12,
        },
    )

    assert response.status_code == 400
    assert "DIARIZATION_HYBRID_OVERLAP_SECONDS" in response.json()["detail"]


# --- Автоисправление опечаток (стадия correction, pymorphy3) ------------------


def test_enable_correction_settings_roundtrip() -> None:
    base = default_settings({})
    # По умолчанию стадия выключена (как ENABLE_CORRECTION=false в config.env).
    assert base.enable_correction is False

    merged = settings_from_mapping({"enable_correction": True}, base=base)
    assert merged.enable_correction is True

    env = merged.env_overrides()
    assert env["ENABLE_CORRECTION"] == "true"


def test_enable_correction_settings_from_env() -> None:
    assert default_settings({"ENABLE_CORRECTION": "true"}).enable_correction is True
    assert default_settings({"ENABLE_CORRECTION": "false"}).enable_correction is False
    assert default_settings({}).enable_correction is False


def test_put_settings_persists_enable_correction(client: TestClient) -> None:
    """PUT реально сохраняет ``enable_correction`` (защита от pydantic-дропа)."""
    assert client.get("/api/settings").json()["enable_correction"] is False

    response = client.put("/api/settings", json={"enable_correction": True})

    assert response.status_code == 200, response.text
    assert response.json()["enable_correction"] is True
    assert client.get("/api/settings").json()["enable_correction"] is True

    # Обратное выключение тоже сохраняется именно как false.
    response = client.put("/api/settings", json={"enable_correction": False})

    assert response.status_code == 200, response.text
    assert client.get("/api/settings").json()["enable_correction"] is False


def test_build_job_config_maps_enable_correction(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    disabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"ENABLE_CORRECTION": "false"},
    )
    enabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"ENABLE_CORRECTION": "true"},
    )

    assert disabled.enable_correction is False
    assert enabled.enable_correction is True



# --- #88: полнота SettingsUpdate (merge_same_name_speakers и гибрид) -------


def test_put_settings_persists_merge_same_name_speakers(client: TestClient) -> None:
    """#88: ``merge_same_name_speakers`` не должен молча теряться при PUT."""
    assert client.get("/api/settings").json()["merge_same_name_speakers"] is True

    response = client.put(
        "/api/settings", json={"merge_same_name_speakers": False}
    )

    assert response.status_code == 200
    assert response.json()["merge_same_name_speakers"] is False
    assert client.get("/api/settings").json()["merge_same_name_speakers"] is False


def test_put_settings_persists_hybrid_linkage_and_threshold(
    client: TestClient,
) -> None:
    """#88: linkage/порог гибрида сохраняются через PUT и читаются обратно."""
    response = client.put(
        "/api/settings",
        json={
            "diarization_hybrid_linkage": "complete",
            "diarization_hybrid_threshold": 1.1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["diarization_hybrid_linkage"] == "complete"
    assert body["diarization_hybrid_threshold"] == pytest.approx(1.1)

    saved = client.get("/api/settings").json()
    assert saved["diarization_hybrid_linkage"] == "complete"
    assert saved["diarization_hybrid_threshold"] == pytest.approx(1.1)


# --- #87: SSRF в проверке внешней LLM --------------------------------------

def test_llm_check_rejects_non_http_scheme(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[int] = []
    monkeypatch.setattr(
        "audio_transcriber.web.app.probe_openai_server",
        lambda *_a, **_k: (called.append(1), (True, "ok", []))[1],
    )

    payload = client.post(
        "/api/llm/check", json={"base_url": "file:///etc/passwd"}
    ).json()

    assert payload["status"] == "error"
    assert called == []


def test_llm_check_rejects_link_local_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[int] = []
    monkeypatch.setattr(
        "audio_transcriber.web.app.probe_openai_server",
        lambda *_a, **_k: (called.append(1), (True, "ok", []))[1],
    )

    payload = client.post(
        "/api/llm/check", json={"base_url": "http://169.254.169.254/v1"}
    ).json()

    assert payload["status"] == "error"
    assert called == []


def test_llm_check_allows_localhost(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_probe(base_url: str, *, api_key: str | None = None):
        captured["base_url"] = base_url
        return True, "Доступно.", ["m"]

    monkeypatch.setattr("audio_transcriber.web.app.probe_openai_server", fake_probe)

    payload = client.post(
        "/api/llm/check", json={"base_url": "http://localhost:11434/v1"}
    ).json()

    assert payload["status"] == "ok"
    assert captured["base_url"] == "http://localhost:11434/v1"


# --- #73: тумблер «протокол/резюме по завершении» (protocol_auto) -----------


def test_protocol_auto_defaults_to_false() -> None:
    """По умолчанию веб-задачи протокол не считают (кнопка в UI)."""
    assert default_settings({}).protocol_auto is False


def test_protocol_auto_settings_from_env() -> None:
    assert default_settings({"PROTOCOL_AUTO": "true"}).protocol_auto is True
    assert default_settings({"PROTOCOL_AUTO": "false"}).protocol_auto is False


def test_protocol_auto_settings_roundtrip() -> None:
    base = default_settings({})
    merged = settings_from_mapping({"protocol_auto": True}, base=base)
    assert merged.protocol_auto is True
    assert merged.env_overrides()["PROTOCOL_AUTO"] == "true"


def test_put_settings_persists_protocol_auto(client: TestClient) -> None:
    """PUT реально сохраняет ``protocol_auto`` (защита от pydantic-дропа)."""
    assert client.get("/api/settings").json()["protocol_auto"] is False

    enabled = client.put("/api/settings", json={"protocol_auto": True})
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["protocol_auto"] is True

    disabled = client.put("/api/settings", json={"protocol_auto": False})
    assert disabled.status_code == 200, disabled.text
    assert client.get("/api/settings").json()["protocol_auto"] is False


def test_build_job_config_maps_protocol_auto(
    audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("audio_transcriber.web.config.env_defaults", lambda: {})

    disabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"PROTOCOL_AUTO": "false"},
    )
    enabled = build_job_config(
        audio_file,
        output_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        overrides={"PROTOCOL_AUTO": "true"},
    )

    assert disabled.protocol_auto is False
    assert enabled.protocol_auto is True
