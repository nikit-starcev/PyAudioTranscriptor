"""Тесты самопроверки окружения (команда ``doctor``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_transcriber import doctor
from audio_transcriber.cli.app import app
from audio_transcriber.diarization import nemo_speech_assets

runner = CliRunner()


def _patch_modules(monkeypatch: pytest.MonkeyPatch, *, available: bool = True) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda _name: available)


def _patch_writable(monkeypatch: pytest.MonkeyPatch, *, ok: bool = True) -> None:
    monkeypatch.setattr(doctor, "_dir_writable", lambda _path: ok)


def _base_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "ASR_BACKEND": "faster-whisper",
        "DIARIZATION_ENABLED": "false",
        "OUTPUT_DIR": str(tmp_path / "out"),
    }
    env.update(overrides)
    return env


def _find(checks: list[doctor.DoctorCheck], key: str) -> doctor.DoctorCheck:
    return next(check for check in checks if check.key == key)


def _patch_vulkan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    devices: list[str] | None = None,
    library: bool = False,
    help_text: str | None = None,
    info: list[str] | None = None,
) -> None:
    """Подменяет все пробы Vulkan, чтобы не запускать реальные бинарники."""
    monkeypatch.setattr(doctor, "_vulkan_devices", lambda _binary, _lib: devices)
    monkeypatch.setattr(doctor, "_vulkan_library_present", lambda _lib, _binary: library)
    monkeypatch.setattr(doctor, "_whisper_help", lambda _binary, _lib: help_text)
    monkeypatch.setattr(doctor, "_vulkaninfo_devices", lambda: info)


def test_all_good_env_has_no_critical_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)

    checks = doctor.run_doctor(None, _base_env(tmp_path))

    assert not doctor.has_critical_failures(checks)
    assert "✓" in doctor.format_report(checks)


def test_missing_critical_dependency_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != "av")
    _patch_writable(monkeypatch)

    checks = doctor.run_doctor(None, _base_env(tmp_path))

    assert not _find(checks, "dep:av").ok
    assert _find(checks, "dep:av").critical
    assert doctor.has_critical_failures(checks)


def test_optional_dependency_missing_is_not_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != "df")
    _patch_writable(monkeypatch)

    checks = doctor.run_doctor(None, _base_env(tmp_path))

    deepfilter = _find(checks, "dep:df")
    assert not deepfilter.ok
    assert not deepfilter.critical


def test_sherpa_dependency_listed_and_optional_for_auto_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != "sherpa_onnx")
    _patch_writable(monkeypatch)

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="auto")
    checks = doctor.run_doctor(None, env)

    sherpa = _find(checks, "dep:sherpa_onnx")
    assert not sherpa.ok
    # Без sherpa-onnx auto безопасно уходит на pyannote — не критично.
    assert not sherpa.critical


def test_sherpa_dependency_absent_when_estimate_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)

    env = _base_env(
        tmp_path,
        DIARIZATION_ENABLED="true",
        DIARIZATION_ESTIMATE_ENABLED="false",
    )
    checks = doctor.run_doctor(None, env)

    assert all(check.key != "dep:sherpa_onnx" for check in checks)


def test_whisper_cpp_missing_binary_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: False)
    monkeypatch.setattr(doctor, "_is_file", lambda _path: False)
    monkeypatch.setattr(doctor, "_vulkan_devices", lambda _binary, _lib: None)

    env = _base_env(
        tmp_path,
        ASR_BACKEND="whisper-cpp",
        WHISPER_CPP_BINARY="/no/whisper-cli",
        WHISPER_CPP_MODEL="/no/model.bin",
    )
    checks = doctor.run_doctor(None, env)

    assert not _find(checks, "bin:whisper-cli").ok
    assert not _find(checks, "model:whisper").ok
    assert doctor.has_critical_failures(checks)


def test_vulkan_device_reported_when_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    _patch_vulkan(monkeypatch, devices=["Vulkan0: AMD Radeon (b)"])

    env = _base_env(tmp_path, ASR_BACKEND="whisper-cpp", WHISPER_CPP_BINARY="whisper-cli")
    checks = doctor.run_doctor(None, env)

    vulkan = _find(checks, "vulkan")
    assert vulkan.ok
    assert "Vulkan0" in vulkan.detail


def test_vulkan_detected_via_vulkaninfo_when_list_devices_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сценарий (а): --list-devices не поддержан, устройства дал vulkaninfo."""
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    _patch_vulkan(
        monkeypatch,
        devices=None,
        library=True,
        info=["AMD Radeon RX 590 Series (RADV POLARIS10)"],
    )

    env = _base_env(tmp_path, ASR_BACKEND="whisper-cpp", WHISPER_CPP_BINARY="whisper-cli")
    checks = doctor.run_doctor(None, env)

    vulkan = _find(checks, "vulkan")
    assert vulkan.ok
    assert not vulkan.critical
    assert "AMD Radeon RX 590 Series" in vulkan.detail


def test_vulkan_absent_warns_with_cpu_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сценарий (б): нет libggml-vulkan и билд без GPU-флагов → warn + CPU-hint."""
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    _patch_vulkan(
        monkeypatch,
        devices=None,
        library=False,
        help_text="usage: whisper-cli\n  -t N  --threads N\n",
    )

    env = _base_env(tmp_path, ASR_BACKEND="whisper-cpp", WHISPER_CPP_BINARY="whisper-cli")
    checks = doctor.run_doctor(None, env)

    vulkan = _find(checks, "vulkan")
    assert not vulkan.ok
    assert not vulkan.critical
    assert "CPU" in vulkan.hint


def test_vulkan_built_but_device_undetermined_is_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сценарий (в): Vulkan собран, устройства определить не удалось → ok."""
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    _patch_vulkan(monkeypatch, devices=None, library=True, info=None)

    env = _base_env(tmp_path, ASR_BACKEND="whisper-cpp", WHISPER_CPP_BINARY="whisper-cli")
    checks = doctor.run_doctor(None, env)

    vulkan = _find(checks, "vulkan")
    assert vulkan.ok
    assert not vulkan.critical
    assert "Vulkan собран" in vulkan.detail


def test_vulkan_gpu_flags_in_help_count_as_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPU-флаги в --help достаточно, чтобы считать Vulkan собранным."""
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: True)
    _patch_vulkan(
        monkeypatch,
        devices=None,
        library=False,
        help_text="  -ng, --no-gpu    disable GPU\n  -dev N, --device N   GPU device ID\n",
        info=None,
    )

    env = _base_env(tmp_path, ASR_BACKEND="whisper-cpp", WHISPER_CPP_BINARY="whisper-cli")
    checks = doctor.run_doctor(None, env)

    vulkan = _find(checks, "vulkan")
    assert vulkan.ok
    assert not vulkan.critical


def test_vulkan_binary_not_set_degrades_softly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сценарий (г): бинарник не задан/не найден → мягко, без исключений."""
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: False)
    _patch_vulkan(monkeypatch)

    env = _base_env(
        tmp_path,
        ASR_BACKEND="whisper-cpp",
        WHISPER_CPP_BINARY="",
        WHISPER_CPP_LIB_PATH="",
    )
    checks = doctor.run_doctor(None, env)

    vulkan = _find(checks, "vulkan")
    assert not vulkan.ok
    assert not vulkan.critical
    assert "не найден" in vulkan.detail


def test_vulkan_library_probe_handles_missing_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Хелпер ``_vulkan_library_present`` не падает на пустых/битых путях."""
    assert doctor._vulkan_library_present(None, "") is False
    assert doctor._vulkan_library_present("/no/such/dir", "no-such-binary") is False


def test_vulkaninfo_parser_reads_device_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/vulkaninfo")

    class _Proc:
        returncode = 0
        stdout = (
            "Devices:\n========\nGPU0:\n"
            "\tapiVersion = 1.4.354\n"
            "\tdeviceName = AMD Radeon RX 590 Series (RADV POLARIS10)\n"
            "\tdriverName = radv\n"
        )
        stderr = ""

    monkeypatch.setattr(doctor, "_run_command", lambda *_args, **_kwargs: _Proc())

    assert doctor._vulkaninfo_devices() == ["AMD Radeon RX 590 Series (RADV POLARIS10)"]


def test_missing_hf_token_is_critical_and_value_never_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true")
    checks = doctor.run_doctor(None, env)

    token_check = _find(checks, "hf_token")
    assert not token_check.ok
    assert token_check.critical
    assert doctor.has_critical_failures(checks)


def test_present_hf_token_is_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    secret = "hf_supersecretvalue123"

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", HF_TOKEN=secret)
    checks = doctor.run_doctor(None, env)
    report = doctor.format_report(checks)

    assert _find(checks, "hf_token").ok
    assert secret not in report


def test_unwritable_output_dir_is_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch, ok=False)

    checks = doctor.run_doctor(None, _base_env(tmp_path))

    output_check = _find(checks, "output_dir")
    assert not output_check.ok
    assert output_check.critical
    assert doctor.has_critical_failures(checks)


def test_config_env_check_reads_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    config_path = tmp_path / "config.env"
    config_path.write_text("OUTPUT_DIR=out\n", encoding="utf-8")
    config_path.chmod(0o600)

    checks = doctor.run_doctor(config_path, _base_env(tmp_path))

    assert _find(checks, "config_env").ok


@pytest.mark.skipif(os.name != "posix", reason="POSIX-права не применяются")
def test_config_env_wide_permissions_warn_not_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Права шире 0600 — предупреждение (не критично), код возврата не меняется."""
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    config_path = tmp_path / "config.env"
    config_path.write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")
    config_path.chmod(0o644)

    checks = doctor.run_doctor(config_path, _base_env(tmp_path))
    check = _find(checks, "config_env")

    assert not check.ok
    assert not check.critical
    assert "0644" in check.detail
    assert "chmod 600" in check.hint
    assert not doctor.has_critical_failures(checks)
    report = doctor.format_report(checks)
    assert "config.env" in report and "[не критично]" in report


@pytest.mark.skipif(os.name != "posix", reason="POSIX-права не применяются")
def test_config_env_owner_only_has_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    config_path = tmp_path / "config.env"
    config_path.write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")
    config_path.chmod(0o600)

    checks = doctor.run_doctor(config_path, _base_env(tmp_path))

    check = _find(checks, "config_env")
    assert check.ok
    assert "права" not in check.detail


def test_format_report_marks_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_available", lambda name: name != "av")
    _patch_writable(monkeypatch)

    report = doctor.format_report(doctor.run_doctor(None, _base_env(tmp_path)))

    assert "✗" in report
    assert "→" in report  # подсказка


# --- CLI-команда doctor ----------------------------------------------------


def _patch_cli(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *, modules: bool = True):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(doctor, "load_config_env", lambda: (None, env))
    monkeypatch.setattr(doctor, "_module_available", lambda _name: modules)
    monkeypatch.setattr(doctor, "_dir_writable", lambda _path: True)
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: False)
    monkeypatch.setattr(doctor, "_is_file", lambda _path: False)
    monkeypatch.setattr(doctor, "_is_dir", lambda _path: False)
    monkeypatch.setattr(doctor, "_vulkan_devices", lambda _binary, _lib: None)
    monkeypatch.setattr(doctor, "_vulkan_library_present", lambda _lib, _binary: False)
    monkeypatch.setattr(doctor, "_whisper_help", lambda _binary, _lib: None)
    monkeypatch.setattr(doctor, "_vulkaninfo_devices", lambda: None)


def test_doctor_cli_exit_zero_when_critical_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cli(monkeypatch, _base_env(tmp_path))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "✓" in result.stdout


def test_doctor_cli_exit_one_on_critical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_cli(monkeypatch, _base_env(tmp_path), modules=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "✗" in result.stdout


def test_doctor_cli_does_not_print_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "hf_clisecret999"
    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", HF_TOKEN=secret)
    _patch_cli(monkeypatch, env)

    result = runner.invoke(app, ["doctor"])

    assert secret not in result.stdout


# --- Движок диаризации NeMo-Speech.cpp (#62) ---------------------------------


def _patch_nemo(
    monkeypatch: pytest.MonkeyPatch,
    *,
    binary: bool,
    doctor_result: tuple[list[str], bool] | None,
    model_cached: bool = False,
) -> None:
    monkeypatch.setattr(doctor, "_binary_available", lambda _binary: binary)
    monkeypatch.setattr(doctor, "_nemo_speech_doctor", lambda _binary, _lib: doctor_result)
    status = nemo_speech_assets.NemoSpeechModelStatus(
        model="nvidia/diar_streaming_sortformer_4spk-v2",
        repo="nvidia/diar_streaming_sortformer_4spk-v2",
        present=model_cached,
        path="/cache/sortformer.q8_0.gguf" if model_cached else None,
        size=147_075_776 if model_cached else 0,
        files=("sortformer.q8_0.gguf",) if model_cached else (),
        source="cache",
    )
    monkeypatch.setattr(doctor, "_nemo_speech_model_status", lambda _model: status)


def test_nemo_speech_binary_and_gpu_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(
        monkeypatch,
        binary=True,
        doctor_result=(["AMD Radeon RX 590 Series"], True),
    )

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="nemo-speech")
    checks = doctor.run_doctor(None, env)

    assert _find(checks, "bin:nemo-speech").ok
    devices = _find(checks, "bin:nemo-speech-doctor")
    assert devices.ok
    assert "AMD Radeon" in devices.detail


def test_nemo_speech_missing_binary_is_critical_when_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(monkeypatch, binary=False, doctor_result=None)

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="nemo-speech")
    checks = doctor.run_doctor(None, env)

    binary_check = _find(checks, "bin:nemo-speech")
    assert not binary_check.ok
    assert binary_check.critical


def test_nemo_speech_missing_binary_not_critical_in_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(monkeypatch, binary=False, doctor_result=None)

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="auto")
    checks = doctor.run_doctor(None, env)

    binary_check = _find(checks, "bin:nemo-speech")
    assert binary_check.ok
    assert "pyannote" in binary_check.detail


def test_nemo_speech_model_cached_reflected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(
        monkeypatch, binary=True, doctor_result=(["GPU"], True), model_cached=True
    )

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="nemo-speech")
    checks = doctor.run_doctor(None, env)

    model = _find(checks, "model:nemo-speech")
    assert model.ok
    assert not model.critical
    assert "в кэше" in model.detail


def test_nemo_speech_model_absent_is_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(
        monkeypatch, binary=True, doctor_result=(["GPU"], True), model_cached=False
    )

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="nemo-speech")
    checks = doctor.run_doctor(None, env)

    model = _find(checks, "model:nemo-speech")
    assert not model.ok
    assert not model.critical
    assert "нет в кэше" in model.detail
    assert "Скачать модель" in model.hint


def test_nemo_speech_vulkan_without_gpu_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(monkeypatch, binary=True, doctor_result=([], True))

    env = _base_env(
        tmp_path,
        DIARIZATION_ENABLED="true",
        DIARIZATION_ENGINE="nemo-speech",
        NEMO_SPEECH_DEVICE="vulkan",
    )
    checks = doctor.run_doctor(None, env)

    devices = _find(checks, "bin:nemo-speech-doctor")
    assert not devices.ok
    assert not devices.critical
    assert "CPU" in devices.detail


def test_nemo_speech_skips_pyannote_dependency_and_token(
    tmp_path: Path,
) -> None:
    env = _base_env(
        tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="nemo-speech"
    )

    deps = {label: critical for _module, label, critical in doctor._dependencies(env)}
    assert deps["pyannote.audio (диаризация)"] is False
    assert deps["torch"] is False
    assert doctor._check_hf_token(env).ok


def test_nemo_speech_absent_when_diarization_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)

    checks = doctor.run_doctor(None, _base_env(tmp_path))

    assert all(check.key != "bin:nemo-speech" for check in checks)


# --- гибридная диаризация ----------------------------------------------------


def test_hybrid_missing_binary_is_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_modules(monkeypatch)
    _patch_writable(monkeypatch)
    _patch_nemo(monkeypatch, binary=False, doctor_result=None)

    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="hybrid")
    checks = doctor.run_doctor(None, env)

    binary_check = _find(checks, "bin:nemo-speech")
    assert not binary_check.ok
    assert binary_check.critical


def test_hybrid_skips_pyannote_dependency_and_token(tmp_path: Path) -> None:
    env = _base_env(tmp_path, DIARIZATION_ENABLED="true", DIARIZATION_ENGINE="hybrid")

    deps = {label: critical for _module, label, critical in doctor._dependencies(env)}
    assert deps["pyannote.audio (диаризация)"] is False
    assert deps["torch"] is False
    assert doctor._check_hf_token(env).ok


def test_hybrid_requires_sherpa_even_without_estimate(tmp_path: Path) -> None:
    env = _base_env(
        tmp_path,
        DIARIZATION_ENABLED="true",
        DIARIZATION_ENGINE="hybrid",
        DIARIZATION_ESTIMATE_ENABLED="false",
    )

    modules = {module for module, _label, _critical in doctor._dependencies(env)}
    assert "sherpa_onnx" in modules
