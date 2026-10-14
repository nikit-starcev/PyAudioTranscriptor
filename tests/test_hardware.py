"""Тесты определения железа и рекомендации настроек (#146)."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber import hardware


def _profile(
    *,
    total_ram_gb: float | None,
    available_ram_gb: float | None = None,
    gpu: hardware.GpuProfile | None = None,
    physical_cores: int | None = 8,
    logical_cores: int | None = 16,
    free_disk_gb: float | None = 100.0,
) -> hardware.HardwareProfile:
    return hardware.HardwareProfile(
        os_name="Linux",
        cpu=hardware.CpuProfile(
            name="Test CPU",
            logical_cores=logical_cores,
            physical_cores=physical_cores,
            max_freq_mhz=3200.0,
        ),
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        gpu=gpu or hardware.GpuProfile(),
        free_disk_gb=free_disk_gb,
    )


# --- парсинг проб -----------------------------------------------------------


def test_cpuinfo_parsing() -> None:
    text = (
        "processor\t: 0\n"
        "model name\t: AMD Ryzen 5 3600\n"
        "physical id\t: 0\n"
        "core id\t\t: 0\n"
        "cpu MHz\t\t: 3600.000\n"
        "\n"
        "processor\t: 1\n"
        "physical id\t: 0\n"
        "core id\t\t: 1\n"
        "cpu MHz\t\t: 4000.000\n"
    )

    assert hardware._cpuinfo_field(text, "model name") == "AMD Ryzen 5 3600"
    assert hardware._linux_physical_cores(text) == 2
    assert hardware._linux_max_freq_mhz(text) == pytest.approx(4000.0)
    assert hardware._cpuinfo_field(text, "missing") is None


def test_linux_physical_cores_without_ids_is_none() -> None:
    assert hardware._linux_physical_cores("processor\t: 0\nprocessor\t: 1\n") is None


def test_nvidia_gpu_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Proc:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3060, 12288\n"
        stderr = ""

    monkeypatch.setattr(hardware, "_run_command", lambda *_a, **_k: _Proc())

    assert hardware._nvidia_gpu() == ("NVIDIA GeForce RTX 3060", 12.0)


def test_nvidia_gpu_absent_when_command_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_command", lambda *_a, **_k: None)

    assert hardware._nvidia_gpu() is None


def test_rocm_gpu_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Proc:
        returncode = 0
        stdout = (
            '{"card0": {"Card series": "AMD Radeon RX 7900 XTX",'
            ' "VRAM Total Memory (B)": "25753026560"}}'
        )
        stderr = ""

    monkeypatch.setattr(hardware, "_run_command", lambda *_a, **_k: _Proc())

    result = hardware._rocm_smi_gpu()
    assert result is not None
    name, vram = result
    assert name == "AMD Radeon RX 7900 XTX"
    assert vram == pytest.approx(25753026560 / hardware._GB)


def test_vulkan_gpu_names_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: "/usr/bin/vulkaninfo")

    class _Proc:
        returncode = 0
        stdout = "Devices:\nGPU0:\n\tdeviceName = AMD Radeon RX 590 Series\n"
        stderr = ""

    monkeypatch.setattr(hardware, "_run_command", lambda *_a, **_k: _Proc())

    assert hardware._vulkan_gpu_names() == ["AMD Radeon RX 590 Series"]


def test_linux_drm_vram_parsing(tmp_path: Path) -> None:
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text("8589934592\n", encoding="utf-8")

    assert hardware._linux_drm_vram_gb(tmp_path) == pytest.approx(8.0)
    assert hardware._linux_drm_vram_gb(tmp_path / "missing") is None


def test_detect_hardware_combines_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: 16.0)
    monkeypatch.setattr(hardware, "_available_ram_gb", lambda: 12.0)
    monkeypatch.setattr(hardware, "_cpu_profile", lambda: hardware.CpuProfile(logical_cores=8))
    monkeypatch.setattr(
        hardware,
        "_gpu_profile",
        lambda: hardware.GpuProfile(present=True, vendor="nvidia", vram_gb=8.0),
    )
    monkeypatch.setattr(hardware, "_free_disk_gb", lambda _path: 50.0)

    profile = hardware.detect_hardware(Path("/tmp"))

    assert profile.total_ram_gb == 16.0
    assert profile.available_ram_gb == 12.0
    assert profile.gpu.present is True
    assert profile.free_disk_gb == 50.0
    payload = profile.as_dict()
    assert payload["os"]
    assert payload["cpu"]["logical_cores"] == 8


def test_detect_hardware_notes_undetermined_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_total_ram_gb", lambda: None)
    monkeypatch.setattr(hardware, "_available_ram_gb", lambda: None)
    monkeypatch.setattr(hardware, "_cpu_profile", lambda: hardware.CpuProfile(logical_cores=4))
    monkeypatch.setattr(hardware, "_gpu_profile", lambda: hardware.GpuProfile())
    monkeypatch.setattr(hardware, "_free_disk_gb", lambda _path: None)

    profile = hardware.detect_hardware()

    joined = " ".join(profile.notes)
    assert "RAM" in joined
    assert "диск" in joined


# --- рекомендация -----------------------------------------------------------


def test_recommend_weak_profile_without_gpu() -> None:
    rec = hardware.recommend_hardware(_profile(total_ram_gb=4.0))

    assert rec.profile == hardware.PROFILE_WEAK
    assert rec.asr_backend == "whisper-cpp"
    assert rec.device == "cpu"
    assert rec.whisper_model_id == "whisper-base"
    assert rec.diarization_engine == "sherpa"
    assert rec.denoise is False
    assert rec.llm_enabled is False
    assert rec.engine_isolation == "off"
    assert rec.found.startswith("нашли:")
    assert rec.proposed.startswith("предлагаем:")
    assert rec.why


def test_recommend_balanced_profile_without_gpu() -> None:
    rec = hardware.recommend_hardware(_profile(total_ram_gb=8.0))

    assert rec.profile == hardware.PROFILE_BALANCED
    assert rec.asr_backend == "whisper-cpp"
    assert rec.device == "cpu"
    assert rec.whisper_model_id == "whisper-small"
    assert rec.diarization_engine == "auto"
    assert rec.denoise is True
    assert rec.llm_enabled is False


def test_recommend_performance_profile_with_cuda() -> None:
    gpu = hardware.GpuProfile(present=True, vendor="nvidia", vram_gb=12.0, accelerators=("cuda",))
    rec = hardware.recommend_hardware(
        _profile(total_ram_gb=32.0, gpu=gpu),
        available_binaries=frozenset({"whisper-cli", "llama-server", "deep-filter"}),
    )

    assert rec.profile == hardware.PROFILE_PERFORMANCE
    assert rec.asr_backend == "faster-whisper"
    assert rec.device == "cuda"
    assert rec.whisper_model_id is None
    assert rec.llm_enabled is True
    assert rec.engine_isolation == "off"


def test_recommend_vulkan_gpu_uses_whisper_cpp_auto() -> None:
    gpu = hardware.GpuProfile(
        present=True, vendor="amd", name="Radeon", vram_gb=8.0, accelerators=("vulkan",)
    )
    rec = hardware.recommend_hardware(_profile(total_ram_gb=16.0, gpu=gpu))

    assert rec.asr_backend == "whisper-cpp"
    assert rec.device == "auto"
    assert rec.whisper_model_id == "whisper-large-v3-turbo"


def test_recommend_cuda_on_low_ram_enables_isolation() -> None:
    gpu = hardware.GpuProfile(present=True, vendor="nvidia", vram_gb=6.0, accelerators=("cuda",))
    rec = hardware.recommend_hardware(_profile(total_ram_gb=8.0, gpu=gpu))

    assert rec.asr_backend == "faster-whisper"
    assert rec.engine_isolation == "subprocess"


def test_recommend_disables_llm_without_llama_server() -> None:
    gpu = hardware.GpuProfile(present=True, vendor="nvidia", vram_gb=12.0, accelerators=("cuda",))
    rec = hardware.recommend_hardware(
        _profile(total_ram_gb=32.0, gpu=gpu),
        available_binaries=frozenset({"whisper-cli"}),
    )

    assert rec.llm_enabled is False
    assert any("llama-server" in line for line in rec.rationale)


def test_recommend_picks_available_whisper_model_not_heavier() -> None:
    rec = hardware.recommend_hardware(
        _profile(total_ram_gb=16.0),
        available_models=frozenset({"whisper-base", "whisper-small", "whisper-large-v3"}),
    )

    # Рекомендован turbo, но его нет; тяжёлый large-v3 не берём — лучший доступный small.
    assert rec.whisper_model_id == "whisper-small"
    assert rec.whisper_model_available is True


def test_recommend_marks_model_unavailable() -> None:
    # Скачаны только модели тяжелее рекомендованной (base) — берём base, помечаем «нет».
    rec = hardware.recommend_hardware(
        _profile(total_ram_gb=4.0), available_models=frozenset({"whisper-large-v3"})
    )

    assert rec.whisper_model_id == "whisper-base"
    assert rec.whisper_model_available is False
    assert any("не скачана" in line for line in rec.rationale)


def test_recommendation_apply_matches_profile() -> None:
    rec = hardware.recommend_hardware(_profile(total_ram_gb=32.0))
    apply = rec.apply()

    assert apply["asr_backend"] == rec.asr_backend
    assert apply["engine_isolation"] == rec.engine_isolation
    assert apply["diarization_hybrid_enabled"] is True
    assert apply["diarization_estimate_enabled"] is True
    # weak-профиль: оценщик и гибрид выключены.
    weak = hardware.recommend_hardware(_profile(total_ram_gb=4.0)).apply()
    assert weak["diarization_hybrid_enabled"] is False
    assert weak["diarization_estimate_enabled"] is False
