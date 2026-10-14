"""Автоопределение конфигурации железа и рекомендация настроек (issue #146).

Модуль не зависит от веб-слоя и тяжёлых библиотек: пробы идут через stdlib
(``/proc``, ``sysctl``, ``ctypes``) и внешние утилиты (``nvidia-smi``,
``rocm-smi``, ``vulkaninfo``), если они установлены. Где определить нельзя —
best-effort, с пометкой в ``notes``.

Рекомендация только предлагается: ни одна настройка не меняется автоматически —
её применяет пользователь (кнопка «Применить» в UI попадает в ``PUT /api/settings``
с профилем из :data:`audio_transcriber.web.settings.HARDWARE_PRESETS`).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "PROFILE_BALANCED",
    "PROFILE_PERFORMANCE",
    "PROFILE_WEAK",
    "CpuProfile",
    "GpuProfile",
    "HardwareProfile",
    "HardwareRecommendation",
    "detect_hardware",
    "recommend_hardware",
]

#: Байт в гибибайте.
_GB = 1024**3

#: Идентификаторы профилей железа (совпадают с пресетами настроек, issue #144).
PROFILE_WEAK = "weak"
PROFILE_BALANCED = "balanced"
PROFILE_PERFORMANCE = "performance"

#: Ориентиры «слабое железо» из docs/architecture/memory.md: 4/8/16 ГБ RAM.
RAM_WEAK_MAX_GB = 4.5
RAM_BALANCED_MAX_GB = 10.0

#: Ранг моделей whisper.cpp по «весу» — для выбора лучшей доступной модели.
_WHISPER_RANK = {
    "whisper-base": 1,
    "whisper-small": 2,
    "whisper-medium": 3,
    "whisper-large-v3-turbo": 4,
    "whisper-large-v3": 5,
}

#: Модель whisper.cpp под каждый профиль (id из каталога моделей).
_PROFILE_WHISPER_MODEL = {
    PROFILE_WEAK: "whisper-base",
    PROFILE_BALANCED: "whisper-small",
    PROFILE_PERFORMANCE: "whisper-large-v3-turbo",
}

#: Человекочитаемые метки профилей.
_PROFILE_LABEL = {
    PROFILE_WEAK: "Слабое железо",
    PROFILE_BALANCED: "Сбалансированный",
    PROFILE_PERFORMANCE: "Производительный",
}


@dataclass(frozen=True, slots=True)
class CpuProfile:
    """Параметры CPU."""

    name: str | None = None
    logical_cores: int | None = None
    physical_cores: int | None = None
    max_freq_mhz: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API."""
        return {
            "name": self.name,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "max_freq_mhz": self.max_freq_mhz,
        }


@dataclass(frozen=True, slots=True)
class GpuProfile:
    """Параметры GPU (наличие, вендор, VRAM, доступные ускорители)."""

    present: bool = False
    vendor: str | None = None
    name: str | None = None
    vram_gb: float | None = None
    accelerators: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API."""
        return {
            "present": self.present,
            "vendor": self.vendor,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "accelerators": list(self.accelerators),
        }


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Определённая конфигурация железа."""

    os_name: str
    cpu: CpuProfile
    total_ram_gb: float | None
    available_ram_gb: float | None
    gpu: GpuProfile
    free_disk_gb: float | None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API."""
        return {
            "os": self.os_name,
            "cpu": self.cpu.as_dict(),
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "gpu": self.gpu.as_dict(),
            "free_disk_gb": self.free_disk_gb,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class HardwareRecommendation:
    """Рекомендация настроек под определённое железо.

    Ничего не применяет: содержит готовый набор полей для ``PUT /api/settings``
    (:meth:`apply`) и обоснование «нашли / предлагаем / потому что».
    """

    profile: str
    label: str
    asr_backend: str
    device: str
    whisper_model_id: str | None
    whisper_model_available: bool
    diarization_engine: str
    denoise: bool
    llm_enabled: bool
    engine_isolation: str
    found: str
    proposed: str
    why: str
    rationale: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API (включая ``apply``)."""
        return {
            "profile": self.profile,
            "label": self.label,
            "asr_backend": self.asr_backend,
            "device": self.device,
            "whisper_model_id": self.whisper_model_id,
            "whisper_model_available": self.whisper_model_available,
            "diarization_engine": self.diarization_engine,
            "denoise": self.denoise,
            "llm_enabled": self.llm_enabled,
            "engine_isolation": self.engine_isolation,
            "found": self.found,
            "proposed": self.proposed,
            "why": self.why,
            "rationale": list(self.rationale),
            "apply": self.apply(),
        }

    def apply(self) -> dict[str, object]:
        """Поля ``WebSettings`` для применения рекомендации (точное совпадение)."""
        return {
            "asr_backend": self.asr_backend,
            "device": self.device,
            "diarization_engine": self.diarization_engine,
            "denoise": self.denoise,
            "llm_enabled": self.llm_enabled,
            "llm_summary": self.llm_enabled,
            "engine_isolation": self.engine_isolation,
            "diarization_hybrid_enabled": self.profile == PROFILE_PERFORMANCE,
            "diarization_estimate_enabled": self.profile != PROFILE_WEAK,
        }


# --- пробы окружения (легко подменяются в тестах) --------------------------


def _run_command(
    args: list[str], *, timeout: float = 5.0
) -> subprocess.CompletedProcess[str] | None:
    """Запускает команду; ``None`` — если утилиты нет или она упала по таймауту."""
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _sysctl_value(name: str) -> str | None:
    """Значение ``sysctl -n <name>`` (macOS) или ``None``."""
    proc = _run_command(["sysctl", "-n", name])
    if proc is None or proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _sysctl_int(name: str) -> int | None:
    """Целочисленное значение ``sysctl`` или ``None``."""
    raw = _sysctl_value(name)
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


def _cpuinfo_text() -> str:
    """Содержимое ``/proc/cpuinfo`` (пустая строка, если недоступно)."""
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _cpuinfo_field(text: str, field: str) -> str | None:
    """Первое значение поля ``/proc/cpuinfo`` вида ``field : value``."""
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == field and value.strip():
            return value.strip()
    return None


def _linux_physical_cores(text: str) -> int | None:
    """Число физических ядер по парам ``(physical id, core id)`` (Linux)."""
    keys: set[tuple[str, str]] = set()
    physical: str | None = None
    core: str | None = None
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        if key == "processor":
            if physical is not None and core is not None:
                keys.add((physical, core))
            physical = core = None
        elif key == "physical id":
            physical = value.strip()
        elif key == "core id":
            core = value.strip()
    if physical is not None and core is not None:
        keys.add((physical, core))
    return len(keys) or None


def _linux_max_freq_mhz(text: str) -> float | None:
    """Максимальная частота CPU из ``/proc/cpuinfo`` (МГц) или ``None``."""
    values: list[float] = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key.strip() != "cpu MHz":
            continue
        try:
            values.append(float(value.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def _windows_cpu_mhz() -> float | None:
    """Частота CPU из реестра Windows (МГц) или ``None``."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(  # type: ignore[attr-defined]
            winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined]
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "~MHz")  # type: ignore[attr-defined]
    except OSError:
        return None
    return float(value) if isinstance(value, int) else None


def _linux_meminfo_gb(field: str) -> float | None:
    """Значение поля ``/proc/meminfo`` (кБ) в ГБ или ``None``."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{field}:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _sysconf_ram_gb() -> float | None:
    """Оценка физической RAM через ``os.sysconf`` (POSIX fallback)."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size / _GB


def _windows_memory() -> tuple[float, float] | None:
    """``(всего, доступно)`` RAM в ГБ через WinAPI (Windows) или ``None``."""
    try:
        import ctypes
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
            ctypes.byref(status)
        ):
            return None
    except (OSError, AttributeError, ImportError):
        return None
    return status.ullTotalPhys / _GB, status.ullAvailPhys / _GB


def _total_ram_gb() -> float | None:
    """Физическая RAM в ГБ (кроссплатформенно, best-effort)."""
    system = platform.system()
    if system == "Linux":
        return _linux_meminfo_gb("MemTotal")
    if system == "Darwin":
        raw = _sysctl_int("hw.memsize")
        return raw / _GB if raw else None
    if system == "Windows":
        memory = _windows_memory()
        return memory[0] if memory else None
    return _sysconf_ram_gb()


def _available_ram_gb() -> float | None:
    """Доступная RAM в ГБ (кроссплатформенно, best-effort)."""
    system = platform.system()
    if system == "Linux":
        return _linux_meminfo_gb("MemAvailable")
    if system == "Windows":
        memory = _windows_memory()
        return memory[1] if memory else None
    return None


def _cpu_profile() -> CpuProfile:
    """Определяет параметры CPU (ядра, частота, имя)."""
    system = platform.system()
    logical = os.cpu_count()
    if system == "Linux":
        text = _cpuinfo_text()
        return CpuProfile(
            name=_cpuinfo_field(text, "model name"),
            logical_cores=logical,
            physical_cores=_linux_physical_cores(text),
            max_freq_mhz=_linux_max_freq_mhz(text),
        )
    if system == "Darwin":
        freq_hz = _sysctl_int("hw.cpufrequency")
        return CpuProfile(
            name=_sysctl_value("machdep.cpu.brand_string"),
            logical_cores=logical,
            physical_cores=_sysctl_int("hw.physicalcpu"),
            max_freq_mhz=freq_hz / 1e6 if freq_hz else None,
        )
    if system == "Windows":
        return CpuProfile(
            name=os.environ.get("PROCESSOR_IDENTIFIER") or None,
            logical_cores=logical,
            physical_cores=None,
            max_freq_mhz=_windows_cpu_mhz(),
        )
    return CpuProfile(logical_cores=logical)


def _megabytes_to_gb(raw: str) -> float | None:
    """«12288» (МБ) → гибибайты; ``None`` при неразборчивом значении."""
    try:
        return float(raw.strip()) / 1024
    except ValueError:
        return None


def _nvidia_gpu() -> tuple[str, float | None] | None:
    """Первый GPU NVIDIA по ``nvidia-smi``: ``(имя, VRAM ГБ)`` или ``None``."""
    proc = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
    name = parts[0] if parts else "NVIDIA GPU"
    vram = _megabytes_to_gb(parts[1]) if len(parts) > 1 else None
    return name or "NVIDIA GPU", vram


def _rocm_smi_gpu() -> tuple[str, float | None] | None:
    """Первый GPU AMD по ``rocm-smi --json``: ``(имя, VRAM ГБ)`` или ``None``."""
    proc = _run_command(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for card in payload.values():
        if not isinstance(card, dict):
            continue
        name = card.get("Card series") or card.get("Card Series") or card.get("Card model")
        vram_b: object = None
        for key, value in card.items():
            if "VRAM Total Memory" in key:
                vram_b = value
                break
        vram_gb = None
        if isinstance(vram_b, str):
            try:
                vram_gb = float(vram_b.strip()) / _GB
            except ValueError:
                vram_gb = None
        return (name if isinstance(name, str) and name else "AMD GPU", vram_gb)
    return None


def _vulkan_gpu_names() -> list[str]:
    """Имена GPU из ``vulkaninfo --summary`` (пусто, если недоступно)."""
    if shutil.which("vulkaninfo") is None:
        return []
    proc = _run_command(["vulkaninfo", "--summary"])
    if proc is None or proc.returncode != 0:
        return []
    names: list[str] = []
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "deviceName" and value.strip():
            names.append(value.strip())
    return names


def _linux_drm_vram_gb(base: Path | None = None) -> float | None:
    """VRAM из sysfs amdgpu/i915 (Linux) или ``None``.

    Драйверы amdgpu/i915 публикуют ``mem_info_vram_total`` — это единственный
    кроссплатформенный способ узнать VRAM, когда ``nvidia-smi``/``rocm-smi``
    недоступны, а GPU найден только через Vulkan.
    """
    root = base or Path("/sys/class/drm")
    if not root.is_dir():
        return None
    try:
        candidates = sorted(root.glob("card*/device/mem_info_vram_total"))
    except OSError:
        return None
    values: list[float] = []
    for path in candidates:
        try:
            values.append(float(path.read_text().strip()) / _GB)
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def _gpu_profile() -> GpuProfile:
    """Определяет GPU и доступные ускорители (CUDA/ROCm/Vulkan/Metal)."""
    nvidia = _nvidia_gpu()
    if nvidia is not None:
        return GpuProfile(
            present=True,
            vendor="nvidia",
            name=nvidia[0],
            vram_gb=nvidia[1],
            accelerators=("cuda", "vulkan"),
        )
    rocm = _rocm_smi_gpu()
    if rocm is not None:
        return GpuProfile(
            present=True,
            vendor="amd",
            name=rocm[0],
            vram_gb=rocm[1] if rocm[1] is not None else _linux_drm_vram_gb(),
            accelerators=("rocm", "vulkan"),
        )
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return GpuProfile(
            present=True,
            vendor="apple",
            name=platform.processor() or "Apple Silicon",
            vram_gb=_total_ram_gb(),
            accelerators=("metal",),
        )
    vulkan = _vulkan_gpu_names()
    if vulkan:
        return GpuProfile(
            present=True,
            vendor=None,
            name=vulkan[0],
            vram_gb=_linux_drm_vram_gb(),
            accelerators=("vulkan",),
        )
    return GpuProfile()


def _free_disk_gb(path: Path | None) -> float | None:
    """Свободное место на диске для пути (ГБ) или ``None``."""
    probe = path or Path.cwd()
    try:
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free / _GB
    except OSError:
        return None


def detect_hardware(disk_path: Path | None = None) -> HardwareProfile:
    """Собирает конфигурацию железа (CPU, RAM, GPU/VRAM, свободный диск).

    Все пробы — best-effort: недоступные значения возвращаются как ``None`` и
    отмечаются в ``notes``, чтобы UI мог показать «не определено».
    """
    notes: list[str] = []
    total_ram = _total_ram_gb()
    if total_ram is None:
        notes.append("объём RAM определить не удалось")
    available_ram = _available_ram_gb()
    cpu = _cpu_profile()
    if cpu.physical_cores is None and cpu.logical_cores is not None:
        notes.append("физические ядра не определены — учитываются логические")
    gpu = _gpu_profile()
    if gpu.present and gpu.vram_gb is None:
        notes.append("VRAM определить не удалось")
    free_disk = _free_disk_gb(disk_path)
    if free_disk is None:
        notes.append("свободное место на диске определить не удалось")
    return HardwareProfile(
        os_name=platform.system(),
        cpu=cpu,
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        gpu=gpu,
        free_disk_gb=free_disk,
        notes=tuple(notes),
    )


# --- рекомендация ----------------------------------------------------------


def _select_whisper_model(recommended: str, available_models: frozenset[str]) -> str:
    """Лучшая из доступных whisper.cpp-моделей, не тяжелее рекомендованной."""
    if not available_models:
        return recommended
    recommended_rank = _WHISPER_RANK.get(recommended, 0)
    candidates = [
        model for model in available_models if _WHISPER_RANK.get(model, 0) <= recommended_rank
    ]
    if not candidates:
        return recommended
    return max(candidates, key=lambda model: _WHISPER_RANK.get(model, 0))


def _describe_found(hw: HardwareProfile) -> str:
    """Строка «нашли: …» с определённой конфигурацией."""
    parts: list[str] = []
    if hw.total_ram_gb is not None:
        ram = f"RAM {hw.total_ram_gb:.1f} ГБ"
        if hw.available_ram_gb is not None:
            ram += f" (доступно {hw.available_ram_gb:.1f})"
        parts.append(ram)
    else:
        parts.append("RAM неизвестно")
    cores = hw.cpu.physical_cores or hw.cpu.logical_cores
    label = "физических" if hw.cpu.physical_cores else "логических"
    if cores is not None:
        cpu = f"CPU {cores} {label} ядер"
        if hw.cpu.max_freq_mhz:
            cpu += f", до {hw.cpu.max_freq_mhz / 1000:.1f} ГГц"
        parts.append(cpu)
    if hw.gpu.present:
        gpu = hw.gpu.name or "GPU"
        if hw.gpu.vram_gb is not None:
            gpu += f", VRAM {hw.gpu.vram_gb:.1f} ГБ"
        parts.append(f"GPU: {gpu}")
    else:
        parts.append("GPU не обнаружен")
    if hw.free_disk_gb is not None:
        parts.append(f"диск {hw.free_disk_gb:.0f} ГБ свободно")
    return "нашли: " + ", ".join(parts)


def _describe_proposed(rec: HardwareRecommendation) -> str:
    """Строка «предлагаем: …» с рекомендованными настройками."""
    if rec.asr_backend == "whisper-cpp":
        asr = f"whisper.cpp {rec.whisper_model_id or ''}".strip()
    else:
        asr = rec.asr_backend
    details = [
        f"{asr} на {'GPU' if rec.device != 'cpu' else 'CPU'}",
        f"диаризация {rec.diarization_engine}",
        f"денойз {'вкл' if rec.denoise else 'выкл'}",
        f"LLM {'вкл' if rec.llm_enabled else 'выкл'}",
        f"изоляция движков {rec.engine_isolation}",
    ]
    return "предлагаем: " + ", ".join(details)


def recommend_hardware(
    hw: HardwareProfile,
    *,
    available_models: frozenset[str] = frozenset(),
    available_binaries: frozenset[str] | None = None,
) -> HardwareRecommendation:
    """Подбирает engine/модель/устройство/стадии под железо с обоснованием.

    Профиль выбирается по RAM (ориентиры memory.md: 4/8/16 ГБ) и наличию GPU;
    доступные модели/бинарники учитываются, чтобы не предлагать заведомо
    отсутствующее. Ничего не применяет.
    """
    rationale: list[str] = []
    ram = hw.total_ram_gb
    accelerators = set(hw.gpu.accelerators)
    has_cuda = "cuda" in accelerators
    gpu_accel = bool(accelerators)

    if ram is not None and ram <= RAM_WEAK_MAX_GB:
        profile = PROFILE_WEAK
        rationale.append(f"RAM {ram:.1f} ГБ ≤ {RAM_WEAK_MAX_GB:.0f} ГБ → лёгкий профиль")
    elif ram is not None and ram <= RAM_BALANCED_MAX_GB:
        profile = PROFILE_BALANCED
        rationale.append(f"RAM {ram:.1f} ГБ ≤ {RAM_BALANCED_MAX_GB:.0f} ГБ → сбалансированный профиль")
    else:
        profile = PROFILE_PERFORMANCE
        if ram is not None:
            rationale.append(f"RAM {ram:.1f} ГБ ≥ {RAM_BALANCED_MAX_GB:.0f} ГБ → производительный профиль")
        else:
            rationale.append("RAM неизвестна → производительный профиль (по умолчанию)")

    if has_cuda:
        asr_backend = "faster-whisper"
        device = "cuda"
        whisper_model = None
        rationale.append("GPU NVIDIA (CUDA) → faster-whisper на CUDA")
    else:
        asr_backend = "whisper-cpp"
        device = "auto" if (gpu_accel or hw.gpu.present) else "cpu"
        recommended = _PROFILE_WHISPER_MODEL[profile]
        whisper_model = _select_whisper_model(recommended, available_models)
        if gpu_accel:
            accelerator = sorted(accelerators)[0]
            rationale.append(f"GPU ({accelerator}) → whisper.cpp с аппаратным ускорением")
        else:
            rationale.append("GPU не обнаружен → whisper.cpp на CPU")

    model_available = whisper_model is None or whisper_model in available_models
    if whisper_model is not None and not model_available:
        rationale.append(f"модель {whisper_model} не скачана — будет предложено скачать")

    diarization_engine = "sherpa" if profile == PROFILE_WEAK else "auto"
    if profile == PROFILE_WEAK:
        rationale.append("мало RAM → диаризация sherpa (ONNX/CPU, без torch)")
    else:
        rationale.append("диаризация auto (оценка N и маршрутизация по движкам)")

    denoise = profile != PROFILE_WEAK
    rationale.append("денойз выключен под слабое железо" if not denoise else "денойз включён")

    llm_ready = available_binaries is None or "llama-server" in available_binaries
    llm_enabled = profile == PROFILE_PERFORMANCE and llm_ready
    if profile == PROFILE_PERFORMANCE and not llm_ready:
        rationale.append("LLM выключен: llama-server не найден (установите, чтобы включить)")
    elif llm_enabled:
        rationale.append("RAM с запасом → LLM-постобработка включена")
    else:
        rationale.append("LLM выключен (нет запаса RAM под модель)")

    if asr_backend == "faster-whisper" or diarization_engine == "pyannote":
        tight = ram is not None and ram <= RAM_BALANCED_MAX_GB
        engine_isolation = "subprocess" if tight else "off"
        if engine_isolation == "subprocess":
            rationale.append("тяжёлый движок в процессе + мало RAM → изоляция в subprocess")
    else:
        engine_isolation = "off"

    rec = HardwareRecommendation(
        profile=profile,
        label=_PROFILE_LABEL[profile],
        asr_backend=asr_backend,
        device=device,
        whisper_model_id=whisper_model,
        whisper_model_available=model_available,
        diarization_engine=diarization_engine,
        denoise=denoise,
        llm_enabled=llm_enabled,
        engine_isolation=engine_isolation,
        found=_describe_found(hw),
        proposed="",
        why="; ".join(rationale),
        rationale=tuple(rationale),
    )
    return replace(rec, proposed=_describe_proposed(rec))
