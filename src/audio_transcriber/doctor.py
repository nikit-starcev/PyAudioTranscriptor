"""Самопроверка окружения (команда ``audio-transcriber doctor``).

Проверяет версию Python, ключевые зависимости, бинарники и модели, наличие
GPU-Vulkan, ``config.env``, токен Hugging Face (значение не печатается) и
доступность на запись каталогов результатов и кэша.

Особенности:

- никаких тяжёлых вычислений и загрузки моделей — проверки быстрые;
- ничего не роняет: отсутствие компонента — это и есть результат проверки;
- каждый пункт имеет признак «критичности»: код возврата 1, только если
  что-то критичное не в порядке; необязательные компоненты дают лишь ✗/предупреждение.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.domain.enums import AsrBackend
from audio_transcriber.utils.config_env import load_config_env

__all__ = [
    "DoctorCheck",
    "format_report",
    "has_critical_failures",
    "load_config_env",
    "run_doctor",
]

logger = logging.getLogger(__name__)

#: Минимально поддерживаемая версия Python (синхронизировано с pyproject.toml).
MIN_PYTHON = (3, 14)

#: Значения по умолчанию, если ``config.env`` отсутствует.
DEFAULT_ASR_BACKEND = AsrBackend.FASTER_WHISPER
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_WHISPER_BINARY = "whisper-cli"
DEFAULT_LLM_BINARY = "llama-server"

#: Ссылки, помогающие устранить проблему (используются веб-интерфейсом).
LINK_WHISPER_CPP = "https://github.com/ggml-org/whisper.cpp"
LINK_LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"
LINK_GGML_MODELS = "https://huggingface.co/ggerganov/whisper.cpp"
LINK_PYANNOTE_MODEL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
LINK_HF_TOKENS = "https://huggingface.co/settings/tokens"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """Один пункт отчёта самопроверки."""

    key: str
    label: str
    ok: bool
    critical: bool
    detail: str = ""
    hint: str = ""
    links: tuple[str, ...] = ()


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "да")


# --- пробы окружения (легко подменяются в тестах) -------------------------


def _module_available(name: str) -> bool:
    """Доступен ли модуль; сам модуль не импортируется (``find_spec``)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _binary_available(binary: str) -> bool:
    """Бинарник задан путём и существует либо находится в ``PATH``."""
    if not binary:
        return False
    candidate = Path(binary).expanduser()
    if candidate.is_file():
        return True
    return shutil.which(binary) is not None


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _dir_writable(path: Path) -> bool:
    """Проверяет запись в каталог (создаёт его при необходимости)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _subprocess_env(library_path: str | None) -> dict[str, str]:
    """Окружение для запуска бинарника с учётом каталога библиотек.

    На Linux ``WHISPER_CPP_LIB_PATH`` добавляется в ``LD_LIBRARY_PATH``, чтобы
    динамический загрузчик нашёл ``libwhisper``/``libggml*`` (включая Vulkan).
    На macOS используется ``DYLD_LIBRARY_PATH``, на Windows — ``PATH``. Пустой
    путь ничего не меняет и не приводит к ошибке.
    """
    env = dict(os.environ)
    if not library_path:
        return env
    if sys.platform == "darwin":
        variable = "DYLD_LIBRARY_PATH"
    elif sys.platform.startswith("win"):
        variable = "PATH"
    else:
        variable = "LD_LIBRARY_PATH"
    existing = env.get(variable)
    env[variable] = library_path + (os.pathsep + existing if existing else "")
    return env


def _run_command(
    command: list[str], library_path: str | None = None, *, timeout: int = 10
) -> subprocess.CompletedProcess[str] | None:
    """Запускает бинарник, возвращая результат или ``None`` при сбое запуска."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_subprocess_env(library_path),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _vulkan_devices(binary: str, library_path: str | None) -> list[str] | None:
    """Список Vulkan-устройств из ``whisper-cli --list-devices``.

    ``None`` — определить не удалось: бинарник не отработал, не поддерживает
    такой аргумент или его запуск завершился ошибкой. Пустой список означает,
    что проверка выполнилась, но устройств не нашлось.
    """
    proc = _run_command([binary, "--list-devices"], library_path)
    if proc is None:
        return None
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "unknown argument" in output.lower():
        return None
    return [
        line.strip() for line in output.splitlines() if "vulkan" in line.lower()
    ]


def _library_names_present(directory: Path) -> bool:
    """Есть ли в каталоге ``libggml-vulkan.*`` (собранный Vulkan-бэкенд ggml)."""
    if not _is_dir(directory):
        return False
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    return any(
        name.startswith("libggml-vulkan") or name.startswith("ggml-vulkan")
        for name in entries
    )


def _vulkan_library_present(library_path: str | None, binary: str) -> bool:
    """Найден ли ``libggml-vulkan`` в каталоге библиотек или рядом с бинарником."""
    candidates: list[Path] = []
    if library_path:
        candidates.append(Path(library_path).expanduser())
    resolved = Path(binary).expanduser()
    if not resolved.is_file():
        found = shutil.which(binary)
        if found:
            resolved = Path(found)
    if resolved.is_file():
        candidates.append(resolved.parent)
    return any(_library_names_present(directory) for directory in candidates)


#: GPU-флаги, которые поддерживает билд whisper.cpp с включённым ускорением.
_GPU_FLAG_TOKENS = ("-dev", "--device", "-ng", "--no-gpu")


def _supports_gpu_flags(help_text: str | None) -> bool:
    """Поддерживает ли билд GPU-флаги (``-dev/--device``, ``-ng/--no-gpu``)."""
    if not help_text:
        return False
    lowered = help_text.lower()
    return any(token in lowered for token in _GPU_FLAG_TOKENS)


def _whisper_help(binary: str, library_path: str | None) -> str | None:
    """Вывод ``whisper-cli --help`` (или ``None``, если запустить не удалось)."""
    proc = _run_command([binary, "--help"], library_path)
    if proc is None:
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def _vulkaninfo_devices() -> list[str] | None:
    """Имена устройств из ``vulkaninfo --summary`` (``None``, если недоступно)."""
    if shutil.which("vulkaninfo") is None:
        return None
    proc = _run_command(["vulkaninfo", "--summary"])
    if proc is None:
        return None
    output = (proc.stdout or "") + (proc.stderr or "")
    devices: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("devicename"):
            _, _, value = stripped.partition("=")
            value = value.strip()
            if value and value not in devices:
                devices.append(value)
    return devices


# --- проверки --------------------------------------------------------------


def _check_python() -> DoctorCheck:
    required = ".".join(str(part) for part in MIN_PYTHON)
    current = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= MIN_PYTHON
    hint = "" if ok else f"Требуется Python {required}+. Установите более новую версию."
    return DoctorCheck(
        key="python",
        label=f"Python {required}+",
        ok=ok,
        critical=True,
        detail=f"текущая {current}",
        hint=hint,
    )


def _dependencies(env: Mapping[str, str]) -> list[tuple[str, str, bool]]:
    """Список (имя модуля, подпись, критичность) для проверки зависимостей."""
    backend = env.get("ASR_BACKEND", DEFAULT_ASR_BACKEND.value).strip()
    diarization = _truthy(env.get("DIARIZATION_ENABLED"), default=True)
    correction = _truthy(env.get("ENABLE_CORRECTION"), default=False)
    return [
        ("av", "av (декодирование аудио)", True),
        ("faster_whisper", "faster-whisper", backend == AsrBackend.FASTER_WHISPER.value),
        ("pyannote.audio", "pyannote.audio (диаризация)", diarization),
        ("torch", "torch", diarization),
        ("textual", "textual (TUI)", False),
        ("pymorphy3", "pymorphy3 (автоисправление)", correction),
        ("df", "deepfilternet (денойз)", False),
    ]


def _check_dependencies(env: Mapping[str, str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for module, label, critical in _dependencies(env):
        available = _module_available(module)
        if available:
            hint = ""
        elif critical:
            hint = f"Установите зависимость: uv sync (модуль '{module}')."
        else:
            hint = f"Необязательно; соответствующая функция будет недоступна ('{module}')."
        checks.append(
            DoctorCheck(
                key=f"dep:{module}",
                label=f"Зависимость: {label}",
                ok=available,
                critical=critical,
                detail="доступна" if available else "не найдена",
                hint=hint,
            )
        )
    return checks


def _check_binaries(env: Mapping[str, str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    backend = env.get("ASR_BACKEND", DEFAULT_ASR_BACKEND.value).strip()
    llm_enabled = _truthy(env.get("LLM_ENABLED"), default=False)

    if backend == AsrBackend.WHISPER_CPP.value:
        binary = env.get("WHISPER_CPP_BINARY", DEFAULT_WHISPER_BINARY).strip()
        available = _binary_available(binary)
        lib_path = env.get("WHISPER_CPP_LIB_PATH", "").strip()
        lib_note = ""
        if available and lib_path:
            lib_note = " (каталог библиотек " + ("найден)" if _is_dir(Path(lib_path)) else "не найден)")
        checks.append(
            DoctorCheck(
                key="bin:whisper-cli",
                label="Бинарник whisper-cli",
                ok=available,
                critical=True,
                detail=f"{binary or 'не задан'}{lib_note}",
                hint="" if available else "Задайте WHISPER_CPP_BINARY/--whisper-cpp-binary.",
                links=() if available else (LINK_WHISPER_CPP,),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                key="bin:whisper-cli",
                label="Бинарник whisper-cli",
                ok=True,
                critical=False,
                detail="не требуется (бэкенд faster-whisper)",
            )
        )

    if llm_enabled:
        binary = env.get("LLM_BINARY", DEFAULT_LLM_BINARY).strip()
        available = _binary_available(binary)
        checks.append(
            DoctorCheck(
                key="bin:llama-server",
                label="Бинарник llama-server",
                ok=available,
                critical=True,
                detail=binary or "не задан",
                hint="" if available else "Задайте LLM_BINARY/--llm-binary.",
                links=() if available else (LINK_LLAMA_CPP,),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                key="bin:llama-server",
                label="Бинарник llama-server",
                ok=True,
                critical=False,
                detail="не требуется (LLM выключена)",
            )
        )
    return checks


def _check_models(env: Mapping[str, str]) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    backend = env.get("ASR_BACKEND", DEFAULT_ASR_BACKEND.value).strip()
    diarization = _truthy(env.get("DIARIZATION_ENABLED"), default=True)
    llm_enabled = _truthy(env.get("LLM_ENABLED"), default=False)

    if backend == AsrBackend.WHISPER_CPP.value:
        raw = env.get("WHISPER_CPP_MODEL", "").strip()
        path = Path(raw) if raw else None
        ok = path is not None and _is_file(path)
        checks.append(
            DoctorCheck(
                key="model:whisper",
                label="Модель whisper.cpp (ggml)",
                ok=ok,
                critical=True,
                detail=raw or "не задана",
                hint="" if ok else "Задайте WHISPER_CPP_MODEL/--whisper-cpp-model.",
                links=() if ok else (LINK_GGML_MODELS,),
            )
        )

    if llm_enabled:
        raw = env.get("LLM_MODEL", "").strip()
        path = Path(raw) if raw else None
        ok = path is not None and _is_file(path)
        checks.append(
            DoctorCheck(
                key="model:llm",
                label="Модель LLM (GGUF)",
                ok=ok,
                critical=True,
                detail=raw or "не задана",
                hint="" if ok else "Задайте LLM_MODEL/--llm-model.",
                links=() if ok else (LINK_LLAMA_CPP,),
            )
        )

    if diarization:
        raw = env.get("PYANNOTE_LOCAL_MODEL", "").strip()
        if raw:
            path = Path(raw)
            ok = _is_dir(path)
            checks.append(
                DoctorCheck(
                    key="model:pyannote",
                    label="Локальная модель диаризации",
                    ok=ok,
                    critical=True,
                    detail=raw,
                    hint="" if ok else "Путь PYANNOTE_LOCAL_MODEL не найден.",
                    links=() if ok else (LINK_PYANNOTE_MODEL,),
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    key="model:pyannote",
                    label="Локальная модель диаризации",
                    ok=True,
                    critical=False,
                    detail="не задана (модель будет загружена с Hugging Face)",
                )
            )
    return checks


def _check_vulkan(env: Mapping[str, str]) -> DoctorCheck:
    backend = env.get("ASR_BACKEND", DEFAULT_ASR_BACKEND.value).strip()
    if backend != AsrBackend.WHISPER_CPP.value:
        return DoctorCheck(
            key="vulkan",
            label="GPU Vulkan",
            ok=True,
            critical=False,
            detail="не применимо (бэкенд faster-whisper)",
        )
    binary = env.get("WHISPER_CPP_BINARY", DEFAULT_WHISPER_BINARY).strip()
    if not _binary_available(binary):
        return DoctorCheck(
            key="vulkan",
            label="GPU Vulkan",
            ok=False,
            critical=False,
            detail="не удалось проверить: whisper-cli не найден",
            hint="Сначала установите/укажите whisper-cli.",
        )
    lib_path = env.get("WHISPER_CPP_LIB_PATH", "").strip() or None

    # 1. Устройства, если билд поддерживает --list-devices.
    devices = _vulkan_devices(binary, lib_path)
    if devices:
        return DoctorCheck(
            key="vulkan",
            label="GPU Vulkan",
            ok=True,
            critical=False,
            detail="Vulkan: " + "; ".join(devices[:3]),
        )

    # 2. Собран ли Vulkan вообще: libggml-vulkan рядом с бинарником/в lib
    #    либо GPU-флаги в --help (дешёвая проверка, модель не грузим).
    vulkan_built = _vulkan_library_present(lib_path, binary) or _supports_gpu_flags(
        _whisper_help(binary, lib_path)
    )
    if not vulkan_built:
        return DoctorCheck(
            key="vulkan",
            label="GPU Vulkan",
            ok=False,
            critical=False,
            detail="Vulkan не обнаружен (нет libggml-vulkan и билд без GPU-флагов)",
            hint=(
                "Соберите whisper.cpp с Vulkan. "
                "Без Vulkan ASR пойдёт на CPU (медленнее)."
            ),
            links=(LINK_WHISPER_CPP,),
        )

    # 3. Vulkan собран, но whisper-device не перечислил — пробуем vulkaninfo.
    info_devices = _vulkaninfo_devices()
    if info_devices:
        return DoctorCheck(
            key="vulkan",
            label="GPU Vulkan",
            ok=True,
            critical=False,
            detail="Vulkan: " + "; ".join(info_devices[:3]),
        )
    return DoctorCheck(
        key="vulkan",
        label="GPU Vulkan",
        ok=True,
        critical=False,
        detail="Vulkan собран (libggml-vulkan), устройство не определено",
    )


def _check_config_env(config_env_path: Path | None) -> DoctorCheck:
    if config_env_path is None:
        return DoctorCheck(
            key="config_env",
            label="config.env",
            ok=False,
            critical=False,
            detail="не найден",
            hint="Скопируйте config.example.env в config.env (необязательно при CLI-флагах).",
        )
    readable = False
    try:
        with open(config_env_path, encoding="utf-8") as handle:
            handle.read(1)
        readable = True
    except OSError:
        readable = False
    return DoctorCheck(
        key="config_env",
        label="config.env",
        ok=readable,
        critical=False,
        detail=str(config_env_path),
        hint="" if readable else "Файл не читается — проверьте права доступа.",
    )


def _check_hf_token(env: Mapping[str, str]) -> DoctorCheck:
    token = (env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    diarization = _truthy(env.get("DIARIZATION_ENABLED"), default=True)
    local_model = env.get("PYANNOTE_LOCAL_MODEL", "").strip()
    needed = diarization and not local_model
    if not needed:
        return DoctorCheck(
            key="hf_token",
            label="Токен Hugging Face",
            ok=True,
            critical=False,
            detail="не требуется",
        )
    return DoctorCheck(
        key="hf_token",
        label="Токен Hugging Face",
        ok=bool(token),
        critical=True,
        detail="задан" if token else "не задан",
        hint="" if token else "Задайте HF_TOKEN в config.env или --hf-token (значение не выводится).",
        links=() if token else (LINK_HF_TOKENS, LINK_PYANNOTE_MODEL),
    )


def _check_writable(env: Mapping[str, str]) -> list[DoctorCheck]:
    output_raw = env.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR).strip() or DEFAULT_OUTPUT_DIR
    output_dir = Path(output_raw)
    checks = [
        DoctorCheck(
            key="output_dir",
            label="Каталог результатов (запись)",
            ok=_dir_writable(output_dir),
            critical=True,
            detail=str(output_dir),
            hint="Проверьте права на запись в каталог результатов.",
        )
    ]
    cache_dir = output_dir / ".cache"
    checks.append(
        DoctorCheck(
            key="cache_dir",
            label="Каталог кэша (запись)",
            ok=_dir_writable(cache_dir),
            critical=False,
            detail=str(cache_dir),
            hint="Без кэша транскрибация просто пересчитывается каждый раз.",
        )
    )
    return checks


def run_doctor(
    config_env_path: Path | None, env: Mapping[str, str]
) -> list[DoctorCheck]:
    """Прогоняет все проверки и возвращает отчёт в порядке вывода."""
    checks: list[DoctorCheck] = [_check_python()]
    checks.extend(_check_dependencies(env))
    checks.extend(_check_binaries(env))
    checks.extend(_check_models(env))
    checks.append(_check_vulkan(env))
    checks.append(_check_config_env(config_env_path))
    checks.append(_check_hf_token(env))
    checks.extend(_check_writable(env))
    return checks


def has_critical_failures(checks: list[DoctorCheck]) -> bool:
    """Есть ли среди проверок критичные провалы (определяет код возврата)."""
    return any((not check.ok) and check.critical for check in checks)


def format_report(checks: list[DoctorCheck]) -> str:
    """Форматирует отчёт со статусами ✓/✗ и подсказками."""
    lines: list[str] = []
    for check in checks:
        mark = "✓" if check.ok else "✗"
        suffix = "" if check.ok or check.critical else " [не критично]"
        line = f"{mark} {check.label}{suffix}"
        if check.detail:
            line += f" — {check.detail}"
        lines.append(line)
        if not check.ok and check.hint:
            lines.append(f"    → {check.hint}")

    failed = sum(1 for check in checks if not check.ok)
    critical = sum(1 for check in checks if not check.ok and check.critical)
    summary = f"Итог: проверок {len(checks)}, проблем {failed} (критичных {critical})."
    if critical == 0:
        summary += " Критичных проблем нет."
    lines.append(summary)
    return "\n".join(lines)
