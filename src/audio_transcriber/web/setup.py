"""План мастера первого запуска (шаги онбординга).

Модуль собирает из отчёта готовности, текущих настроек и состояния каталога
моделей единый «план мастера»: какие шаги ещё не выполнены и что нужно
скачать/указать. Это единственный источник истины для SPA — фронт только
отображает план, а тестируется он здесь (без браузера).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from audio_transcriber.doctor import (
    LINK_DEEP_FILTER,
    LINK_LLAMA_CPP,
    LINK_NEMO_SPEECH,
    LINK_SHERPA_ONNX,
    LINK_WHISPER_CPP,
)
from audio_transcriber.domain.enums import AsrBackend
from audio_transcriber.web import assets as assets_registry
from audio_transcriber.web import deps as deps_registry
from audio_transcriber.web import models as models_registry

#: Идентификаторы шагов мастера (порядок отображения).
STEP_HARDWARE = "hardware"
STEP_HF_TOKEN = "hf_token"
STEP_MODELS = "models"
STEP_BINARIES = "binaries"
STEP_READINESS = "readiness"


@dataclass(frozen=True, slots=True)
class HardwareOption:
    """Предлагаемый вариант железа и соответствующие настройки."""

    id: str
    label: str
    asr_backend: str
    device: str
    note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "asr_backend": self.asr_backend,
            "device": self.device,
            "note": self.note,
        }


#: Варианты железа в порядке предложения.
HARDWARE_OPTIONS: tuple[HardwareOption, ...] = (
    HardwareOption(
        "nvidia",
        "NVIDIA / CUDA",
        AsrBackend.FASTER_WHISPER.value,
        "cuda",
        "faster-whisper на GPU NVIDIA (нужна CUDA-сборка torch).",
    ),
    HardwareOption(
        "amd",
        "AMD / Vulkan",
        AsrBackend.WHISPER_CPP.value,
        "auto",
        "whisper.cpp + llama.cpp со сборками под Vulkan (Radeon или любой GPU).",
    ),
    HardwareOption(
        "cpu",
        "Только CPU",
        AsrBackend.FASTER_WHISPER.value,
        "cpu",
        "Работает везде, но медленнее; подходит как запасной вариант.",
    ),
)

#: Модель whisper.cpp, предлагаемая по умолчанию в мастере.
DEFAULT_WHISPER_MODEL_ID = "whisper-large-v3-turbo"
#: Модель LLM, предлагаемая по умолчанию в мастере.
DEFAULT_LLM_MODEL_ID = "qwen2.5-7b-instruct-q4_k_m"
#: Модель диаризации.
PYANNOTE_MODEL_ID = "pyannote-community-1"
#: Снимок GigaAM v3 (onnx-asr) — нужен при бэкенде ``gigaam``.
GIGAAM_MODEL_ID = "gigaam-v3-onnx"


def current_hardware(settings: object) -> str:
    """Определяет выбранный вариант железа по сохранённым настройкам."""
    backend = getattr(settings, "asr_backend", AsrBackend.FASTER_WHISPER.value)
    device = getattr(settings, "device", "auto")
    if backend == AsrBackend.WHISPER_CPP.value:
        return "amd"
    if device == "cuda":
        return "nvidia"
    return "cpu"


def hardware_option(option_id: str) -> HardwareOption | None:
    """Вариант железа по идентификатору."""
    for option in HARDWARE_OPTIONS:
        if option.id == option_id:
            return option
    return None


def required_model_ids(settings: object) -> list[str]:
    """Модели, нужные для текущего режима (для шага «Модели»)."""
    ids: list[str] = []
    backend = getattr(settings, "asr_backend", AsrBackend.FASTER_WHISPER.value)
    if backend == AsrBackend.WHISPER_CPP.value:
        ids.append(DEFAULT_WHISPER_MODEL_ID)
    elif backend == AsrBackend.GIGAAM.value:
        ids.append(GIGAAM_MODEL_ID)
    if bool(getattr(settings, "llm_enabled", False)):
        ids.append(DEFAULT_LLM_MODEL_ID)
    ids.append(PYANNOTE_MODEL_ID)
    return ids


#: Ссылки на upstream-репозитории для бинарных ресурсов.
_BINARY_LINKS: dict[str, tuple[str, ...]] = {
    "whisper-cli": (LINK_WHISPER_CPP,),
    "llama-server": (LINK_LLAMA_CPP,),
    "deep-filter": (LINK_DEEP_FILTER,),
    "nemo-speech": (LINK_NEMO_SPEECH,),
}


def _format_size(size: int) -> str:
    """Человекочитаемый размер (``~12.3 МБ``)."""
    if size <= 0:
        return "—"
    units = ("Б", "КБ", "МБ", "ГБ")
    value = float(size)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    if index == 0:
        return f"{int(value)} {units[index]}"
    return f"{value:.1f} {units[index]}"


def prefer_vulkan(settings: object) -> bool:
    """Выбирать Vulkan-сборку бинарников (ветка AMD/Vulkan, whisper-cpp)."""
    return getattr(settings, "asr_backend", "") == AsrBackend.WHISPER_CPP.value


def _binary_requirement(
    key: str, settings: object, *, needed: bool
) -> dict[str, object]:
    """Пункт шага «Бинарники» для ресурса из реестра :mod:`assets` (#98)."""
    asset = assets_registry.find_asset(key)
    if asset is None:  # pragma: no cover — ключи заданы в коде
        raise KeyError(key)
    artifact = assets_registry.resolve_artifact(asset, prefer_vulkan=prefer_vulkan(settings))
    configured = assets_registry.configured_binary(asset, settings)
    if artifact is not None:
        instructions = (
            f"Скачивается кнопкой «Скачать» "
            f"({asset.label.split('(')[0].strip()}, ~{_format_size(artifact.size)}). "
            "После установки путь пропишется в настройках автоматически. "
            f"{asset.note}".strip()
        )
    else:
        instructions = asset.manual_hint or asset.note
    return {
        "key": key,
        "asset_key": key,
        "kind": assets_registry.KIND_BINARY,
        "check_id": asset.check_id,
        "label": asset.label,
        "needed": needed,
        "optional": asset.optional,
        "downloadable": artifact is not None,
        "artifact": artifact.as_dict() if artifact is not None else None,
        "platform": f"{assets_registry.current_os()}/{assets_registry.current_arch()}",
        "setting_key": asset.env_key,
        "lib_setting_key": asset.lib_env_key,
        "installed_path": configured,
        "instructions": instructions,
        "links": list(_BINARY_LINKS.get(key, ())),
    }


def _pip_requirement(
    dep_key: str,
    plan_key: str,
    *,
    needed: bool,
    instructions: str,
    links: list[str],
) -> dict[str, object] | None:
    """Пункт шага «Пакеты» для pip-ресурса из allowlist (#66)."""
    dep = deps_registry.find_dependency(dep_key)
    if dep is None:
        return None
    return {
        "key": plan_key,
        "asset_key": dep.key,
        "kind": assets_registry.KIND_PIP,
        "check_id": dep.check_id,
        "label": dep.label,
        "needed": needed,
        "dep_key": dep.key,
        "spec": dep.spec,
        "installable": deps_registry.installer_available(),
        "downloadable": False,
        "instructions": instructions,
        "links": links,
    }


def binary_requirements(settings: object) -> list[dict[str, object]]:
    """Нужные внешние компоненты (бинарники/пакеты), их доступность и инструкции.

    Бинарники (#98) скачиваются кнопкой из allowlist :mod:`assets`; pip-пакеты
    (#66) ставятся кнопкой из allowlist :mod:`deps`. Поле ``check_id`` задаёт
    идентификатор проверки доктора (по умолчанию ``bin:<key>``); для пакета
    используется ``dep:<модуль>``.
    """
    backend = getattr(settings, "asr_backend", AsrBackend.FASTER_WHISPER.value)
    llm_enabled = bool(getattr(settings, "llm_enabled", False))
    requirements: list[dict[str, object]] = []
    if backend == AsrBackend.WHISPER_CPP.value:
        requirements.append(_binary_requirement("whisper-cli", settings, needed=True))
    if backend == AsrBackend.GIGAAM.value:
        onnx = _pip_requirement(
            "gigaam",
            "onnx-asr",
            needed=True,
            instructions=(
                "Пакет ставится кнопкой «Установить» (onnx-asr[cpu,hub]). "
                "Вручную: uv pip install 'onnx-asr[cpu,hub]'. Модель GigaAM "
                "подтянется из каталога моделей или с Hugging Face."
            ),
            links=[],
        )
        if onnx is not None:
            requirements.append(onnx)
    if llm_enabled:
        requirements.append(_binary_requirement("llama-server", settings, needed=True))
    requirements.extend(_diarization_requirements(settings))
    return requirements


def _diarization_requirements(settings: object) -> list[dict[str, object]]:
    """Внешние компоненты диаризации (#62/#64) для шага «Бинарники/Пакеты».

    sherpa-onnx — пакет из allowlist (#66): показываем кнопку «Установить»,
    но не блокируем запуск (без него оценка N деградирует, а ``auto``
    безопасно уходит на pyannote). Бинарник nemo-speech нужен явным движкам
    ``nemo-speech``/``hybrid``.
    """
    engine = (
        str(getattr(settings, "diarization_engine", "auto")).strip().casefold() or "auto"
    )
    estimate_enabled = bool(getattr(settings, "diarization_estimate_enabled", True))
    hybrid_enabled = bool(getattr(settings, "diarization_hybrid_enabled", True))
    requirements: list[dict[str, object]] = []

    if estimate_enabled or (hybrid_enabled and engine == "hybrid"):
        sherpa = _pip_requirement(
            "sherpa",
            "sherpa-onnx",
            needed=False,
            instructions=(
                "Пакет ставится кнопкой «Установить» (sherpa-onnx). "
                "Без него оценка числа говорящих деградирует, "
                "а гибридная диаризация недоступна."
            ),
            links=[LINK_SHERPA_ONNX],
        )
        if sherpa is not None:
            requirements.append(sherpa)

    if engine in {"nemo-speech", "hybrid"}:
        requirements.append(_binary_requirement("nemo-speech", settings, needed=True))
    return requirements


def _check_status(report: Mapping[str, object], check_id: str) -> str:
    checks = report.get("checks")
    if not isinstance(checks, Sequence):
        return "ok"
    for check in checks:
        if isinstance(check, Mapping) and check.get("id") == check_id:
            raw = check.get("status")
            return raw if isinstance(raw, str) else "ok"
    return "ok"


def _critical_failures(report: Mapping[str, object]) -> int:
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        raw = summary.get("critical_failures")
        if isinstance(raw, int):
            return raw
    return 0


def _binary_available(report: Mapping[str, object], check_id: str) -> bool:
    return _check_status(report, check_id) == "ok"


def build_setup_steps(
    *,
    settings: object,
    report: Mapping[str, object],
    models: Sequence[Mapping[str, object]],
    hf_token_set: bool,
) -> dict[str, object]:
    """Собирает план мастера: шаги, варианты железа, нужные модели и бинарники."""
    model_by_id: dict[str, Mapping[str, object]] = {
        str(model.get("id")): model for model in models if model.get("id")
    }

    def _present(model_id: str) -> bool:
        entry = model_by_id.get(model_id)
        status = entry.get("status") if isinstance(entry, Mapping) else None
        return bool(status.get("present")) if isinstance(status, Mapping) else False

    required = required_model_ids(settings)
    missing_models = [model_id for model_id in required if not _present(model_id)]

    binaries = binary_requirements(settings)
    for requirement in binaries:
        check_id = str(requirement.get("check_id") or f"bin:{requirement['key']}")
        requirement["available"] = _binary_available(report, check_id)
        requirement["status"] = "ok" if requirement["available"] else "fail"
    binaries_ok = all(
        bool(requirement["available"])
        for requirement in binaries
        if requirement.get("needed")
    )

    hf_check_status = _check_status(report, "hf_token")
    hf_ok = hf_check_status != "fail"
    readiness_ok = _critical_failures(report) == 0

    steps: list[dict[str, object]] = [
        {
            "id": STEP_HARDWARE,
            "title": "Железо и бэкенд",
            "description": "Выберите, на чём считать: NVIDIA/CUDA, AMD/Vulkan или CPU.",
            "status": "ok",
            "action": STEP_HARDWARE,
        },
        {
            "id": STEP_HF_TOKEN,
            "title": "Токен Hugging Face",
            "description": (
                "Нужен для скачивания gated-модели pyannote. Если модель уже "
                "скачана и указана в настройках — токен не требуется."
            ),
            "status": "ok" if hf_ok else "todo",
            "action": STEP_HF_TOKEN,
            "required": hf_check_status == "fail",
            "set": hf_token_set,
        },
        {
            "id": STEP_MODELS,
            "title": "Модели",
            "description": "Скачайте недостающие модели (с предупреждением о размере).",
            "status": "ok" if not missing_models else "todo",
            "action": STEP_MODELS,
            "required": required,
            "missing": missing_models,
        },
        {
            "id": STEP_BINARIES,
            "title": "Бинарники и пакеты",
            "description": (
                "Внешние бинарники (whisper-cli, llama-server, deep-filter) "
                "скачиваются кнопкой «Скачать», пакеты (onnx-asr и др.) ставятся "
                "кнопкой «Установить»."
            ),
            "status": "ok" if binaries_ok else "todo",
            "action": STEP_BINARIES,
        },
        {
            "id": STEP_READINESS,
            "title": "Готовность",
            "description": (
                "Сводка: что установлено и чего не хватает (модели, пакеты, "
                "бинарники), доступность путей и размеры."
            ),
            "status": "ok" if readiness_ok else "todo",
            "action": STEP_READINESS,
        },
    ]

    readiness = _readiness_summary(
        required=required,
        model_by_id=model_by_id,
        binaries=binaries,
    )

    summary = report.get("summary")
    return {
        "hardware": {
            "options": [option.as_dict() for option in HARDWARE_OPTIONS],
            "current": current_hardware(settings),
        },
        "steps": steps,
        "required_models": required,
        "missing_models": missing_models,
        "binaries": binaries,
        "readiness": readiness,
        "hf_token": {"required": hf_check_status == "fail", "set": hf_token_set},
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }


def _readiness_summary(
    *,
    required: list[str],
    model_by_id: Mapping[str, Mapping[str, object]],
    binaries: list[dict[str, object]],
) -> dict[str, object]:
    """Сводка шага «Готовность»: модели, пакеты и бинарники — что есть/чего нет."""
    models: list[dict[str, object]] = []
    for model_id in required:
        entry = models_registry.find_model(model_id)
        status = model_by_id.get(model_id, {}).get("status")
        status_map = status if isinstance(status, Mapping) else {}
        present = bool(status_map.get("present"))
        path = status_map.get("path")
        models.append(
            {
                "id": model_id,
                "title": entry.title if entry is not None else model_id,
                "present": present,
                "size": int(status_map.get("size", 0) or 0),
                "expected_size": int(
                    status_map.get("expected_size")
                    or (entry.approx_size if entry is not None else 0)
                    or 0
                ),
                "path": str(path) if path else "",
            }
        )
    missing_models = [model["id"] for model in models if not model["present"]]

    dependencies: list[dict[str, object]] = []
    for dep in deps_registry.DEPENDENCIES:
        dependencies.append(
            {
                "key": dep.key,
                "label": dep.label,
                "spec": dep.spec,
                "installed": deps_registry.module_available(dep.module),
                "needed_for": dep.needed_for,
            }
        )
    missing_dependencies = [
        dep["key"] for dep in dependencies if not dep["installed"]
    ]

    binaries_summary: list[dict[str, object]] = []
    for requirement in binaries:
        artifact = requirement.get("artifact")
        artifact_map = artifact if isinstance(artifact, Mapping) else {}
        binaries_summary.append(
            {
                "key": requirement.get("key"),
                "label": requirement.get("label"),
                "needed": bool(requirement.get("needed")),
                "available": bool(requirement.get("available")),
                "downloadable": bool(requirement.get("downloadable")),
                "installed_path": requirement.get("installed_path") or "",
                "size": int(artifact_map.get("size", 0) or 0),
                "platform": requirement.get("platform") or "",
            }
        )
    missing_binaries = [
        item["key"] for item in binaries_summary if item["needed"] and not item["available"]
    ]

    return {
        "models": models,
        "dependencies": dependencies,
        "binaries": binaries_summary,
        "missing_models": missing_models,
        "missing_dependencies": missing_dependencies,
        "missing_binaries": missing_binaries,
    }
