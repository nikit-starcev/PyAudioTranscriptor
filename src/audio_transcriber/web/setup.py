"""План мастера первого запуска (шаги онбординга).

Модуль собирает из отчёта готовности, текущих настроек и состояния каталога
моделей единый «план мастера»: какие шаги ещё не выполнены и что нужно
скачать/указать. Это единственный источник истины для SPA — фронт только
отображает план, а тестируется он здесь (без браузера).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from audio_transcriber.doctor import LINK_LLAMA_CPP, LINK_WHISPER_CPP
from audio_transcriber.domain.enums import AsrBackend
from audio_transcriber.web import deps as deps_registry

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


def binary_requirements(settings: object) -> list[dict[str, object]]:
    """Нужные внешние компоненты (бинарники/пакеты), их доступность и инструкции.

    Авто-скачивания нет: бинарники whisper.cpp/llama.cpp собираются под ОС/GPU
    вручную, а пакет ``onnx-asr`` ставится из PyPI. Поле ``check_id`` задаёт
    идентификатор проверки доктора (по умолчанию ``bin:<key>``); для пакета
    используется ``dep:<модуль>``.
    """
    backend = getattr(settings, "asr_backend", AsrBackend.FASTER_WHISPER.value)
    llm_enabled = bool(getattr(settings, "llm_enabled", False))
    requirements: list[dict[str, object]] = []
    if backend == AsrBackend.WHISPER_CPP.value:
        requirements.append(
            {
                "key": "whisper-cli",
                "label": "Бинарник whisper-cli (whisper.cpp)",
                "needed": True,
                "instructions": (
                    "Соберите whisper.cpp под свою ОС/GPU, положите whisper-cli и его "
                    "библиотеки в один каталог и укажите путь в настройках "
                    "(WHISPER_CPP_BINARY / WHISPER_CPP_LIB_PATH)."
                ),
                "links": [LINK_WHISPER_CPP],
            }
        )
    if backend == AsrBackend.GIGAAM.value:
        dep = deps_registry.find_dependency("gigaam")
        if dep is not None:
            requirements.append(
                {
                    "key": "onnx-asr",
                    "check_id": dep.check_id,
                    "label": dep.label,
                    "needed": True,
                    # Пакет можно поставить кнопкой из мастера (#66): spec берётся
                    # из allowlist реестра, признак „есть установщик“ — из deps.
                    "dep_key": dep.key,
                    "spec": dep.spec,
                    "installable": deps_registry.installer_available(),
                    "instructions": (
                        f"Пакет ставится кнопкой «Установить» ({dep.spec}). "
                        "Вручную: uv pip install "
                        f"'{dep.spec}'. Модель GigaAM подтянется из каталога "
                        "моделей или с Hugging Face."
                    ),
                    "links": [],
                }
            )
    if llm_enabled:
        requirements.append(
            {
                "key": "llama-server",
                "label": "Бинарник llama-server (llama.cpp)",
                "needed": True,
                "instructions": (
                    "Соберите llama.cpp, положите llama-server и его библиотеки "
                    "(libggml-vulkan.so и т.п.) в один каталог и укажите путь в "
                    "настройках (LLM_BINARY / LLM_LIB_PATH)."
                ),
                "links": [LINK_LLAMA_CPP],
            }
        )
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
                "whisper-cli и llama-server собираются под вашу ОС/GPU вручную, "
                "а опциональные пакеты (onnx-asr и др.) ставятся кнопкой."
            ),
            "status": "ok" if binaries_ok else "todo",
            "action": STEP_BINARIES,
        },
        {
            "id": STEP_READINESS,
            "title": "Готовность",
            "description": "Повторная проверка окружения доктором.",
            "status": "ok" if readiness_ok else "todo",
            "action": STEP_READINESS,
        },
    ]

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
        "hf_token": {"required": hf_check_status == "fail", "set": hf_token_set},
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }
