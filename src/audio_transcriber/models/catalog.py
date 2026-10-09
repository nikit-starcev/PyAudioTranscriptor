"""Каталог известных моделей: реестр, пути, локальный статус, удаление.

Модуль не зависит от веб-слоя и сети: его переиспользуют веб-API (через
``audio_transcriber.web.models``) и CLI-команда ``models``. Скачивание
(единственная часть, ходящая в интернет) вынесено в
:mod:`audio_transcriber.models.download`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

#: Категории моделей.
KIND_WHISPER = "whisper-cpp"
KIND_LLM = "llm"
KIND_PYANNOTE = "pyannote"
KIND_GIGAAM = "gigaam"
KIND_SHERPA = "sherpa"


class ModelError(RuntimeError):
    """Ошибка каталога/загрузки модели с понятным для пользователя текстом."""


@dataclass(frozen=True, slots=True)
class ModelFile:
    """Один файл модели внутри целевого каталога."""

    filename: str
    approx_size: int = 0


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Запись каталога: описание модели и куда её класть."""

    id: str
    kind: str
    title: str
    repo: str
    files: tuple[ModelFile, ...]
    target_dir: str
    approx_size: int
    note: str = ""
    gated: bool = False
    setting_key: str = ""
    snapshot: bool = False

    @property
    def primary_filename(self) -> str | None:
        """Имя главного файла (для GGUF со шардами — первый)."""
        return self.files[0].filename if self.files else None

    @property
    def dir_name(self) -> str:
        """Имя целевого каталога (последний компонент ``target_dir``)."""
        return Path(self.target_dir).name


#: Каталог известных моделей. Скачиваются в ``<каталог моделей>/<target_dir>``,
#: если в настройках не задан явный путь (тогда используется он).
MODEL_CATALOG: tuple[ModelEntry, ...] = (
    ModelEntry(
        id="whisper-large-v3-turbo",
        kind=KIND_WHISPER,
        title="whisper.cpp large-v3-turbo (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-large-v3-turbo.bin", 1_624_555_275),),
        target_dir="whisper-models",
        approx_size=1_624_555_275,
        setting_key="whisper_cpp_model",
        note="Лучший баланс качество/скорость для whisper.cpp; требует бэкенд whisper-cpp.",
    ),
    ModelEntry(
        id="whisper-large-v3",
        kind=KIND_WHISPER,
        title="whisper.cpp large-v3 (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-large-v3.bin", 3_095_033_483),),
        target_dir="whisper-models",
        approx_size=3_095_033_483,
        setting_key="whisper_cpp_model",
        note="Максимальная точность whisper.cpp; заметно больше памяти и медленнее turbo.",
    ),
    ModelEntry(
        id="whisper-medium",
        kind=KIND_WHISPER,
        title="whisper.cpp medium (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-medium.bin", 1_533_763_059),),
        target_dir="whisper-models",
        approx_size=1_533_763_059,
        setting_key="whisper_cpp_model",
        note="Компромисс между качеством и потреблением памяти.",
    ),
    ModelEntry(
        id="whisper-small",
        kind=KIND_WHISPER,
        title="whisper.cpp small (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-small.bin", 487_601_967),),
        target_dir="whisper-models",
        approx_size=487_601_967,
        setting_key="whisper_cpp_model",
        note="Быстрая и компактная модель; качество ниже.",
    ),
    ModelEntry(
        id="whisper-base",
        kind=KIND_WHISPER,
        title="whisper.cpp base (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-base.bin", 147_951_465),),
        target_dir="whisper-models",
        approx_size=147_951_465,
        setting_key="whisper_cpp_model",
        note="Самая лёгкая из рекомендуемых моделей whisper.cpp; для быстрых черновиков.",
    ),
    ModelEntry(
        id="qwen2.5-7b-instruct-q4_k_m",
        kind=KIND_LLM,
        title="Qwen2.5-7B-Instruct Q4_K_M (GGUF)",
        repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        files=(
            ModelFile("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", 3_993_201_344),
            ModelFile("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", 689_872_288),
        ),
        target_dir="llama-models",
        approx_size=4_683_073_632,
        setting_key="llm_model",
        note="Модель для LLM-постобработки (llama-server); состоит из двух шардов.",
    ),
    ModelEntry(
        id="pyannote-community-1",
        kind=KIND_PYANNOTE,
        title="pyannote speaker-diarization-community-1",
        repo="pyannote/speaker-diarization-community-1",
        files=(),
        target_dir="pyannote-models/speaker-diarization-community-1",
        approx_size=33_554_432,
        gated=True,
        snapshot=True,
        setting_key="pyannote_local_model",
        note="Gated-модель: нужен токен HF и принятые условия использования.",
    ),
    ModelEntry(
        id="gigaam-v3-onnx",
        kind=KIND_GIGAAM,
        title="GigaAM v3 (ONNX, onnx-asr)",
        repo="istupakov/gigaam-v3-onnx",
        files=(),
        target_dir="gigaam-models/gigaam-v3-onnx",
        approx_size=4_455_303_019,
        snapshot=True,
        setting_key="gigaam_model_path",
        note=(
            "GigaAM v3 для русского через onnx-asr: нужен пакет "
            "onnx-asr[cpu,hub] и бэкенд gigaam. Снимок репозитория включает "
            "варианты ctc/rnnt/e2e в fp32 и int8; при выборе квантизации "
            "используется int8-часть."
        ),
    ),
    ModelEntry(
        id="sherpa-campplus-advanced",
        kind=KIND_SHERPA,
        title="3D-Speaker CAM++ (sherpa-onnx, эмбеддинги)",
        repo="csukuangfj/speaker-embedding-models",
        files=(
            ModelFile(
                "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                28_281_164,
            ),
        ),
        target_dir="sherpa-models",
        approx_size=28_281_164,
        setting_key="diarization_estimate_model",
        note=(
            "Модель эмбеддингов говорящего для оценщика числа говорящих и "
            "гибридной диаризации. Требуется пакет sherpa-onnx — без него "
            "оценка недоступна, а гибрид не выбирается."
        ),
    ),
)


def find_model(model_id: str) -> ModelEntry | None:
    """Запись каталога по идентификатору."""
    for entry in MODEL_CATALOG:
        if entry.id == model_id:
            return entry
    return None


def configured_path(entry: ModelEntry, settings: object) -> str:
    """Путь модели из настроек (``setting_key``) или пустая строка."""
    if not entry.setting_key:
        return ""
    raw = getattr(settings, entry.setting_key, "")
    return raw.strip() if isinstance(raw, str) else ""


def resolve_target(entry: ModelEntry, *, models_root: Path, settings: object) -> Path:
    """Каталог модели: заданный в настройках или ``<models_root>/<target_dir>``.

    Для файловых моделей заданный путь указывает на файл — берётся его каталог;
    для снимков (pyannote) — каталог как есть.
    """
    configured = configured_path(entry, settings)
    if configured:
        path = Path(configured).expanduser()
        if entry.snapshot:
            return path
        # Голое имя файла без каталога — это имя модели по умолчанию
        # (например, модель эмбеддингов sherpa-onnx задаётся именем), а не
        # путь: кладём/ищем её в каталоге каталога моделей, а не в CWD.
        if not path.parent.parts and not path.is_file():
            return models_root / entry.target_dir
        return path.parent
    return models_root / entry.target_dir


def primary_path(entry: ModelEntry, target_dir: Path) -> Path:
    """Главный артефакт модели: файл (первый шард) или каталог снимка."""
    if entry.snapshot:
        return target_dir
    name = entry.primary_filename
    return target_dir / name if name else target_dir


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def dir_size(directory: Path) -> int:
    """Суммарный размер файлов каталога (0 при ошибке доступа)."""
    total = 0
    try:
        for child in directory.rglob("*"):
            if child.is_file():
                total += _file_size(child)
    except OSError:
        return total
    return total


def _snapshot_present(directory: Path) -> bool:
    """Есть ли в каталоге снимка хотя бы один содержательный файл (без ``.cache``)."""
    if not directory.is_dir():
        return False
    try:
        for child in directory.rglob("*"):
            if child.is_file() and ".cache" not in child.parts:
                return True
    except OSError:
        return False
    return False


@dataclass(frozen=True, slots=True)
class LocalModelStatus:
    """Локальное состояние модели на диске."""

    present: bool
    size: int
    expected_size: int
    missing: tuple[str, ...]
    path: Path
    partial: bool = False


def local_status(entry: ModelEntry, target_dir: Path) -> LocalModelStatus:
    """Проверяет наличие файлов модели в ``target_dir`` (без сети)."""
    if entry.snapshot:
        present = _snapshot_present(target_dir)
        size = dir_size(target_dir)
        missing: tuple[str, ...] = () if present else (entry.dir_name,)
        return LocalModelStatus(
            present=present,
            size=size,
            expected_size=entry.approx_size,
            missing=missing,
            path=target_dir,
            partial=bool(size > 0 and not present),
        )
    missing = tuple(
        model_file.filename
        for model_file in entry.files
        if not (target_dir / model_file.filename).is_file()
    )
    size = sum(_file_size(target_dir / model_file.filename) for model_file in entry.files)
    return LocalModelStatus(
        present=not missing,
        size=size,
        expected_size=entry.approx_size,
        missing=missing,
        path=primary_path(entry, target_dir),
        partial=bool(missing and size > 0),
    )


def delete_model_files(entry: ModelEntry, target_dir: Path) -> bool:
    """Удаляет файлы модели; ``False`` — если удалять было нечего.

    Удаление ограничено ожидаемым артефактом: для снимка — только каталог с
    ожидаемым именем, для файлов — только файлы из каталога модели.
    """
    if entry.snapshot:
        if target_dir.name != entry.dir_name or not target_dir.is_dir():
            return False
        shutil.rmtree(target_dir, ignore_errors=True)
        return not target_dir.exists()
    removed = False
    for model_file in entry.files:
        candidate = target_dir / model_file.filename
        try:
            if candidate.is_file():
                candidate.unlink()
                removed = True
        except OSError:
            continue
    return removed


def free_space(path: Path) -> int:
    """Свободное место на диске для каталога (0, если определить не удалось)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0
