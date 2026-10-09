"""CLI-команда ``models``: просмотр, скачивание и удаление моделей.

Переиспользует тот же движок, что и веб-интерфейс
(:mod:`audio_transcriber.models`): каталог, проверку локального статуса,
менеджер фоновой загрузки с докачкой и удаление файлов. Это позволяет
получать модели без веб-UI — для Docker и скриптов, а также как шаг
first-run (флаг ``--download-models`` у команды ``web``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import typer

from audio_transcriber.models import (
    KIND_GIGAAM,
    KIND_LLM,
    KIND_PYANNOTE,
    KIND_SHERPA,
    KIND_WHISPER,
    MODEL_CATALOG,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_DOWNLOADING,
    STATUS_ERROR,
    Downloader,
    DownloadState,
    HfDownloader,
    ModelDownloadManager,
    ModelEntry,
    delete_model_files,
    find_model,
    free_space,
    local_status,
    resolve_target,
)
from audio_transcriber.utils.config_env import load_config_env, parse_bool
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.secrets import SecretsStore, effective_hf_token

#: Повторы загрузки при ошибке сети (число дополнительных попыток).
DEFAULT_DOWNLOAD_RETRIES = 2

#: Интервал опроса состояния загрузки при выводе прогресса (секунды).
_POLL_INTERVAL = 0.2

#: Человекочитаемое назначение категории модели.
_PURPOSE: Mapping[str, str] = {
    KIND_WHISPER: "ASR (whisper.cpp)",
    KIND_LLM: "LLM-постобработка",
    KIND_PYANNOTE: "Диаризация (pyannote)",
    KIND_GIGAAM: "ASR (GigaAM)",
    KIND_SHERPA: "Эмбеддинги говорящего",
}


@dataclass(frozen=True, slots=True)
class ModelPathSettings:
    """Пути моделей из ``config.env`` (те же поля, что у веб-настроек)."""

    whisper_cpp_model: str = ""
    llm_model: str = ""
    pyannote_local_model: str = ""
    gigaam_model_path: str = ""
    diarization_estimate_model: str = ""

    @classmethod
    def from_env(cls, defaults: Mapping[str, str]) -> ModelPathSettings:
        return cls(
            whisper_cpp_model=defaults.get("WHISPER_CPP_MODEL", "").strip(),
            llm_model=defaults.get("LLM_MODEL", "").strip(),
            pyannote_local_model=defaults.get("PYANNOTE_LOCAL_MODEL", "").strip(),
            gigaam_model_path=defaults.get("GIGAAM_MODEL_PATH", "").strip(),
            diarization_estimate_model=defaults.get("DIARIZATION_ESTIMATE_MODEL", "").strip(),
        )


def _format_size(num_bytes: int) -> str:
    """Человекочитаемый размер (Б/КБ/МБ/ГБ/ТБ)."""
    value = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024
    return f"{int(num_bytes)} Б"


def _default_models_root() -> Path:
    """Каталог моделей по умолчанию — как у веб-интерфейса."""
    return WebPaths.default().models_dir


def _resolve_models_root(raw: Path | None) -> Path:
    return raw if raw is not None else _default_models_root()


def _load_context() -> tuple[ModelPathSettings, str | None, dict[str, str]]:
    """Настройки путей, HF-токен и объединённое окружение ``config.env``/process."""
    _, file_env = load_config_env()
    merged = {**os.environ, **file_env}
    settings = ModelPathSettings.from_env(merged)
    token = effective_hf_token(SecretsStore(WebPaths.default().secrets_json), merged)
    return settings, token, merged


def _make_manager(
    models_root: Path, settings: ModelPathSettings, downloader: Downloader
) -> ModelDownloadManager:
    return ModelDownloadManager(
        models_root=models_root,
        downloader=downloader,
        resolve_target=lambda entry: resolve_target(
            entry, models_root=models_root, settings=settings
        ),
    )


def _print_progress(model_id: str, state: DownloadState, *, final: bool = False) -> None:
    if state.status != STATUS_DOWNLOADING or state.total <= 0:
        return
    typer.echo(
        f"\r  {model_id}: {state.fraction * 100:5.1f}% "
        f"({_format_size(state.bytes_done)} / {_format_size(state.total)})",
        nl=False,
    )
    if final:
        typer.echo("")


def _wait_and_report(manager: ModelDownloadManager, entry: ModelEntry) -> DownloadState:
    while manager.is_running(entry.id):
        _print_progress(entry.id, manager.state(entry.id))
        time.sleep(_POLL_INTERVAL)
    state = manager.state(entry.id)
    _print_progress(entry.id, state, final=True)
    return state


def _download_one(
    entry: ModelEntry,
    *,
    manager: ModelDownloadManager,
    token: str | None,
    retries: int,
) -> tuple[bool, str]:
    """Скачивает одну модель с повторами; возвращает ``(успех, сообщение)``."""
    attempts = max(retries, 0) + 1
    for attempt in range(attempts):
        if not manager.start(entry, token=token):
            manager.wait(entry.id)
        state = _wait_and_report(manager, entry)
        if state.status == STATUS_DONE:
            return True, f"скачана ({_format_size(state.bytes_done)})"
        if state.status == STATUS_CANCELLED:
            return False, "отменено"
        if state.status == STATUS_ERROR and attempt + 1 < attempts:
            typer.echo(
                f"  ! {entry.id}: {state.error or 'ошибка загрузки'}; "
                f"повтор {attempt + 2}/{attempts}"
            )
            continue
        return False, state.error or state.message or "не удалось скачать"
    return False, "не удалось скачать"


def _download_entries(
    entries: list[ModelEntry],
    *,
    models_root: Path,
    settings: ModelPathSettings,
    token: str | None,
    retries: int,
    skip_gated: bool,
) -> int:
    """Скачивает отсутствующие модели; возвращает число неудач."""
    manager = _make_manager(models_root, settings, HfDownloader())
    failures = 0
    for entry in entries:
        target = resolve_target(entry, models_root=models_root, settings=settings)
        status = local_status(entry, target)
        if status.present:
            typer.echo(f"✓ {entry.id}: уже скачана ({_format_size(status.size)}) — пропуск")
            continue
        if entry.gated and not token:
            if skip_gated:
                typer.echo(f"– {entry.id}: пропуск — нужен токен Hugging Face (gated)")
                continue
            typer.echo(
                f"✗ {entry.id}: нужен токен Hugging Face для gated-модели "
                "(HF_TOKEN в config.env или токен в web-data/secrets.json).",
                err=True,
            )
            failures += 1
            continue
        remaining = max(entry.approx_size - status.size, 0)
        available = free_space(models_root)
        if available and remaining and available < remaining:
            typer.echo(
                f"✗ {entry.id}: недостаточно места на диске — нужно ещё "
                f"~{_format_size(remaining)}, свободно ~{_format_size(available)}.",
                err=True,
            )
            failures += 1
            continue
        typer.echo(f"↓ {entry.id} ({_format_size(entry.approx_size)}): {entry.title}")
        ok, message = _download_one(entry, manager=manager, token=token, retries=retries)
        if ok:
            typer.echo(f"✓ {entry.id}: {message}")
        else:
            typer.echo(f"✗ {entry.id}: {message}", err=True)
            failures += 1
    return failures


def _whisper_entry_for(defaults: Mapping[str, str]) -> ModelEntry | None:
    """Whisper-модель каталога, соответствующая ``WHISPER_CPP_MODEL`` (или turbo)."""
    raw = defaults.get("WHISPER_CPP_MODEL", "").strip()
    if raw:
        name = Path(raw).name
        for entry in MODEL_CATALOG:
            if entry.kind == KIND_WHISPER and entry.primary_filename == name:
                return entry
    return find_model("whisper-large-v3-turbo")


def required_model_entries(defaults: Mapping[str, str]) -> list[ModelEntry]:
    """Модели каталога, нужные текущему ``config.env`` (для first-run)."""
    entries: list[ModelEntry] = []

    def add(entry: ModelEntry | None) -> None:
        if entry is not None and entry not in entries:
            entries.append(entry)

    backend = defaults.get("ASR_BACKEND", "").strip().casefold() or "faster-whisper"
    if backend == "whisper-cpp":
        add(_whisper_entry_for(defaults))
    elif backend == "gigaam":
        add(find_model("gigaam-v3-onnx"))
    if parse_bool(defaults.get("LLM_ENABLED")):
        add(find_model("qwen2.5-7b-instruct-q4_k_m"))
    if parse_bool(defaults.get("DIARIZATION_ESTIMATE_ENABLED"), default=True):
        add(find_model("sherpa-campplus-advanced"))
    engine = defaults.get("DIARIZATION_ENGINE", "").strip().casefold() or "auto"
    if engine in {"auto", "pyannote", "hybrid"}:
        add(find_model("pyannote-community-1"))
    return entries


def download_for_web(*, models_root: Path | None = None) -> int:
    """First-run: скачать отсутствующие модели, нужные ``config.env``.

    Возвращает число неудач (gated-модели без токена мягко пропускаются).
    Используется флагом ``--download-models`` команды ``web``.
    """
    root = _resolve_models_root(models_root)
    settings, token, env = _load_context()
    entries = required_model_entries(env)
    if not entries:
        typer.echo("Для текущего config.env дополнительные модели каталога не требуются.")
        return 0
    typer.echo(f"Каталог моделей: {root}")
    return _download_entries(
        entries,
        models_root=root,
        settings=settings,
        token=token,
        retries=DEFAULT_DOWNLOAD_RETRIES,
        skip_gated=True,
    )


def _print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    typer.echo(template.format(*headers))
    typer.echo("  ".join("-" * width for width in widths))
    for row in rows:
        typer.echo(template.format(*row))


models_app = typer.Typer(
    name="models",
    help=(
        "Скачивание, просмотр и удаление моделей (whisper.cpp, LLM, pyannote, "
        "GigaAM, sherpa) без веб-интерфейса."
    ),
    add_completion=False,
    no_args_is_help=True,
)

_MODELS_ROOT_OPTION = typer.Option(
    None,
    "--models-root",
    help=(
        "Каталог моделей. По умолчанию как у веб-интерфейса: "
        "AUDIO_TRANSCRIBER_MODELS_DIR или <web-data>/models."
    ),
)
_MODEL_IDS_ARGUMENT = typer.Argument(
    [],
    help="Идентификаторы моделей (можно несколько).",
)
_DELETE_ID_ARGUMENT = typer.Argument(..., help="Идентификатор модели.")
_ALL_OPTION = typer.Option(
    False,
    "--all",
    help="Скачать все отсутствующие модели каталога.",
)
_RETRIES_OPTION = typer.Option(
    DEFAULT_DOWNLOAD_RETRIES,
    "--retries",
    min=0,
    help=f"Число повторов при ошибке сети (по умолчанию {DEFAULT_DOWNLOAD_RETRIES}).",
)


@models_app.command("list")
def models_list(
    models_root: Path | None = _MODELS_ROOT_OPTION,
) -> None:
    """Показать каталог моделей: назначение, размер, статус и путь."""
    root = _resolve_models_root(models_root)
    settings, _token, _env = _load_context()
    rows: list[tuple[str, ...]] = []
    for entry in MODEL_CATALOG:
        target = resolve_target(entry, models_root=root, settings=settings)
        status = local_status(entry, target)
        rows.append(
            (
                entry.id,
                _PURPOSE.get(entry.kind, entry.kind),
                _format_size(entry.approx_size),
                "✓ present" if status.present else "✗ missing",
                "да" if entry.gated else "—",
                str(status.path),
            )
        )
    _print_table(("ID", "НАЗНАЧЕНИЕ", "РАЗМЕР", "СТАТУС", "GATED", "ПУТЬ"), rows)
    typer.echo(f"\nКаталог моделей: {root}")


@models_app.command("download")
def models_download(
    model_ids: list[str] = _MODEL_IDS_ARGUMENT,
    all_: bool = _ALL_OPTION,
    models_root: Path | None = _MODELS_ROOT_OPTION,
    retries: int = _RETRIES_OPTION,
) -> None:
    """Скачать отсутствующие модели (идемпотентно: уже скачанные пропускаются)."""
    if all_ and model_ids:
        typer.echo("Ошибка: укажите либо --all, либо идентификаторы моделей.", err=True)
        raise typer.Exit(code=2)
    if not all_ and not model_ids:
        typer.echo("Ошибка: укажите идентификаторы моделей или --all.", err=True)
        raise typer.Exit(code=2)

    root = _resolve_models_root(models_root)
    settings, token, _env = _load_context()

    if all_:
        entries = list(MODEL_CATALOG)
        skip_gated = True
    else:
        entries = []
        unknown = [model_id for model_id in model_ids if find_model(model_id) is None]
        for model_id in model_ids:
            entry = find_model(model_id)
            if entry is not None:
                entries.append(entry)
        for model_id in unknown:
            typer.echo(f"Модель «{model_id}» не найдена в каталоге.", err=True)
        if unknown:
            raise typer.Exit(code=1)
        skip_gated = False

    failures = _download_entries(
        entries,
        models_root=root,
        settings=settings,
        token=token,
        retries=retries,
        skip_gated=skip_gated,
    )
    if failures:
        raise typer.Exit(code=1)


@models_app.command("delete")
def models_delete(
    model_id: str = _DELETE_ID_ARGUMENT,
    models_root: Path | None = _MODELS_ROOT_OPTION,
) -> None:
    """Удалить файлы модели с диска."""
    entry = find_model(model_id)
    if entry is None:
        typer.echo(f"Модель «{model_id}» не найдена в каталоге.", err=True)
        raise typer.Exit(code=1)
    root = _resolve_models_root(models_root)
    settings, _token, _env = _load_context()
    target = resolve_target(entry, models_root=root, settings=settings)
    if delete_model_files(entry, target):
        typer.echo(f"✓ {model_id}: файлы удалены ({target})")
    else:
        typer.echo(f"– {model_id}: локальных файлов нет — удалять нечего.")
