"""Реестр внешних ресурсов веб-мастера: Python-пакеты и бинарные архивы (#98).

Раньше кнопкой из интерфейса ставились только модели (Hugging Face) и
pip-пакеты (gigaam/sherpa). Issue #98 требует, чтобы **всё внешнее**, что
скачивается со стороны, ставилось кнопкой. Здесь вводится единый реестр
:data:`ASSETS`:

* ``kind="pip"`` — Python-пакеты (тонкая обёртка над allowlist
  :data:`~audio_transcriber.web.deps.DEPENDENCIES`, кнопка «Установить»);
* ``kind="binary"`` — бинарные архивы/файлы (кнопка «Скачать»): streamed
  download → проверка ``sha256`` → распаковка в ``web-data/bin/<key>/`` →
  ``chmod +x`` → авто-прописывание пути в настройках.

Безопасность: URL и хеши берутся **только** из этого модуля (фиксированный
allowlist по ОС/архитектуре), пользовательский ввод в сетевой запрос не
попадает. Распаковка защищена от path traversal (``tarfile`` с
``filter="data"``, ручная проверка членов zip).

Прогресс установки идёт в общую шину :class:`~audio_transcriber.web.models.DownloadBus`
(SSE ``/api/assets/events``, он же ``/api/deps/events``) — с монотонным
``seq``/``Last-Event-ID``, как у моделей и pip-пакетов.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform as platform_module
import shutil
import stat
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from audio_transcriber.web.deps import DEPENDENCIES, DependencySpec, module_available
from audio_transcriber.web.models import DownloadBus

logger = logging.getLogger(__name__)

__all__ = [
    "ASSETS",
    "BINARY_ASSETS",
    "KIND_BINARY",
    "KIND_PIP",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_IDLE",
    "STATUS_RUNNING",
    "AssetArtifact",
    "AssetError",
    "AssetSpec",
    "BinaryInstaller",
    "BinaryState",
    "ChecksumError",
    "UnsupportedPlatform",
    "artifact_payload",
    "asset_binary_path",
    "asset_payload",
    "current_arch",
    "current_os",
    "find_asset",
    "install_binary_asset",
    "resolve_artifact",
    "sha256_file",
    "stream_download",
]

#: Виды внешних ресурсов.
KIND_PIP = "pip"
KIND_BINARY = "binary"

#: Статусы операции установки (совпадают с пакетами — общий SSE/UI).
STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

#: Порог публикации прогресса (доля целого) и минимальный интервал событий.
_PROGRESS_STEP = 0.01
_PROGRESS_INTERVAL = 0.25

#: Читаемый размер сетевой загрузки (байт).
_CHUNK = 262_144


class AssetError(RuntimeError):
    """Ошибка внешнего ресурса с понятным пользователю текстом."""


class ChecksumError(AssetError):
    """Скачанный файл не совпал с ожидаемым ``sha256``."""


class UnsupportedPlatform(AssetError):
    """Для текущей ОС/архитектуры нет готового артефакта (нужна инструкция)."""


def current_os() -> str:
    """ОС в терминах allowlist: ``linux`` / ``windows`` / ``macos``."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def current_arch() -> str:
    """Архитектура в терминах allowlist: ``x86_64`` или ``arm64``."""
    machine = platform_module.machine().strip().lower()
    if machine in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return machine or "unknown"


@dataclass(frozen=True, slots=True)
class AssetArtifact:
    """Один разрешённый артефакт: URL + ``sha256`` + способ распаковки."""

    os: str
    arch: str
    url: str
    sha256: str
    size: int = 0
    #: ``none`` (одиночный файл) / ``zip`` / ``tar.gz``.
    archive: str = "none"
    #: Имя бинарника внутри архива (ищется рекурсивно после распаковки).
    member: str = ""
    #: Имя сохраняемого файла для ``archive="none"``.
    filename: str = ""
    #: Вариант сборки: ``cpu`` / ``vulkan`` / ``metal``.
    variant: str = "cpu"

    def as_dict(self) -> dict[str, object]:
        return {
            "os": self.os,
            "arch": self.arch,
            "url": self.url,
            "sha256": self.sha256,
            "size": self.size,
            "archive": self.archive,
            "variant": self.variant,
        }


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Элемент единого allowlist внешних ресурсов."""

    key: str
    label: str
    kind: str
    needed_for: str
    check_id: str
    optional: bool = False
    note: str = ""
    #: Инструкция для случаев, когда готового артефакта нет вовсе.
    manual_hint: str = ""
    # --- pip ---
    spec: str = ""
    module: str = ""
    # --- binary ---
    artifacts: tuple[AssetArtifact, ...] = ()
    #: Имя бинарника (для поиска в PATH и внутри архива).
    bin_name: str = ""
    #: Атрибут :class:`~audio_transcriber.web.settings.WebSettings` с путём.
    settings_field: str = ""
    #: Атрибут настроек с каталогом библиотек (если есть).
    lib_settings_field: str = ""
    #: Соответствующие ключи ``config.env``.
    env_key: str = ""
    lib_env_key: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "needed_for": self.needed_for,
            "check_id": self.check_id,
            "optional": self.optional,
            "note": self.note,
        }


def _pip_asset(dep: DependencySpec) -> AssetSpec:
    """Представление pip-пакета из :data:`DEPENDENCIES` в едином реестре."""
    return AssetSpec(
        key=dep.key,
        label=dep.label,
        kind=KIND_PIP,
        needed_for=dep.needed_for,
        check_id=dep.check_id,
        spec=dep.spec,
        module=dep.module,
    )


#: Ссылка на релизы DeepFilterNet (используется в инструкциях).
_DEEP_FILTER_BASE = "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6"
#: Ссылка на релизы llama.cpp.
_LLAMA_BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11466"
#: Ссылка на релизы whisper.cpp.
_WHISPER_BASE = "https://github.com/ggml-org/whisper.cpp/releases/download/b5454"

#: Бинарные ресурсы (кнопка «Скачать»). URL/хеши зафиксированы по ОС/арх.
BINARY_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        key="deep-filter",
        label="Бинарник deep-filter (DeepFilterNet, денойз)",
        kind=KIND_BINARY,
        needed_for="Шумоподавление на этапе denoise",
        check_id="bin:deep-filter",
        optional=True,
        bin_name="deep-filter",
        settings_field="deep_filter_binary",
        env_key="DEEP_FILTER_BINARY",
        note="Опционально: без бинарника этап денойза мягко пропускается.",
        manual_hint=(
            "Для вашей ОС/архитектуры готового артефакта нет. Соберите "
            "DeepFilterNet v0.5.6 или скачайте вручную с релизов и укажите путь "
            "в DEEP_FILTER_BINARY (настройки → «Шумоподавление»)."
        ),
        artifacts=(
            AssetArtifact(
                os="linux",
                arch="x86_64",
                url=f"{_DEEP_FILTER_BASE}/deep-filter-0.5.6-x86_64-unknown-linux-musl",
                sha256="70775e251eee44c0f2451a1e833326cf8bcbbe304d3e7cd12851e6fce72ef7da",
                size=36_417_296,
                archive="none",
                filename="deep-filter",
            ),
            AssetArtifact(
                os="linux",
                arch="arm64",
                url=f"{_DEEP_FILTER_BASE}/deep-filter-0.5.6-aarch64-unknown-linux-gnu",
                sha256="14e02a1c0028f3ca0bdf83b62b3336e56ba0556894ef295a95e8573f06557166",
                size=39_238_496,
                archive="none",
                filename="deep-filter",
            ),
            AssetArtifact(
                os="windows",
                arch="x86_64",
                url=f"{_DEEP_FILTER_BASE}/deep-filter-0.5.6-x86_64-pc-windows-msvc.exe",
                sha256="75e11fa16445f560cb6b021521ddb89e89270d13b83089705d98776f58fd7915",
                size=26_912_256,
                archive="none",
                filename="deep-filter.exe",
            ),
            AssetArtifact(
                os="macos",
                arch="arm64",
                url=f"{_DEEP_FILTER_BASE}/deep-filter-0.5.6-aarch64-apple-darwin",
                sha256="4601e7f4e4c03e59a4c5b5000216ef3add3e808799cfccd95e14e83ea4611081",
                size=27_877_081,
                archive="none",
                filename="deep-filter",
            ),
            AssetArtifact(
                os="macos",
                arch="x86_64",
                url=f"{_DEEP_FILTER_BASE}/deep-filter-0.5.6-x86_64-apple-darwin",
                sha256="d3be84003acb7c23e738ad7f70a158ec779a8d233a82e7fa3e717d112eb5b50f",
                size=29_933_512,
                archive="none",
                filename="deep-filter",
            ),
        ),
    ),
    AssetSpec(
        key="whisper-cli",
        label="Бинарник whisper-cli (whisper.cpp)",
        kind=KIND_BINARY,
        needed_for="Бэкенд распознавания whisper-cpp (CPU)",
        check_id="bin:whisper-cli",
        bin_name="whisper-cli",
        settings_field="whisper_cpp_binary",
        lib_settings_field="whisper_cpp_lib_path",
        env_key="WHISPER_CPP_BINARY",
        lib_env_key="WHISPER_CPP_LIB_PATH",
        note=(
            "Готовые сборки whisper.cpp — только CPU. Для Vulkan/GPU соберите "
            "whisper.cpp вручную и укажите WHISPER_CPP_BINARY/"
            "WHISPER_CPP_LIB_PATH."
        ),
        manual_hint=(
            "Для вашей ОС нет готовой CPU-сборки whisper.cpp. Соберите "
            "whisper.cpp (см. README) и укажите WHISPER_CPP_BINARY/"
            "WHISPER_CPP_LIB_PATH."
        ),
        artifacts=(
            AssetArtifact(
                os="linux",
                arch="x86_64",
                url=f"{_WHISPER_BASE}/whisper-bin-ubuntu-x64.tar.gz",
                sha256="a72becf15d7917f990f6313867a52638b82b7f9ef237fb0c980dac56a135781c",
                size=10_364_195,
                archive="tar.gz",
                member="whisper-cli",
            ),
            AssetArtifact(
                os="windows",
                arch="x86_64",
                url=f"{_WHISPER_BASE}/whisper-bin-x64.zip",
                sha256="6ba69e3482d7826214f90a6a9c84ca07782aec1e1d0c6a7c30c994fd5d816ccb",
                size=8_928_640,
                archive="zip",
                member="whisper-cli.exe",
            ),
        ),
    ),
    AssetSpec(
        key="llama-server",
        label="Бинарник llama-server (llama.cpp)",
        kind=KIND_BINARY,
        needed_for="Локальная LLM-постобработка (llama-server)",
        check_id="bin:llama-server",
        bin_name="llama-server",
        settings_field="llm_binary",
        lib_settings_field="llm_lib_path",
        env_key="LLM_BINARY",
        lib_env_key="LLM_LIB_PATH",
        note=(
            "По умолчанию берётся Vulkan-сборка на Linux/Windows (AMD/Radeon) и "
            "Metal на macOS; при CPU-режиме — CPU-сборка."
        ),
        manual_hint=(
            "Для вашей ОС/архитектуры готового артефакта нет. Соберите "
            "llama.cpp и укажите LLM_BINARY/LLM_LIB_PATH."
        ),
        artifacts=(
            AssetArtifact(
                os="linux",
                arch="x86_64",
                url=f"{_LLAMA_BASE}/llama-b11466-bin-ubuntu-vulkan-x64.tar.gz",
                sha256="5875e5d9d8f9d6dfc0537e9840e327ec2adbaafb38e608b542c57e3edd714651",
                size=31_678_354,
                archive="tar.gz",
                member="llama-server",
                variant="vulkan",
            ),
            AssetArtifact(
                os="linux",
                arch="x86_64",
                url=f"{_LLAMA_BASE}/llama-b11466-bin-ubuntu-x64.tar.gz",
                sha256="bfd3560ee562783fd8c1215a3a3f2c6b4c4e04a27f1955673f66c89182e06877",
                size=17_732_626,
                archive="tar.gz",
                member="llama-server",
                variant="cpu",
            ),
            AssetArtifact(
                os="windows",
                arch="x86_64",
                url=f"{_LLAMA_BASE}/llama-b11466-bin-win-vulkan-x64.zip",
                sha256="a703cbb4ec25329864640b05c0b4db58a67eb1759954cc6fdd4f51e6d0871e86",
                size=33_377_550,
                archive="zip",
                member="llama-server.exe",
                variant="vulkan",
            ),
            AssetArtifact(
                os="windows",
                arch="x86_64",
                url=f"{_LLAMA_BASE}/llama-b11466-bin-win-cpu-x64.zip",
                sha256="625b40212bcfbeb072f9ff6ad9f41e1234b2cd6d03f19dc9f0978495d58acbf8",
                size=19_436_661,
                archive="zip",
                member="llama-server.exe",
                variant="cpu",
            ),
            AssetArtifact(
                os="macos",
                arch="arm64",
                url=f"{_LLAMA_BASE}/llama-b11466-bin-macos-arm64.tar.gz",
                sha256="3a241f477fe65cbaacbccc8e4e8bb469b7cd9587934b856907ba7c2284d58795",
                size=12_007_670,
                archive="tar.gz",
                member="llama-server",
                variant="metal",
            ),
            AssetArtifact(
                os="macos",
                arch="x86_64",
                url=f"{_LLAMA_BASE}/llama-b11466-bin-macos-x64.tar.gz",
                sha256="796ce7e49e899360b1b31dd1f493678c09e174b48e86776909d4938cd8dd61fe",
                size=11_525_387,
                archive="tar.gz",
                member="llama-server",
                variant="metal",
            ),
        ),
    ),
    AssetSpec(
        key="nemo-speech",
        label="Бинарник nemo-speech (NeMo-Speech.cpp)",
        kind=KIND_BINARY,
        needed_for="Точная диаризация NeMo (движки nemo-speech/hybrid)",
        check_id="bin:nemo-speech",
        bin_name="nemo-speech",
        settings_field="nemo_speech_binary",
        lib_settings_field="nemo_speech_lib_path",
        env_key="NEMO_SPEECH_BINARY",
        lib_env_key="NEMO_SPEECH_LIB_PATH",
        note=(
            "Готовый бандл под стабильным URL пока не публикуется: бинарник "
            "собирается отдельно, модель Sortformer тянется командой "
            "'nemo-speech pull …'."
        ),
        manual_hint=(
            "Скачать нельзя: соберите NeMo-Speech.cpp под свою ОС/GPU, положите "
            "бинарник и его библиотеки и укажите NEMO_SPEECH_BINARY/"
            "NEMO_SPEECH_LIB_PATH."
        ),
        artifacts=(),
    ),
)

#: Единый реестр: pip-пакеты (из allowlist :mod:`deps`) + бинарные ресурсы.
ASSETS: tuple[AssetSpec, ...] = tuple(_pip_asset(dep) for dep in DEPENDENCIES) + BINARY_ASSETS

_ASSETS_BY_KEY: dict[str, AssetSpec] = {asset.key: asset for asset in ASSETS}


def find_asset(key: str) -> AssetSpec | None:
    """Ресурс реестра по ключу (``None`` — ключ не разрешён)."""
    return _ASSETS_BY_KEY.get(key)


def resolve_artifact(
    asset: AssetSpec,
    *,
    os_name: str | None = None,
    arch: str | None = None,
    prefer_vulkan: bool = False,
) -> AssetArtifact | None:
    """Подбирает артефакт под ОС/архитектуру (``None`` — готового нет).

    ``prefer_vulkan`` выбирает Vulkan-вариант (Linux/Windows, AMD); иначе —
    CPU/Metal. Если нужного варианта нет, возвращается первый доступный.
    """
    target_os = os_name or current_os()
    target_arch = arch or current_arch()
    candidates = [
        artifact
        for artifact in asset.artifacts
        if artifact.os == target_os and artifact.arch == target_arch
    ]
    if not candidates:
        return None
    if prefer_vulkan:
        for artifact in candidates:
            if artifact.variant == "vulkan":
                return artifact
    for artifact in candidates:
        if artifact.variant in {"cpu", "metal", ""}:
            return artifact
    return candidates[0]


def configured_binary(asset: AssetSpec, settings: object) -> str:
    """Путь бинарника из настроек (``settings_field``) или пустая строка."""
    if not asset.settings_field:
        return ""
    raw = getattr(settings, asset.settings_field, "")
    return raw.strip() if isinstance(raw, str) else ""


def _find_member(directory: Path, name: str) -> Path | None:
    """Рекурсивно ищет файл с именем ``name`` внутри ``directory``."""
    if not directory.is_dir():
        return None
    direct = directory / name
    if direct.is_file():
        return direct
    try:
        for child in directory.rglob(name):
            if child.is_file():
                return child
    except OSError:
        return None
    return None


def asset_binary_path(
    asset: AssetSpec,
    *,
    bin_root: Path | None = None,
    settings: object | None = None,
) -> Path | None:
    """Локальный путь бинарника: из настроек → из ``web-data/bin`` → из ``PATH``."""
    if settings is not None:
        configured = configured_binary(asset, settings)
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_file():
                return candidate
    if bin_root is not None and asset.bin_name:
        found = _find_member(bin_root / asset.key, asset.bin_name)
        if found is not None:
            return found
    if asset.bin_name:
        which = shutil.which(asset.bin_name)
        if which:
            return Path(which)
    return None


def sha256_file(path: Path) -> str:
    """``sha256`` файла (потоково, не читая целиком в память)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_download(
    url: str, destination: Path, on_progress: Callable[[int], None]
) -> None:
    """Скачивает URL потоком в файл, сообщая о приросте байт (реальная сеть)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            on_progress(len(chunk))


#: Функция сетевой загрузки (подменяется в тестах).
FetchFn = Callable[[str, Path, Callable[[int], None]], None]


def _reset_dir(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def _ensure_executable(path: Path) -> None:
    """Выставляет бит выполнения (на Windows — no-op по смыслу)."""
    if os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        logger.warning("Не удалось выставить +x для %s", path)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise AssetError(f"Небезопасный путь в архиве: {member.filename}")
        zip_file.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive) as tar_file:
            tar_file.extractall(destination, filter="data")
    except tarfile.TarError as exc:
        raise AssetError(f"Повреждённый или небезопасный tar-архив: {exc}") from exc


def _extract_archive(archive_type: str, archive: Path, destination: Path) -> None:
    if archive_type == "zip":
        _safe_extract_zip(archive, destination)
        return
    if archive_type in {"tar.gz", "tgz", "tar"}:
        _safe_extract_tar(archive, destination)
        return
    raise AssetError(f"Неизвестный тип архива: {archive_type!r}")


def install_binary_asset(
    asset: AssetSpec,
    artifact: AssetArtifact,
    *,
    bin_root: Path,
    fetch: FetchFn,
    on_progress: Callable[[int], None],
    on_message: Callable[[str], None],
) -> Path:
    """Скачивает и устанавливает бинарный ресурс, возвращая путь к бинарнику.

    :raises ChecksumError: если ``sha256`` не совпал (файл удаляется).
    :raises AssetError: при небезопасном/повреждённом архиве или отсутствии
        ожидаемого бинарника внутри.
    """
    destination = bin_root / asset.key
    _reset_dir(destination)
    filename = artifact.filename or Path(artifact.url).name
    downloaded = destination / filename
    on_message(f"Скачивание {filename}…")
    fetch(artifact.url, downloaded, on_progress)
    on_message("Проверка контрольной суммы…")
    actual = sha256_file(downloaded)
    if actual != artifact.sha256:
        downloaded.unlink(missing_ok=True)
        raise ChecksumError(
            f"sha256 не совпал (ожидался {artifact.sha256[:12]}…, получен {actual[:12]}…)"
        )
    if artifact.archive == "none":
        _ensure_executable(downloaded)
        return downloaded
    on_message("Распаковка…")
    _extract_archive(artifact.archive, downloaded, destination)
    downloaded.unlink(missing_ok=True)
    member = artifact.member or asset.bin_name
    target = _find_member(destination, member)
    if target is None:
        raise AssetError(f"В архиве не найден бинарник {member!r}")
    _ensure_executable(target)
    return target


@dataclass(slots=True)
class BinaryState:
    """Текущее состояние установки бинарного ресурса."""

    key: str
    status: str = STATUS_IDLE
    message: str = ""
    error: str | None = None
    bytes_done: int = 0
    total: int = 0
    fraction: float = 0.0
    path: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "bytes_done": self.bytes_done,
            "total": self.total,
            "fraction": round(self.fraction, 4) if self.status != STATUS_IDLE else None,
            "path": self.path,
        }


@dataclass(slots=True)
class _Tracker:
    """Троттлинг публикации прогресса по байтам."""

    installer: BinaryInstaller
    key: str
    total: int
    done: int = 0
    _last_fraction: float = 0.0
    _last_event: float = field(default_factory=time.monotonic)

    def __call__(self, delta: int) -> None:
        self.done += max(int(delta), 0)
        now = time.monotonic()
        with self.installer.lock:
            state = self.installer._state_locked(self.key)
            state.bytes_done = self.done
            state.total = self.total
            if self.total > 0:
                state.fraction = min(self.done / self.total, 1.0)
            fraction = state.fraction
            changed = abs(fraction - self._last_fraction) >= _PROGRESS_STEP
            if not changed and (now - self._last_event) < _PROGRESS_INTERVAL:
                return
            self._last_event = now
            self._last_fraction = fraction
        self.installer._publish(
            self.key,
            STATUS_RUNNING,
            f"Скачивание… {fraction * 100:.0f}%",
            bytes_done=self.done,
            total=self.total,
            fraction=fraction,
        )


class BinaryInstaller:
    """Фоновые установки бинарных ресурсов (не более одной одновременно)."""

    def __init__(
        self,
        *,
        bus: DownloadBus,
        bin_root: Path,
        settings_store: object | None = None,
        fetch: FetchFn | None = None,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        self._bus = bus
        self._bin_root = bin_root
        self._settings_store = settings_store
        self._fetch = fetch or stream_download
        self._on_success = on_success
        self.lock = threading.Lock()
        self._states: dict[str, BinaryState] = {}
        self._thread: threading.Thread | None = None
        self._active_key: str | None = None

    @property
    def bus(self) -> DownloadBus:
        return self._bus

    @property
    def bin_root(self) -> Path:
        return self._bin_root

    def _state_locked(self, key: str) -> BinaryState:
        state = self._states.get(key)
        if state is None:
            state = BinaryState(key=key)
            self._states[key] = state
        return state

    def state(self, key: str) -> BinaryState:
        """Снимок состояния операции (по ключу реестра)."""
        with self.lock:
            return replace(self._state_locked(key))

    def is_running(self) -> bool:
        """Идёт ли установка прямо сейчас (одна за раз)."""
        with self.lock:
            return self._thread is not None and self._thread.is_alive()

    def active_key(self) -> str | None:
        """Ключ текущей установки или ``None``."""
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                return self._active_key
            return None

    def wait(self, timeout: float | None = None) -> None:
        """Ожидает завершения текущей установки (используется в тестах)."""
        with self.lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def start(self, asset: AssetSpec, artifact: AssetArtifact) -> bool:
        """Запускает скачивание артефакта; ``False`` — если установка уже идёт."""
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            state = self._state_locked(asset.key)
            state.status = STATUS_RUNNING
            state.message = "Скачивание…"
            state.error = None
            state.bytes_done = 0
            state.total = artifact.size
            state.fraction = 0.0
            state.path = ""
            self._active_key = asset.key
            thread = threading.Thread(
                target=self._run,
                args=(asset, artifact),
                name=f"asset-install-{asset.key}",
                daemon=True,
            )
            self._thread = thread
        self._publish(
            asset.key,
            STATUS_RUNNING,
            "Скачивание…",
            bytes_done=0,
            total=artifact.size,
            fraction=0.0,
        )
        thread.start()
        return True

    def _publish(
        self,
        key: str,
        status: str,
        message: str,
        error: str | None = None,
        *,
        bytes_done: int | None = None,
        total: int | None = None,
        fraction: float | None = None,
        path: str | None = None,
    ) -> None:
        with self.lock:
            state = self._state_locked(key)
            state.status = status
            state.message = message
            state.error = error
            if bytes_done is not None:
                state.bytes_done = bytes_done
            if total is not None:
                state.total = total
            if fraction is not None:
                state.fraction = fraction
            if path is not None:
                state.path = path
            event: dict[str, object] = {
                "key": key,
                "status": status,
                "message": message,
                "error": error,
                "bytes_done": state.bytes_done,
                "total": state.total,
                "fraction": round(state.fraction, 4) if status != STATUS_IDLE else None,
                "path": state.path,
            }
        self._bus.publish(event)

    def _apply_settings(self, asset: AssetSpec, binary: Path) -> None:
        """Прописывает путь бинарника (и каталог библиотек) в настройках."""
        if self._settings_store is None or not asset.settings_field:
            return
        load = getattr(self._settings_store, "load", None)
        save = getattr(self._settings_store, "save", None)
        if load is None or save is None:
            return
        current = load()
        patch: dict[str, str] = {asset.settings_field: str(binary)}
        if asset.lib_settings_field:
            patch[asset.lib_settings_field] = str(binary.parent)
        updated = replace(current, **patch)
        save(updated)

    def _run(self, asset: AssetSpec, artifact: AssetArtifact) -> None:
        tracker = _Tracker(installer=self, key=asset.key, total=artifact.size)

        def on_message(message: str) -> None:
            self._publish(asset.key, STATUS_RUNNING, message)

        try:
            binary = install_binary_asset(
                asset,
                artifact,
                bin_root=self._bin_root,
                fetch=self._fetch,
                on_progress=tracker,
                on_message=on_message,
            )
            self._apply_settings(asset, binary)
        except ChecksumError as exc:
            logger.warning("Проверка sha256 для %s не прошла", asset.key)
            self._publish(
                asset.key,
                STATUS_ERROR,
                "Контрольная сумма не совпала — файл удалён",
                str(exc),
            )
            return
        except Exception as exc:  # noqa: BLE001 — внешняя сеть/архив: показываем причину
            message = _shorten(str(exc))
            logger.warning("Установка %s не удалась: %s", asset.key, message)
            self._publish(asset.key, STATUS_ERROR, "Ошибка установки", message)
            return
        if self._on_success is not None:
            try:
                self._on_success()
            except Exception:
                logger.exception("Сбой после установки %s", asset.key)
        self._publish(
            asset.key,
            STATUS_DONE,
            f"Установлено: {binary}",
            path=str(binary),
            fraction=1.0,
        )


def _shorten(text: str, limit: int = 300) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def artifact_payload(artifact: AssetArtifact | None) -> dict[str, object] | None:
    """JSON-представление артефакта (без секретов — URL/хеш публичны)."""
    return artifact.as_dict() if artifact is not None else None


def asset_payload(
    asset: AssetSpec,
    *,
    bin_root: Path,
    settings: object,
    status: str = STATUS_IDLE,
    message: str = "",
    error: str | None = None,
    bytes_done: int = 0,
    total: int = 0,
    fraction: float | None = None,
    prefer_vulkan: bool = False,
    os_name: str | None = None,
    arch: str | None = None,
) -> dict[str, object]:
    """JSON-представление ресурса с локальным статусом и прогрессом."""
    payload: dict[str, object] = dict(asset.as_dict())
    payload.update(
        {
            "status": status,
            "message": message,
            "error": error,
            "bytes_done": bytes_done,
            "total": total,
            "fraction": fraction,
        }
    )
    if asset.kind == KIND_PIP:
        payload.update(
            {
                "installed": module_available(asset.module) if asset.module else False,
                "downloadable": False,
                "spec": asset.spec,
                "module": asset.module,
            }
        )
        return payload
    target_os = os_name or current_os()
    target_arch = arch or current_arch()
    artifact = resolve_artifact(
        asset, os_name=target_os, arch=target_arch, prefer_vulkan=prefer_vulkan
    )
    path = asset_binary_path(asset, bin_root=bin_root, settings=settings)
    payload.update(
        {
            "installed": path is not None,
            "downloadable": artifact is not None,
            "path": str(path) if path is not None else "",
            "platform": f"{target_os}/{target_arch}",
            "artifact": artifact_payload(artifact),
            "settings_field": asset.settings_field,
            "lib_settings_field": asset.lib_settings_field,
            "env_key": asset.env_key,
            "lib_env_key": asset.lib_env_key,
        }
    )
    return payload
