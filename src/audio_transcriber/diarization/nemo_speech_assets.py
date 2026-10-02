"""Обнаружение бинарника nemo-speech и локальной модели Sortformer.

Модуль не зависит от веб-слоя: поиском бинарника и проверкой кэша модели
пользуются и ``doctor`` (CLI), и локальный веб-API (автодетект, статус и
скачивание). Это позволяет держать ``doctor`` быстрым и не тянуть FastAPI.

Кэш моделей ``nemo-speech`` лежит в каталоге пользовательского кэша ОС
(``~/.cache/nemo-speech/models/<repo>/<commit>/<model>.gguf``); переменная
``NEMO_SPEECH_MODEL_DIR`` переопределяет корень кэша.

Запуск ``nemo-speech`` (``--version``/``doctor``) выполняется только для
явно найденных кандидатов и с коротким таймаутом; в тестах эти функции
подменяются, чтобы не дёргать сторонние бинарники.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from audio_transcriber.config.defaults import (
    DEFAULT_NEMO_SPEECH_MODEL,
    NEMO_SPEECH_FORBIDDEN_LIBS,
)
from audio_transcriber.utils.env import effective_library_path, with_library_path

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PROBE_TIMEOUT",
    "KNOWN_MODEL_REPOS",
    "MODEL_DIR_ENV",
    "NEMO_SPEECH_BINARY_NAME",
    "NemoSpeechCandidate",
    "NemoSpeechModelStatus",
    "NemoSpeechProbe",
    "bundle_library_path",
    "cached_model_files",
    "cached_model_size",
    "detect_candidates",
    "model_cache_root",
    "model_status",
    "parse_version",
    "probe_binary",
    "resolve_model_repo",
    "safe_library_path",
    "search_candidates",
]

#: Имя бинарника в ``PATH``/типовых каталогах.
NEMO_SPEECH_BINARY_NAME = "nemo-speech"

#: Таймаут одной пробы (``--version``/``doctor``), секунды.
DEFAULT_PROBE_TIMEOUT = 12.0

#: Переменная окружения nemo-speech с корнем кэша моделей.
MODEL_DIR_ENV = "NEMO_SPEECH_MODEL_DIR"

#: Репозиторий модели Sortformer по умолчанию (совпадает с ``config.defaults``).
DEFAULT_SORTFORMER_REPO = DEFAULT_NEMO_SPEECH_MODEL

#: Короткие имена моделей из ``nemo-speech model list`` → HF-репозиторий.
KNOWN_MODEL_REPOS: Mapping[str, str] = {
    "sortformer": DEFAULT_SORTFORMER_REPO,
    "sortformer-diar": DEFAULT_SORTFORMER_REPO,
}

#: Каталог проекта (``.../PyAudioTranscriptor``): ``src/audio_transcriber/diarization`` → вверх 3.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Типовые системные каталоги, куда кладут собранный бинарник.
_FALLBACK_BIN_DIRS = ("/usr/local/bin", "/usr/bin")

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.]+)?)")
_GGUF_RE = re.compile(r"(\S+\.gguf)\b", re.IGNORECASE)


def model_cache_root(env: Mapping[str, str] | None = None) -> Path:
    """Корень кэша моделей nemo-speech.

    Приоритет: ``NEMO_SPEECH_MODEL_DIR`` → ``$XDG_CACHE_HOME/nemo-speech/models``
    → ``~/.cache/nemo-speech/models``.
    """
    source = env if env is not None else os.environ
    override = source.get(MODEL_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    xdg = source.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "nemo-speech" / "models"


def resolve_model_repo(model: str) -> str:
    """Преобразует короткое имя модели/путь в HF-репозиторий для кэша."""
    value = model.strip()
    if not value:
        return DEFAULT_SORTFORMER_REPO
    return KNOWN_MODEL_REPOS.get(value, value)


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _find_repo_dir(cache_root: Path, repo: str) -> Path:
    """Каталог репозитория в кэше; для короткого имени — поиск по дереву."""
    direct = cache_root / repo
    if _is_dir(direct):
        return direct
    if "/" not in repo and _is_dir(cache_root):
        try:
            matches = sorted(
                (child for child in cache_root.glob(f"**/{repo}") if child.is_dir()),
                key=str,
            )
        except OSError:
            matches = []
        if matches:
            return matches[0]
    return direct


def _gguf_files(directory: Path) -> list[Path]:
    if not _is_dir(directory):
        return []
    try:
        return sorted(
            (child for child in directory.rglob("*.gguf") if child.is_file()),
            key=str,
        )
    except OSError:
        return []


def cached_model_files(
    model: str, *, env: Mapping[str, str] | None = None
) -> tuple[list[Path], Path, str]:
    """Возвращает ``(gguf-файлы, каталог кэша, repo)`` для модели (или пусто)."""
    root = model_cache_root(env)
    repo = resolve_model_repo(model)
    directory = _find_repo_dir(root, repo)
    return _gguf_files(directory), root, repo


def cached_model_size(model: str, *, env: Mapping[str, str] | None = None) -> int:
    """Суммарный размер ``.gguf``-файлов модели в кэше (0, если нет)."""
    files, _root, _repo = cached_model_files(model, env=env)
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


@dataclass(frozen=True, slots=True)
class NemoSpeechModelStatus:
    """Локальное состояние модели Sortformer."""

    model: str
    repo: str
    present: bool
    path: str | None
    size: int
    files: tuple[str, ...]
    #: ``cache`` — модель в кэше nemo-speech; ``file`` — явный путь к ``.gguf``.
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "repo": self.repo,
            "present": self.present,
            "path": self.path,
            "size": self.size,
            "files": list(self.files),
            "source": self.source,
        }


def model_status(
    model: str, *, env: Mapping[str, str] | None = None
) -> NemoSpeechModelStatus:
    """Проверяет наличие модели: явный ``.gguf`` либо кэш nemo-speech.

    При наличии в кэше предпочитаются верифицированные файлы (рядом лежит
    ``<file>.verified``) — как их помечает сам ``nemo-speech pull``.
    """
    value = model.strip() or DEFAULT_NEMO_SPEECH_MODEL
    explicit = Path(value).expanduser()
    if explicit.suffix.casefold() == ".gguf":
        present = _is_file(explicit)
        size = 0
        if present:
            try:
                size = explicit.stat().st_size
            except OSError:
                size = 0
        return NemoSpeechModelStatus(
            model=value,
            repo=value,
            present=present,
            path=str(explicit) if present else None,
            size=size,
            files=(explicit.name,) if present else (),
            source="file",
        )

    files, _root, repo = cached_model_files(value, env=env)
    verified = [path for path in files if _is_file(Path(f"{path}.verified"))]
    chosen = verified or files
    size = 0
    for path in chosen:
        try:
            size += path.stat().st_size
        except OSError:
            continue
    primary = chosen[0] if chosen else None
    return NemoSpeechModelStatus(
        model=value,
        repo=repo,
        present=primary is not None,
        path=str(primary) if primary is not None else None,
        size=size,
        files=tuple(path.name for path in files),
        source="cache",
    )


@dataclass(frozen=True, slots=True)
class NemoSpeechCandidate:
    """Найденный бинарник nemo-speech вместе с каталогом библиотек и пробой."""

    binary: str
    lib_path: str | None
    #: Откуда взят путь: ``настройки``/``PATH``/``рядом с проектом`` и т.п.
    source: str
    version: str | None = None
    devices: tuple[str, ...] = ()
    has_vulkan: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "binary": self.binary,
            "lib_path": self.lib_path,
            "source": self.source,
            "version": self.version,
            "devices": list(self.devices),
            "has_vulkan": self.has_vulkan,
        }


def bundle_library_path(binary: Path) -> str | None:
    """Каталог ``lib/`` бандла рядом с бинарником, если это похоже на бандл.

    Бандит только раскладка ``<bundle>/bin/nemo-speech`` + ``<bundle>/lib``,
    подкреплённая либо ``share``/``include``, либо движковыми библиотеками
    (``libggml*``/``libnemo*``/``libspeech*``). Иначе (например, ``/usr/bin``)
    системный ``/lib`` не подмешивается.
    """
    if binary.parent.name != "bin":
        return None
    bundle = binary.parent.parent
    lib = bundle / "lib"
    if not _is_dir(lib):
        return None
    marker = _is_dir(bundle / "share") or _is_dir(bundle / "include")
    engine_lib = False
    try:
        engine_lib = any(
            child.name.startswith(("libggml", "libnemo", "libspeech"))
            for child in lib.iterdir()
        )
    except OSError:
        engine_lib = False
    if not (marker or engine_lib):
        return None
    return str(lib)


def _dedup_key(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(Path(path).expanduser())


def _locations(
    explicit: str,
    *,
    root: Path,
    home: Path,
    which: Callable[[str], str | None],
) -> list[tuple[str, Path]]:
    """Проверяемые пути в порядке приоритета: настройки → PATH → типовые."""
    name = NEMO_SPEECH_BINARY_NAME
    locations: list[tuple[str, Path]] = []
    value = explicit.strip()
    if value:
        candidate = Path(value).expanduser()
        if _is_file(candidate):
            locations.append(("настройки", candidate))
        else:
            found = which(value)
            if found:
                locations.append(("настройки", Path(found)))
    path_found = which(name)
    if path_found:
        locations.append(("PATH", Path(path_found)))
    locations.append(("~/.local/bin", home / ".local" / "bin" / name))
    for raw in _FALLBACK_BIN_DIRS:
        locations.append((raw, Path(raw) / name))
    locations.append(("рядом с проектом", root.parent / "nemo-speech-native" / "bin" / name))
    locations.append(("в проекте", root / "nemo-speech-native" / "bin" / name))
    locations.append(("текущий каталог", Path.cwd() / "nemo-speech-native" / "bin" / name))
    return locations


def search_candidates(
    explicit: str = "",
    *,
    root: Path | None = None,
    home: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[NemoSpeechCandidate]:
    """Ищет nemo-speech в настройках, ``PATH`` и типовых каталогах.

    Результат дедуплицирован по каноническому пути; ``lib_path`` заполняется
    для бандловой раскладки (``bin/`` + ``lib/``).
    """
    project_root = root or _PROJECT_ROOT
    home_dir = home or Path.home()
    seen: set[str] = set()
    candidates: list[NemoSpeechCandidate] = []
    for source, path in _locations(explicit, root=project_root, home=home_dir, which=which):
        if not _is_file(path):
            continue
        key = _dedup_key(str(path))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            NemoSpeechCandidate(
                binary=str(path),
                lib_path=bundle_library_path(path),
                source=source,
            )
        )
    return candidates


@dataclass(frozen=True, slots=True)
class NemoSpeechProbe:
    """Результат пробы бинарника: версия, GPU-устройства, наличие Vulkan."""

    version: str | None
    devices: tuple[str, ...]
    has_vulkan: bool


def safe_library_path(lib_path: str | None) -> str | None:
    """Каталог lib без затеняющих системных libstdc++/libgcc_s (или ``None``)."""
    return effective_library_path(lib_path, forbidden_libs=NEMO_SPEECH_FORBIDDEN_LIBS)


def parse_version(output: str) -> str | None:
    """Первая версия вида ``X.Y[.Z]`` в выводе ``--version``/``doctor``."""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _VERSION_RE.search(stripped)
        if match:
            return match.group(1)
    return None


def _run(binary: str, args: list[str], env: Mapping[str, str], timeout: float) -> str | None:
    try:
        proc = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            env=dict(env),
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nemo-speech %s не удалось запустить (%s): %s", args, binary, exc)
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def probe_binary(
    binary: str,
    lib_path: str | None = None,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    env: Mapping[str, str] | None = None,
) -> NemoSpeechProbe:
    """Запускает ``--version`` и ``doctor`` найденного бинарника.

    Каталог библиотек подмешивается безопасно (см. :func:`safe_library_path`).
    Любая ошибка запуска не роняет детект: поля окажутся пустыми/``None``.
    """
    base = dict(env if env is not None else os.environ)
    run_env = with_library_path(base, safe_library_path(lib_path))
    version_output = _run(binary, ["--version"], run_env, timeout)
    doctor_output = _run(binary, ["doctor"], run_env, timeout) or ""
    devices = tuple(
        line.strip()
        for line in doctor_output.splitlines()
        if line.strip().startswith("[") and "gpu" in line.casefold()
    )
    version = parse_version(version_output) if version_output else None
    if version is None and doctor_output:
        version = parse_version(doctor_output)
    return NemoSpeechProbe(
        version=version,
        devices=devices,
        has_vulkan="backend_vulkan" in doctor_output.casefold(),
    )


def detect_candidates(
    explicit: str = "",
    *,
    probe: bool = True,
    root: Path | None = None,
    home: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[NemoSpeechCandidate]:
    """Ищет кандидатов и (по желанию) обогащает их версией/устройствами.

    Проба запускается не более чем для пяти кандидатов: обычно бинарник один.
    """
    candidates = search_candidates(explicit, root=root, home=home, which=which)
    if not probe:
        return candidates
    probed: list[NemoSpeechCandidate] = []
    for candidate in candidates[:5]:
        result = probe_binary(candidate.binary, candidate.lib_path)
        probed.append(
            replace(
                candidate,
                version=result.version,
                devices=result.devices,
                has_vulkan=result.has_vulkan,
            )
        )
    return probed
