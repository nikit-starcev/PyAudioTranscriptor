"""Сборка portable-бандла ``audio-transcriber`` (PyInstaller one-dir).

Скрипт готовит изолированное build-окружение через ``uv`` (CPU-сборка PyTorch,
издание проекта с extra ``web``, сам PyInstaller), запускает PyInstaller по
спеке ``packaging/pyinstaller/audio-transcriber.spec`` и складывает результат в
``<OUT>/audio-transcriber/``.

PyInstaller не умеет кросс-компиляцию, поэтому сборка возможна только под
host-ОС: ``--target`` обязан совпадать с ней (в CI runner — целевая ОС).

Пример::

    python scripts/build_portable.py --target linux --out ./dist-portable
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

BUNDLE_DIRNAME = "audio-transcriber"

SUPPORTED_TARGETS = ("linux", "windows", "macos")

EXECUTABLE_NAMES = {
    "linux": "audio-transcriber",
    "windows": "audio-transcriber.exe",
    "macos": "audio-transcriber",
}

_OS_TO_TARGET = {
    "linux": "linux",
    "windows": "windows",
    "darwin": "macos",
}

_PROJECT_EXTRA = "web"

# PyTorch ставим отдельно под CPU-индекс, поэтому в constraints из lock его нет.
_EXCLUDED_CONSTRAINTS = ("torch", "torchaudio")

# CUDA/ROCm-пакеты и triton не нужны для CPU-сборки и лишь раздувают окружение.
_EXCLUDED_PREFIXES = ("nvidia-", "amd-", "triton")


def host_target() -> str:
    """Целевая ОС текущей машины в терминах ``--target``."""
    os_name = platform.system().lower()
    target = _OS_TO_TARGET.get(os_name)
    if target is None:
        raise SystemExit(f"Неподдерживаемая host-ОС: {platform.system()!r}")
    return target


def repo_root() -> Path:
    """Корень репозитория (на два уровня выше ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


def spec_path() -> Path:
    """Путь к PyInstaller-спеке."""
    return repo_root() / "packaging" / "pyinstaller" / "audio-transcriber.spec"


def executable_name(target: str) -> str:
    """Имя исполняемого файла для целевой ОС."""
    return EXECUTABLE_NAMES[target]


def bundle_dir(out_dir: Path) -> Path:
    """Каталог one-dir бандла внутри ``out_dir``."""
    return Path(out_dir) / BUNDLE_DIRNAME


def validate_target(target: str, host: str) -> None:
    """Проверяет, что целевая ОС совпадает с host (иначе — понятная ошибка)."""
    if target not in SUPPORTED_TARGETS:
        raise SystemExit(
            f"Неизвестный --target {target!r}. Допустимо: {', '.join(SUPPORTED_TARGETS)}."
        )
    if target != host:
        raise SystemExit(
            f"PyInstaller не кросс-компилирует: запрошен --target {target!r}, "
            f"а эта машина — {host!r}. Запускайте сборку на целевой ОС "
            "(в CI runner совпадает с matrix-ОС)."
        )


def build_parser() -> argparse.ArgumentParser:
    """CLI-парсер аргументов сборки."""
    parser = argparse.ArgumentParser(
        prog="build_portable.py",
        description="Собрать portable one-dir бандл audio-transcriber (PyInstaller).",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=SUPPORTED_TARGETS,
        help="Целевая ОС (обязана совпадать с host-ОС).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Каталог, куда положить <out>/audio-transcriber/.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Удалить прежние артефакты и build-окружение перед сборкой.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    return build_parser().parse_args(argv)


def _venv_python(venv_dir: Path) -> Path:
    """Путь к интерпретатору внутри venv (``bin`` на POSIX, ``Scripts`` на Windows)."""
    if platform.system().lower() == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _format_size(num_bytes: int) -> str:
    """Человекочитаемый размер каталога."""
    size = float(num_bytes)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ"):
        if size < 1024.0 or unit == "ГиБ":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} ГиБ"


def _directory_size(path: Path) -> int:
    """Суммарный размер всех файлов в каталоге."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Запускает команду, печатает её и падает с понятной ошибкой при сбое."""
    print("+ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Не найдена команда {cmd[0]!r}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Команда завершилась с кодом {exc.returncode}: {cmd[0]}") from exc


def _uv() -> str:
    """Путь к ``uv`` или понятная ошибка."""
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "Не найден uv. Установите его: https://docs.astral.sh/uv/getting-started/installation/"
        )
    return uv


def _clean(out_dir: Path, build_root: Path) -> None:
    """Удаляет прежний бандл и build-окружение."""
    for target in (bundle_dir(out_dir), build_root):
        if target.exists():
            print(f"Очистка: {target}")
            shutil.rmtree(target)


def constraint_lines(export_text: str) -> list[str]:
    """Отбирает из вывода ``uv export`` строки-ограничения для build-окружения.

    Оставляет зафиксированные lock-ом версии всех зависимостей, кроме PyTorch
    (его ставит CPU-индекс) и CUDA/ROCm/triton (не нужны для portable-сборки).
    """
    lines: list[str] = []
    for raw in export_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[=;<\[ ]", line, maxsplit=1)[0].lower()
        if name in _EXCLUDED_CONSTRAINTS or name.startswith(_EXCLUDED_PREFIXES):
            continue
        lines.append(line)
    return lines


def _write_constraints(project: Path, uv: str, dest: Path) -> Path:
    """Выгружает зафиксированные версии из ``uv.lock`` в файл constraints."""
    print("+ uv export (constraints из uv.lock)")
    try:
        result = subprocess.run(
            [
                uv,
                "export",
                "--frozen",
                "--no-dev",
                "--extra",
                _PROJECT_EXTRA,
                "--format",
                "requirements-txt",
                "--no-emit-project",
                "--no-hashes",
            ],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"uv export завершился с кодом {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(constraint_lines(result.stdout)) + "\n", encoding="utf-8")
    return dest


def _prepare_build_env(project: Path, build_root: Path) -> Path:
    """Создаёт venv и ставит проект с extra ``web`` и PyInstaller (CPU-torch).

    Версии зависимостей берутся из ``uv.lock`` через constraints, чтобы
    окружение совпадало с CI и не разъезжалось при свежем resolve. CPU-индекс
    PyTorch включается через ``--torch-backend cpu``, а ``--no-sources``
    отключает ``[tool.uv.sources]``, который на Linux/Windows тянет CUDA.
    """
    uv = _uv()
    venv_dir = build_root / "venv"
    pyversion = f"{sys.version_info.major}.{sys.version_info.minor}"
    _run([uv, "venv", "--clear", "--python", pyversion, str(venv_dir)])
    constraints = _write_constraints(project, uv, build_root / "constraints.txt")
    python = _venv_python(venv_dir)
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-sources",
            "--torch-backend",
            "cpu",
            "--constraints",
            str(constraints),
            f"{project}[{_PROJECT_EXTRA}]",
            "pyinstaller",
        ]
    )
    return python


def build(target: str, out_dir: Path, *, clean: bool = False) -> Path:
    """Собирает one-dir бандл и возвращает путь к каталогу с исполняемым файлом."""
    host = host_target()
    validate_target(target, host)

    project = repo_root()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    build_root = out_dir / ".build"
    if clean:
        _clean(out_dir, build_root)

    python = _prepare_build_env(project, build_root)

    spec = spec_path()
    if not spec.is_file():
        raise SystemExit(f"Не найдена спека PyInstaller: {spec}")

    _run(
        [
            str(python),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--log-level",
            "WARN",
            "--distpath",
            str(out_dir),
            "--workpath",
            str(build_root / "pyinstaller"),
            str(spec),
        ]
    )

    bundle = bundle_dir(out_dir)
    exe = bundle / executable_name(target)
    if not exe.is_file():
        raise SystemExit(f"Сборка завершилась, но исполняемый файл не найден: {exe}")
    return bundle


def _force_utf8_stdout() -> None:
    """Не падать на cp1252-консоли Windows при выводе кириллицы."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа скрипта."""
    _force_utf8_stdout()
    args = parse_args(argv)
    bundle = build(args.target, args.out, clean=args.clean)
    size = _format_size(_directory_size(bundle))
    print(f"\nГотово: {bundle}")
    print(f"Исполняемый файл: {bundle / executable_name(args.target)}")
    print(f"Размер: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
