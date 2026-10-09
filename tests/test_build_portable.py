"""Тесты аргументов и утилит скрипта сборки portable-бандла.

Реальная сборка PyInstaller не запускается: проверяются только чистые функции
(валидация цели, имена, разбор аргументов).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "build_portable.py"


def _load_build_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_portable", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_portable = _load_build_module()


def test_host_target_supported() -> None:
    assert build_portable.host_target() in build_portable.SUPPORTED_TARGETS


@pytest.mark.parametrize("target", ["linux", "windows", "macos"])
def test_validate_target_accepts_host(target: str) -> None:
    build_portable.validate_target(target, target)


def test_validate_target_rejects_cross_compile() -> None:
    with pytest.raises(SystemExit):
        build_portable.validate_target("windows", "linux")


def test_validate_target_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        build_portable.validate_target("plan9", "plan9")


def test_executable_names() -> None:
    assert build_portable.executable_name("windows") == "audio-transcriber.exe"
    assert build_portable.executable_name("linux") == "audio-transcriber"
    assert build_portable.executable_name("macos") == "audio-transcriber"


def test_bundle_dir() -> None:
    assert build_portable.bundle_dir(Path("/tmp/out")) == Path("/tmp/out/audio-transcriber")


def test_parse_args() -> None:
    args = build_portable.parse_args(["--target", "linux", "--out", "/tmp/out", "--clean"])
    assert args.target == "linux"
    assert args.out == Path("/tmp/out")
    assert args.clean is True


def test_parse_args_requires_target_and_out() -> None:
    with pytest.raises(SystemExit):
        build_portable.parse_args([])


def test_constraint_lines_filters_torch_and_cuda() -> None:
    export = "\n".join(
        [
            "huggingface-hub==1.24.0",
            "    # via pyannote-audio",
            "torch==2.12.1+cu126 ; sys_platform != 'darwin'",
            "torchaudio==2.11.0+cu126 ; sys_platform != 'darwin'",
            "nvidia-cublas-cu12==12.6.4.1 ; sys_platform == 'linux'",
            "triton==3.7.1 ; sys_platform == 'linux'",
            "fsspec==2026.6.0",
            "",
        ]
    )
    assert build_portable.constraint_lines(export) == [
        "huggingface-hub==1.24.0",
        "fsspec==2026.6.0",
    ]
