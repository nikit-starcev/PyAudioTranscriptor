"""Проверки автоматического версионирования пакета (issue #23).

Версия не хардкодится в ``pyproject.toml``, а выводится ``hatch-vcs`` из
git-тега ``vX.Y.Z``; между тегами добавляется dev-суффикс. Эти тесты
фиксируют конфигурацию и формат итоговой версии.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from audio_transcriber import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# PEP 440 в объёме, который выдаёт hatch-vcs: X.Y.Z[(.devN|.postN|.rcN)][+local].
_PEP440 = re.compile(r"^\d+\.\d+\.\d+(?:\.(?:dev|post|rc|a|b)\d+)?(?:\+[A-Za-z0-9.]+)?$")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_version_is_dynamic_and_sourced_from_vcs() -> None:
    data = _pyproject()

    project = data["project"]
    assert "version" not in project, "статическую version нужно заменить на dynamic"
    assert "version" in project.get("dynamic", [])

    assert data["tool"]["hatch"]["version"]["source"] == "vcs"

    build_requires = {req.split("==")[0].split(">=")[0].strip() for req in data["build-system"]["requires"]}
    assert "hatch-vcs" in build_requires


def test_installed_version_is_pep440() -> None:
    assert __version__
    assert _PEP440.match(__version__), f"неожиданный формат версии: {__version__!r}"
