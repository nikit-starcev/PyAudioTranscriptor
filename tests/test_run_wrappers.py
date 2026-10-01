"""Согласованность обёрток запуска ``run.sh`` и ``run.ps1``.

Обёртки по отдельности перечисляют переменные ``config.env`` и пробрасывают их
в CLI, поэтому список легко может разойтись. Эти тесты держат его
синхронизированным: множества переменных в ``run.sh`` и ``run.ps1`` обязаны
совпадать, а все значения ``config.env`` — попадать в окружение процесса так
же, как это делает ``set -a; source config.env`` на Linux/macOS.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# В run.sh каждая переменная config.env берётся как "${VAR...}".
_SH_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)[:}]")
# В run.ps1 — доступ к разобранному словарю и хелпер «флаг + значение».
_PS_VAR_RE = re.compile(r'\$config\["([A-Z][A-Z0-9_]*)"\]')
_PS_VALUE_ARG_RE = re.compile(r'Get-ValueArg\s+"[^"]+"\s+"([A-Z][A-Z0-9_]*)"')

#: Переменные, критичные для текущих возможностей; должны быть в обеих обёртках.
_REQUIRED_VARS = {
    "HF_TOKEN",
    "MODEL",
    "ASR_BACKEND",
    "WHISPER_CPP_MODEL",
    "LLM_ENABLED",
    "LLM_CONTEXT",
    "LLM_REQUEST_TIMEOUT",
    "GLOSSARY_PATH",
}


def _run_sh_vars() -> set[str]:
    text = (_PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")
    return set(_SH_VAR_RE.findall(text))


def _run_ps1_vars() -> set[str]:
    text = (_PROJECT_ROOT / "run.ps1").read_text(encoding="utf-8")
    return set(_PS_VAR_RE.findall(text)) | set(_PS_VALUE_ARG_RE.findall(text))


def test_run_wrappers_reference_same_config_vars() -> None:
    sh_vars = _run_sh_vars()
    ps_vars = _run_ps1_vars()

    assert sh_vars == ps_vars, (
        "Списки переменных config.env в run.sh и run.ps1 разошлись: "
        f"только в run.sh — {sorted(sh_vars - ps_vars)}, "
        f"только в run.ps1 — {sorted(ps_vars - sh_vars)}"
    )


def test_run_wrappers_forward_required_vars() -> None:
    sh_vars = _run_sh_vars()
    ps_vars = _run_ps1_vars()

    missing_sh = _REQUIRED_VARS - sh_vars
    missing_ps = _REQUIRED_VARS - ps_vars
    assert not missing_sh, f"run.sh не пробрасывает: {sorted(missing_sh)}"
    assert not missing_ps, f"run.ps1 не пробрасывает: {sorted(missing_ps)}"


def test_run_wrappers_export_all_config_env() -> None:
    """Обе обёртки кладут *все* значения config.env в окружение процесса.

    Так переменные, которые код читает напрямую из окружения
    (``WHISPER_CPP_VAD_MODEL``, ``WHISPER_CPP_CHUNK_*``, ``GLOSSARY_DB`` и др.),
    не теряются на платформах без явного проброса.
    """
    sh_text = (_PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")
    ps_text = (_PROJECT_ROOT / "run.ps1").read_text(encoding="utf-8")

    # run.sh: set -a; source config.env; set +a.
    assert "set -a" in sh_text
    assert "source config.env" in sh_text
    # run.ps1: переносит разобранные значения в окружение процесса.
    assert "SetEnvironmentVariable" in ps_text
