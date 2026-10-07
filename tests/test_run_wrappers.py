"""Согласованность обёрток запуска ``run.sh`` и ``run.ps1``.

Маппинг ``config.env`` → флаги CLI живёт в одном месте — в коде
(:mod:`audio_transcriber.cli.env_config`). Обёртки больше не перечисляют
переменные и не пробрасывают флаги: они только экспортируют ``config.env`` в
окружение и вызывают ``audio-transcriber transcribe <файл>``, а настройки
подхватывает CLI (единый источник, issue #90). Эти тесты следят за тем, чтобы
дублирования не вернули.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Флаги, которые раньше встречались в дублирующем маппинге обёрток. Теперь их
#: быть не должно — «канарейка» против возврата дублирования.
_PREVIOUSLY_MAPPED_FLAGS = (
    "--model",
    "--asr-backend",
    "--whisper-cpp-model",
    "--llm",
    "--llm-context",
    "--llm-request-timeout",
    "--glossary",
    "--clear-cache",
    "--no-word-timestamps",
    "--no-denoise",
)

#: Секреты, которые нельзя передавать флагами argv — они видны в
#: ``ps``/``/proc/<pid>/cmdline`` (issue #81).
_SECRET_FLAGS = ("--hf-token", "--llm-api-key")


def _sh_text() -> str:
    return (_PROJECT_ROOT / "run.sh").read_text(encoding="utf-8")


def _ps_text() -> str:
    return (_PROJECT_ROOT / "run.ps1").read_text(encoding="utf-8")


def test_run_wrappers_export_all_config_env() -> None:
    """Обе обёртки кладут *все* значения config.env в окружение процесса.

    Так переменные, которые код читает напрямую из окружения
    (``WHISPER_CPP_VAD_MODEL``, ``WHISPER_CPP_CHUNK_*``, ``GLOSSARY_DB`` и др.),
    не теряются на платформах без явного проброса.
    """
    sh_text = _sh_text()
    ps_text = _ps_text()

    # run.sh: set -a; source config.env; set +a.
    assert "set -a" in sh_text
    assert "source config.env" in sh_text
    # run.ps1: переносит разобранные значения в окружение процесса.
    assert "SetEnvironmentVariable" in ps_text


def test_run_wrappers_do_not_duplicate_flag_mapping() -> None:
    """Обёртки не дублируют маппинг env→флаг — он только в коде CLI."""
    sh_text = _sh_text()
    ps_text = _ps_text()

    for name, text in (("run.sh", sh_text), ("run.ps1", ps_text)):
        for flag in _PREVIOUSLY_MAPPED_FLAGS:
            assert flag not in text, (
                f"{name} снова пробрасывает флаг {flag} (дублирование маппинга env→флаг)"
            )

    # run.ps1 больше не строит argv из config.env через хелпер.
    assert "Get-ValueArg" not in ps_text
    # run.sh больше не накапливает env-флаги в массив ARGS.
    assert "ARGS+=(" not in sh_text


def test_run_wrappers_pass_only_audio_file_to_cli() -> None:
    """Обёртки вызывают только ``transcribe <файл>`` — без настроек-флагов."""
    sh_text = _sh_text()
    ps_text = _ps_text()

    assert re.search(r'audio-transcriber transcribe "\$AUDIO_FILE"', sh_text)
    assert re.search(r"audio-transcriber transcribe \$AudioFile", ps_text)


def test_run_wrappers_do_not_pass_secrets_in_argv() -> None:
    """Обёртки не пробрасывают секреты флагами — только через окружение/файл."""
    sh_text = _sh_text()
    ps_text = _ps_text()

    for flag in _SECRET_FLAGS:
        assert flag not in sh_text, f"run.sh пробрасывает секрет флагом {flag}"
        assert flag not in ps_text, f"run.ps1 пробрасывает секрет флагом {flag}"


def test_run_wrappers_still_forward_secret_env_vars() -> None:
    """Секреты остаются доступны CLI через config.env/окружение, не argv."""
    sh_text = _sh_text()
    ps_text = _ps_text()
    assert "HF_TOKEN" in sh_text
    assert "HF_TOKEN" in ps_text
    # run.sh экспортирует весь config.env (`set -a`), run.ps1 — через окружение.
    assert "set -a" in sh_text and "source config.env" in sh_text
    assert "SetEnvironmentVariable" in ps_text
