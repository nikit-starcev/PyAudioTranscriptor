"""Тесты хелперов окружения внешних процессов (#91).

Регрессия: раньше денойз временно мутировал **глобальный** ``os.environ``
(``_sanitized_env``), из-за чего параллельные стадии могли увидеть чужое
окружение. Сейчас окружение для дочерних процессов собирается копией и
передаётся через ``env=`` — глобальное состояние процесса не трогается.
"""

from __future__ import annotations

import os

from audio_transcriber.utils.env import with_library_path


def test_with_library_path_does_not_mutate_global_environ(
    monkeypatch,
) -> None:
    """Сборка окружения не меняет ``os.environ`` процесса."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    original = os.environ.get("LD_LIBRARY_PATH")

    result = with_library_path(os.environ, "/opt/engine/lib")

    assert result["LD_LIBRARY_PATH"] == "/opt/engine/lib"
    assert os.environ.get("LD_LIBRARY_PATH") == original
    assert result is not os.environ


def test_with_library_path_prepends_and_preserves_existing() -> None:
    """Каталог движка ставится первым, прежнее значение сохраняется после него."""
    source = {"LD_LIBRARY_PATH": "/usr/lib", "PATH": "/bin"}

    result = with_library_path(source, "/opt/engine/lib")

    assert result == {
        "LD_LIBRARY_PATH": "/opt/engine/lib:/usr/lib",
        "PATH": "/bin",
    }
    assert source == {"LD_LIBRARY_PATH": "/usr/lib", "PATH": "/bin"}


def test_with_library_path_without_library_returns_copy() -> None:
    """Без каталога библиотек возвращается именно копия, а не тот же объект."""
    source = {"PATH": "/bin"}

    result = with_library_path(source, None)

    assert result == {"PATH": "/bin"}
    assert result is not source
