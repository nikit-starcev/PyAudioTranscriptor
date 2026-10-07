"""Определение корня ``config.env`` для исходников и wheel-установки (issue #90).

Раньше ``utils/config_env.py`` жёстко брал ``parents[3]`` от файла модуля: для
``src``-layout это корень репозитория, но для wheel это оказывался
``.../lib/pythonX.Y`` — несуществующий «корень». Тесты фиксируют корректное
поведение в обоих режимах.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.utils.config_env import (
    detect_project_root,
    find_config_env,
    parse_config_env,
    project_root,
)


def test_detect_project_root_from_source_checkout() -> None:
    package_dir = Path("/repo/PyAudioTranscriptor/src/audio_transcriber")

    root = detect_project_root(package_dir, cwd=Path("/somewhere/else"))

    assert root == Path("/repo/PyAudioTranscriptor")


def test_detect_project_root_for_wheel_uses_cwd() -> None:
    """Wheel-установка: жёсткого корня репо нет — берём рабочий каталог."""
    package_dir = Path("/venv/lib/python3.12/site-packages/audio_transcriber")

    root = detect_project_root(package_dir, cwd=Path("/work/project"))

    assert root == Path("/work/project")
    # И это не бессмысленный каталог вида site-packages/lib.
    assert "site-packages" not in str(root)


def test_project_root_is_repo_root_in_sources() -> None:
    """При запуске из исходников ``project_root`` — корень репозитория."""
    tests_dir = Path(__file__).resolve().parents[1]
    assert project_root() == tests_dir


def test_find_config_env_prefers_first_candidate(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "config.env").write_text("MODEL=medium\n", encoding="utf-8")
    (other / "config.env").write_text("MODEL=large\n", encoding="utf-8")

    found = find_config_env([tmp_path, other])

    assert found == tmp_path / "config.env"


def test_find_config_env_returns_none_when_absent(tmp_path: Path) -> None:
    assert find_config_env([tmp_path / "empty"]) is None


def test_parse_config_env_ignores_comments_and_quotes() -> None:
    parsed = parse_config_env(
        '# комментарий\nMODEL = "medium"\n\nLANGUAGE=ru\nnot-a-pair\n'
    )

    assert parsed == {"MODEL": "medium", "LANGUAGE": "ru"}
