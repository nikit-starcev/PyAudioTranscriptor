"""Тесты обнаружения бинарника/модели nemo-speech (:mod:`nemo_speech_assets`).

Реальные сторонние команды не запускаются: ``which`` и пробу бинарника
подменяем, файловую систему строим в ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config import defaults as config_defaults
from audio_transcriber.diarization import nemo_speech_assets as assets

DEFAULT_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"


def _make_binary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return path


# --- кэш модели ---------------------------------------------------------------


def test_model_cache_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "models"))
    assert assets.model_cache_root() == tmp_path / "models"


def test_model_cache_root_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEMO_SPEECH_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert assets.model_cache_root() == tmp_path / "cache" / "nemo-speech" / "models"


def test_resolve_model_repo() -> None:
    assert assets.resolve_model_repo("sortformer") == DEFAULT_REPO
    assert assets.resolve_model_repo(DEFAULT_REPO) == DEFAULT_REPO
    assert assets.resolve_model_repo("") == DEFAULT_REPO
    assert assets.resolve_model_repo("custom/model") == "custom/model"


def test_model_status_reads_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(root))
    gguf = root / DEFAULT_REPO / "abc123" / "diar_streaming_sortformer_4spk-v2.q8_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"x" * 2048)
    Path(f"{gguf}.verified").write_text("sha256=...", encoding="utf-8")

    status = assets.model_status(DEFAULT_REPO)

    assert status.present
    assert status.repo == DEFAULT_REPO
    assert status.source == "cache"
    assert status.path == str(gguf)
    assert status.size == 2048
    assert status.files == ("diar_streaming_sortformer_4spk-v2.q8_0.gguf",)
    assert status.as_dict()["present"] is True


def test_model_status_short_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(root))
    gguf = root / DEFAULT_REPO / "abc" / "model.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"y" * 16)

    status = assets.model_status("sortformer")

    assert status.present
    assert status.repo == DEFAULT_REPO
    assert status.size == 16


def test_model_status_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "empty"))
    status = assets.model_status(DEFAULT_REPO)

    assert not status.present
    assert status.path is None
    assert status.size == 0
    assert assets.cached_model_size(DEFAULT_REPO) == 0


def test_model_status_explicit_gguf(tmp_path: Path) -> None:
    gguf = tmp_path / "my.gguf"
    gguf.write_bytes(b"z" * 100)

    status = assets.model_status(str(gguf))

    assert status.present
    assert status.source == "file"
    assert status.path == str(gguf)
    assert status.size == 100
    assert assets.model_status(str(tmp_path / "missing.gguf")).present is False


# --- бандл и библиотеки -------------------------------------------------------


def test_bundle_library_path_detects_bundle(tmp_path: Path) -> None:
    binary = _make_binary(tmp_path / "bundle" / "bin" / "nemo-speech")
    lib = tmp_path / "bundle" / "lib"
    lib.mkdir()
    (lib / "libggml-base.so.0").write_text("", encoding="utf-8")

    assert assets.bundle_library_path(binary) == str(lib)


def test_bundle_library_path_none_for_non_bundle(tmp_path: Path) -> None:
    binary = _make_binary(tmp_path / "plain" / "nemo-speech")
    (tmp_path / "plain" / "lib").mkdir()
    assert assets.bundle_library_path(binary) is None

    # bin/ + lib/ без признака бандла (нет share/include и движковых .so).
    binary2 = _make_binary(tmp_path / "weird" / "bin" / "nemo-speech")
    (tmp_path / "weird" / "lib").mkdir()
    (tmp_path / "weird" / "lib" / "libother.so").write_text("", encoding="utf-8")
    assert assets.bundle_library_path(binary2) is None


def test_safe_library_path_rejects_system_shadow(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libggml.so").write_text("", encoding="utf-8")
    assert assets.safe_library_path(str(lib)) == str(lib)

    (lib / "libstdc++.so.6").write_text("", encoding="utf-8")
    assert assets.safe_library_path(str(lib)) is None
    assert assets.safe_library_path(None) is None


# --- поиск бинарника ----------------------------------------------------------


def test_search_candidates_project_sibling(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    bundle = tmp_path / "nemo-speech-native"
    binary = _make_binary(bundle / "bin" / "nemo-speech")
    lib = bundle / "lib"
    lib.mkdir()
    (lib / "libnemo_speech.so").write_text("", encoding="utf-8")

    candidates = assets.search_candidates(
        root=project, home=tmp_path / "home", which=lambda _name: None
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.binary == str(binary)
    assert candidate.lib_path == str(lib)
    assert candidate.source == "рядом с проектом"


def test_search_candidates_prefers_settings_and_dedups(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    binary = _make_binary(tmp_path / "custom" / "nemo-speech")

    candidates = assets.search_candidates(
        str(binary), root=project, home=tmp_path / "home", which=lambda _name: str(binary)
    )

    # Настройки и PATH указывают на один файл — остаётся один кандидат.
    assert len(candidates) == 1
    assert candidates[0].source == "настройки"
    assert candidates[0].binary == str(binary)


def test_search_candidates_uses_which_and_home(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    in_path = _make_binary(tmp_path / "toolchain" / "nemo-speech")
    home_bin = _make_binary(tmp_path / "home" / ".local" / "bin" / "nemo-speech")

    candidates = assets.search_candidates(
        root=project, home=tmp_path / "home", which=lambda name: str(in_path) if name == "nemo-speech" else None
    )

    binaries = {candidate.binary for candidate in candidates}
    assert in_path.as_posix() in {Path(b).as_posix() for b in binaries}
    assert home_bin.as_posix() in {Path(b).as_posix() for b in binaries}


def test_search_candidates_explicit_name_via_which(tmp_path: Path) -> None:
    found = _make_binary(tmp_path / "opt" / "nemo-speech")
    candidates = assets.search_candidates(
        "nemo-speech",
        root=tmp_path / "proj",
        home=tmp_path / "home",
        which=lambda name: str(found) if name == "nemo-speech" else None,
    )
    assert [Path(c.binary) for c in candidates] == [found]


# --- версия/проба -------------------------------------------------------------


def test_parse_version() -> None:
    assert assets.parse_version("nemo-speech 0.1.0\n") == "0.1.0"
    assert (
        assets.parse_version("NeMo-Speech.cpp 0.1.0\nFeatures: diarization\n")
        == "0.1.0"
    )
    assert assets.parse_version("no numbers here") is None


def test_probe_binary_parses_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        ("--version",): "nemo-speech 0.1.0\n",
        ("doctor",): (
            "NeMo-Speech.cpp 0.1.0\n"
            "Features: diarization backend_vulkan\n"
            "Devices:\n"
            "  [0] gpu   AMD Radeon RX 590 (8.0 GiB)\n"
            "  [1] cpu   AMD Ryzen 5\n"
        ),
    }

    class _Proc:
        returncode = 0

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command: list[str], **_kwargs: object) -> _Proc:
        key = tuple(command[1:])
        return _Proc(outputs.get(key, ""))

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    probe = assets.probe_binary("/opt/nemo-speech/bin/nemo-speech")

    assert probe.version == "0.1.0"
    assert probe.has_vulkan is True
    assert probe.devices == ("[0] gpu   AMD Radeon RX 590 (8.0 GiB)",)


def test_detect_candidates_probes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = _make_binary(tmp_path / "bin" / "nemo-speech")
    monkeypatch.setattr(
        assets,
        "search_candidates",
        lambda *_args, **_kwargs: [
            assets.NemoSpeechCandidate(binary=str(binary), lib_path=None, source="тест")
        ],
    )
    monkeypatch.setattr(
        assets,
        "probe_binary",
        lambda *_args, **_kwargs: assets.NemoSpeechProbe(
            version="9.9.9", devices=("gpu",), has_vulkan=True
        ),
    )

    (candidate,) = assets.detect_candidates()

    assert candidate.version == "9.9.9"
    assert candidate.devices == ("gpu",)
    assert candidate.has_vulkan is True
    assert candidate.as_dict()["binary"] == str(binary)


def test_model_default_constant_matches_config() -> None:
    assert config_defaults.DEFAULT_NEMO_SPEECH_MODEL == DEFAULT_REPO
