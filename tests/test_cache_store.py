"""Тесты файлового хранилища постадийного кэша (``StageCache``)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from audio_transcriber.cache.store import (
    CACHE_FORMAT_VERSION,
    StageCache,
    compute_cache_key,
)


def _audio(tmp_path: Path, name: str = "sample.mp3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"audio")
    return path


def test_key_is_stable_for_same_inputs(tmp_path: Path) -> None:
    source = _audio(tmp_path)
    params = {"model": "large", "language": "ru", "denoise": True}

    first = compute_cache_key("asr", source, params)
    second = compute_cache_key("asr", source, dict(reversed(list(params.items()))))

    assert first == second


def test_key_changes_with_params(tmp_path: Path) -> None:
    source = _audio(tmp_path)

    assert compute_cache_key("asr", source, {"language": "ru"}) != compute_cache_key(
        "asr", source, {"language": "en"}
    )


def test_key_changes_with_stage(tmp_path: Path) -> None:
    source = _audio(tmp_path)

    assert compute_cache_key("asr", source, {}) != compute_cache_key("diarization", source, {})


def test_key_changes_with_file_size(tmp_path: Path) -> None:
    source = _audio(tmp_path)
    key_before = compute_cache_key("asr", source, {})

    source.write_bytes(b"longer audio payload")

    assert compute_cache_key("asr", source, {}) != key_before


def test_key_changes_with_mtime(tmp_path: Path) -> None:
    source = _audio(tmp_path)
    key_before = compute_cache_key("asr", source, {})

    os.utime(source, (1_000_000, 1_000_000))

    assert compute_cache_key("asr", source, {}) != key_before


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    payload = {"segments": [{"start": 0.0, "end": 1.0}], "language": "ru"}

    cache.save("asr", "key1", payload)

    assert cache.load("asr", "key1") == payload


def test_load_missing_returns_none(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    assert cache.load("asr", "absent") is None


def test_disabled_cache_does_not_write_or_read(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache", enabled=False)

    cache.save("asr", "key1", {"x": 1})

    assert cache.load("asr", "key1") is None
    assert not (tmp_path / ".cache").exists()


def test_broken_json_returns_none(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    cache.directory.mkdir(parents=True)
    (cache.directory / "asr-broken.json").write_text("{not json", encoding="utf-8")

    assert cache.load("asr", "broken") is None


def test_incompatible_version_returns_none(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    cache.directory.mkdir(parents=True)
    path = cache.directory / "asr-old.json"
    path.write_text(
        json.dumps({"cache_version": CACHE_FORMAT_VERSION + 1, "stage": "asr", "data": {}}),
        encoding="utf-8",
    )

    assert cache.load("asr", "old") is None


def test_wrong_stage_in_file_returns_none(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    cache.directory.mkdir(parents=True)
    path = cache.directory / "asr-wrong.json"
    path.write_text(
        json.dumps({"cache_version": CACHE_FORMAT_VERSION, "stage": "diarization", "data": {}}),
        encoding="utf-8",
    )

    assert cache.load("asr", "wrong") is None


def test_audio_roundtrip(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    source = tmp_path / "denoised.wav"
    source.write_bytes(b"RIFFdata")

    saved = cache.save_audio("denoise", "k", source)

    assert saved is not None
    assert cache.load_audio("denoise", "k") == saved
    assert saved.read_bytes() == b"RIFFdata"


def test_audio_load_missing_returns_none(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    assert cache.load_audio("denoise", "absent") is None


def test_clear_removes_cache_files(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    cache.save("asr", "a", {"x": 1})
    cache.save("diarization", "b", {"y": 2})
    cache.save_audio("denoise", "c", _audio(tmp_path, "d.wav"))

    removed = cache.clear()

    assert removed == 3
    assert list(cache.directory.iterdir()) == []


def test_clear_on_missing_dir_returns_zero(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / ".cache")
    assert cache.clear() == 0
