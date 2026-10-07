"""Тесты файлового хранилища постадийного кэша (``StageCache``)."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
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


# --- Параллельные записи (#91) ----------------------------------------------


def test_temp_paths_are_unique(tmp_path: Path) -> None:
    """Каждая запись получает уникальный временный файл (pid + uuid)."""
    cache = StageCache(tmp_path / ".cache")
    target = cache.directory / "asr-key.json"

    paths = {cache._temp_path(target) for _ in range(50)}

    assert len(paths) == 50
    assert all(path != target for path in paths)
    assert all(path.name.endswith(".tmp") for path in paths)


def test_concurrent_atomic_writes_use_distinct_temp_files(tmp_path: Path) -> None:
    """Две одновременные записи не делят один ``.tmp`` (иначе порча файла)."""
    cache = StageCache(tmp_path / ".cache")
    target = cache.directory / "asr-key.json"
    barrier = threading.Barrier(2)
    seen: list[Path] = []
    lock = threading.Lock()

    def producer(tmp: Path) -> None:
        with lock:
            seen.append(tmp)
        # Обе записи внутри «критической секции» одновременно: при общем
        # временном пути оба producer'а получили бы один и тот же файл.
        barrier.wait(timeout=5)
        tmp.write_text("payload", encoding="utf-8")

    def writer() -> None:
        cache._write_atomic(target, producer)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer) for _ in range(2)]
        for future in futures:
            future.result()

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert target.read_text(encoding="utf-8") == "payload"


def test_concurrent_saves_keep_file_valid_and_no_leftover_tmp(tmp_path: Path) -> None:
    """Много параллельных ``save`` на один ключ: файл валиден, ``.tmp`` не остаются."""
    cache = StageCache(tmp_path / ".cache")
    barrier = threading.Barrier(8)

    def writer(index: int) -> None:
        barrier.wait(timeout=5)
        cache.save("asr", "same-key", {"writer": index, "payload": "x" * 2048})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(writer, index) for index in range(8)]
        for future in futures:
            future.result()

    loaded = cache.load("asr", "same-key")
    assert isinstance(loaded, dict)
    assert loaded["writer"] in range(8)
    assert list(cache.directory.glob("*.tmp")) == []
