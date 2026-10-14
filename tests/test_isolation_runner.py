"""Интеграционные (но быстрые) тесты запуска воркера изоляции (#143).

Реальный subprocess запускается с подставным модулем-воркером: проверяются
разбор прогресса из stderr, чтение результата, таймаут и понятные ошибки.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from audio_transcriber.isolation.protocol import KIND_ASR, WorkerRequest
from audio_transcriber.isolation.runner import run_worker
from audio_transcriber.utils.exceptions import EngineIsolationError

#: Подставной воркер: эмулирует протокол без загрузки моделей.
_FAKE_WORKER = '''\
import argparse
import json
import os
import sys
import time
from pathlib import Path

from audio_transcriber.isolation.protocol import progress_line
from audio_transcriber.progress import ProgressEvent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    mode = request.get("options", {}).get("mode", "ok")
    sys.stderr.write(progress_line(ProgressEvent("asr", "старт", fraction=0.5)) + "\\n")
    sys.stderr.flush()
    if mode == "sleep":
        time.sleep(60)
    if mode == "crash":
        os._exit(7)
    if mode == "fail":
        Path(args.result).write_text(
            json.dumps({"ok": False, "error": "сломалось"}), encoding="utf-8"
        )
        return 3
    Path(args.result).write_text(
        json.dumps(
            {
                "ok": True,
                "kind": request["kind"],
                "payload": {"language": "ru", "duration": 1.0, "segments": []},
            }
        ),
        encoding="utf-8",
    )
    return 0


raise SystemExit(main())
'''


def _install_fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "fake_engine_worker.py").write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))


def _request(mode: str) -> WorkerRequest:
    return WorkerRequest(
        kind=KIND_ASR,
        audio_path="/tmp/a.wav",
        options={"mode": mode},
    )


def test_run_worker_returns_payload_and_streams_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_worker(tmp_path, monkeypatch)
    events: list[object] = []

    result = run_worker(
        _request("ok"),
        on_progress=events.append,
        timeout=30.0,
        python=sys.executable,
        module="fake_engine_worker",
    )

    assert result.ok is True
    assert result.payload is not None
    assert result.payload["language"] == "ru"
    assert events and events[0].stage == "asr"


def test_run_worker_timeout_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_worker(tmp_path, monkeypatch)

    with pytest.raises(EngineIsolationError, match="таймаут"):
        run_worker(
            _request("sleep"),
            timeout=0.5,
            python=sys.executable,
            module="fake_engine_worker",
        )


def test_run_worker_reports_worker_side_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_worker(tmp_path, monkeypatch)

    result = run_worker(
        _request("fail"),
        timeout=30.0,
        python=sys.executable,
        module="fake_engine_worker",
    )

    assert result.ok is False
    assert result.error == "сломалось"


def test_run_worker_without_result_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_worker(tmp_path, monkeypatch)

    with pytest.raises(EngineIsolationError):
        run_worker(
            _request("crash"),
            timeout=30.0,
            python=sys.executable,
            module="fake_engine_worker",
        )


def test_run_worker_reports_missing_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_worker(tmp_path, monkeypatch)

    with pytest.raises(EngineIsolationError):
        run_worker(
            _request("ok"),
            timeout=30.0,
            python=sys.executable,
            module="module_that_does_not_exist_143",
        )
