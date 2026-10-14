"""Запуск воркера изоляции и разбор его результата (#143).

Основной процесс не загружает тяжёлые модели: он пишет JSON-запрос во
временный файл, поднимает ``python -m audio_transcriber.isolation.worker``,
читает прогресс из stderr и JSON-результат из файла. Воркер живёт в пределах
одного прохода (load → process → exit), гасится по таймауту и регистрируется
в :mod:`audio_transcriber.utils.subprocess_registry`, чтобы atexit/сигналы не
оставляли его висеть.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from audio_transcriber.isolation.protocol import (
    DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    WorkerRequest,
    parse_progress_line,
    request_to_json,
)
from audio_transcriber.progress import ProgressCallback
from audio_transcriber.utils.exceptions import EngineIsolationError
from audio_transcriber.utils.subprocess_registry import register_process, terminate_process

logger = logging.getLogger(__name__)

#: Модуль воркера, запускаемый через ``python -m``.
WORKER_MODULE = "audio_transcriber.isolation.worker"

#: Пакетный корень для ``PYTHONPATH`` дочернего процесса: ``.../src`` или
#: ``site-packages``. Гарантирует импортируемость пакета из исходников.
_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])

_STDERR_TAIL_SIZE = 40
_STDERR_BUFFER_SIZE = 4000


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Результат прохода воркера: успех с payload либо ошибка."""

    ok: bool
    payload: dict[str, object] | None = None
    error: str | None = None
    error_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WorkerResult:
        """Разбирает ответ воркера; без ``ok`` — ошибка формата."""
        ok = data.get("ok")
        if not isinstance(ok, bool):
            raise EngineIsolationError("Воркер вернул ответ без поля 'ok'")
        if ok:
            payload = data.get("payload")
            if not isinstance(payload, dict):
                raise EngineIsolationError("Воркер вернул успех без объекта payload")
            return cls(ok=True, payload=payload)
        error = data.get("error")
        error_type = data.get("error_type")
        return cls(
            ok=False,
            error=error if isinstance(error, str) else "движок завершился ошибкой",
            error_type=error_type if isinstance(error_type, str) else None,
        )


class _StderrReader(threading.Thread):
    """Читает stderr воркера: строки прогресса → колбэк, прочее → лог/хвост."""

    def __init__(self, stream: IO[str], on_progress: ProgressCallback | None) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._on_progress = on_progress
        self._lines: deque[str] = deque(maxlen=_STDERR_BUFFER_SIZE)

    def run(self) -> None:
        try:
            for raw in self._stream:
                line = raw.rstrip("\n")
                event = parse_progress_line(line)
                if event is not None:
                    if self._on_progress is not None:
                        try:
                            self._on_progress(event)
                        except Exception as exc:  # noqa: BLE001 — колбэк прогресса не должен ронять чтение
                            logger.debug("Колбэк прогресса воркера упал: %s", exc)
                    continue
                self._lines.append(line)
                logger.debug("воркер #143: %s", line)
        except (OSError, ValueError) as exc:
            logger.debug("Чтение stderr воркера прервано: %s", exc)

    def tail(self) -> str:
        """Последние строки stderr для сообщения об ошибке."""
        return "\n".join(list(self._lines)[-_STDERR_TAIL_SIZE:])


class _StreamDrain(threading.Thread):
    """Опустошает поток воркера в фоне, чтобы pipe не переполнился."""

    def __init__(self, stream: IO[str]) -> None:
        super().__init__(daemon=True)
        self._stream = stream

    def run(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            for _ in self._stream:
                pass


def _worker_env() -> dict[str, str]:
    """Окружение воркера с гарантированным доступом к пакету по ``PYTHONPATH``."""
    env = dict(os.environ)
    existing = [part for part in (env.get("PYTHONPATH") or "").split(os.pathsep) if part]
    if _PACKAGE_ROOT not in existing:
        env["PYTHONPATH"] = os.pathsep.join([_PACKAGE_ROOT, *existing])
    return env


def _read_result(
    result_path: Path, returncode: int, stderr_tail: str
) -> WorkerResult:
    """Читает файл-результат; при его отсутствии — понятная ошибка с хвостом stderr."""
    if not result_path.is_file():
        detail = f" Хвост stderr: {stderr_tail}" if stderr_tail else ""
        raise EngineIsolationError(
            f"Воркер изоляции завершился с кодом {returncode} без результата.{detail}"
        )
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineIsolationError(f"Не удалось прочитать результат воркера: {exc}") from exc
    if not isinstance(data, dict):
        raise EngineIsolationError("Результат воркера имеет неожидаемый формат")
    return WorkerResult.from_dict(data)


def run_worker(
    request: WorkerRequest,
    *,
    on_progress: ProgressCallback | None = None,
    timeout: float | None = DEFAULT_ISOLATION_TIMEOUT_SECONDS,
    python: str | None = None,
    module: str = WORKER_MODULE,
) -> WorkerResult:
    """Запускает воркер на один запрос и возвращает его результат.

    :param timeout: таймаут в секундах; ``None`` или неположительное значение —
        без ограничения.
    :raises EngineIsolationError: воркер не запустился, превысил таймаут или
        вернул повреждённый/отсутствующий результат.
    """
    python = python or sys.executable
    effective_timeout = timeout if timeout and timeout > 0 else None

    with tempfile.TemporaryDirectory(prefix="engine-isolation-") as tmp:
        tmp_dir = Path(tmp)
        request_path = tmp_dir / "request.json"
        result_path = tmp_dir / "result.json"
        request_path.write_text(request_to_json(request), encoding="utf-8")

        command = [python, "-m", module, "--request", str(request_path), "--result", str(result_path)]
        try:
            proc: subprocess.Popen[str] = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=_worker_env(),
            )
        except OSError as exc:
            raise EngineIsolationError(
                f"Не удалось запустить воркер изоляции ({module}): {exc}"
            ) from exc

        register_process(proc)
        stderr_stream = proc.stderr
        stdout_stream = proc.stdout
        if stderr_stream is None or stdout_stream is None:
            terminate_process(proc)
            raise EngineIsolationError("Не удалось получить потоки ввода-вывода воркера")
        stderr_reader = _StderrReader(stderr_stream, on_progress)
        stderr_reader.start()
        stdout_drain = _StreamDrain(stdout_stream)
        stdout_drain.start()

        try:
            try:
                proc.wait(timeout=effective_timeout)
            except subprocess.TimeoutExpired as exc:
                terminate_process(proc)
                raise EngineIsolationError(
                    f"Воркер изоляции превысил таймаут {effective_timeout:.0f} с "
                    "и был остановлен"
                ) from exc
        finally:
            stderr_reader.join(timeout=5.0)
            stdout_drain.join(timeout=5.0)
            terminate_process(proc)

        return _read_result(result_path, proc.returncode, stderr_reader.tail())
