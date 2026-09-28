"""Реестр внешних процессов и гарантированная очистка ресурсов.

Тяжёлые движки (``llama-server``, ``whisper-cli``) запускаются как внешние
процессы и держат GPU/VRAM. Чтобы процесс **никогда не оставался висеть**,
все они регистрируются здесь, а модуль:

* останавливает их через :mod:`atexit` и по сигналам ``SIGINT``/``SIGTERM``;
* даёт единый :class:`StderrReader` для чтения stderr в фоне (хвост и полный
  буфер — для определения устройства и диагностики).

Вынесено из клиента LLM, чтобы тем же механизмом пользовался и whisper.cpp.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import subprocess
import threading
from collections import deque
from types import FrameType
from typing import IO, Any

logger = logging.getLogger(__name__)

_STDERR_TAIL_SIZE = 40
# Хвост для сообщений об ошибках и полный буфер — для определения устройства
# и слоёв, выгруженных на GPU (строка «offloaded N/M layers to GPU»).
_STDERR_BUFFER_SIZE = 4000

_active_processes: dict[int, subprocess.Popen[str]] = {}
_registry_lock = threading.Lock()
_cleanup_handlers_ready = False
_signal_handlers_installed = False
_previous_signal_handlers: dict[int, Any] = {}


def terminate_process(proc: subprocess.Popen[str], *, timeout: float = 10.0) -> None:
    """Останавливает процесс: SIGTERM, при необходимости — SIGKILL."""
    with _registry_lock:
        _active_processes.pop(proc.pid, None)
    if proc.poll() is not None:
        return
    with contextlib.suppress(OSError):
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    logger.warning("Процесс не завершился за %.0f с — снимаю принудительно", timeout)
    try:
        proc.kill()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired, OSError:
        pass


def terminate_all_processes() -> None:
    """Останавливает все зарегистрированные процессы (atexit/сигналы)."""
    with _registry_lock:
        processes = list(_active_processes.values())
    for proc in processes:
        terminate_process(proc)


def _handle_signal(signum: int, frame: FrameType | None) -> None:
    """Сначала гасит зарегистрированные процессы, затем передаёт сигнал дальше."""
    terminate_all_processes()
    previous = _previous_signal_handlers.get(signum)
    if previous is signal.SIG_IGN:
        return
    if callable(previous):
        previous(signum, frame)
        return
    try:
        signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
    except ValueError, OSError:
        signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _ensure_cleanup_handlers() -> None:
    """Один раз регистрирует atexit и обработчики SIGINT/SIGTERM."""
    global _cleanup_handlers_ready, _signal_handlers_installed

    if not _cleanup_handlers_ready:
        atexit.register(terminate_all_processes)
        _cleanup_handlers_ready = True

    if _signal_handlers_installed:
        return
    if threading.current_thread() is not threading.main_thread():
        return

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except ValueError, OSError:
            continue
        _previous_signal_handlers[signum] = previous
    _signal_handlers_installed = True


def register_process(proc: subprocess.Popen[str]) -> None:
    """Регистрирует процесс для гарантированной очистки."""
    with _registry_lock:
        _active_processes[proc.pid] = proc
    _ensure_cleanup_handlers()


class StderrReader:
    """Читает stderr подпроцесса в фоне, хранит хвост и полный буфер."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=_STDERR_BUFFER_SIZE)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            for line in self._stream:
                self._lines.append(line.rstrip("\n"))
        except OSError, ValueError:
            pass

    def tail(self) -> str:
        return "\n".join(list(self._lines)[-_STDERR_TAIL_SIZE:])

    def full_text(self) -> str:
        return "\n".join(self._lines)
