"""Реестр и установщик опциональных пакетов веб-мастера.

Мастер первого запуска умеет не только показывать подсказки, но и ставить
недостающие Python-пакеты (issue #66). Устанавливаемый spec берётся **только**
из фиксированного allowlist (:data:`DEPENDENCIES`) — пользовательский ввод в
команду установки не попадает.

Установка идёт фоновым потоком, а её вывод публикуется в шину событий
(:class:`~audio_transcriber.web.models.DownloadBus`) и раздаётся как SSE
(``GET /api/deps/events``) с монотонным ``seq``/``id`` и поддержкой
``Last-Event-ID`` — тем же приёмом, что и загрузка моделей.

Менеджер установщика (:class:`DependencyInstaller`) намеренно не выполняет
команды сам: реальный запуск вынесен в ``runner``, который подменяется в тестах,
чтобы не дёргать ``pip``/``uv``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from audio_transcriber.utils.subprocess_registry import (
    register_process,
    terminate_process,
)
from audio_transcriber.web.models import DownloadBus

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_INSTALL_TIMEOUT",
    "DEPENDENCIES",
    "DEP_DONE",
    "DEP_ERROR",
    "DEP_IDLE",
    "DEP_RUNNING",
    "DependencyInstaller",
    "DependencySpec",
    "DependencyState",
    "InstallerTimeout",
    "InstallerUnavailable",
    "find_dependency",
    "installer_available",
    "installer_name",
    "module_available",
    "resolve_install_command",
    "run_installer",
]


#: Статусы операции установки.
DEP_IDLE = "idle"
DEP_RUNNING = "running"
DEP_DONE = "done"
DEP_ERROR = "error"

#: Таймаут установки по умолчанию (секунды): зависший установщик гасится (#87).
#: Переопределяется ``DEP_INSTALL_TIMEOUT``; ``<= 0`` — без таймаута.
DEFAULT_INSTALL_TIMEOUT = 1800.0

#: Типовые каталоги, куда ставится ``uv`` без прав root.
_UV_FALLBACK_PATHS = ("~/.local/bin/uv", "~/.cargo/bin/uv")


@dataclass(frozen=True, slots=True)
class DependencySpec:
    """Разрешённая к автоустановке зависимость (элемент allowlist)."""

    key: str
    spec: str
    label: str
    module: str
    check_id: str
    needed_for: str

    def as_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "spec": self.spec,
            "label": self.label,
            "check_id": self.check_id,
            "needed_for": self.needed_for,
        }


#: Единственный источник истины для автоустановки: ключ → spec → проверка доктора.
#: Устанавливать что-либо вне этого списка запрещено.
DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec(
        key="gigaam",
        spec="onnx-asr[cpu,hub]",
        label="Пакет onnx-asr (GigaAM)",
        module="onnx_asr",
        check_id="dep:onnx_asr",
        needed_for="Бэкенд распознавания GigaAM",
    ),
    DependencySpec(
        key="sherpa",
        spec="sherpa-onnx",
        label="Пакет sherpa-onnx (оценка числа говорящих)",
        module="sherpa_onnx",
        check_id="dep:sherpa_onnx",
        needed_for="Быстрый выбор движка диаризации (auto)",
    ),
)

_DEPENDENCIES_BY_KEY = {spec.key: spec for spec in DEPENDENCIES}


class InstallerUnavailable(RuntimeError):
    """Ни ``uv``, ни ``pip`` не найдены — установить пакет нечем."""


class InstallerTimeout(RuntimeError):
    """Установщик не завершился за отведённый таймаут (процесс остановлен)."""


def find_dependency(key: str) -> DependencySpec | None:
    """Зависимость реестра по ключу (``None`` — ключ не разрешён)."""
    return _DEPENDENCIES_BY_KEY.get(key)


def module_available(name: str) -> bool:
    """Доступен ли модуль (без импорта — ``find_spec``), как в докторе."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def find_uv() -> str | None:
    """Путь к ``uv``: из ``PATH`` либо из типовых пользовательских каталогов."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in _UV_FALLBACK_PATHS:
        path = Path(candidate).expanduser()
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return None


def pip_available() -> bool:
    """Есть ли модуль ``pip`` в текущем интерпретаторе."""
    return module_available("pip")


def installer_name() -> str | None:
    """Предпочитаемый установщик: ``uv``, иначе ``pip``, иначе ``None``."""
    if find_uv() is not None:
        return "uv"
    if pip_available():
        return "pip"
    return None


def installer_available() -> bool:
    """Найден ли хоть какой-то установщик (``uv``/``pip``)."""
    return installer_name() is not None


def resolve_install_command(spec: str) -> list[str]:
    """Собирает команду установки для фиксированного spec.

    Предпочитается ``uv`` (в venv проекта ``pip`` может отсутствовать), иначе
    ``python -m pip``. Если нет ни того, ни другого — :class:`InstallerUnavailable`.
    """
    uv = find_uv()
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, spec]
    if pip_available():
        return [sys.executable, "-m", "pip", "install", spec]
    raise InstallerUnavailable("не найден установщик (uv/pip)")


#: Запуск команды установки: возвращает код возврата, отдавая строки вывода.
InstallRunner = Callable[[list[str], Callable[[str], None]], int]


def _install_timeout() -> float:
    """Таймаут установки: ``DEP_INSTALL_TIMEOUT`` или :data:`DEFAULT_INSTALL_TIMEOUT`."""
    raw = os.environ.get("DEP_INSTALL_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_INSTALL_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        logger.warning("DEP_INSTALL_TIMEOUT=%r — не число, использую дефолт", raw)
        return DEFAULT_INSTALL_TIMEOUT
    return max(value, 0.0)


def run_installer(
    command: list[str],
    on_line: Callable[[str], None],
    *,
    timeout: float | None = None,
) -> int:
    """Реальный запуск установщика: stdout+stderr построчно уходят в ``on_line``.

    Процесс регистрируется в общем реестре и ограничен дедлайном (#87): при
    зависании он принудительно останавливается (:class:`InstallerTimeout`), а
    при остановке сервера гасится вместе с остальными (``terminate_all_processes``).
    """
    limit = _install_timeout() if timeout is None else timeout
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    register_process(process)

    def _drain() -> None:
        if process.stdout is None:
            return
        try:
            for raw in process.stdout:
                on_line(raw.rstrip("\r\n"))
        except (OSError, ValueError):
            # Поток закрыт при принудительной остановке процесса.
            pass

    reader = threading.Thread(target=_drain, name="dep-install-output", daemon=True)
    reader.start()
    wait_timeout = limit if limit and limit > 0 else None
    try:
        try:
            process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process(process)
            raise InstallerTimeout(
                f"Установка не завершилась за {limit:g} с — процесс остановлен"
            ) from exc
    finally:
        terminate_process(process)
    reader.join(timeout=5.0)
    return process.returncode


@dataclass(slots=True)
class DependencyState:
    """Текущее состояние операции установки одной зависимости."""

    key: str
    status: str = DEP_IDLE
    message: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "status": self.status,
            "message": self.message,
            "error": self.error,
        }


class DependencyInstaller:
    """Фоновые установки опциональных пакетов (не более одной одновременно)."""

    def __init__(
        self,
        *,
        bus: DownloadBus,
        runner: InstallRunner | None = None,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        self._bus = bus
        self._runner = runner or run_installer
        self._on_success = on_success
        self._lock = threading.Lock()
        self._states: dict[str, DependencyState] = {}
        self._thread: threading.Thread | None = None
        self._active_key: str | None = None

    @property
    def bus(self) -> DownloadBus:
        return self._bus

    def _state_locked(self, key: str) -> DependencyState:
        state = self._states.get(key)
        if state is None:
            state = DependencyState(key=key)
            self._states[key] = state
        return state

    def state(self, key: str) -> DependencyState:
        """Снимок состояния операции (по ключу реестра)."""
        with self._lock:
            return replace(self._state_locked(key))

    def is_running(self) -> bool:
        """Идёт ли установка прямо сейчас (одна на весь сервер)."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def active_key(self) -> str | None:
        """Ключ текущей установки или ``None``, если ничего не выполняется."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._active_key
            return None

    def wait(self, timeout: float | None = None) -> None:
        """Ожидает завершения текущей установки (используется в тестах)."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def start(self, key: str, spec: str) -> bool:
        """Запускает установку spec; ``False`` — если установка уже идёт.

        :raises InstallerUnavailable: если не найден ни ``uv``, ни ``pip``.
        """
        command = resolve_install_command(spec)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            state = self._state_locked(key)
            state.status = DEP_RUNNING
            state.message = f"Установка {spec}…"
            state.error = None
            self._active_key = key
            thread = threading.Thread(
                target=self._run,
                args=(key, spec, command),
                name=f"dep-install-{key}",
                daemon=True,
            )
            self._thread = thread
        self._publish(key, DEP_RUNNING, f"Установка {spec}…")
        thread.start()
        return True

    def _publish(self, key: str, status: str, message: str, error: str | None = None) -> None:
        """Обновляет состояние и публикует событие SSE."""
        with self._lock:
            state = self._state_locked(key)
            state.status = status
            state.message = message
            state.error = error
        self._bus.publish(
            {"key": key, "status": status, "message": message, "error": error}
        )

    def _run(self, key: str, spec: str, command: list[str]) -> None:
        def on_line(line: str) -> None:
            if line:
                self._publish(key, DEP_RUNNING, line)

        try:
            code = self._runner(command, on_line)
        except InstallerTimeout as exc:
            logger.warning("Установка %s: %s", spec, exc)
            self._publish(key, DEP_ERROR, str(exc), str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — установщик внешний: мягкая ошибка
            logger.warning("Установка %s не запустилась: %s", spec, exc)
            self._publish(key, DEP_ERROR, f"Не удалось запустить установщик: {exc}", str(exc))
            return
        if code == 0:
            if self._on_success is not None:
                try:
                    self._on_success()
                except Exception:
                    logger.exception("Сбой после успешной установки %s", spec)
            self._publish(key, DEP_DONE, "Установлено")
        else:
            message = f"Установщик завершился с кодом {code}"
            logger.warning("Установка %s: %s", spec, message)
            self._publish(key, DEP_ERROR, message, message)
