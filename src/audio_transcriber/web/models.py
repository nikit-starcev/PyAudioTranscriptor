"""Каталог известных моделей и фоновое скачивание с Hugging Face.

Модуль решает две задачи веб-интерфейса:

* **реестр моделей** (:data:`MODEL_CATALOG`) — какие файлы нужны для каждого
  режима работы, откуда их брать и каким полем настроек они задаются;
* **менеджер загрузок** (:class:`ModelDownloadManager`) — скачивание через
  ``huggingface_hub`` с докачкой, прогрессом по байтам и отменой. Прогресс
  публикуется в :class:`DownloadBus`, который HTTP-слой раздаёт как SSE.

Модуль намеренно не делает сетевых запросов сам по себе: ``HfDownloader`` —
единственный класс, который ходит в интернет, и он подменяется в тестах.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Категории моделей.
KIND_WHISPER = "whisper-cpp"
KIND_LLM = "llm"
KIND_PYANNOTE = "pyannote"
KIND_GIGAAM = "gigaam"

#: Статусы загрузки.
STATUS_IDLE = "idle"
STATUS_DOWNLOADING = "downloading"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

#: Интервал keep-alive SSE по умолчанию (секунды).
DEFAULT_HEARTBEAT = 15.0

#: Порог публикации прогресса: доля от целого, после которой отправляется событие.
_PROGRESS_STEP = 0.005
#: Минимальный интервал между событиями прогресса (секунды).
_PROGRESS_INTERVAL = 0.25


class ModelError(RuntimeError):
    """Ошибка каталога/загрузки модели с понятным для пользователя текстом."""


class DownloadCancelled(Exception):
    """Скачивание прервано пользователем (внутренний сигнал потока загрузки)."""


@dataclass(frozen=True, slots=True)
class ModelFile:
    """Один файл модели внутри целевого каталога."""

    filename: str
    approx_size: int = 0


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Запись каталога: описание модели и куда её класть."""

    id: str
    kind: str
    title: str
    repo: str
    files: tuple[ModelFile, ...]
    target_dir: str
    approx_size: int
    note: str = ""
    gated: bool = False
    #: Поле :class:`~audio_transcriber.web.settings.WebSettings`, куда пишется путь.
    setting_key: str = ""
    #: ``True`` — скачивается снимок каталога репозитория (pyannote), а не файлы.
    snapshot: bool = False

    @property
    def primary_filename(self) -> str | None:
        """Имя главного файла (для GGUF со шардами — первый)."""
        return self.files[0].filename if self.files else None

    @property
    def dir_name(self) -> str:
        """Имя целевого каталога (последний компонент ``target_dir``)."""
        return Path(self.target_dir).name


#: Каталог известных моделей. Скачиваются в ``<каталог моделей>/<target_dir>``,
#: если в настройках не задан явный путь (тогда используется он).
MODEL_CATALOG: tuple[ModelEntry, ...] = (
    ModelEntry(
        id="whisper-large-v3-turbo",
        kind=KIND_WHISPER,
        title="whisper.cpp large-v3-turbo (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-large-v3-turbo.bin", 1_624_555_275),),
        target_dir="whisper-models",
        approx_size=1_624_555_275,
        setting_key="whisper_cpp_model",
        note="Лучший баланс качество/скорость для whisper.cpp; требует бэкенд whisper-cpp.",
    ),
    ModelEntry(
        id="whisper-medium",
        kind=KIND_WHISPER,
        title="whisper.cpp medium (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-medium.bin", 1_533_774_781),),
        target_dir="whisper-models",
        approx_size=1_533_774_781,
        setting_key="whisper_cpp_model",
        note="Компромисс между качеством и потреблением памяти.",
    ),
    ModelEntry(
        id="whisper-small",
        kind=KIND_WHISPER,
        title="whisper.cpp small (ggml)",
        repo="ggerganov/whisper.cpp",
        files=(ModelFile("ggml-small.bin", 487_601_967),),
        target_dir="whisper-models",
        approx_size=487_601_967,
        setting_key="whisper_cpp_model",
        note="Быстрая и компактная модель; качество ниже.",
    ),
    ModelEntry(
        id="qwen2.5-7b-instruct-q4_k_m",
        kind=KIND_LLM,
        title="Qwen2.5-7B-Instruct Q4_K_M (GGUF)",
        repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        files=(
            ModelFile("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", 3_993_201_344),
            ModelFile("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", 689_872_288),
        ),
        target_dir="llama-models",
        approx_size=4_683_073_632,
        setting_key="llm_model",
        note="Модель для LLM-постобработки (llama-server); состоит из двух шардов.",
    ),
    ModelEntry(
        id="pyannote-community-1",
        kind=KIND_PYANNOTE,
        title="pyannote speaker-diarization-community-1",
        repo="pyannote/speaker-diarization-community-1",
        files=(),
        target_dir="pyannote-models/speaker-diarization-community-1",
        approx_size=33_554_432,
        gated=True,
        snapshot=True,
        setting_key="pyannote_local_model",
        note="Gated-модель: нужен токен HF и принятые условия использования.",
    ),
    ModelEntry(
        id="gigaam-v3-onnx",
        kind=KIND_GIGAAM,
        title="GigaAM v3 (ONNX, onnx-asr)",
        repo="istupakov/gigaam-v3-onnx",
        files=(),
        target_dir="gigaam-models/gigaam-v3-onnx",
        approx_size=4_455_303_019,
        snapshot=True,
        setting_key="gigaam_model_path",
        note=(
            "GigaAM v3 для русского через onnx-asr: нужен пакет "
            "onnx-asr[cpu,hub] и бэкенд gigaam. Снимок репозитория включает "
            "варианты ctc/rnnt/e2e в fp32 и int8; при выборе квантизации "
            "используется int8-часть."
        ),
    ),
)


def find_model(model_id: str) -> ModelEntry | None:
    """Запись каталога по идентификатору."""
    for entry in MODEL_CATALOG:
        if entry.id == model_id:
            return entry
    return None


def configured_path(entry: ModelEntry, settings: object) -> str:
    """Путь модели из настроек (``setting_key``) или пустая строка."""
    if not entry.setting_key:
        return ""
    raw = getattr(settings, entry.setting_key, "")
    return raw.strip() if isinstance(raw, str) else ""


def resolve_target(entry: ModelEntry, *, models_root: Path, settings: object) -> Path:
    """Каталог модели: заданный в настройках или ``<models_root>/<target_dir>``.

    Для файловых моделей заданный путь указывает на файл — берётся его каталог;
    для снимков (pyannote) — каталог как есть.
    """
    configured = configured_path(entry, settings)
    if configured:
        path = Path(configured).expanduser()
        return path if entry.snapshot else path.parent
    return models_root / entry.target_dir


def primary_path(entry: ModelEntry, target_dir: Path) -> Path:
    """Главный артефакт модели: файл (первый шард) или каталог снимка."""
    if entry.snapshot:
        return target_dir
    name = entry.primary_filename
    return target_dir / name if name else target_dir


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def dir_size(directory: Path) -> int:
    """Суммарный размер файлов каталога (0 при ошибке доступа)."""
    total = 0
    try:
        for child in directory.rglob("*"):
            if child.is_file():
                total += _file_size(child)
    except OSError:
        return total
    return total


def _snapshot_present(directory: Path) -> bool:
    """Есть ли в каталоге снимка хотя бы один содержательный файл (без ``.cache``)."""
    if not directory.is_dir():
        return False
    try:
        for child in directory.rglob("*"):
            if child.is_file() and ".cache" not in child.parts:
                return True
    except OSError:
        return False
    return False


@dataclass(frozen=True, slots=True)
class LocalModelStatus:
    """Локальное состояние модели на диске."""

    present: bool
    size: int
    expected_size: int
    missing: tuple[str, ...]
    path: Path
    partial: bool = False


def local_status(entry: ModelEntry, target_dir: Path) -> LocalModelStatus:
    """Проверяет наличие файлов модели в ``target_dir`` (без сети)."""
    if entry.snapshot:
        present = _snapshot_present(target_dir)
        size = dir_size(target_dir)
        missing: tuple[str, ...] = () if present else (entry.dir_name,)
        return LocalModelStatus(
            present=present,
            size=size,
            expected_size=entry.approx_size,
            missing=missing,
            path=target_dir,
            partial=bool(size > 0 and not present),
        )
    missing = tuple(
        model_file.filename
        for model_file in entry.files
        if not (target_dir / model_file.filename).is_file()
    )
    size = sum(_file_size(target_dir / model_file.filename) for model_file in entry.files)
    return LocalModelStatus(
        present=not missing,
        size=size,
        expected_size=entry.approx_size,
        missing=missing,
        path=primary_path(entry, target_dir),
        partial=bool(missing and size > 0),
    )


def delete_model_files(entry: ModelEntry, target_dir: Path) -> bool:
    """Удаляет файлы модели; ``False`` — если удалять было нечего.

    Удаление ограничено ожидаемым артефактом: для снимка — только каталог с
    ожидаемым именем, для файлов — только файлы из каталога модели.
    """
    if entry.snapshot:
        if target_dir.name != entry.dir_name or not target_dir.is_dir():
            return False
        shutil.rmtree(target_dir, ignore_errors=True)
        return not target_dir.exists()
    removed = False
    for model_file in entry.files:
        candidate = target_dir / model_file.filename
        try:
            if candidate.is_file():
                candidate.unlink()
                removed = True
        except OSError:
            continue
    return removed


def free_space(path: Path) -> int:
    """Свободное место на диске для каталога (0, если определить не удалось)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


@runtime_checkable
class Downloader(Protocol):
    """Источник файлов модели (реальный — Hugging Face; в тестах — заглушка)."""

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        """Скачивает один файл в ``destination`` (с докачкой)."""
        ...

    def fetch_snapshot(
        self,
        *,
        repo: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        """Скачивает весь снимок репозитория в ``destination``."""
        ...


def _redact(text: str, token: str | None) -> str:
    if token and token in text:
        return text.replace(token, "***")
    return text


def _shorten(text: str, limit: int = 300) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _progress_tqdm(on_progress: Callable[[int], None]) -> type:
    """Класс tqdm, вызывающий ``on_progress(скачано_байт)`` на каждом шаге."""
    from tqdm.auto import tqdm as base_tqdm

    class _ProgressTqdm(base_tqdm):
        def update(self, n: int | float | None = 1) -> bool | None:
            result = super().update(n)
            if n:
                on_progress(int(n))
            return result

    return _ProgressTqdm


class HfDownloader:
    """Реальный загрузчик через ``huggingface_hub`` (без сети — не вызывается)."""

    def fetch(
        self,
        *,
        repo: str,
        filename: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        from huggingface_hub import hf_hub_download

        destination.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(destination.parent),
            token=token,
            tqdm_class=_progress_tqdm(on_progress),
        )

    def fetch_snapshot(
        self,
        *,
        repo: str,
        destination: Path,
        token: str | None,
        on_progress: Callable[[int], None],
    ) -> None:
        from huggingface_hub import snapshot_download

        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(destination),
            token=token,
            tqdm_class=_progress_tqdm(on_progress),
        )


@dataclass(slots=True)
class DownloadState:
    """Состояние загрузки одной модели."""

    model_id: str
    status: str = STATUS_IDLE
    fraction: float = 0.0
    bytes_done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fraction": round(self.fraction, 4) if self.status != STATUS_IDLE else None,
            "bytes_done": self.bytes_done,
            "total": self.total,
            "message": self.message,
            "error": self.error,
        }


@dataclass(slots=True)
class _Subscriber:
    queue: asyncio.Queue[dict[str, object] | None]
    loop: asyncio.AbstractEventLoop


class DownloadBus:
    """Глобальная (по всем моделям) потокобезопасная шина событий прогресса."""

    def __init__(self, *, heartbeat: float = DEFAULT_HEARTBEAT) -> None:
        self._heartbeat = heartbeat
        self._lock = threading.Lock()
        self._subscribers: list[_Subscriber] = []
        self._history: deque[dict[str, object]] = deque(maxlen=500)

    def publish(self, event: Mapping[str, object]) -> None:
        """Отправляет событие подписчикам и запоминает его в истории."""
        payload = dict(event)
        with self._lock:
            self._history.append(payload)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, payload)
            except RuntimeError:
                # Event loop закрыт (остановка сервера) — отдавать некому.
                continue

    def history(self) -> list[dict[str, object]]:
        """Снимок накопленных событий (для подключившихся с задержкой)."""
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        """Очищает историю (при остановке сервера/тестах)."""
        with self._lock:
            self._history.clear()

    async def subscribe(self) -> AsyncIterator[dict[str, object] | None]:
        """Живой поток событий; ``None`` — сигнал keep-alive по таймауту."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        subscriber = _Subscriber(queue=queue, loop=loop)
        with self._lock:
            self._subscribers.append(subscriber)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=self._heartbeat)
                except TimeoutError:
                    yield None
                    continue
                yield item
        finally:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)


@dataclass(slots=True)
class _ProgressTracker:
    """Накопитель прогресса по байтам с троттлингом публикации событий."""

    manager: ModelDownloadManager
    model_id: str
    cancel: threading.Event
    total: int
    done: int = 0
    last_event: float = field(default_factory=time.monotonic)

    def __call__(self, delta: int) -> None:
        if self.cancel.is_set():
            raise DownloadCancelled(self.model_id)
        increment = max(int(delta), 0)
        with self.manager.lock:
            self.done += increment
            state = self.manager.state_locked(self.model_id)
            state.bytes_done = self.done
            state.total = self.total
            if self.total > 0:
                state.fraction = min(self.done / self.total, 1.0)
            now = time.monotonic()
            changed = abs(state.fraction - self._last_fraction) >= _PROGRESS_STEP
            if not changed and (now - self.last_event) < _PROGRESS_INTERVAL:
                return
            self.last_event = now
            self._last_fraction = state.fraction
            event = self.manager.event_locked(state, "Скачивание…")
        self.manager.bus.publish(event)

    _last_fraction: float = 0.0


class ModelDownloadManager:
    """Фоновые загрузки моделей с прогрессом, отменой и удалением файлов."""

    def __init__(
        self,
        *,
        models_root: Path,
        downloader: Downloader,
        resolve_target: Callable[[ModelEntry], Path],
        bus: DownloadBus | None = None,
    ) -> None:
        self._models_root = models_root
        self._downloader = downloader
        self._resolve_target = resolve_target
        self._bus = bus or DownloadBus()
        self.lock = threading.Lock()
        self._states: dict[str, DownloadState] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancels: dict[str, threading.Event] = {}

    @property
    def bus(self) -> DownloadBus:
        return self._bus

    @property
    def models_root(self) -> Path:
        return self._models_root

    def state_locked(self, model_id: str) -> DownloadState:
        """Состояние модели (вызывается под ``self.lock``)."""
        state = self._states.get(model_id)
        if state is None:
            state = DownloadState(model_id=model_id)
            self._states[model_id] = state
        return state

    def state(self, model_id: str) -> DownloadState:
        """Снимок состояния модели."""
        with self.lock:
            return replace(self.state_locked(model_id))

    def is_running(self, model_id: str) -> bool:
        with self.lock:
            thread = self._threads.get(model_id)
            return thread is not None and thread.is_alive()

    def event_locked(
        self, state: DownloadState, message: str | None = None
    ) -> dict[str, object]:
        """Событие SSE по состоянию (вызывается под ``self.lock``)."""
        return {
            "id": state.model_id,
            "status": state.status,
            "fraction": round(state.fraction, 4) if state.status != STATUS_IDLE else None,
            "bytes_done": state.bytes_done,
            "total": state.total,
            "message": message if message is not None else state.message,
        }

    def start(self, entry: ModelEntry, *, token: str | None) -> bool:
        """Запускает загрузку модели; ``False`` — если она уже идёт."""
        with self.lock:
            existing = self._threads.get(entry.id)
            if existing is not None and existing.is_alive():
                return False
            target = self._resolve_target(entry)
            state = self.state_locked(entry.id)
            state.status = STATUS_DOWNLOADING
            state.fraction = 0.0
            state.bytes_done = 0
            state.total = entry.approx_size
            state.message = "Скачивание…"
            state.error = None
            cancel = threading.Event()
            self._cancels[entry.id] = cancel
            thread = threading.Thread(
                target=self._run,
                args=(entry, target, token, cancel),
                name=f"model-download-{entry.id}",
                daemon=True,
            )
            self._threads[entry.id] = thread
            event = self.event_locked(state)
        self._bus.publish(event)
        thread.start()
        return True

    def cancel(self, model_id: str) -> bool:
        """Просит прервать загрузку; ``False`` — если она не выполняется."""
        with self.lock:
            thread = self._threads.get(model_id)
            if thread is None or not thread.is_alive():
                return False
            cancel = self._cancels.get(model_id)
        if cancel is not None:
            cancel.set()
        return True

    def wait(self, model_id: str, *, timeout: float = 10.0) -> None:
        """Дожидается завершения потока загрузки (используется в тестах)."""
        with self.lock:
            thread = self._threads.get(model_id)
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(
        self, entry: ModelEntry, target: Path, token: str | None, cancel: threading.Event
    ) -> None:
        tracker = _ProgressTracker(
            manager=self,
            model_id=entry.id,
            cancel=cancel,
            total=entry.approx_size,
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            for model_file in entry.files:
                if cancel.is_set():
                    raise DownloadCancelled(entry.id)
                self._downloader.fetch(
                    repo=entry.repo,
                    filename=model_file.filename,
                    destination=target / model_file.filename,
                    token=token,
                    on_progress=tracker,
                )
            if entry.snapshot:
                if cancel.is_set():
                    raise DownloadCancelled(entry.id)
                self._downloader.fetch_snapshot(
                    repo=entry.repo,
                    destination=target,
                    token=token,
                    on_progress=tracker,
                )
            if cancel.is_set():
                raise DownloadCancelled(entry.id)
        except DownloadCancelled:
            self._finish(entry.id, STATUS_CANCELLED, message="Скачивание отменено")
            return
        except Exception as exc:  # noqa: BLE001 — внешняя сеть/диск: показываем причину
            message = _shorten(_redact(str(exc), token))
            logger.warning("Загрузка модели %s не удалась: %s", entry.id, message)
            self._finish(entry.id, STATUS_ERROR, message="Ошибка загрузки", error=message)
            return
        self._finish(entry.id, STATUS_DONE, message="Готово", actual_size=dir_size(target))

    def _finish(
        self,
        model_id: str,
        status: str,
        *,
        message: str,
        error: str | None = None,
        actual_size: int | None = None,
    ) -> None:
        with self.lock:
            state = self.state_locked(model_id)
            state.status = status
            state.message = message
            state.error = error
            if status == STATUS_DONE:
                state.fraction = 1.0
                if actual_size is not None:
                    state.bytes_done = actual_size
            event = self.event_locked(state)
            self._threads.pop(model_id, None)
            self._cancels.pop(model_id, None)
        self._bus.publish(event)


def model_payload(
    entry: ModelEntry,
    *,
    models_root: Path,
    settings: object,
    state: DownloadState | None = None,
) -> dict[str, object]:
    """JSON-представление записи каталога с локальным статусом и прогрессом."""
    target = resolve_target(entry, models_root=models_root, settings=settings)
    status = local_status(entry, target)
    primary = primary_path(entry, target)
    payload: dict[str, object] = {
        "id": entry.id,
        "kind": entry.kind,
        "title": entry.title,
        "repo": entry.repo,
        "files": [model_file.filename for model_file in entry.files],
        "approx_size": entry.approx_size,
        "note": entry.note,
        "gated": entry.gated,
        "setting_key": entry.setting_key,
        "target_dir": entry.target_dir,
        "target_path": str(target),
        "primary_path": str(primary),
        "status": {
            "present": status.present,
            "size": status.size,
            "expected_size": status.expected_size,
            "missing_files": list(status.missing),
            "path": str(status.path),
            "partial": status.partial,
        },
    }
    if state is not None:
        payload["download"] = state.as_dict()
    return payload
