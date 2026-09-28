"""Textual-интерфейс в стиле htop для транскрибера.

Запуск: ``audio-transcriber tui``.

Позволяет выбрать аудио/видеофайлы, выстроить очередь, запустить обработку
и наблюдать живой прогресс по этапам (распознавание, диаризация, объединение,
автоисправление, экспорт). Пока идёт транскрибация, в очередь можно добавлять
новые файлы — они обработаются следом. Результаты показываются после запуска.
"""

from __future__ import annotations

import logging
import queue as queue_module
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    Switch,
)

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.client import (
    DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE,
)
from audio_transcriber.llm.client import create_llm_client
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple

logger = logging.getLogger(__name__)

STAGES: list[tuple[str, str]] = [
    ("asr", "Распознавание речи"),
    ("diarization", "Определение говорящих"),
    ("merge", "Объединение сегментов"),
    ("correction", "Автоисправление опечаток"),
    ("llm", "LLM-постобработка"),
    ("export", "Экспорт"),
]

_FORMAT_OPTIONS = [
    ("txt", "txt"),
    ("txt + docx", "txt,docx"),
    ("txt + docx + json + srt", "txt,docx,json,srt"),
]

_LANG_OPTIONS = [
    ("авто", ""),
    ("ru", "ru"),
    ("en", "en"),
]

_BACKEND_OPTIONS = [
    ("whisper-cpp (Vulkan)", "whisper-cpp"),
    ("faster-whisper", "faster-whisper"),
]

_DEVICE_OPTIONS = [
    ("auto", "auto"),
    ("cpu", "cpu"),
    ("cuda", "cuda"),
]

_VALID_FORMATS = {fmt.value for fmt in ExportFormat}


def _format_select(raw: str) -> tuple[list[tuple[str, str]], str]:
    """Строит опции форматов и выбранное значение из строки ``FORMATS``.

    Если в ``config.env`` задан набор, которого нет среди стандартных, он
    добавляется отдельным пунктом — настройка не теряется.
    """
    normalized = ",".join(part.strip() for part in raw.split(",") if part.strip())
    parts = normalized.split(",") if normalized else []
    if parts and all(part in _VALID_FORMATS for part in parts):
        for _, value in _FORMAT_OPTIONS:
            if value == normalized:
                return _FORMAT_OPTIONS, normalized
        label = " + ".join(parts)
        return [*_FORMAT_OPTIONS, (label, normalized)], normalized
    return _FORMAT_OPTIONS, ExportFormat.TXT.value


def _select_value(raw: str, options: list[tuple[str, str]], default: str) -> str:
    """Возвращает ближайшее допустимое значение выпадающего списка."""
    values = {value for _, value in options}
    candidate = (raw or "").strip()
    if candidate in values:
        return candidate
    for value in values:
        if value.lower() == candidate.lower():
            return value
    return default


# Расширения аудио- и видеофайлов, которые показываются в дереве файлов.
MEDIA_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".ape",
    ".amr",
    ".mp2",
    ".mpga",
    ".wv",
    ".mp4",
    ".m4v",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".ts",
    ".mts",
    ".m2ts",
    ".3gp",
    ".3g2",
    ".ogv",
    ".vob",
}

_STATUS_MARK = {
    "pending": Text("·", style="dim"),
    "running": Text("→", style="bold cyan"),
    "done": Text("✓", style="green"),
    "error": Text("✗", style="red"),
}


class MediaDirectoryTree(DirectoryTree):
    """Дерево файлов, показывающее только папки и аудио/видеофайлы."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        for path in paths:
            try:
                if path.is_dir() or path.suffix.lower() in MEDIA_EXTENSIONS:
                    yield path
            except OSError:
                continue


def _load_env_defaults() -> dict[str, str]:
    """Читает ``config.env`` рядом с проектом и возвращает словарь настроек."""
    candidates = (Path.cwd(), Path(__file__).resolve().parents[3])
    for root in candidates:
        config_path = root / "config.env"
        if config_path.is_file():
            defaults: dict[str, str] = {}
            for line in config_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                defaults[key.strip()] = value.strip().strip('"').strip("'")
            return defaults
    return {}


class ProgressUpdate(Message):
    def __init__(self, event: ProgressEvent) -> None:
        super().__init__()
        self.event = event


class FileStarted(Message):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


class FileDone(Message):
    def __init__(
        self, path: Path, result: TranscriptionResult | None = None, error: str | None = None
    ) -> None:
        super().__init__()
        self.path = path
        self.result = result
        self.error = error


class QueueDone(Message):
    pass


class ProgressPanel(Vertical):
    """Панель живого прогресса: текущая стадия, полоса и чек-лист с временем."""

    def __init__(self) -> None:
        super().__init__(id="progress")
        self._done: set[str] = set()
        self._current: str | None = None
        self._started: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}
        self._overall_start: float | None = None

    def compose(self) -> ComposeResult:
        yield Static("Ожидание запуска", id="cur_stage")
        yield ProgressBar(show_eta=False, id="bar")
        yield Static("", id="stage_list")

    def reset(self) -> None:
        self._done = set()
        self._current = None
        self._started = {}
        self._elapsed = {}
        self._overall_start = None
        self.query_one("#cur_stage", Static).update("Ожидание запуска")
        self.query_one("#bar", ProgressBar).update(total=None)
        self._render_list()

    def apply_event(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        if self._overall_start is None:
            self._overall_start = now

        order = {stage: i for i, (stage, _) in enumerate(STAGES)}
        if event.stage == "done":
            for stage, _ in STAGES:
                self._finish_stage(stage, now)
            self._done = {stage for stage, _ in STAGES}
            self._current = None
            total = now - self._overall_start
            self.query_one("#cur_stage", Static).update(f"Готово — {_fmt_duration(total)}")
            self.query_one("#bar", ProgressBar).update(total=100, progress=100)
        elif event.stage in order:
            idx = order[event.stage]
            if event.stage not in self._started:
                self._started[event.stage] = now
            # завершаем все предыдущие стадии
            for stage, _ in STAGES:
                if order[stage] < idx:
                    self._finish_stage(stage, now)

            self._current = event.stage
            self._done = {stage for stage, _ in STAGES if order[stage] < idx}
            # ВАЖНО: не помечаем стадию завершённой по fraction>=1.0 —
            # стадии бывают составными (диаризация: segmentation → embeddings),
            # и завершение под-этапа ≠ завершение стадии. Стадия закрывается,
            # когда стартует следующая стадия или приходит событие "done".

            label = STAGES[idx][1]
            detail = f" · {event.detail}" if event.detail else ""
            self.query_one("#cur_stage", Static).update(f"{label}{detail}")

            bar = self.query_one("#bar", ProgressBar)
            if event.fraction is not None:
                bar.update(total=100, progress=round(event.fraction * 100))
            else:
                bar.update(total=None)
        self._render_list()

    def _finish_stage(self, stage: str, now: float) -> None:
        if stage in self._started and stage not in self._elapsed:
            self._elapsed[stage] = now - self._started[stage]

    def refresh_times(self) -> None:
        """Обновляет живое время текущей стадии (вызывается по таймеру)."""
        if self._current is not None:
            self._render_list()

    def _stage_time(self, stage: str) -> str | None:
        if stage == self._current and stage in self._started:
            return _fmt_duration(time.monotonic() - self._started[stage]) + "…"
        if stage in self._elapsed:
            return _fmt_duration(self._elapsed[stage])
        return None

    def _render_list(self) -> None:
        lines = []
        for stage, label in STAGES:
            if stage in self._done:
                mark = "[green]✓[/green]"
            elif stage == self._current:
                mark = "[bold cyan]●[/bold cyan]"
            else:
                mark = "[dim]·[/dim]"
            elapsed = self._stage_time(stage)
            time_str = f"  [dim]— {elapsed}[/dim]" if elapsed else ""
            lines.append(f"{mark} {label}{time_str}")
        self.query_one("#stage_list", Static).update("\n".join(lines))


class QueueController:
    """Состояние очереди файлов TUI: строки таблицы, статусы и очередь на обработку.

    Вынесено из :class:`TranscriberApp`, чтобы логика добавления/статусов не
    смешивалась с обработкой сообщений и виджетами. Повторное добавление уже
    известного файла не создаёт дублирующей строки.
    """

    def __init__(self) -> None:
        self.pending: queue_module.Queue[Path] = queue_module.Queue()
        self.rows: list[Path] = []
        self.status: dict[Path, str] = {}

    def is_waiting(self, path: Path) -> bool:
        """Файл уже в очереди или обрабатывается прямо сейчас."""
        return self.status.get(path) in ("pending", "running")

    def is_empty(self) -> bool:
        return self.pending.empty()

    def first(self) -> Path:
        """Первый файл очереди (без извлечения)."""
        return self.pending.queue[0]

    def add(self, path: Path) -> None:
        """Ставит файл в очередь, переиспользуя существующую строку."""
        if path not in self.rows:
            self.rows.append(path)
        self.status[path] = "pending"
        self.pending.put(path)

    def mark(self, path: Path, status: str) -> None:
        self.status[path] = status


def build_config_from_widgets(app: TranscriberApp, input_file: Path) -> AppConfig:
    """Собирает :class:`AppConfig` из значений виджетов TUI.

    Вынесено из :meth:`TranscriberApp._build_config`, чтобы сборка конфигурации
    не разрасталась внутри god-объекта приложения.
    """
    num_speakers_raw = app.query_one("#num_speakers", Input).value.strip()
    num_speakers = int(num_speakers_raw) if num_speakers_raw else None

    formats_value = app.query_one("#formats", Select).value
    formats_raw = formats_value if isinstance(formats_value, str) else "txt"
    export_formats = tuple(
        dict.fromkeys(ExportFormat(f.strip()) for f in formats_raw.split(",") if f.strip())
    )

    enable_correction = app.query_one("#correction", Switch).value
    diarization_enabled = app.query_one("#diarization", Switch).value

    backend_raw = str(app.query_one("#backend", Select).value or "")
    backend = (
        AsrBackend(backend_raw)
        if backend_raw in {item.value for item in AsrBackend}
        else AsrBackend.FASTER_WHISPER
    )
    device_raw = str(app.query_one("#device", Select).value or "")
    device = Device(device_raw) if device_raw in {item.value for item in Device} else Device.AUTO

    model = app.query_one("#model", Input).value.strip() or "large-v3-turbo"
    wcp_model = app.query_one("#wcp_model", Input).value.strip()
    wcp_binary = app.query_one("#wcp_binary", Input).value.strip() or "whisper-cli"
    wcp_lib = app.query_one("#wcp_lib", Input).value.strip()

    speaker_names_raw = app.query_one("#speaker_names", Input).value.strip()
    speaker_names = (
        AppConfig.parse_speaker_names(
            [p.strip() for p in speaker_names_raw.split(",") if p.strip()]
        )
        if speaker_names_raw
        else {}
    )

    hotwords = app.query_one("#hotwords", Input).value.strip() or None
    output_dir = app.query_one("#output_dir", Input).value.strip() or "output"

    llm_enabled = app.query_one("#llm", Switch).value
    llm_extract_names = app.query_one("#llm_names", Switch).value
    llm_model = app.query_one("#llm_model", Input).value.strip()
    llm_binary = app.query_one("#llm_binary", Input).value.strip() or "llama-server"
    llm_lib = app.query_one("#llm_lib", Input).value.strip()
    glossary_path = app.query_one("#glossary_path", Input).value.strip()

    language_value = app.query_one("#language", Select).value
    language = language_value if isinstance(language_value, str) and language_value else None

    defaults = app._defaults
    pyannote_local_raw = defaults.get("PYANNOTE_LOCAL_MODEL", "").strip()
    pyannote_local_model = Path(pyannote_local_raw) if pyannote_local_raw else None

    wcp_threads_raw = defaults.get("WHISPER_CPP_THREADS", "").strip()
    whisper_cpp_threads = int(wcp_threads_raw) if wcp_threads_raw.isdigit() else None

    return AppConfig(
        input_file=input_file,
        output_dir=Path(output_dir),
        model_name=model,
        language=language,
        device=device,
        export_formats=export_formats,
        num_speakers=num_speakers,
        diarization_enabled=diarization_enabled,
        speaker_names=speaker_names,
        hf_token=defaults.get("HF_TOKEN") or None,
        pyannote_local_model=pyannote_local_model,
        initial_prompt=defaults.get("INITIAL_PROMPT") or None,
        hotwords=hotwords,
        enable_correction=enable_correction,
        correction_min_word_length=_to_int(
            defaults.get("CORRECTION_MIN_WORD_LENGTH"),
            DEFAULT_CORRECTION_MIN_WORD_LENGTH,
        ),
        correction_min_similarity=_to_float(
            defaults.get("CORRECTION_MIN_SIMILARITY"),
            DEFAULT_CORRECTION_MIN_SIMILARITY,
        ),
        correction_max_candidates=_to_int(
            defaults.get("CORRECTION_MAX_CANDIDATES"),
            DEFAULT_CORRECTION_MAX_CANDIDATES,
        ),
        verbose=_to_bool(defaults.get("VERBOSE")),
        asr_backend=backend,
        whisper_cpp_model=Path(wcp_model) if wcp_model else None,
        whisper_cpp_binary=wcp_binary,
        whisper_cpp_lib_path=wcp_lib or None,
        whisper_cpp_threads=whisper_cpp_threads,
        llm_enabled=llm_enabled,
        llm_model=Path(llm_model) if llm_model else None,
        llm_binary=llm_binary,
        llm_lib_path=llm_lib or None,
        llm_gpu=_to_bool(defaults.get("LLM_GPU"), default=True),
        llm_context_size=_to_int(defaults.get("LLM_CONTEXT"), DEFAULT_LLM_CONTEXT_SIZE),
        llm_suggest_terms=_to_bool(defaults.get("LLM_SUGGEST_TERMS")),
        llm_extract_names=llm_extract_names,
        glossary_path=normalize_glossary_paths_tuple(glossary_path or None),
    )


class TranscriberApp(App):
    """Основное приложение TUI транскрибера."""

    TITLE = "AudioTranscriptor"
    SUB_TITLE = "локальная транскрибация с разметкой говорящих"

    CSS = """
    #body { height: 1fr; }

    #sidebar { width: 42; border-right: solid $primary; padding: 0 1; }
    #sidebar.hidden { display: none; }
    #tree { height: 1fr; }
    #selected_file { height: 1; color: $text-muted; }
    #add_to_queue { width: 100%; }
    #queue { height: 10; }

    #main { width: 1fr; height: 1fr; }

    #settings { height: 1fr; padding: 1 2; }
    #settings.hidden { display: none; }
    #settings Horizontal { height: auto; }
    .field-label { color: $text-muted; }
    #settings .field-label { width: 22; height: 3; content-align: left middle; }
    #settings Input, #settings Select { width: 1fr; }

    #advanced { height: auto; }
    #advanced.hidden { display: none; }

    #progress { height: auto; padding: 1 2; border: solid $primary; }
    #progress.hidden { display: none; }
    #cur_stage { height: 1; text-style: bold; color: $accent; }
    #bar { height: 1; }
    #stage_list { height: auto; }

    #results { height: 1fr; }
    #results.hidden { display: none; }
    #status { height: 1; color: $text-muted; }
    #run { width: 100%; }

    .hidden { display: none; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Выход"),
        ("r", "run", "Запустить"),
        ("a", "add_to_queue", "В очередь"),
        ("m", "toggle_mode", "Режим"),
        ("h", "toggle_settings", "Настройки"),
        ("t", "toggle_sidebar", "Файлы"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._defaults = _load_env_defaults()
        self._selected_file: Path | None = None
        self._queue = QueueController()
        self._transcribing = False
        self._run_start_time: float | None = None

    # --- очередь (тонкие обёртки над QueueController) ------------------

    @property
    def _pending(self) -> queue_module.Queue[Path]:
        return self._queue.pending

    @_pending.setter
    def _pending(self, value: queue_module.Queue[Path]) -> None:
        self._queue.pending = value

    @property
    def _queue_rows(self) -> list[Path]:
        return self._queue.rows

    @property
    def _queue_status(self) -> dict[Path, str]:
        return self._queue.status

    def compose(self) -> ComposeResult:
        format_options, format_value = _format_select(self._defaults.get("FORMATS", ""))
        backend_value = _select_value(
            self._defaults.get("ASR_BACKEND", ""),
            _BACKEND_OPTIONS,
            AsrBackend.FASTER_WHISPER.value,
        )
        device_value = _select_value(
            self._defaults.get("DEVICE", ""), _DEVICE_OPTIONS, Device.AUTO.value
        )
        yield Header(show_clock=True)

        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("Файлы (аудио/видео)", classes="field-label")
                yield MediaDirectoryTree(Path.home(), id="tree")
                yield Static("выберите файл стрелками", id="selected_file")
                yield Button("＋ Добавить в очередь  [a]", id="add_to_queue")
                yield Label("Очередь", classes="field-label")
                yield DataTable(id="queue")

            with Vertical(id="main"):
                with VerticalScroll(id="settings"):
                    with Horizontal():
                        yield Label("Режим", classes="field-label")
                        yield Select(
                            [("Простой", "simple"), ("Продвинутый", "advanced")],
                            id="mode",
                            value="simple",
                            allow_blank=False,
                        )
                    with Horizontal():
                        yield Label("Язык", classes="field-label")
                        yield Select(
                            _LANG_OPTIONS,
                            id="language",
                            value=self._defaults.get("LANGUAGE", "ru"),
                            allow_blank=False,
                        )
                    with Horizontal():
                        yield Label("Формат", classes="field-label")
                        yield Select(
                            format_options,
                            id="formats",
                            value=format_value,
                            allow_blank=False,
                        )
                    with Horizontal():
                        yield Label("Результаты", classes="field-label")
                        yield Input(
                            value=self._defaults.get("OUTPUT_DIR", "output"), id="output_dir"
                        )
                    with Horizontal():
                        yield Label("Говорящих", classes="field-label")
                        yield Input(placeholder="пусто = авто", id="num_speakers", type="integer")
                    with Horizontal():
                        yield Label("Диаризация", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("DIARIZATION_ENABLED"), default=True),
                            id="diarization",
                        )
                    with Horizontal():
                        yield Label("Автоисправление", classes="field-label")
                        yield Switch(value=False, id="correction")
                    with Horizontal():
                        yield Label("LLM-обработка", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("LLM_ENABLED")),
                            id="llm",
                        )
                    with Horizontal():
                        yield Label("Определять имена", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("LLM_EXTRACT_NAMES"), default=True),
                            id="llm_names",
                        )
                    with Horizontal():
                        yield Label("Модель LLM", classes="field-label")
                        yield Input(value=self._defaults.get("LLM_MODEL", ""), id="llm_model")
                    with Horizontal():
                        yield Label("Бинарник LLM", classes="field-label")
                        yield Input(
                            value=self._defaults.get("LLM_BINARY", "llama-server"), id="llm_binary"
                        )
                    with Horizontal():
                        yield Label("Библиотеки LLM", classes="field-label")
                        yield Input(value=self._defaults.get("LLM_LIB_PATH", ""), id="llm_lib")
                    with Horizontal():
                        yield Label("Глоссарий", classes="field-label")
                        yield Input(
                            value=self._defaults.get("GLOSSARY_PATH", ""), id="glossary_path"
                        )

                    with Container(id="advanced", classes="hidden"):
                        with Horizontal():
                            yield Label("Бэкенд", classes="field-label")
                            yield Select(
                                _BACKEND_OPTIONS,
                                id="backend",
                                value=backend_value,
                                allow_blank=False,
                            )
                        with Horizontal():
                            yield Label("Модель", classes="field-label")
                            yield Input(
                                value=self._defaults.get("MODEL", "large-v3-turbo"), id="model"
                            )
                        with Horizontal():
                            yield Label("Модель whisper.cpp", classes="field-label")
                            yield Input(
                                value=self._defaults.get("WHISPER_CPP_MODEL", ""), id="wcp_model"
                            )
                        with Horizontal():
                            yield Label("Бинарник whisper", classes="field-label")
                            yield Input(
                                value=self._defaults.get("WHISPER_CPP_BINARY", "whisper-cli"),
                                id="wcp_binary",
                            )
                        with Horizontal():
                            yield Label("Библиотеки whisper", classes="field-label")
                            yield Input(
                                value=self._defaults.get("WHISPER_CPP_LIB_PATH", ""), id="wcp_lib"
                            )
                        with Horizontal():
                            yield Label("Устройство", classes="field-label")
                            yield Select(
                                _DEVICE_OPTIONS,
                                id="device",
                                value=device_value,
                                allow_blank=False,
                            )
                        with Horizontal():
                            yield Label("Имена говорящих", classes="field-label")
                            yield Input(
                                value=self._defaults.get("SPEAKER_NAMES", ""), id="speaker_names"
                            )
                        with Horizontal():
                            yield Label("Hotwords", classes="field-label")
                            yield Input(value=self._defaults.get("HOTWORDS", ""), id="hotwords")

                    yield Button("Запустить очередь  [r]", id="run", variant="primary")

                yield ProgressPanel()
                yield DataTable(id="results", zebra_stripes=True)
                yield Static("", id="status")

        yield Footer()

    def on_mount(self) -> None:
        results = self.query_one("#results", DataTable)
        results.add_column("Время", key="time", width=10)
        results.add_column("Говорящий", key="speaker", width=16)
        results.add_column("Текст", key="text")

        queue = self.query_one("#queue", DataTable)
        queue.add_column("Статус", key="status", width=6)
        queue.add_column("Файл", key="file")

        self._progress = self.query_one(ProgressPanel)
        self._progress.reset()
        self.query_one("#progress", ProgressPanel).add_class("hidden")
        self.query_one("#results", DataTable).add_class("hidden")
        self.set_interval(0.5, self._tick)

    # --- действия (клавиши) ---

    def action_run(self) -> None:
        self._start_run()

    def action_add_to_queue(self) -> None:
        self._add_selected_to_queue()

    def action_toggle_mode(self) -> None:
        select = self.query_one("#mode", Select)
        select.value = "advanced" if select.value == "simple" else "simple"

    def action_toggle_settings(self) -> None:
        self.query_one("#settings", VerticalScroll).toggle_class("hidden")

    def action_toggle_sidebar(self) -> None:
        self.query_one("#sidebar", Vertical).toggle_class("hidden")

    # --- сообщения ---

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected_file = event.path
        self.query_one("#selected_file", Static).update(f"выбран: {event.path.name}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mode":
            advanced = self.query_one("#advanced", Container)
            if event.value == "advanced":
                advanced.remove_class("hidden")
            else:
                advanced.add_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self._start_run()
        elif event.button.id == "add_to_queue":
            self._add_selected_to_queue()

    def on_progress_update(self, message: ProgressUpdate) -> None:
        self._progress.apply_event(message.event)

    def on_file_started(self, message: FileStarted) -> None:
        self._queue.mark(message.path, "running")
        self._render_queue()
        self._progress.reset()
        self.query_one("#progress", ProgressPanel).remove_class("hidden")
        self.query_one("#status", Static).update(f"Обработка: {message.path.name}")

    def on_file_done(self, message: FileDone) -> None:
        self._queue.mark(message.path, "error" if message.error else "done")
        self._render_queue()

        if message.error is not None:
            self.query_one("#status", Static).update(f"Ошибка: {message.error}")
            self.notify(message.error, severity="error", timeout=10)
            return

        assert message.result is not None
        self._populate_results(message.result)
        self.query_one("#results", DataTable).remove_class("hidden")
        elapsed = time.time() - (self._run_start_time or time.time())
        self.query_one("#status", Static).update(
            f"{message.path.name}: {len(message.result.entries)} реплик, "
            f"{len(message.result.speakers)} говорящих, за {elapsed:.0f} с"
        )

    def on_queue_done(self, _message: QueueDone) -> None:
        self._transcribing = False
        run_button = self.query_one("#run", Button)
        run_button.disabled = False
        run_button.label = "Запустить очередь  [r]"
        self.query_one("#progress", ProgressPanel).add_class("hidden")
        self.query_one("#settings", VerticalScroll).remove_class("hidden")
        self.query_one("#status", Static).update(
            "Очередь обработана. Можно добавить файлы и запустить снова."
        )

    # --- внутреннее ---

    def _add_selected_to_queue(self) -> None:
        if self._selected_file is None:
            self.notify("Сначала выберите файл в дереве", severity="warning", timeout=5)
            return
        if self._queue.is_waiting(self._selected_file):
            self.notify("Файл уже в очереди", severity="warning", timeout=5)
            return
        # Повторное добавление уже обработанного файла не создаёт вторую
        # строку — контроллер переиспользует существующую запись.
        self._queue.add(self._selected_file)
        self._render_queue()
        self.notify(f"В очередь: {self._selected_file.name}")

    def _render_queue(self) -> None:
        table = self.query_one("#queue", DataTable)
        table.clear()
        for path in self._queue.rows:
            status = self._queue.status.get(path, "pending")
            table.add_row(_STATUS_MARK[status], path.name)

    def _start_run(self) -> None:
        if self._transcribing:
            return
        if self._queue.is_empty():
            self.notify("Очередь пуста — добавьте файлы", severity="warning", timeout=5)
            return
        try:
            first = self._queue.first()
            base_config = self._build_config(first)
            base_config.ensure_output_dir()
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error", timeout=8)
            self.query_one("#status", Static).update(f"Ошибка: {exc}")
            return

        self._transcribing = True
        self._run_start_time = time.time()
        run_button = self.query_one("#run", Button)
        run_button.disabled = True
        run_button.label = "Идёт транскрибация..."
        self.query_one("#settings", VerticalScroll).add_class("hidden")
        self.query_one("#results", DataTable).add_class("hidden")
        self._run_queue_worker(base_config)

    def _build_config(self, input_file: Path) -> AppConfig:
        return build_config_from_widgets(self, input_file)

    def _populate_results(self, result: TranscriptionResult) -> None:
        table = self.query_one("#results", DataTable)
        table.clear()
        from audio_transcriber.export.timestamps import format_timestamp

        for entry in result.entries:
            speaker = entry.speaker.display_name if entry.speaker else "?"
            table.add_row(format_timestamp(entry.start), speaker, entry.text)

    def _tick(self) -> None:
        if self._transcribing and self._run_start_time is not None:
            elapsed = time.time() - self._run_start_time
            self.query_one("#status", Static).update(f"Обработка... {elapsed:.0f} с")
            self._progress.refresh_times()

    def _emit_progress(self, event: ProgressEvent) -> None:
        self.post_message(ProgressUpdate(event))

    @work(thread=True, group="transcribe")
    def _run_queue_worker(self, base_config: AppConfig) -> None:
        # Один LLM-клиент (и, значит, один llama-server) на всю очередь:
        # модель грузится один раз, а не на каждый файл. Закрываем его в
        # finally — после обработки не должно оставаться процессов.
        llm_client = self._create_llm_client(base_config)
        try:
            while True:
                try:
                    path = self._pending.get(timeout=2.0)
                except queue_module.Empty:
                    break

                self.post_message(FileStarted(path))
                config = replace(base_config, input_file=path)
                try:
                    result = run_pipeline(
                        config,
                        llm_client=llm_client,
                        on_progress=self._emit_progress,
                    )
                    self.post_message(FileDone(path=path, result=result))
                except Exception as exc:  # noqa: BLE001 — показываем ошибку и продолжаем
                    self.post_message(FileDone(path=path, error=str(exc)))
        finally:
            if llm_client is not None:
                try:
                    llm_client.close()
                except Exception as exc:  # noqa: BLE001 — не блокируем QueueDone
                    logger.warning("Не удалось закрыть LLM-клиент: %s", exc)
            # Очередь завершена (даже при неожиданной ошибке) — UI разблокируем,
            # но только после того, как llama-server гарантированно закрыт.
            self.post_message(QueueDone())

    @staticmethod
    def _create_llm_client(config: AppConfig) -> LlmClient | None:
        """Создаёт один LLM-клиент на очередь (или ``None``, если LLM выключен)."""
        if not config.llm_enabled:
            return None
        try:
            return create_llm_client(
                model_path=config.llm_model,
                binary=config.llm_binary,
                library_path=config.llm_lib_path,
                gpu=config.llm_gpu,
                context_size=config.llm_context_size,
            )
        except Exception as exc:  # noqa: BLE001 — при сбое обработаем без LLM
            logger.warning("Не удалось создать LLM-клиент: %s", exc)
            return None


def _fmt_duration(seconds: float) -> str:
    """Форматирует длительность: «<1 с», «12 с» или «1:23» для минут."""
    if seconds < 1:
        return "<1 с"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs} с"


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        return default


def _to_bool(value: str | None, default: bool = False) -> bool:
    """Разбирает булево значение из ``config.env`` («true», «1», «да», …)."""
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "да")


def main() -> None:
    TranscriberApp().run()


if __name__ == "__main__":
    main()
