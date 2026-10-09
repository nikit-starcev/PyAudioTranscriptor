"""Сборка Textual-приложения транскрибера.

Здесь остаётся только :class:`TranscriberApp` (компоновка, биндинги, обработка
сообщений и делегирование) и точка входа ``main``. Детали вынесены в соседние
модули пакета:

* :mod:`~audio_transcriber.tui.screens` — модальные экраны;
* :mod:`~audio_transcriber.tui.widgets` — дерево файлов, прогресс, проигрыватель;
* :mod:`~audio_transcriber.tui.config_builder` — сборка :class:`AppConfig`;
* :mod:`~audio_transcriber.tui.messages` — сообщения воркер → интерфейс;
* :mod:`~audio_transcriber.tui.queue` — состояние очереди;
* :mod:`~audio_transcriber.tui.options` — опции выпадающих списков;
* :mod:`~audio_transcriber.tui.results` — строки таблицы результатов;
* :mod:`~audio_transcriber.tui.constants` — этапы, расширения, метки;
* :mod:`~audio_transcriber.tui.formatting` — форматирование и разбор значений.

Имена из вынесенных модулей ре-экспортируются здесь для совместимости импортов
(``audio_transcriber.tui.app``). Запуск: ``audio-transcriber tui``.
"""

from __future__ import annotations

import logging
import queue as queue_module
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    Switch,
)

from audio_transcriber.config import defaults as config_defaults
from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_VOICES_DIR,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.enrollment import EnrollmentOutcome
from audio_transcriber.diarization.samples import find_speaker_samples, samples_directory
from audio_transcriber.domain.editing import merge_speakers, rename_speaker
from audio_transcriber.domain.enums import AsrBackend, Device
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.client import create_llm_client
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.protocol import ProtocolArtifacts, generate_protocol
from audio_transcriber.storage.glossary_db import GlossaryDB
from audio_transcriber.tui.config_builder import build_config_from_widgets
from audio_transcriber.tui.constants import _STATUS_MARK, STAGES
from audio_transcriber.tui.formatting import _amplitude_line, _progress_bar, _to_bool
from audio_transcriber.tui.messages import FileDone, FileStarted, ProgressUpdate, QueueDone
from audio_transcriber.tui.options import (
    _BACKEND_OPTIONS,
    _DEVICE_OPTIONS,
    _LANG_OPTIONS,
    _format_select,
    _select_value,
)
from audio_transcriber.tui.queue import QueueController
from audio_transcriber.tui.results import ResultRow, filter_result_rows
from audio_transcriber.tui.screens import (
    ConfirmDeleteScreen,
    SpeakerEditorScreen,
    VoicesLibraryScreen,
)
from audio_transcriber.tui.widgets import MediaDirectoryTree, PlayerPanel, ProgressPanel
from audio_transcriber.utils.config_env import load_config_env
from audio_transcriber.utils.notifications import notify
from audio_transcriber.utils.text import sanitize_filename

logger = logging.getLogger(__name__)

# Ре-экспорт имён из вынесенных модулей — для совместимости импортов и тестов.
__all__ = [
    "STAGES",
    "ConfirmDeleteScreen",
    "EnrollmentOutcome",
    "MediaDirectoryTree",
    "PlayerPanel",
    "ProtocolArtifacts",
    "QueueController",
    "QueueDone",
    "SpeakerEditorScreen",
    "TranscriberApp",
    "VoicesLibraryScreen",
    "_amplitude_line",
    "_load_env_defaults",
    "_progress_bar",
    "merge_speakers",
    "rename_speaker",
]


def _load_env_defaults() -> dict[str, str]:
    """Читает ``config.env`` рядом с проектом и возвращает словарь настроек."""
    _, defaults = load_config_env()
    return defaults


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

    #search_bar { height: 3; }
    #search { width: 1fr; }
    #search_status { width: auto; padding: 0 1; content-align: left middle; color: $text-muted; }
    #legend { width: auto; padding: 0 2; content-align: left middle; color: $text-muted; }
    #results { height: 1fr; }
    #results.hidden { display: none; }
    #status { height: 1; color: $text-muted; }
    #run_actions { height: auto; }
    #run { width: 1fr; }
    #build_protocol { width: 1fr; margin-left: 1; }
    #glossary_sources { height: auto; margin-left: 22; }
    #glossary_status { height: auto; margin-left: 22; color: $text-muted; }

    .hidden { display: none; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Выход"),
        ("r", "run", "Запустить"),
        ("a", "add_to_queue", "В очередь"),
        ("m", "toggle_mode", "Режим"),
        ("h", "toggle_settings", "Настройки"),
        ("t", "toggle_sidebar", "Файлы"),
        ("slash", "focus_search", "Поиск (/)"),
        ("ctrl+f", "focus_search", "Поиск"),
        ("e", "edit_speakers", "Спикеры (e)"),
        ("v", "open_voices_library", "Голоса (v)"),
        ("ctrl+p", "generate_protocol", "Протокол"),
        ("escape", "clear_search", "Сброс поиска"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._defaults = _load_env_defaults()
        self._selected_file: Path | None = None
        self._queue = QueueController()
        self._transcribing = False
        self._run_start_time: float | None = None
        # Полный (нефильтрованный) набор строк таблицы результатов; фильтр
        # поиска применяется к нему при каждой отрисовке.
        self._result_rows: list[ResultRow] = []
        # Уведомление по завершении очереди (берётся из собранного конфига).
        self._notify_enabled = False
        self._files_done = 0
        self._files_failed = 0
        # Последний завершённый результат и соответствующий ему конфиг — нужны
        # для правки говорящих (переименование/объединение) и переэкспорта без
        # повторного прогона транскрибации.
        self._last_result: TranscriptionResult | None = None
        self._last_config: AppConfig | None = None
        # Образцы голоса говорящих последнего результата (speaker_id -> WAV),
        # найденные в <output>/<файл>.speakers/ — для проигрывания и сохранения
        # в библиотеку из редактора говорящих.
        self._last_samples: dict[str, Path] = {}
        self._base_config: AppConfig | None = None
        # Источники глоссария, показанные в настройках: id чекбокса -> имя
        # источника. Нужно, чтобы переключение чекбокса адресовало БД.
        self._glossary_source_ids: dict[str, str] = {}
        # Протокол формируется по кнопке; после правок имён он «устаревает».
        self._protocol_stale = False
        self._protocol_building = False

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
                        yield Input(
                            value=self._defaults.get("NUM_SPEAKERS", ""),
                            placeholder="пусто = авто",
                            id="num_speakers",
                            type="integer",
                        )
                    with Horizontal():
                        yield Label("Шумоподавление (денойз)", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("DENOISE"), default=True),
                            id="denoise",
                        )
                    with Horizontal():
                        yield Label("Кэш результатов", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("USE_CACHE"), default=True),
                            id="cache",
                        )
                    with Horizontal():
                        yield Label("Уведомления", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("NOTIFICATIONS"), default=True),
                            id="notifications",
                        )
                    with Horizontal():
                        yield Label("Диаризация", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("DIARIZATION_ENABLED"), default=True),
                            id="diarization",
                        )
                    with Horizontal():
                        yield Label("Автоисправление", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("ENABLE_CORRECTION"), default=False),
                            id="correction",
                        )
                    with Horizontal():
                        yield Label("Очистка артефактов", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("CLEAN_ARTIFACTS"), default=True),
                            id="clean",
                        )
                    with Horizontal():
                        yield Label("Схлопывать повторы", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("COLLAPSE_REPEATS"), default=True),
                            id="collapse_repeats",
                        )
                    with Horizontal():
                        yield Label("Нормализация текста", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("NORMALIZE_TEXT"), default=True),
                            id="normalize",
                        )
                    with Horizontal():
                        yield Label("Наложение речи", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("MARK_OVERLAP"), default=True),
                            id="overlap",
                        )
                    with Horizontal():
                        yield Label("Таймлайн говорящих", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("TIMELINE"), default=True),
                            id="timeline",
                        )
                    with Horizontal():
                        yield Label("Образцы голоса (файлы)", classes="field-label")
                        yield Switch(
                            value=_to_bool(
                                self._defaults.get("EXPORT_SPEAKER_SAMPLES"), default=True
                            ),
                            id="speaker_samples",
                        )
                    with Horizontal():
                        yield Label("Порог уверенности", classes="field-label")
                        yield Input(
                            value=self._defaults.get("LOW_CONFIDENCE_THRESHOLD", "-1.0"),
                            id="low_conf",
                        )
                    with Horizontal():
                        yield Label("Макс. пауза склейки, с", classes="field-label")
                        yield Input(
                            value=self._defaults.get("SENTENCE_MERGE_MAX_GAP", "5.0"),
                            id="sentence_merge_max_gap",
                        )
                    with Horizontal():
                        yield Label("LLM-обработка", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("LLM_ENABLED")),
                            id="llm",
                        )
                    with Horizontal():
                        yield Label("Имена (эксперим.)", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("LLM_EXTRACT_NAMES"), default=False),
                            id="llm_names",
                        )
                    with Horizontal():
                        yield Label("Резюме встречи", classes="field-label")
                        yield Switch(
                            value=_to_bool(self._defaults.get("LLM_SUMMARY"), default=True),
                            id="llm_summary",
                        )
                    with Horizontal():
                        yield Label("Семант. правки (LLM)", classes="field-label")
                        yield Switch(
                            value=_to_bool(
                                self._defaults.get("LLM_CORRECT_SEMANTIC"), default=False
                            ),
                            id="llm_semantic",
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
                        yield Label("Доп. промпт (текст)", classes="field-label")
                        yield Input(
                            value=self._defaults.get("LLM_PROMPT_EXTRA", ""), id="llm_prompt_extra"
                        )
                    with Horizontal():
                        yield Label("Доп. промпт (файл)", classes="field-label")
                        yield Input(
                            value=self._defaults.get("LLM_PROMPT_FILE", ""), id="llm_prompt_file"
                        )
                    with Horizontal():
                        yield Label("Глоссарий", classes="field-label")
                        yield Checkbox(
                            "Использовать глоссарий",
                            value=_to_bool(
                                self._defaults.get("GLOSSARY_ENABLED"), default=True
                            ),
                            id="glossary_enabled",
                        )
                    with Horizontal():
                        yield Label("БД глоссария", classes="field-label")
                        yield Input(
                            value=self._defaults.get("GLOSSARY_DB", ""),
                            placeholder="glossary.db",
                            id="glossary_db",
                        )
                    with Horizontal():
                        yield Label("Источники", classes="field-label")
                        yield Button("Обновить источники", id="glossary_refresh")
                    yield Vertical(id="glossary_sources")
                    yield Static("", id="glossary_status")

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
                            yield Label("Образцы голоса", classes="field-label")
                            yield Input(
                                value=self._defaults.get("SPEAKER_REFERENCES", ""),
                                placeholder="Имя=путь.wav,Имя2=путь2.wav",
                                id="speaker_references",
                            )
                        with Horizontal():
                            yield Label("Библиотека голосов", classes="field-label")
                            yield Input(
                                value=self._defaults.get("VOICES_DIR", ""),
                                placeholder="voices",
                                id="voices_dir",
                            )
                        with Horizontal():
                            yield Label("Порог голоса", classes="field-label")
                            yield Input(
                                value=self._defaults.get(
                                    "ENROLLMENT_MIN_SIMILARITY",
                                    str(DEFAULT_ENROLLMENT_MIN_SIMILARITY),
                                ),
                                id="enrollment_min_similarity",
                            )
                        with Horizontal():
                            yield Label("Hotwords", classes="field-label")
                            yield Input(value=self._defaults.get("HOTWORDS", ""), id="hotwords")
                        with Horizontal():
                            yield Label("Доп. файлы глоссария", classes="field-label")
                            yield Input(
                                value=self._defaults.get("GLOSSARY_PATH", ""),
                                placeholder="файл.txt,файл.csv",
                                id="glossary_path",
                            )

                    with Horizontal(id="run_actions"):
                        yield Button(
                            "Запустить очередь  [r]", id="run", variant="primary"
                        )
                        yield Button(
                            "Сформировать протокол  [Ctrl+P]",
                            id="build_protocol",
                            disabled=True,
                        )

                yield ProgressPanel()
                with Horizontal(id="search_bar"):
                    yield Input(placeholder="Поиск по стенограмме (/)…", id="search")
                    yield Static("", id="search_status")
                    yield Static(
                        "e — спикеры, v — голоса, Ctrl+P — протокол  ·  "
                        "⚠ низкая уверенность  ·  ⇄ наложение речи  ·  "
                        "? говорящий под вопросом",
                        id="legend",
                    )
                yield DataTable(id="results", zebra_stripes=True)
                yield Static("", id="status")

        yield Footer()

    def on_mount(self) -> None:
        results = self.query_one("#results", DataTable)
        results.add_column("Время", key="time", width=10)
        results.add_column("Говорящий", key="speaker", width=24)
        results.add_column("Метки", key="marks", width=10)
        results.add_column("Текст", key="text")

        queue = self.query_one("#queue", DataTable)
        queue.add_column("Статус", key="status", width=6)
        queue.add_column("Файл", key="file")

        self._progress = self.query_one(ProgressPanel)
        self._progress.reset()
        self.query_one("#progress", ProgressPanel).add_class("hidden")
        self.query_one("#results", DataTable).add_class("hidden")
        self._refresh_glossary_sources()
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

    def action_focus_search(self) -> None:
        """Переносит фокус в поле поиска по стенограмме (``/`` или ``Ctrl+F``)."""
        self.query_one("#search", Input).focus()

    def action_clear_search(self) -> None:
        """Сбрасывает фильтр поиска (``Esc``) и показывает все строки."""
        search = self.query_one("#search", Input)
        if search.value:
            search.value = ""
        self._render_results()

    def action_edit_speakers(self) -> None:
        """Открывает редактор говорящих по последнему результату (``e``)."""
        if self._last_result is None:
            self.notify(
                "Нет результата — сначала выполните транскрибацию",
                severity="warning",
                timeout=5,
            )
            return
        config = self._last_config
        if config is not None:
            voices_dir: Path | None = config.resolved_voices_dir()
            audio_path: Path | None = config.input_file
            references: Mapping[str, Sequence[Path]] = config.speaker_references
            min_similarity = config.enrollment_min_similarity
        else:
            voices_dir = None
            audio_path = None
            references = {}
            min_similarity = DEFAULT_ENROLLMENT_MIN_SIMILARITY
        self.push_screen(
            SpeakerEditorScreen(
                self._last_result,
                samples=self._last_samples,
                voices_dir=voices_dir,
                audio_path=audio_path,
                references=references,
                min_similarity=min_similarity,
            ),
            self._on_speaker_editor_closed,
        )

    def action_open_voices_library(self) -> None:
        """Открывает библиотеку голосов напрямую из главного экрана (``v``)."""
        voices_dir = (
            self._last_config.resolved_voices_dir()
            if self._last_config is not None
            else Path(DEFAULT_VOICES_DIR)
        )
        self.push_screen(VoicesLibraryScreen(voices_dir=voices_dir))

    def action_generate_protocol(self) -> None:
        """Собирает протокол по текущему результату (``Ctrl+P``).

        Резюме пересчитывается по актуальной стенограмме (с учётом правок
        имён), документы выгружаются в форматы последнего прогона. Работа
        идёт в фоновом потоке, чтобы не блокировать интерфейс.
        """
        if self._last_result is None or self._last_config is None:
            self.notify(
                "Нет результата — сначала выполните транскрибацию",
                severity="warning",
                timeout=5,
            )
            return
        if self._protocol_building:
            return
        self._protocol_building = True
        self.query_one("#build_protocol", Button).disabled = True
        self.query_one("#status", Static).update("Формирую протокол…")
        self._generate_protocol_worker(self._last_config, self._last_result)

    @work(thread=True, group="protocol")
    def _generate_protocol_worker(
        self, config: AppConfig, result: TranscriptionResult
    ) -> None:
        try:
            artifacts = generate_protocol(config, result, on_progress=self._emit_progress)
        except Exception as exc:  # noqa: BLE001 — показываем ошибку в UI
            self.call_from_thread(self._on_protocol_failed, str(exc))
            return
        self.call_from_thread(self._on_protocol_ready, artifacts)

    def _on_protocol_ready(self, artifacts: ProtocolArtifacts) -> None:
        self._protocol_building = False
        self._protocol_stale = False
        self.query_one("#build_protocol", Button).disabled = False
        names = ", ".join(path.name for path in artifacts.paths) or "нет форматов"
        summary_note = " с резюме" if artifacts.summary else ""
        message = f"Протокол сформирован{summary_note}: {names}"
        self.query_one("#status", Static).update(message)
        self.notify(message)

    def _on_protocol_failed(self, error: str) -> None:
        self._protocol_building = False
        self.query_one("#build_protocol", Button).disabled = False
        self.query_one("#status", Static).update(f"Протокол не сформирован: {error}")
        self.notify(f"Протокол не сформирован: {error}", severity="error", timeout=8)

    def _mark_protocol_stale(self) -> None:
        """Отмечает, что после правок имён протокол нужно пересобрать."""
        self._protocol_stale = True
        self.query_one("#status", Static).update(
            "имена изменены — нажмите [Ctrl+P], чтобы пересобрать протокол"
        )

    # --- сообщения ---

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search":
            self._render_results()

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
        elif event.button.id == "glossary_refresh":
            self._refresh_glossary_sources()
        elif event.button.id == "build_protocol":
            self.action_generate_protocol()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Сохраняет переключение источника глоссария в его БД."""
        checkbox_id = event.checkbox.id
        if checkbox_id is None:
            return
        source_name = self._glossary_source_ids.get(checkbox_id)
        if source_name is None:
            return
        try:
            with GlossaryDB(self._resolved_glossary_db_path()) as db:
                db.set_source_enabled(source_name, event.value)
        except (OSError, sqlite3.Error) as exc:
            self.notify(
                f"Не удалось сохранить источник глоссария: {exc}",
                severity="error",
                timeout=8,
            )
            return
        state = "включён" if event.value else "отключён"
        self.query_one("#glossary_status", Static).update(
            f"Источник «{source_name}» {state}"
        )

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
            self._files_failed += 1
            self.query_one("#status", Static).update(f"Ошибка: {message.error}")
            self.notify(message.error, severity="error", timeout=10)
            return

        assert message.result is not None
        self._files_done += 1
        # Запоминаем последний результат и конфиг для редактора говорящих.
        self._last_result = message.result
        if self._base_config is not None:
            self._last_config = replace(self._base_config, input_file=message.path)
        self._last_samples = self._collect_samples(message.result)
        self._populate_results(message.result)
        self.query_one("#results", DataTable).remove_class("hidden")
        self.query_one("#build_protocol", Button).disabled = False
        self._protocol_stale = False
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
        self._notify_queue_finished()

    def _notify_queue_finished(self) -> None:
        """Десктоп-уведомление о завершении очереди (если оно включено)."""
        if not self._notify_enabled:
            return
        if self._files_failed:
            notify(
                "Транскрибация завершена с ошибками",
                f"Обработано {self._files_done} из "
                f"{self._files_done + self._files_failed} файл(ов), "
                f"ошибок: {self._files_failed}",
            )
        else:
            notify("Транскрибация завершена", f"Обработано {self._files_done} файл(ов)")

    # --- внутреннее ---

    def _resolved_glossary_db_path(self) -> Path:
        """Путь к БД глоссария из поля настроек (или значение по умолчанию)."""
        raw = self.query_one("#glossary_db", Input).value.strip()
        if raw:
            return Path(raw)
        return Path(config_defaults.DEFAULT_GLOSSARY_DB)

    def _refresh_glossary_sources(self) -> None:
        """Перечитывает источники глоссария из БД и перерисовывает список.

        На каждый источник создаётся чекбокс с именем, типом и числом записей;
        его состояние отражает ``enabled`` источника. Пустая БД — подсказка об
        импорте.
        """
        container = self.query_one("#glossary_sources", Vertical)
        status = self.query_one("#glossary_status", Static)
        container.remove_children()
        self._glossary_source_ids = {}
        path = self._resolved_glossary_db_path()
        try:
            with GlossaryDB(path) as db:
                sources = db.list_sources()
                counts = db.entry_counts()
        except (OSError, sqlite3.Error) as exc:
            status.update(f"Не удалось открыть БД глоссария ({path}): {exc}")
            return

        if not sources:
            status.update(
                "БД глоссария пуста — импортируйте источник: "
                "audio-transcriber glossary import <файл>"
            )
            return

        for index, source in enumerate(sources):
            widget_id = f"glossary_src_{index}"
            self._glossary_source_ids[widget_id] = source.name
            count = counts.get(source.name, 0)
            container.mount(
                Checkbox(
                    f"{source.name} · {source.kind} · записей: {count}",
                    value=source.enabled,
                    id=widget_id,
                )
            )
        status.update(f"Источников: {len(sources)} (снятая галочка отключает источник)")

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
        self._base_config = base_config
        self._notify_enabled = base_config.notifications
        self._files_done = 0
        self._files_failed = 0
        run_button = self.query_one("#run", Button)
        run_button.disabled = True
        run_button.label = "Идёт транскрибация..."
        self.query_one("#settings", VerticalScroll).add_class("hidden")
        self.query_one("#results", DataTable).add_class("hidden")
        self._run_queue_worker(base_config)

    def _build_config(self, input_file: Path) -> AppConfig:
        return build_config_from_widgets(self, input_file)

    def _collect_samples(self, result: TranscriptionResult) -> dict[str, Path]:
        """Ищет сохранённые образцы голоса последнего прогона (``<файл>.speakers``)."""
        if self._last_config is None:
            return {}
        directory = samples_directory(self._last_config.output_dir, result.source_path)
        return find_speaker_samples(result, directory)

    def _populate_results(self, result: TranscriptionResult) -> None:
        from audio_transcriber.export.annotations import (
            DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD,
            is_low_confidence,
            is_low_speaker_confidence,
        )
        from audio_transcriber.export.timestamps import format_timestamp

        threshold = result.low_confidence_threshold
        rows: list[ResultRow] = []
        for entry in result.entries:
            speaker = entry.speaker_label
            marks = ""
            if is_low_confidence(entry, threshold):
                marks += "⚠"
            if is_low_speaker_confidence(entry, DEFAULT_SPEAKER_CONFIDENCE_THRESHOLD):
                marks += "?"
            if entry.overlap:
                marks += "⇄"
            rows.append(ResultRow(format_timestamp(entry.start), speaker, marks, entry.text))
        self._result_rows = rows
        self._render_results()

    def _render_results(self) -> None:
        """Рисует таблицу результатов с учётом текущего фильтра поиска."""
        query = self.query_one("#search", Input).value
        table = self.query_one("#results", DataTable)
        table.clear()
        rows = filter_result_rows(self._result_rows, query)
        for row in rows:
            table.add_row(row.time, row.speaker, row.marks, row.text)

        status = self.query_one("#search_status", Static)
        if query.strip():
            status.update(f"Совпадений: {len(rows)} / {len(self._result_rows)}")
        else:
            status.update(f"Реплик: {len(self._result_rows)}")

    def _on_speaker_editor_closed(self, result: TranscriptionResult | None) -> None:
        """Применяет правки редактора говорящих и помечает протокол устаревшим."""
        if result is None:  # Esc/«Отмена» — оставляем всё как было
            return
        self._last_result = result
        self._refresh_samples_after_rename(result)
        self._populate_results(result)
        self.query_one("#results", DataTable).remove_class("hidden")
        # Протокол в TUI собирается отдельно: правка имён делает его устаревшим.
        self._mark_protocol_stale()
        self.notify("Спикеры сохранены. Нажмите Ctrl+P, чтобы собрать протокол.")

    def _refresh_samples_after_rename(self, result: TranscriptionResult) -> None:
        """Приводит образцы говорящих в соответствие новым именам.

        Файлы образцов называются по ``display_name``; после переименования
        говорящего (например, «Применить имена») старые имена перестают
        находиться. Переименовываем файлы и заново собираем карту.
        """
        previous = dict(self._last_samples)
        for speaker in result.speakers:
            sample = previous.get(speaker.id)
            if sample is None or not sample.is_file():
                continue
            target = sample.parent / f"{sanitize_filename(speaker.display_name)}.wav"
            if target == sample or target.exists():
                continue
            try:
                sample.rename(target)
            except OSError as exc:
                logger.warning("Не удалось переименовать образец %s: %s", sample, exc)
        collected = self._collect_samples(result)
        for speaker_id, sample in previous.items():
            if speaker_id not in collected and sample.is_file():
                collected[speaker_id] = sample
        self._last_samples = collected

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
            return create_llm_client(config)
        except Exception as exc:  # noqa: BLE001 — при сбое обработаем без LLM
            logger.warning("Не удалось создать LLM-клиент: %s", exc)
            return None


def main() -> None:
    TranscriberApp().run()


if __name__ == "__main__":
    main()
