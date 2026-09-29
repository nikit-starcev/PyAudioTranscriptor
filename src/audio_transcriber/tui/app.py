"""Textual-интерфейс в стиле htop для транскрибера.

Запуск: ``audio-transcriber tui``.

Позволяет выбрать аудио/видеофайлы, выстроить очередь, запустить обработку
и наблюдать живой прогресс по этапам (шумоподавление, распознавание,
диаризация, объединение, очистка артефактов, автоисправление, экспорт). Пока
идёт транскрибация, в очередь можно добавлять новые файлы — они обработаются
следом. Результаты показываются после запуска.
"""

from __future__ import annotations

import logging
import queue as queue_module
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Literal, NamedTuple

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
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

from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_VOICES_DIR,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.diarization.enrollment import EnrollmentOutcome, enroll_speakers
from audio_transcriber.diarization.samples import find_speaker_samples, samples_directory
from audio_transcriber.diarization.voices import (
    collect_voice_library,
    delete_voice_sample,
    merge_references,
    save_speaker_sample,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.domain.models import SpeakerSegment, TranscriptionResult
from audio_transcriber.export.factory import create_exporter
from audio_transcriber.export.timeline import build_speaker_tracks, write_timeline
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.client import (
    DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE,
)
from audio_transcriber.llm.client import create_llm_client
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.progress import ProgressEvent
from audio_transcriber.utils.config_env import load_config_env
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple
from audio_transcriber.utils.notifications import notify
from audio_transcriber.utils.playback import (
    SILENCE_RMS_THRESHOLD,
    PlaybackHandle,
    amplitude_envelope,
    read_duration,
    start_playback,
)
from audio_transcriber.utils.text import sanitize_filename

logger = logging.getLogger(__name__)

STAGES: list[tuple[str, str]] = [
    ("denoise", "Шумоподавление"),
    ("asr", "Распознавание речи"),
    ("diarization", "Определение говорящих"),
    ("merge", "Объединение сегментов"),
    ("clean", "Очистка артефактов"),
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


class ResultRow(NamedTuple):
    """Строка таблицы результатов: время, говорящий, метки, текст реплики."""

    time: str
    speaker: str
    marks: str
    text: str


def _search_terms(query: str) -> list[str]:
    """Разбивает запрос поиска на нормализованные (casefold) слова."""
    return [term.casefold() for term in query.split() if term]


def filter_result_rows(rows: Sequence[ResultRow], query: str) -> list[ResultRow]:
    """Фильтрует строки стенограммы по запросу без учёта регистра.

    Запрос разбивается по пробелам, и строка попадает в выборку, только если
    **все** слова запроса встречаются как подстроки в тексте реплики или в
    имени говорящего (логика «И»: ``"иван привет"`` найдёт реплики Ивана, в
    которых есть «привет»). Пустой запрос возвращает все строки.
    """
    terms = _search_terms(query)
    if not terms:
        return list(rows)

    matching: list[ResultRow] = []
    for row in rows:
        haystack = f"{row.speaker}\n{row.text}".casefold()
        if all(term in haystack for term in terms):
            matching.append(row)
    return matching


def rename_speaker(
    result: TranscriptionResult, speaker_id: str, new_name: str
) -> TranscriptionResult:
    """Возвращает копию результата с новым именем говорящего у него и реплик.

    Ничего не делает, если говорящий с таким идентификатором не найден.
    """

    if not any(speaker.id == speaker_id for speaker in result.speakers):
        return result
    speakers = [
        replace(speaker, display_name=new_name) if speaker.id == speaker_id else speaker
        for speaker in result.speakers
    ]
    renamed = next(speaker for speaker in speakers if speaker.id == speaker_id)
    entries = [
        replace(entry, speaker=renamed)
        if entry.speaker is not None and entry.speaker.id == speaker_id
        else entry
        for entry in result.entries
    ]
    return replace(result, speakers=speakers, entries=entries)


def merge_speakers(
    result: TranscriptionResult, source_id: str, target_id: str
) -> TranscriptionResult:
    """Сливает говорящего ``source_id`` в ``target_id``.

    Все реплики источника переназначаются целевому говорящему, источник
    удаляется из списка. Если целевого говорящего нет, результат не меняется.
    """

    target = next((speaker for speaker in result.speakers if speaker.id == target_id), None)
    if target is None or source_id == target_id:
        return result
    speakers = [speaker for speaker in result.speakers if speaker.id != source_id]
    entries = [
        replace(entry, speaker=target)
        if entry.speaker is not None and entry.speaker.id == source_id
        else entry
        for entry in result.entries
    ]
    return replace(result, speakers=speakers, entries=entries)


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
    _, defaults = load_config_env()
    return defaults


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
    clean_artifacts = app.query_one("#clean", Switch).value
    collapse_repeats = app.query_one("#collapse_repeats", Switch).value
    normalize_text = app.query_one("#normalize", Switch).value
    mark_overlap = app.query_one("#overlap", Switch).value
    diarization_enabled = app.query_one("#diarization", Switch).value
    denoise_enabled = app.query_one("#denoise", Switch).value
    use_cache = app.query_one("#cache", Switch).value
    notifications_enabled = app.query_one("#notifications", Switch).value
    timeline_enabled = app.query_one("#timeline", Switch).value

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

    speaker_references_raw = app.query_one("#speaker_references", Input).value.strip()
    speaker_references = (
        AppConfig.parse_speaker_references(
            [p.strip() for p in speaker_references_raw.split(",") if p.strip()]
        )
        if speaker_references_raw
        else {}
    )
    enrollment_min_raw = app.query_one("#enrollment_min_similarity", Input).value.strip()
    enrollment_min_similarity = _to_float(
        enrollment_min_raw or None, DEFAULT_ENROLLMENT_MIN_SIMILARITY
    )
    export_speaker_samples = app.query_one("#speaker_samples", Switch).value
    voices_dir_raw = app.query_one("#voices_dir", Input).value.strip()
    voices_dir = Path(voices_dir_raw) if voices_dir_raw else None

    hotwords = app.query_one("#hotwords", Input).value.strip() or None
    output_dir = app.query_one("#output_dir", Input).value.strip() or "output"

    llm_enabled = app.query_one("#llm", Switch).value
    llm_extract_names = app.query_one("#llm_names", Switch).value
    llm_summary = app.query_one("#llm_summary", Switch).value
    llm_model = app.query_one("#llm_model", Input).value.strip()
    llm_binary = app.query_one("#llm_binary", Input).value.strip() or "llama-server"
    llm_lib = app.query_one("#llm_lib", Input).value.strip()
    llm_prompt_extra = app.query_one("#llm_prompt_extra", Input).value.strip() or None
    llm_prompt_file_raw = app.query_one("#llm_prompt_file", Input).value.strip()
    llm_prompt_file = Path(llm_prompt_file_raw) if llm_prompt_file_raw else None
    glossary_path = app.query_one("#glossary_path", Input).value.strip()

    language_value = app.query_one("#language", Select).value
    language = language_value if isinstance(language_value, str) and language_value else None

    defaults = app._defaults
    pyannote_local_raw = defaults.get("PYANNOTE_LOCAL_MODEL", "").strip()
    pyannote_local_model = Path(pyannote_local_raw) if pyannote_local_raw else None

    wcp_threads_raw = defaults.get("WHISPER_CPP_THREADS", "").strip()
    whisper_cpp_threads = int(wcp_threads_raw) if wcp_threads_raw.isdigit() else None

    low_conf_raw = app.query_one("#low_conf", Input).value.strip()
    low_confidence_threshold = _to_float(low_conf_raw or None, DEFAULT_LOW_CONFIDENCE_THRESHOLD)

    cache_dir_raw = defaults.get("CACHE_DIR", "").strip()
    cache_dir = Path(cache_dir_raw) if cache_dir_raw else None

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
        speaker_references=speaker_references,
        enrollment_min_similarity=enrollment_min_similarity,
        voices_dir=voices_dir,
        export_speaker_samples=export_speaker_samples,
        hf_token=defaults.get("HF_TOKEN") or None,
        pyannote_local_model=pyannote_local_model,
        initial_prompt=defaults.get("INITIAL_PROMPT") or None,
        hotwords=hotwords,
        clean_artifacts=clean_artifacts,
        collapse_repeats=collapse_repeats,
        repeat_min_words=_to_int(defaults.get("REPEAT_MIN_WORDS"), DEFAULT_REPEAT_MIN_WORDS),
        repeat_similarity=_to_float(defaults.get("REPEAT_SIMILARITY"), DEFAULT_REPEAT_SIMILARITY),
        normalize_text=normalize_text,
        denoise=denoise_enabled,
        mark_overlap=mark_overlap,
        use_cache=use_cache,
        cache_dir=cache_dir,
        notifications=notifications_enabled,
        timeline=timeline_enabled,
        low_confidence_threshold=low_confidence_threshold,
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
        llm_summary=llm_summary,
        llm_prompt_extra=llm_prompt_extra,
        llm_prompt_file=llm_prompt_file,
        glossary_path=normalize_glossary_paths_tuple(glossary_path or None),
    )


#: Символы для строки амплитуды (от тишины к пику).
_AMP_CHARS = "▁▂▃▄▅▆▇█"


def _amplitude_line(envelope: Sequence[float], width: int | None = None) -> str:
    """Строка амплитуды из блочных символов (по одному на окно)."""
    values = list(envelope)
    if width is not None:
        values = values[:width]
    last = len(_AMP_CHARS) - 1
    chars = []
    for value in values:
        clamped = min(1.0, max(0.0, value))
        chars.append(_AMP_CHARS[round(clamped * last)])
    return "".join(chars)


def _progress_bar(fraction: float, width: int) -> str:
    """Полоса прогресса с курсором ``●`` на позиции ``fraction`` (0..1)."""
    if width <= 0:
        return ""
    position = min(width - 1, max(0, round(fraction * (width - 1))))
    return "─" * position + "●" + "─" * (width - 1 - position)


def _format_size(size: int) -> str:
    """Человекочитаемый размер файла (Б/КБ/МБ/ГБ)."""
    value = float(size)
    for unit in ("Б", "КБ", "МБ"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


class PlayerPanel(Vertical):
    """Панель проигрывания образца: время, амплитуда и полоса прогресса.

    Самодостаточна: владеет процессом плеера, таймером обновления и кэшем
    амплитуд. Экраны лишь вызывают :meth:`play`/:meth:`stop`/:meth:`toggle`.
    Амплитуда считается в фоновом потоке (``@work(thread=True)``), поэтому
    чтение аудио не блокирует интерфейс.
    """

    DEFAULT_CSS = """
    PlayerPanel { height: auto; }
    PlayerPanel .player-time { height: 1; color: $text-muted; }
    PlayerPanel .player-amp { height: 1; color: $accent; }
    PlayerPanel .player-bar { height: 1; color: $accent; }
    PlayerPanel .player-note { height: 1; color: $warning; }
    """

    def __init__(self, *, columns: int = 50, id: str | None = None) -> None:
        super().__init__(id=id)
        self._columns = columns
        self._handle: PlaybackHandle | None = None
        self._path: Path | None = None
        self._timer: Timer | None = None
        self._amp_cache: dict[Path, list[float]] = {}
        self._duration_cache: dict[Path, float] = {}

    def compose(self) -> ComposeResult:
        yield Static("⏱ 0.0 / 0.0 с", classes="player-time")
        yield Static("", classes="player-amp")
        yield Static("", classes="player-bar")
        yield Static("", classes="player-note")

    # --- публичный API ---

    @property
    def playing_path(self) -> Path | None:
        """Путь образца, который проигрывается сейчас (или ``None``)."""
        return self._path

    @property
    def is_playing(self) -> bool:
        """``True``, пока процесс плеера жив."""
        return self._handle is not None and self._handle.is_running()

    def duration(self, path: Path) -> float:
        """Длительность образца в секундах (кэшируется; ``0.0`` для не-WAV)."""
        cached = self._duration_cache.get(path)
        if cached is None:
            cached = read_duration(path)
            self._duration_cache[path] = cached
        return cached

    def play(self, path: Path) -> bool:
        """Начинает проигрывание; ``False`` — плеер недоступен или ошибка запуска."""
        self.stop()
        handle = start_playback(path)
        if handle is None:
            return False
        self._handle = handle
        self._path = path
        self._render_player(0.0)
        self._load_amplitude(path)
        self._start_timer()
        return True

    def toggle(self, path: Path) -> bool:
        """Повторный вызов для того же файла останавливает проигрывание."""
        if self.is_playing and self._path == path:
            self.stop()
            return True
        return self.play(path)

    def stop(self) -> None:
        """Останавливает плеер и возвращает панель в исходный вид."""
        self._stop_timer()
        if self._handle is not None:
            self._handle.stop()
        self._handle = None
        self._path = None
        self._write(".player-time", "⏱ 0.0 / 0.0 с")
        self._write(".player-amp", "")
        self._write(".player-bar", "")
        self._write(".player-note", "")

    # --- внутреннее ---

    def _write(self, selector: str, text: str) -> None:
        self.query_one(selector, Static).update(text)

    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer = self.set_interval(0.1, self._tick)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        handle = self._handle
        if handle is None:
            self._stop_timer()
            return
        if not handle.is_running():
            self.stop()
            return
        elapsed = handle.elapsed
        if handle.duration > 0 and elapsed >= handle.duration:
            self.stop()
            return
        self._render_player(elapsed)

    def _render_player(self, elapsed: float) -> None:
        duration = self._handle.duration if self._handle is not None else 0.0
        if duration <= 0 and self._path is not None:
            duration = self.duration(self._path)
        self._write(".player-time", f"⏱ {elapsed:.1f} / {duration:.1f} с")
        fraction = elapsed / duration if duration > 0 else 0.0
        self._write(".player-bar", _progress_bar(fraction, self._columns))

    def _load_amplitude(self, path: Path) -> None:
        cached = self._amp_cache.get(path)
        if cached is not None:
            self._show_amplitude(path, cached)
            return
        self._compute_amplitude(path)

    @work(thread=True, group="amplitude", exclusive=True)
    def _compute_amplitude(self, path: Path) -> None:
        envelope = amplitude_envelope(path, columns=self._columns)
        self.app.call_from_thread(self._on_amplitude, path, envelope)

    def _on_amplitude(self, path: Path, envelope: list[float]) -> None:
        self._amp_cache[path] = envelope
        if self.is_mounted:
            self._show_amplitude(path, envelope)

    def _show_amplitude(self, path: Path, envelope: list[float]) -> None:
        if self._path != path:
            return
        if not envelope or max(envelope) < SILENCE_RMS_THRESHOLD:
            self._write(".player-amp", "")
            self._write(".player-note", "тишина / нет голоса")
            return
        self._write(".player-amp", _amplitude_line(envelope, self._columns))
        self._write(".player-note", "")


class ConfirmDeleteScreen(ModalScreen[bool]):
    """Мини-подтверждение удаления образца голоса («Да»/«Нет»)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Нет"),
        ("n", "cancel", "Нет"),
        ("y", "confirm", "Да"),
    ]

    CSS = """
    ConfirmDeleteScreen { align: center middle; }
    #confirm_dialog {
        width: 52; height: auto;
        border: thick $error; background: $surface; padding: 1 2;
    }
    #confirm_text { height: auto; margin-bottom: 1; }
    #confirm_actions { height: auto; }
    #confirm_actions Button { margin-right: 1; }
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm_dialog"):
            yield Static(f"Удалить образец «{self._name}»?", id="confirm_text")
            with Horizontal(id="confirm_actions"):
                yield Button("Да [y]", id="confirm_yes", variant="error")
                yield Button("Нет [Esc]", id="confirm_no")
        yield Footer()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_yes":
            self.action_confirm()
        elif event.button.id == "confirm_no":
            self.action_cancel()


class VoicesLibraryScreen(ModalScreen[None]):
    """Просмотр и удаление библиотеки образцов голоса (``voices_dir``).

    Таблица показывает имя (stem), длительность, размер и имя файла. Выбранный
    образец можно проиграть (``p``) с той же визуализацией, что и в редакторе
    говорящих, или удалить (``d``/``Delete``) с подтверждением.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "close", "Закрыть"),
        ("r", "refresh", "Обновить"),
        ("p", "play_voice", "Проиграть"),
        ("d", "delete_voice", "Удалить"),
        ("delete", "delete_voice", "Удалить"),
    ]

    CSS = """
    VoicesLibraryScreen { align: center middle; }
    #voices_library {
        width: 84; height: auto;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    #voices_title { text-style: bold; height: 1; }
    #voices_path { height: 1; color: $text-muted; }
    #voices_table { height: 12; }
    #voices_player { margin-top: 1; }
    #voices_actions { height: auto; margin-top: 1; }
    #voices_actions Button { margin-right: 1; }
    #voices_status { height: auto; margin-top: 1; color: $text-muted; }
    """

    def __init__(self, *, voices_dir: Path | None = None) -> None:
        super().__init__()
        self._voices_dir = (
            Path(voices_dir) if voices_dir is not None else Path(DEFAULT_VOICES_DIR)
        )
        self._files: list[Path] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="voices_library"):
            yield Static("Библиотека голосов", id="voices_title")
            yield Static(str(self._voices_dir), id="voices_path")
            yield DataTable(id="voices_table", zebra_stripes=True, cursor_type="row")
            yield PlayerPanel(id="voices_player")
            with Horizontal(id="voices_actions"):
                yield Button("Проиграть [p]", id="voices_play")
                yield Button("Удалить [d]", id="voices_delete")
                yield Button("Обновить [r]", id="voices_refresh")
                yield Button("Закрыть [Esc]", id="voices_close")
            yield Static("", id="voices_status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#voices_table", DataTable)
        table.add_column("Имя", key="name")
        table.add_column("Длительность", key="duration", width=14)
        table.add_column("Размер", key="size", width=12)
        table.add_column("Файл", key="file")
        self.reload()

    def reload(self) -> None:
        """Перечитывает каталог библиотеки и перерисовывает таблицу."""
        table = self.query_one("#voices_table", DataTable)
        table.clear()
        self._files = []
        directory = self._voices_dir
        try:
            is_dir = directory.is_dir()
        except OSError:
            is_dir = False
        if not is_dir:
            self._set_status(
                f"Каталог библиотеки не найден: {directory}. "
                "Сохраните образец из редактора говорящих ([l])."
            )
            return

        try:
            entries = sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() == ".wav"
                ),
                key=lambda path: path.name.casefold(),
            )
        except OSError as exc:
            self._set_status(f"Не удалось прочитать каталог: {exc}")
            return

        panel = self.query_one("#voices_player", PlayerPanel)
        for path in entries:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            table.add_row(
                path.stem,
                f"{panel.duration(path):.1f} с",
                _format_size(size),
                path.name,
            )
            self._files.append(path)
        self._set_status(f"Образцов: {len(self._files)}")

    def selected_voice(self) -> Path | None:
        """Выбранный в таблице образец (или ``None``)."""
        table = self.query_one("#voices_table", DataTable)
        row = table.cursor_row
        if not self._files or row < 0 or row >= len(self._files):
            return None
        return self._files[row]

    def action_refresh(self) -> None:
        self.reload()

    def action_play_voice(self) -> None:
        path = self.selected_voice()
        if path is None:
            self._set_status("Не выбран образец")
            return
        panel = self.query_one("#voices_player", PlayerPanel)
        playing_before = panel.is_playing
        if not panel.toggle(path):
            self.app.notify(
                "Аудио-плеер не найден (ffplay/paplay/aplay/mpv/afplay) — "
                "установите один из них, чтобы прослушивать образцы",
                severity="warning",
                timeout=8,
            )
            return
        if panel.is_playing:
            self._set_status(f"Проигрываю: {path.name}")
        elif playing_before:
            self._set_status(f"Остановлено: {path.name}")

    def action_delete_voice(self) -> None:
        path = self.selected_voice()
        if path is None:
            self._set_status("Не выбран образец")
            return
        self.app.push_screen(
            ConfirmDeleteScreen(path.name),
            lambda confirmed: self._delete_confirmed(path, confirmed),
        )

    def _delete_confirmed(self, path: Path, confirmed: bool | None) -> None:
        if not confirmed:
            self._set_status("Удаление отменено")
            return
        panel = self.query_one("#voices_player", PlayerPanel)
        if panel.playing_path == path:
            panel.stop()
        if delete_voice_sample(path, self._voices_dir):
            self._set_status(f"Удалено: {path.name}")
            self.reload()
        else:
            self._set_status("Не удалось удалить образец (вне библиотеки или ошибка)")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "voices_play":
            self.action_play_voice()
        elif event.button.id == "voices_delete":
            self.action_delete_voice()
        elif event.button.id == "voices_refresh":
            self.action_refresh()
        elif event.button.id == "voices_close":
            self.action_close()

    def _set_status(self, message: str) -> None:
        self.query_one("#voices_status", Static).update(message)


class SpeakerEditorScreen(ModalScreen[TranscriptionResult | None]):
    """Модальный экран правки говорящих: переименование и объединение.

    Работает с копией результата: правки видны в таблице говорящих, но
    применяются к приложению только по «Сохранить» (возвращает копию). Закрытие
    через ``Esc``/«Отмена» возвращает ``None`` — исходный результат не меняется.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Отмена"),
        ("r", "rename", "Переименовать"),
        ("m", "merge", "Объединить"),
        ("p", "play_sample", "Проиграть"),
        ("l", "save_to_library", "В библиотеку"),
        ("v", "open_voices", "Голоса"),
        ("a", "apply_names", "Применить имена"),
        ("s", "save", "Сохранить"),
    ]

    CSS = """
    SpeakerEditorScreen { align: center middle; }
    #speaker_editor {
        width: 78; height: auto;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    #editor_title { text-style: bold; height: 1; margin-bottom: 1; }
    #speakers_table { height: 10; }
    #rename_row, #merge_row, #editor_actions, #editor_actions2 {
        height: auto; margin-top: 1;
    }
    #new_name, #merge_target { width: 1fr; }
    #rename, #merge { margin-left: 1; }
    #editor_actions Button, #editor_actions2 Button { margin-right: 1; }
    #player { margin-top: 1; }
    #editor_status { height: auto; margin-top: 1; color: $text-muted; }
    """

    def __init__(
        self,
        result: TranscriptionResult,
        *,
        samples: Mapping[str, Path] | None = None,
        voices_dir: Path | None = None,
        audio_path: Path | None = None,
        references: Mapping[str, Sequence[Path]] | None = None,
        min_similarity: float = DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    ) -> None:
        super().__init__()
        # Копия: Esc не должен менять исходный результат приложения.
        self._result = replace(
            result, speakers=list(result.speakers), entries=list(result.entries)
        )
        # Образцы голоса говорящих (speaker_id -> WAV) и куда их сохранять.
        self._samples: dict[str, Path] = dict(samples or {})
        self._voices_dir = voices_dir
        self._speaker_ids: list[str] = []
        # Контекст для «Применить имена» (enrollment без повторной расшифровки).
        self._audio_path = audio_path
        self._explicit_references: dict[str, tuple[Path, ...]] = {
            name: tuple(paths) for name, paths in (references or {}).items()
        }
        self._min_similarity = min_similarity

    @property
    def edited_result(self) -> TranscriptionResult:
        """Текущее (возможно, отредактированное) состояние говорящих."""
        return self._result

    def compose(self) -> ComposeResult:
        with Vertical(id="speaker_editor"):
            yield Static("Правка говорящих", id="editor_title")
            yield DataTable(id="speakers_table", zebra_stripes=True, cursor_type="row")
            with Horizontal(id="rename_row"):
                yield Input(placeholder="Новое имя…", id="new_name")
                yield Button("Переименовать [r]", id="rename")
            with Horizontal(id="merge_row"):
                yield Select([], id="merge_target", allow_blank=True, prompt="Куда объединить")
                yield Button("Объединить [m]", id="merge")
            with Horizontal(id="editor_actions"):
                yield Button("Проиграть [p]", id="play_sample")
                yield Button("Голоса [v]", id="open_voices")
                yield Button("В библиотеку [l]", id="save_to_library")
            with Horizontal(id="editor_actions2"):
                yield Button("Применить имена [a]", id="apply_names")
                yield Button("Сохранить [s]", id="save", variant="primary")
                yield Button("Отмена [Esc]", id="cancel")
            yield PlayerPanel(id="player")
            yield Static("", id="editor_status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#speakers_table", DataTable)
        table.add_column("ID", key="id", width=14)
        table.add_column("Имя", key="name")
        table.add_column("Реплик", key="count", width=8)
        table.add_column("Образец", key="sample", width=9)
        table.add_column("Длит.", key="duration", width=8)
        self.reload()

    def reload(self) -> None:
        """Перерисовывает таблицу говорящих и список целей объединения."""
        counts: Counter[str] = Counter(
            entry.speaker.id for entry in self._result.entries if entry.speaker is not None
        )
        table = self.query_one("#speakers_table", DataTable)
        table.clear()
        self._speaker_ids = [speaker.id for speaker in self._result.speakers]
        for speaker in self._result.speakers:
            sample = self._samples.get(speaker.id)
            exists = self._sample_exists(sample)
            duration_text = f"{self._sample_duration(sample):.1f}" if exists else "—"
            table.add_row(
                speaker.id,
                speaker.display_name,
                str(counts.get(speaker.id, 0)),
                "✓" if exists else "—",
                duration_text,
            )
        self.query_one("#merge_target", Select).set_options(
            [
                (f"{speaker.display_name} ({speaker.id})", speaker.id)
                for speaker in self._result.speakers
            ]
        )
        self._set_status("")

    def _sample_exists(self, sample: Path | None) -> bool:
        if sample is None:
            return False
        try:
            return sample.is_file()
        except OSError:
            return False

    def _sample_duration(self, sample: Path | None) -> float:
        if sample is None:
            return 0.0
        return self.query_one("#player", PlayerPanel).duration(sample)

    def selected_speaker_id(self) -> str | None:
        """Идентификатор говорящего в текущей строке таблицы (или ``None``)."""
        table = self.query_one("#speakers_table", DataTable)
        row = table.cursor_row
        if not self._speaker_ids or row < 0 or row >= len(self._speaker_ids):
            return None
        return self._speaker_ids[row]

    def rename_selected(self, new_name: str) -> bool:
        """Переименовывает выбранного говорящего (и все его реплики)."""
        speaker_id = self.selected_speaker_id()
        name = new_name.strip()
        if speaker_id is None:
            self._set_status("Не выбран говорящий")
            return False
        if not name:
            self._set_status("Введите новое имя")
            return False
        self._result = rename_speaker(self._result, speaker_id, name)
        self.reload()
        self._set_status(f"{speaker_id} → {name}")
        return True

    def merge_selected_into(self, target_id: str | None) -> bool:
        """Сливает выбранного говорящего в указанного ``target_id``."""
        source_id = self.selected_speaker_id()
        if source_id is None:
            self._set_status("Не выбран говорящий")
            return False
        if not target_id or target_id == source_id:
            self._set_status("Выберите другого говорящего")
            return False
        if not any(speaker.id == target_id for speaker in self._result.speakers):
            self._set_status("Целевой говорящий не найден")
            return False
        self._result = merge_speakers(self._result, source_id, target_id)
        self.reload()
        self._set_status(f"{source_id} объединён в {target_id}")
        return True

    def action_rename(self) -> None:
        self.rename_selected(self.query_one("#new_name", Input).value)

    def action_merge(self) -> None:
        value = self.query_one("#merge_target", Select).value
        self.merge_selected_into(value if isinstance(value, str) else None)

    def _selected_sample(self) -> Path | None:
        """Путь к образцу голоса выбранного говорящего (или ``None``)."""
        speaker_id = self.selected_speaker_id()
        if speaker_id is None:
            return None
        sample = self._samples.get(speaker_id)
        if sample is None:
            return None
        try:
            return sample if sample.is_file() else None
        except OSError:
            return None

    def _selected_display_name(self) -> str:
        speaker_id = self.selected_speaker_id() or ""
        return next(
            (
                speaker.display_name
                for speaker in self._result.speakers
                if speaker.id == speaker_id
            ),
            speaker_id,
        )

    def action_play_sample(self) -> None:
        """Неблокирующе проигрывает образец выбранного говорящего (``p``).

        Повторное нажатие останавливает проигрывание; выбор другого говорящего
        переключает панель на его образец.
        """
        if self.selected_speaker_id() is None:
            self._set_status("Не выбран говорящий")
            return
        sample = self._selected_sample()
        if sample is None:
            self.app.notify(
                "Для выбранного говорящего нет образца голоса",
                severity="warning",
                timeout=6,
            )
            return
        panel = self.query_one("#player", PlayerPanel)
        was_playing = panel.is_playing
        if not panel.toggle(sample):
            self.app.notify(
                "Аудио-плеер не найден (ffplay/paplay/aplay/mpv/afplay) — "
                "установите один из них, чтобы прослушивать образцы",
                severity="warning",
                timeout=8,
            )
            return
        if panel.is_playing:
            self._set_status(f"Проигрываю образец: {sample.name}")
        elif was_playing:
            self._set_status(f"Остановлено: {sample.name}")

    def action_open_voices(self) -> None:
        """Открывает библиотеку голосов из редактора говорящих (``v``)."""
        voices_dir = (
            self._voices_dir if self._voices_dir is not None else Path(DEFAULT_VOICES_DIR)
        )
        self.app.push_screen(VoicesLibraryScreen(voices_dir=voices_dir))

    def _resolved_references(self) -> dict[str, tuple[Path, ...]]:
        """Явные образцы + актуальное содержимое библиотеки голосов."""
        library = collect_voice_library(self._voices_dir)
        return merge_references(self._explicit_references, library)

    def _speaker_segments(self) -> list[SpeakerSegment]:
        """Восстанавливает сегменты говорящих из текущих реплик результата.

        Нужны для повторного enrollment без диаризации: у каждой реплики уже
        есть интервал и говорящий.
        """
        return [
            SpeakerSegment(start=entry.start, end=entry.end, speaker_id=entry.speaker.id)
            for entry in self._result.entries
            if entry.speaker is not None and entry.end > entry.start
        ]

    def action_apply_names(self) -> None:
        """Заново сопоставляет голоса на текущем аудио и применяет имена (``a``).

        Повторной расшифровки нет: enrollment сравнивает голос говорящего с
        образцами (библиотека + явные) и переименовывает реплики на месте.
        """
        if self._audio_path is None:
            self._set_status("Нет аудио для сопоставления голосов")
            return
        references = self._resolved_references()
        if not references:
            self._set_status("Нет образцов голоса (библиотека пуста, явные не заданы)")
            return
        speaker_segments = self._speaker_segments()
        if not speaker_segments:
            self._set_status("Нет сегментов говорящих для сопоставления")
            return
        self._set_status("Сопоставляю голоса по образцам…")
        self._apply_names_worker(self._audio_path, dict(references), speaker_segments)

    @work(thread=True, group="apply-names")
    def _apply_names_worker(
        self,
        audio_path: Path,
        references: Mapping[str, Sequence[Path]],
        speaker_segments: Sequence[SpeakerSegment],
    ) -> None:
        outcome = enroll_speakers(
            speaker_segments=speaker_segments,
            references=references,
            audio_path=audio_path,
            min_similarity=self._min_similarity,
        )
        self.app.call_from_thread(self._on_names_applied, outcome)

    @staticmethod
    def _format_best_candidates(
        candidates: Mapping[str, tuple[str, float]],
    ) -> str:
        """Короткая сводка лучших недобранных пар: ``SPEAKER_04≈«Имя» 0.48``."""
        if not candidates:
            return ""
        ordered = sorted(candidates.items(), key=lambda item: (-item[1][1], item[0]))
        return "; ".join(
            f"{speaker_id}≈«{name}» {score:.2f}"
            for speaker_id, (name, score) in ordered[:3]
        )

    def _on_names_applied(self, outcome: EnrollmentOutcome) -> None:
        mapping = outcome.mapping
        total = outcome.speaker_count
        severity: Literal["information", "warning"]
        for speaker_id, name in mapping.items():
            self._result = rename_speaker(self._result, speaker_id, name)
        if mapping:
            self.reload()
            applied = ", ".join(f"{speaker_id} → {name}" for speaker_id, name in mapping.items())
            message = f"Применены имена ({len(mapping)}/{total}): {applied}"
            severity = "information"
        else:
            message = f"Имена по голосу не сопоставлены (0/{total}, ниже порога)"
            severity = "warning"
        unmatched = self._format_best_candidates(outcome.best_candidates)
        if unmatched:
            message += f" | не добрали: {unmatched}"
        self._set_status(message)
        self.app.notify(message, severity=severity, timeout=10)

    def action_save_to_library(self) -> None:
        """Копирует образец выбранного говорящего в библиотеку ``voices_dir`` (``l``)."""
        if self.selected_speaker_id() is None:
            self._set_status("Не выбран говорящий")
            return
        sample = self._selected_sample()
        if sample is None:
            self.app.notify(
                "Для выбранного говорящего нет образца голоса",
                severity="warning",
                timeout=6,
            )
            return
        if self._voices_dir is None:
            self.app.notify(
                "Библиотека голосов не задана (voices_dir)",
                severity="warning",
                timeout=6,
            )
            return
        name = self._selected_display_name()
        try:
            target = save_speaker_sample(sample, self._voices_dir, name)
        except OSError as exc:
            self.app.notify(
                f"Не удалось сохранить образец: {exc}", severity="error", timeout=8
            )
            return
        self._set_status(f"Сохранено в библиотеку: {target.name}")

    def action_save(self) -> None:
        self.dismiss(self._result)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rename":
            self.action_rename()
        elif event.button.id == "merge":
            self.action_merge()
        elif event.button.id == "play_sample":
            self.action_play_sample()
        elif event.button.id == "open_voices":
            self.action_open_voices()
        elif event.button.id == "apply_names":
            self.action_apply_names()
        elif event.button.id == "save_to_library":
            self.action_save_to_library()
        elif event.button.id == "save":
            self.action_save()
        elif event.button.id == "cancel":
            self.action_cancel()

    def _set_status(self, message: str) -> None:
        self.query_one("#editor_status", Static).update(message)


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
        ("slash", "focus_search", "Поиск (/)"),
        ("ctrl+f", "focus_search", "Поиск"),
        ("e", "edit_speakers", "Спикеры (e)"),
        ("v", "open_voices_library", "Голоса (v)"),
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

                    yield Button("Запустить очередь  [r]", id="run", variant="primary")

                yield ProgressPanel()
                with Horizontal(id="search_bar"):
                    yield Input(placeholder="Поиск по стенограмме (/)…", id="search")
                    yield Static("", id="search_status")
                    yield Static(
                        "e — спикеры, v — голоса  ·  ⚠ низкая уверенность  ·  ⇄ наложение",
                        id="legend",
                    )
                yield DataTable(id="results", zebra_stripes=True)
                yield Static("", id="status")

        yield Footer()

    def on_mount(self) -> None:
        results = self.query_one("#results", DataTable)
        results.add_column("Время", key="time", width=10)
        results.add_column("Говорящий", key="speaker", width=16)
        results.add_column("Метки", key="marks", width=8)
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
        from audio_transcriber.export.annotations import is_low_confidence
        from audio_transcriber.export.timestamps import format_timestamp

        threshold = result.low_confidence_threshold
        rows: list[ResultRow] = []
        for entry in result.entries:
            speaker = entry.speaker.display_name if entry.speaker else "?"
            marks = ""
            if is_low_confidence(entry, threshold):
                marks += "⚠"
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
        """Применяет правки редактора говорящих и переэкспортирует результат."""
        if result is None:  # Esc/«Отмена» — оставляем всё как было
            return
        self._last_result = result
        self._refresh_samples_after_rename(result)
        self._populate_results(result)
        self.query_one("#results", DataTable).remove_class("hidden")
        try:
            exported = self._reexport_result(result)
        except Exception as exc:  # noqa: BLE001 — показываем ошибку экспорта в UI
            self.notify(f"Не удалось экспортировать: {exc}", severity="error", timeout=8)
            return
        names = ", ".join(path.name for path in exported)
        self.notify(f"Спикеры сохранены. Экспортировано: {names or 'нет форматов'}")

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

    def _reexport_result(self, result: TranscriptionResult) -> list[Path]:
        """Повторно выгружает результат в форматы (и таймлайн) последнего конфига."""
        config = self._last_config
        if config is None:
            return []
        config.ensure_output_dir()
        exported: list[Path] = []
        for export_format in config.export_formats:
            output_path = config.output_dir / f"{config.input_file.stem}.{export_format.value}"
            create_exporter(export_format).export(result, output_path)
            exported.append(output_path)
        if config.timeline and build_speaker_tracks(result):
            timeline_path = config.output_dir / f"{config.input_file.stem}.timeline.html"
            if write_timeline(result, timeline_path):
                exported.append(timeline_path)
        return exported

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
