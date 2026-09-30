"""Виджеты TUI: дерево файлов, панель прогресса и панель проигрывания."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import DirectoryTree, ProgressBar, Static

from audio_transcriber.progress import ProgressEvent
from audio_transcriber.tui.constants import MEDIA_EXTENSIONS, STAGES
from audio_transcriber.tui.formatting import _amplitude_line, _fmt_duration, _progress_bar
from audio_transcriber.utils.playback import (
    SILENCE_RMS_THRESHOLD,
    PlaybackHandle,
    amplitude_envelope,
    read_duration,
    start_playback,
)


class MediaDirectoryTree(DirectoryTree):
    """Дерево файлов, показывающее только папки и аудио/видеофайлы."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        for path in paths:
            try:
                if path.is_dir() or path.suffix.lower() in MEDIA_EXTENSIONS:
                    yield path
            except OSError:
                continue


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
