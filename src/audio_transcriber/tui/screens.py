"""Модальные экраны TUI: правка говорящих и библиотека образцов голоса."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Literal

from textual import work
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Input, Select, Static

from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_VOICES_DIR,
)
from audio_transcriber.diarization.enrollment import EnrollmentOutcome, enroll_speakers
from audio_transcriber.diarization.voices import (
    collect_voice_library,
    delete_voice_sample,
    merge_references,
    save_reference_sample,
)
from audio_transcriber.domain.editing import merge_speakers, rename_speaker
from audio_transcriber.domain.models import SpeakerSegment, TranscriptionResult
from audio_transcriber.tui.formatting import _format_size
from audio_transcriber.tui.widgets import PlayerPanel


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
            target, quality = save_reference_sample(sample, self._voices_dir, name)
        except OSError as exc:
            self.app.notify(
                f"Не удалось сохранить образец: {exc}", severity="error", timeout=8
            )
            return
        warnings = quality.warnings() if quality is not None else []
        message = f"Сохранено в библиотеку: {target.name}"
        if warnings:
            message += " | качество: " + "; ".join(warnings)
        self._set_status(message)

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
