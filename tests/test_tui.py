"""Тесты TUI: чтение ``config.env``, сборка ``AppConfig`` и жизненный цикл LLM.

Запускаются через headless ``App.run_test`` (Textual не требует реального
терминала). Реальных моделей и сети нет.
"""

from __future__ import annotations

import asyncio
import queue as queue_module
from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.storage.glossary_db import Source
from audio_transcriber.tui import app as tui_app
from audio_transcriber.tui import screens as tui_screens


@pytest.fixture(autouse=True)
def tui_notify_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Перехватывает десктоп-уведомления TUI, чтобы тесты не дёргали notify-send."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tui_app, "notify", lambda title, message: calls.append((title, message)) or True
    )
    return calls


def _defaults(output_dir: Path, **overrides: str) -> dict[str, str]:
    values = {
        "LANGUAGE": "en",
        "FORMATS": "txt,docx,json,srt",
        "OUTPUT_DIR": str(output_dir),
        "DEVICE": "cuda",
        "ASR_BACKEND": "whisper-cpp",
        "MODEL": "medium",
        "WHISPER_CPP_MODEL": "/models/ggml.bin",
        "WHISPER_CPP_BINARY": "/bin/whisper-cli",
        "WHISPER_CPP_LIB_PATH": "/lib/whisper",
        "WHISPER_CPP_THREADS": "8",
        "INITIAL_PROMPT": "привет",
        "HOTWORDS": "ОИБ",
        "LLM_ENABLED": "true",
        "LLM_MODEL": "/models/llm.gguf",
        "LLM_BINARY": "/bin/llama-server",
        "LLM_LIB_PATH": "/lib/llama",
        "LLM_GPU": "false",
        "LLM_CONTEXT": "2048",
        "LLM_SUGGEST_TERMS": "true",
        "LLM_EXTRACT_NAMES": "false",
        "VERBOSE": "true",
        "CORRECTION_MIN_WORD_LENGTH": "7",
        "CORRECTION_MIN_SIMILARITY": "0.9",
        "CORRECTION_MAX_CANDIDATES": "5000",
    }
    values.update(overrides)
    return values


def test_build_config_reads_config_env_defaults(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: _defaults(tmp_path / "out"))

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.language == "en"
    assert config.device is Device.CUDA
    assert config.asr_backend is AsrBackend.WHISPER_CPP
    assert config.model_name == "medium"
    assert config.export_formats == (
        ExportFormat.TXT,
        ExportFormat.DOCX,
        ExportFormat.JSON,
        ExportFormat.SRT,
    )
    assert config.whisper_cpp_model == Path("/models/ggml.bin")
    assert config.whisper_cpp_threads == 8
    assert config.initial_prompt == "привет"
    assert config.hotwords == "ОИБ"
    assert config.llm_gpu is False
    assert config.llm_context_size == 2048
    assert config.llm_suggest_terms is True
    assert config.llm_extract_names is False
    assert config.verbose is True
    assert config.correction_min_word_length == 7
    assert config.correction_min_similarity == 0.9
    assert config.correction_max_candidates == 5000


def test_build_config_uses_safe_defaults_without_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.asr_backend is AsrBackend.FASTER_WHISPER
    assert config.device is Device.AUTO
    assert config.export_formats == (ExportFormat.TXT,)
    assert config.llm_enabled is False
    assert config.llm_extract_names is False


def test_queue_controller_reuses_row_without_duplicates() -> None:
    controller = tui_app.QueueController()
    first = Path("/media/first.mp3")
    second = Path("/media/second.mp3")

    controller.add(first)
    controller.add(second)
    controller.mark(first, "done")
    controller.add(first)  # повторное добавление после обработки

    assert controller.rows == [first, second]  # дублирующей строки нет
    assert controller.status[first] == "pending"
    assert controller.pending.qsize() == 3
    assert controller.first() == first


def test_queue_controller_waiting_and_empty() -> None:
    controller = tui_app.QueueController()
    assert controller.is_empty()

    path = Path("/media/call.wav")
    controller.add(path)
    assert not controller.is_empty()
    assert controller.is_waiting(path)

    controller.mark(path, "running")
    assert controller.is_waiting(path)

    controller.mark(path, "done")
    assert not controller.is_waiting(path)


def test_build_config_diarization_enabled_by_default(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).diarization_enabled is True


def test_build_config_diarization_can_be_disabled_via_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["DIARIZATION_ENABLED"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).diarization_enabled is False


def test_build_config_cleaning_enabled_by_default(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).clean_artifacts is True


def test_build_config_cleaning_can_be_disabled_via_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["CLEAN_ARTIFACTS"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).clean_artifacts is False


def test_build_config_quality_defaults(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.collapse_repeats is True
    assert config.normalize_text is True
    assert config.mark_overlap is True
    assert config.repeat_min_words == 2
    assert config.low_confidence_threshold == pytest.approx(-1.0)


def test_build_config_quality_toggles_from_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults.update(
        {
            "COLLAPSE_REPEATS": "false",
            "NORMALIZE_TEXT": "false",
            "MARK_OVERLAP": "false",
            "LOW_CONFIDENCE_THRESHOLD": "-3.5",
            "REPEAT_MIN_WORDS": "4",
            "REPEAT_SIMILARITY": "0.75",
        }
    )
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.collapse_repeats is False
    assert config.normalize_text is False
    assert config.mark_overlap is False
    assert config.repeat_min_words == 4
    assert config.repeat_similarity == pytest.approx(0.75)
    assert config.low_confidence_threshold == pytest.approx(-3.5)


def test_populate_results_renders_marks_column(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio_transcriber.domain.models import TranscriptEntry

    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app._populate_results(
                TranscriptionResult(
                    source_path=audio_file,
                    language="ru",
                    duration=2.0,
                    entries=[
                        TranscriptEntry(start=0.0, end=1.0, text="плохо", avg_logprob=-2.0),
                        TranscriptEntry(start=1.0, end=2.0, text="спор", overlap=True),
                    ],
                    speakers=[],
                    low_confidence_threshold=-1.0,
                )
            )
            table = app.query_one("#results", tui_app.DataTable)
            return table.row_count, len(table.columns), str(table.get_row_at(0))

    rows, columns, first_row = asyncio.run(_run())

    assert rows == 2
    assert columns == 4
    assert "⚠" in first_row


def test_populate_results_shows_all_speakers_and_speaker_uncertainty(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio_transcriber.domain.models import Speaker, TranscriptEntry

    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    anya = Speaker(id="SPEAKER_00", display_name="Аня")
    boris = Speaker(id="SPEAKER_01", display_name="Боря")

    async def _run() -> str:
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app._populate_results(
                TranscriptionResult(
                    source_path=audio_file,
                    language="ru",
                    duration=1.0,
                    entries=[
                        TranscriptEntry(
                            start=0.0,
                            end=1.0,
                            text="хором",
                            speaker=anya,
                            overlap=True,
                            extra_speakers=[boris],
                            speaker_confidence=0.2,
                        )
                    ],
                    speakers=[anya, boris],
                    low_confidence_threshold=-1.0,
                )
            )
            table = app.query_one("#results", tui_app.DataTable)
            return str(table.get_row_at(0))

    row = asyncio.run(_run())

    assert "Аня + Боря" in row  # основной и доп. говорящий вместе
    assert "?" in row  # низкая уверенность привязки говорящего


def test_stages_start_with_denoise() -> None:
    # Шумоподавление — первый этап конвейера, и он виден в списке стадий TUI.
    assert tui_app.STAGES[0] == ("denoise", "Шумоподавление")


def test_build_config_denoise_enabled_by_default(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).denoise is True


def test_build_config_denoise_can_be_disabled_via_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["DENOISE"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).denoise is False


def test_queue_does_not_duplicate_row_after_completion(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app._selected_file = audio_file
            app._add_selected_to_queue()
            app._queue_status[audio_file] = "done"  # файл обработан
            app._add_selected_to_queue()  # повторное добавление
            return list(app._queue_rows), app._pending.qsize()

    rows, queued = asyncio.run(_run())

    assert rows == [audio_file]
    assert queued == 2


def test_tui_reuses_and_closes_one_llm_client_per_queue(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """На всю очередь создаётся один клиент; после неё он гарантированно закрыт."""
    created: list[object] = []

    class _FakeClient:
        def __init__(self) -> None:
            self.closed = 0

        def chat(self, messages: list[dict[str, str]]) -> str:
            return '{"participants": [], "corrections": []}'

        def close(self) -> None:
            self.closed += 1

    def fake_create(**kwargs):
        client = _FakeClient()
        created.append(client)
        return client

    passed_clients: list[object] = []

    def fake_pipeline(config, **kwargs):
        passed_clients.append(kwargs.get("llm_client"))
        return TranscriptionResult(
            source_path=config.input_file,
            language="ru",
            duration=1.0,
            entries=[],
            speakers=[],
        )

    monkeypatch.setattr(tui_app, "create_llm_client", fake_create)
    monkeypatch.setattr(tui_app, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        tui_app,
        "_load_env_defaults",
        lambda: _defaults(tmp_path / "out"),
    )

    class _FakeQueue:
        def __init__(self, items: list[Path]) -> None:
            self._items = list(items)

        def get(self, timeout: float | None = None) -> Path:
            if self._items:
                return self._items.pop(0)
            raise queue_module.Empty

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            config = app._build_config(audio_file)
            app._pending = _FakeQueue([audio_file, audio_file])  # type: ignore[assignment]
            worker = tui_app.TranscriberApp.__dict__["_run_queue_worker"].__wrapped__
            worker(app, config)

    asyncio.run(_run())

    assert len(created) == 1
    assert created[0].closed == 1  # type: ignore[attr-defined]
    assert len(passed_clients) == 2
    assert all(client is created[0] for client in passed_clients)


# --- Пакет 2 «LLM»: резюме и доп. инструкции -------------------------------


def test_build_config_llm_summary_defaults_to_true(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).llm_summary is True


def test_build_config_llm_summary_from_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["LLM_SUMMARY"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).llm_summary is False


def test_build_config_llm_prompt_extra_and_file(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt_file = tmp_path / "extra.txt"
    prompt_file.write_text("Инструкция", encoding="utf-8")
    defaults = _defaults(tmp_path / "out")
    defaults["LLM_PROMPT_EXTRA"] = "Пиши кратко"
    defaults["LLM_PROMPT_FILE"] = str(prompt_file)
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.llm_prompt_extra == "Пиши кратко"
    assert config.llm_prompt_file == prompt_file


# --- Пакет 3 «надёжность»: кэш результатов ---------------------------------


def test_build_config_cache_enabled_by_default(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).use_cache is True


def test_build_config_cache_toggle_and_dir_from_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "custom-cache"
    defaults = _defaults(tmp_path / "out")
    defaults["USE_CACHE"] = "false"
    defaults["CACHE_DIR"] = str(cache_dir)
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.use_cache is False
    assert config.cache_dir == cache_dir


# --- Пакет 4 «интерфейс»: поиск по стенограмме ------------------------------


def _search_result(audio_file: Path) -> TranscriptionResult:
    from audio_transcriber.domain.models import Speaker, TranscriptEntry

    ivan = Speaker(id="SPEAKER_00", display_name="Иван")
    maria = Speaker(id="SPEAKER_01", display_name="Мария")
    return TranscriptionResult(
        source_path=audio_file,
        language="ru",
        duration=3.0,
        entries=[
            TranscriptEntry(start=0.0, end=1.0, text="Привет, Иван", speaker=ivan),
            TranscriptEntry(start=1.0, end=2.0, text="привет мир", speaker=maria),
            TranscriptEntry(start=2.0, end=3.0, text="пока", speaker=ivan),
        ],
        speakers=[ivan, maria],
        low_confidence_threshold=-1.0,
    )


def test_search_filters_rows_and_shows_count(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> tuple[int, str]:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._populate_results(_search_result(audio_file))
            app.query_one("#search", tui_app.Input).value = "привет"
            await pilot.pause()
            table = app.query_one("#results", tui_app.DataTable)
            status = app.query_one("#search_status", tui_app.Static)
            return table.row_count, str(status.render())

    rows, status = asyncio.run(_run())

    assert rows == 2
    assert status == "Совпадений: 2 / 3"


def test_search_is_case_insensitive_and_matches_speaker(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> tuple[int, int]:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._populate_results(_search_result(audio_file))
            search = app.query_one("#search", tui_app.Input)
            search.value = "МАРИЯ"
            await pilot.pause()
            table = app.query_one("#results", tui_app.DataTable)
            return table.row_count, len(app._result_rows)

    rows, total = asyncio.run(_run())

    assert rows == 1  # нашли по говорящему без учёта регистра
    assert total == 3


def test_search_all_words_must_match(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> int:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._populate_results(_search_result(audio_file))
            app.query_one("#search", tui_app.Input).value = "привет иван"
            await pilot.pause()
            return app.query_one("#results", tui_app.DataTable).row_count

    # обе подстроки («привет» в тексте, «иван» в говорящем) — одна реплика
    assert asyncio.run(_run()) == 1


def test_search_reset_restores_all_rows(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> tuple[str, int, str]:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._populate_results(_search_result(audio_file))
            search = app.query_one("#search", tui_app.Input)
            search.focus()
            search.value = "привет"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            status = str(app.query_one("#search_status", tui_app.Static).render())
            return search.value, app.query_one("#results", tui_app.DataTable).row_count, status

    value, rows, status = asyncio.run(_run())

    assert value == ""
    assert rows == 3
    assert status == "Реплик: 3"


def test_slash_focuses_search(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> str | None:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app.query_one("#tree", tui_app.MediaDirectoryTree).focus()
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            focused = app.focused
            return focused.id if focused is not None else None

    assert asyncio.run(_run()) == "search"


# --- Пакет 4 «интерфейс»: уведомления о завершении --------------------------


def test_build_config_notifications_enabled_by_default(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).notifications is True


def test_build_config_notifications_toggle_from_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["NOTIFICATIONS"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).notifications is False


def test_queue_done_notifies_success(
    tmp_path: Path,
    audio_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tui_notify_calls: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> None:
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app._notify_enabled = True
            app._files_done = 2
            app._files_failed = 0
            app.on_queue_done(tui_app.QueueDone())

    asyncio.run(_run())

    assert len(tui_notify_calls) == 1
    assert tui_notify_calls[0][0] == "Транскрибация завершена"
    assert "2" in tui_notify_calls[0][1]


def test_queue_done_notifies_errors(
    tmp_path: Path,
    audio_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tui_notify_calls: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> None:
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app._notify_enabled = True
            app._files_done = 1
            app._files_failed = 2
            app.on_queue_done(tui_app.QueueDone())

    asyncio.run(_run())

    assert len(tui_notify_calls) == 1
    assert tui_notify_calls[0][0] == "Транскрибация завершена с ошибками"
    assert "ошибок: 2" in tui_notify_calls[0][1]


def test_queue_done_does_not_notify_when_disabled(
    tmp_path: Path,
    audio_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tui_notify_calls: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> None:
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app._notify_enabled = False
            app._files_done = 3
            app.on_queue_done(tui_app.QueueDone())

    asyncio.run(_run())

    assert tui_notify_calls == []


# --- Пакет 5 «enrollment-диаризация»: образцы голоса ------------------------


def test_build_config_speaker_references_and_threshold_from_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "ivan.wav"
    reference.write_bytes(b"")
    defaults = _defaults(tmp_path / "out")
    defaults["SPEAKER_REFERENCES"] = f"Иван={reference}"
    defaults["ENROLLMENT_MIN_SIMILARITY"] = "0.65"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.speaker_references == {"Иван": (reference,)}
    assert config.enrollment_min_similarity == pytest.approx(0.65)


def test_build_config_speaker_references_default_to_none(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.speaker_references == {}
    assert config.enrollment_min_similarity == pytest.approx(0.6)


# --- Пакет 6 «таймлайн говорящих»: настройка TUI ----------------------------


def test_build_config_timeline_enabled_by_default(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).timeline is True


def test_build_config_timeline_can_be_disabled_via_env(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["TIMELINE"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test():
            return app._build_config(audio_file)

    assert asyncio.run(_run()).timeline is False


# --- Пакет 7 «правка говорящих»: чистые операции ----------------------------


def test_rename_speaker_updates_speakers_and_entries(audio_file: Path) -> None:
    original = _search_result(audio_file)

    renamed = tui_app.rename_speaker(original, "SPEAKER_00", "Пётр")

    assert any(
        speaker.id == "SPEAKER_00" and speaker.display_name == "Пётр"
        for speaker in renamed.speakers
    )
    assert all(
        entry.speaker is not None and entry.speaker.display_name == "Пётр"
        for entry in renamed.entries
        if entry.speaker is not None and entry.speaker.id == "SPEAKER_00"
    )
    # Исходный результат не мутирован.
    assert any(speaker.display_name == "Иван" for speaker in original.speakers)


def test_merge_speakers_reassigns_entries_and_drops_source(audio_file: Path) -> None:
    original = _search_result(audio_file)

    merged = tui_app.merge_speakers(original, "SPEAKER_00", "SPEAKER_01")

    assert [speaker.id for speaker in merged.speakers] == ["SPEAKER_01"]
    assert all(
        entry.speaker is not None and entry.speaker.id == "SPEAKER_01"
        for entry in merged.entries
        if entry.speaker is not None
    )


# --- Пакет 7 «правка говорящих»: модальный редактор --------------------------


def test_speaker_editor_rename_updates_entries(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> TranscriptionResult:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.query_one("#speakers_table", tui_app.DataTable).move_cursor(row=0)
            assert screen.rename_selected("Пётр") is True
            return screen.edited_result

    result = asyncio.run(_run())

    assert any(speaker.display_name == "Пётр" for speaker in result.speakers)
    assert all(
        entry.speaker is not None and entry.speaker.display_name == "Пётр"
        for entry in result.entries
        if entry.speaker is not None and entry.speaker.id == "SPEAKER_00"
    )


def test_speaker_editor_merge_reassigns_entries(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> TranscriptionResult:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.query_one("#speakers_table", tui_app.DataTable).move_cursor(row=0)
            assert screen.merge_selected_into("SPEAKER_01") is True
            return screen.edited_result

    result = asyncio.run(_run())

    assert [speaker.id for speaker in result.speakers] == ["SPEAKER_01"]
    assert all(
        entry.speaker is not None and entry.speaker.id == "SPEAKER_01"
        for entry in result.entries
        if entry.speaker is not None
    )


def test_speaker_editor_save_marks_protocol_stale(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "out"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(output_dir)})

    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT, ExportFormat.JSON),
        timeline=False,
        protocol_auto=False,
    )

    async def _run() -> str:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = config
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.rename_selected("Иван II")
            screen.action_save()
            await pilot.pause()
            return str(app.query_one("#status", tui_app.Static).render())

    status = asyncio.run(_run())

    # Сохранение имён не выгружает файлы автоматически — только помечает
    # протокол устаревшим.
    assert "имена изменены" in status
    assert not (output_dir / f"{audio_file.stem}.txt").exists()


def test_speaker_editor_cancel_keeps_original_result(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    original = _search_result(audio_file)

    async def _run() -> TranscriptionResult | None:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = original
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.rename_selected("Пётр")
            screen.action_cancel()
            await pilot.pause()
            return app._last_result

    result = asyncio.run(_run())

    assert result is original
    assert [speaker.display_name for speaker in result.speakers] == ["Иван", "Мария"]


def test_e_key_opens_speaker_editor(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> bool:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            await pilot.press("e")
            await pilot.pause()
            return isinstance(app.screen, tui_app.SpeakerEditorScreen)

    assert asyncio.run(_run()) is True


def test_edit_speakers_without_result_does_not_open_editor(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> bool:
        app = tui_app.TranscriberApp()
        async with app.run_test():
            app.action_edit_speakers()
            return isinstance(app.screen, tui_app.SpeakerEditorScreen)

    assert asyncio.run(_run()) is False


# --- Блок 1 «проигрывание»: панель, колонки образцов ------------------------


def test_speaker_editor_has_player_panel_and_sample_columns(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            table = screen.query_one("#speakers_table", tui_app.DataTable)
            labels = [str(table.columns[key].label) for key in table.columns]
            panel = screen.query_one("#player", tui_app.PlayerPanel)
            return labels, table.get_row_at(0), panel

    labels, first_row, panel = asyncio.run(_run())

    assert "Образец" in labels
    assert "Длит." in labels
    assert len(labels) == 5
    assert first_row[3] == "—"  # образца нет — явно видно
    assert isinstance(panel, tui_app.PlayerPanel)


def test_player_panel_render_helpers() -> None:
    bar = tui_app._progress_bar(0.5, 11)
    assert bar == "─────●─────"
    assert tui_app._amplitude_line([0.0, 1.0]) == "▁█"
    line = tui_app._amplitude_line([0.5] * 50, 50)
    assert len(line) == 50


# --- Блок 2 «библиотека голосов»: открытие, удаление ------------------------


def _make_voices(tmp_path: Path) -> Path:
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "Иван.wav").write_bytes(b"audio")
    (voices / "Мария.wav").write_bytes(b"audio")
    return voices


def test_v_opens_voices_library_from_editor(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> bool:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.query_one("#speakers_table", tui_app.DataTable).focus()
            await pilot.pause()
            await pilot.press("v")
            await pilot.pause()
            return isinstance(app.screen, tui_app.VoicesLibraryScreen)

    assert asyncio.run(_run()) is True


def test_v_opens_voices_library_from_main(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> bool:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app.query_one("#tree", tui_app.MediaDirectoryTree).focus()
            await pilot.pause()
            await pilot.press("v")
            await pilot.pause()
            return isinstance(app.screen, tui_app.VoicesLibraryScreen)

    assert asyncio.run(_run()) is True


def test_voices_library_lists_files_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    voices = _make_voices(tmp_path)

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app.push_screen(tui_app.VoicesLibraryScreen(voices_dir=voices))
            await pilot.pause()
            library = app.screen
            table = library.query_one("#voices_table", tui_app.DataTable)
            path_label = str(library.query_one("#voices_path", tui_app.Static).render())
            return table.row_count, path_label

    rows, path_label = asyncio.run(_run())

    assert rows == 2
    assert str(voices) in path_label


def test_voices_library_missing_dir_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app.push_screen(
                tui_app.VoicesLibraryScreen(voices_dir=tmp_path / "absent")
            )
            await pilot.pause()
            library = app.screen
            status = str(library.query_one("#voices_status", tui_app.Static).render())
            return library.query_one("#voices_table", tui_app.DataTable).row_count, status

    rows, status = asyncio.run(_run())

    assert rows == 0
    assert "не найден" in status


def test_voices_library_delete_with_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    voices = _make_voices(tmp_path)
    target = voices / "Иван.wav"

    async def _run() -> tuple[bool, bool]:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app.push_screen(tui_app.VoicesLibraryScreen(voices_dir=voices))
            await pilot.pause()
            library = app.screen
            library.query_one("#voices_table", tui_app.DataTable).move_cursor(row=0)
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            confirm_shown = isinstance(app.screen, tui_app.ConfirmDeleteScreen)
            app.screen.action_confirm()
            await pilot.pause()
            return confirm_shown, target.exists()

    confirm_shown, exists = asyncio.run(_run())

    assert confirm_shown is True
    assert exists is False


def test_voices_library_delete_can_be_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    voices = _make_voices(tmp_path)
    target = voices / "Иван.wav"

    async def _run() -> bool:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app.push_screen(tui_app.VoicesLibraryScreen(voices_dir=voices))
            await pilot.pause()
            library = app.screen
            library.query_one("#voices_table", tui_app.DataTable).move_cursor(row=0)
            await pilot.pause()
            library.action_delete_voice()
            await pilot.pause()
            app.screen.action_cancel()
            await pilot.pause()
            return target.exists()

    assert asyncio.run(_run()) is True


# --- Блок 3 «применение имён без перезапуска» -------------------------------


def test_apply_names_renames_speakers_with_enrollment(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"audio")
    captured: dict[str, object] = {}

    def fake_enroll(**kwargs):
        captured.update(kwargs)
        return tui_app.EnrollmentOutcome(
            mapping={"SPEAKER_00": "Пётр"}, best_candidates={}, speaker_count=1
        )

    monkeypatch.setattr(tui_screens, "enroll_speakers", fake_enroll)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        voices_dir=tmp_path / "no_voices",
        speaker_references={"Пётр": (reference,)},
        timeline=False,
    )

    async def _run() -> TranscriptionResult:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = config
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.action_apply_names()
            for _ in range(50):
                await asyncio.sleep(0.02)
                await pilot.pause()
                if any(
                    speaker.display_name == "Пётр"
                    for speaker in screen.edited_result.speakers
                ):
                    break
            return screen.edited_result

    result = asyncio.run(_run())

    assert any(speaker.display_name == "Пётр" for speaker in result.speakers)
    assert all(
        entry.speaker is not None and entry.speaker.display_name == "Пётр"
        for entry in result.entries
        if entry.speaker is not None and entry.speaker.id == "SPEAKER_00"
    )
    assert "Пётр" in captured.get("references", {})


def test_apply_names_without_samples_reports_status(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    async def _run() -> str:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = AppConfig(
                input_file=audio_file,
                output_dir=tmp_path / "out",
                voices_dir=tmp_path / "no_voices",
                timeline=False,
            )
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.action_apply_names()
            await pilot.pause()
            return str(screen.query_one("#editor_status", tui_app.Static).render())

    assert "Нет образцов" in asyncio.run(_run())


def test_apply_names_reports_counts_and_best_unmatched(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"audio")

    def fake_enroll(**kwargs):
        return tui_app.EnrollmentOutcome(
            mapping={"SPEAKER_00": "Пётр"},
            best_candidates={"SPEAKER_01": ("Анна", 0.483)},
            speaker_count=2,
        )

    monkeypatch.setattr(tui_screens, "enroll_speakers", fake_enroll)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        voices_dir=tmp_path / "no_voices",
        speaker_references={"Пётр": (reference,)},
        timeline=False,
    )

    async def _run() -> str:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = config
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.action_apply_names()
            for _ in range(50):
                await asyncio.sleep(0.02)
                await pilot.pause()
                if any(
                    speaker.display_name == "Пётр"
                    for speaker in screen.edited_result.speakers
                ):
                    break
            return str(screen.query_one("#editor_status", tui_app.Static).render())

    status = asyncio.run(_run())

    assert "1/2" in status
    assert "Пётр" in status
    assert "Анна" in status
    assert "0.48" in status



def test_speaker_editor_save_renames_sample_files(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )

    output_dir = tmp_path / "out"
    speakers_dir = output_dir / f"{audio_file.stem}.speakers"
    speakers_dir.mkdir(parents=True)
    ivan = speakers_dir / "Иван.wav"
    maria = speakers_dir / "Мария.wav"
    ivan.write_bytes(b"audio")
    maria.write_bytes(b"audio")
    config = AppConfig(
        input_file=audio_file,
        output_dir=output_dir,
        export_formats=(ExportFormat.TXT,),
        timeline=False,
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = config
            app._last_samples = {"SPEAKER_00": ivan, "SPEAKER_01": maria}
            app.action_edit_speakers()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, tui_app.SpeakerEditorScreen)
            screen.rename_selected("Пётр")
            screen.action_save()
            await pilot.pause()
            return app._last_samples.get("SPEAKER_00")

    sample = asyncio.run(_run())

    assert (speakers_dir / "Пётр.wav").is_file()
    assert sample is not None and sample.name == "Пётр.wav"


def test_new_bindings_present() -> None:
    editor_keys = {
        binding[0]
        for binding in tui_app.SpeakerEditorScreen.BINDINGS
        if isinstance(binding, tuple)
    }
    assert {"p", "v", "a"} <= editor_keys

    library_keys = {
        binding[0]
        for binding in tui_app.VoicesLibraryScreen.BINDINGS
        if isinstance(binding, tuple)
    }
    assert {"p", "d", "delete", "r", "escape"} <= library_keys

    app_keys = {
        binding[0]
        for binding in tui_app.TranscriberApp.BINDINGS
        if isinstance(binding, tuple)
    }
    assert {"e", "v", "ctrl+p"} <= app_keys


# --- Блок «галочки глоссария» и «протокол по кнопке» ------------------------


def _install_fake_glossary_db(
    monkeypatch: pytest.MonkeyPatch,
    sources: list[Source],
    counts: dict[str, int] | None = None,
) -> dict[str, list]:
    """Подменяет ``GlossaryDB`` в TUI и возвращает журнал вызовов."""
    recorded: dict[str, list] = {"enabled": [], "opened": []}
    resolved_counts = dict(counts or {})

    class _FakeGlossaryDB:
        def __init__(self, path: Path) -> None:
            self.path = path
            recorded["opened"].append(Path(path))

        def __enter__(self) -> _FakeGlossaryDB:
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

        def list_sources(self) -> list[Source]:
            return list(sources)

        def entry_counts(self) -> dict[str, int]:
            return dict(resolved_counts)

        def set_source_enabled(self, name: str, enabled: bool) -> bool:
            recorded["enabled"].append((name, enabled))
            return True

    monkeypatch.setattr(tui_app, "GlossaryDB", _FakeGlossaryDB)
    return recorded


def _glossary_source(index: int, name: str, *, enabled: bool, kind: str = "txt") -> Source:
    return Source(
        id=index,
        name=name,
        kind=kind,
        path=f"{name}.{kind}",
        enabled=enabled,
        imported_at=None,
    )


def test_tui_has_glossary_checkbox_and_db_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    _install_fake_glossary_db(monkeypatch, [])

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            checkbox = app.query_one("#glossary_enabled", tui_app.Checkbox)
            db_input = app.query_one("#glossary_db", tui_app.Input)
            return checkbox.value, db_input.placeholder

    value, placeholder = asyncio.run(_run())

    assert value is True
    assert placeholder == "glossary.db"


def test_tui_glossary_lists_sources_with_kind_and_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    _install_fake_glossary_db(
        monkeypatch,
        [
            _glossary_source(1, "ТЗ", enabled=True, kind="txt"),
            _glossary_source(2, "manual", enabled=False, kind="manual"),
        ],
        {"ТЗ": 7, "manual": 2},
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            boxes = list(app.query("#glossary_sources Checkbox"))
            return [(box.value, str(box.label)) for box in boxes]

    rows = asyncio.run(_run())

    assert [enabled for enabled, _ in rows] == [True, False]
    assert "ТЗ" in rows[0][1] and "7" in rows[0][1] and "txt" in rows[0][1]
    assert "manual" in rows[1][1]


def test_tui_glossary_toggle_saves_source_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    recorded = _install_fake_glossary_db(
        monkeypatch, [_glossary_source(1, "ТЗ", enabled=True)]
    )

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#glossary_sources Checkbox", tui_app.Checkbox)
            box.value = False
            await pilot.pause()
            return recorded["enabled"]

    enabled_calls = asyncio.run(_run())

    assert enabled_calls[-1] == ("ТЗ", False)


def test_tui_glossary_empty_db_shows_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    _install_fake_glossary_db(monkeypatch, [])

    async def _run() -> str:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            return str(app.query_one("#glossary_status", tui_app.Static).render())

    assert "пуст" in asyncio.run(_run())


def test_tui_build_config_reads_glossary_and_disables_auto_protocol(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = _defaults(tmp_path / "out")
    defaults["GLOSSARY_DB"] = str(tmp_path / "g.db")
    defaults["GLOSSARY_ENABLED"] = "false"
    monkeypatch.setattr(tui_app, "_load_env_defaults", lambda: defaults)
    _install_fake_glossary_db(monkeypatch, [])

    async def _run():
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            return app._build_config(audio_file)

    config = asyncio.run(_run())

    assert config.glossary_db == tmp_path / "g.db"
    assert config.glossary_enabled is False
    # TUI собирает протокол по кнопке, а не по итогам прогона.
    assert config.protocol_auto is False


def test_tui_protocol_action_builds_and_notifies(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    _install_fake_glossary_db(monkeypatch, [])
    built: dict[str, object] = {}

    def fake_generate(config, result, *, on_progress=None):
        built["config"] = config
        built["result"] = result
        return tui_app.ProtocolArtifacts(
            paths=(config.output_dir / f"{audio_file.stem}.txt",), summary="Резюме"
        )

    monkeypatch.setattr(tui_app, "generate_protocol", fake_generate)
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        timeline=False,
        protocol_auto=False,
    )

    async def _run() -> str:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = config
            await pilot.pause()
            app.action_generate_protocol()
            for _ in range(100):
                await asyncio.sleep(0.02)
                await pilot.pause()
                if not app._protocol_building and built:
                    break
            return str(app.query_one("#status", tui_app.Static).render())

    status = asyncio.run(_run())

    assert built["result"] is not None
    assert "сформирован" in status


def test_tui_protocol_stale_after_name_edits(
    tmp_path: Path, audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tui_app, "_load_env_defaults", lambda: {"OUTPUT_DIR": str(tmp_path / "out")}
    )
    _install_fake_glossary_db(monkeypatch, [])
    config = AppConfig(
        input_file=audio_file,
        output_dir=tmp_path / "out",
        timeline=False,
        protocol_auto=False,
    )

    async def _run() -> tuple[bool, str]:
        app = tui_app.TranscriberApp()
        async with app.run_test() as pilot:
            app._last_result = _search_result(audio_file)
            app._last_config = config
            await pilot.pause()
            app._on_speaker_editor_closed(app._last_result)
            await pilot.pause()
            status = str(app.query_one("#status", tui_app.Static).render())
            return app._protocol_stale, status

    stale, status = asyncio.run(_run())

    assert stale is True
    assert "имена изменены" in status
