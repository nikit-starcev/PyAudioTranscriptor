"""Тесты TUI: чтение ``config.env``, сборка ``AppConfig`` и жизненный цикл LLM.

Запускаются через headless ``App.run_test`` (Textual не требует реального
терминала). Реальных моделей и сети нет.
"""

from __future__ import annotations

import asyncio
import queue as queue_module
from pathlib import Path

import pytest

from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.tui import app as tui_app


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
