"""Воркер изоляции тяжёлых движков (#143).

Запускается отдельным процессом тем же интерпретатором, что и основной:
``python -m audio_transcriber.isolation.worker --request <json> --result <json>``.
Загружает ровно один движок (faster-whisper для ASR либо pyannote для
диаризации), выполняет один проход и записывает результат в файл-результат.
По завершении процесс умирает, и весь его RSS (модель, torch/CTranslate2-арены)
возвращается ОС — включая Windows/macOS, где ``malloc_trim`` недоступен.

Прогресс пишется в stderr строками с маркером
:data:`~audio_transcriber.isolation.protocol.PROGRESS_PREFIX`; остальной stderr
считается логом и переиспользуется для диагностики ошибок.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from audio_transcriber.domain.enums import Device
from audio_transcriber.isolation.protocol import (
    KIND_ASR,
    KIND_DIARIZATION,
    WorkerRequest,
    asr_result,
    diarization_result,
    progress_line,
    request_from_json,
)
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.exceptions import EngineIsolationError

logger = logging.getLogger(__name__)


def _stderr_progress(event: ProgressEvent) -> None:
    """Пишет событие прогресса в stderr отдельной строкой с маркером."""
    try:
        sys.stderr.write(progress_line(event) + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def _build_recognizer(options: dict[str, Any], emit: ProgressCallback | None) -> Any:
    """Создаёт движок ASR по параметрам запроса (faster-whisper)."""
    from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer

    return WhisperSpeechRecognizer(
        str(options["model_name"]),
        Device(str(options.get("device", Device.AUTO.value))),
        initial_prompt=options.get("initial_prompt"),
        hotwords=options.get("hotwords"),
        on_progress=emit,
        word_timestamps=bool(options.get("word_timestamps", True)),
    )


def _build_diarizer(options: dict[str, Any], emit: ProgressCallback | None) -> Any:
    """Создаёт движок диаризации по параметрам запроса (pyannote)."""
    from audio_transcriber.config.defaults import DEFAULT_DIARIZATION_MIN_DURATION_OFF
    from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer

    local_model = options.get("local_model_path")
    kwargs: dict[str, Any] = {
        "hf_token": options.get("hf_token"),
        "local_model_path": Path(local_model) if local_model else None,
        "on_progress": emit,
        "min_duration_off": float(
            options.get("min_duration_off", DEFAULT_DIARIZATION_MIN_DURATION_OFF)
        ),
        "clustering_threshold": options.get("clustering_threshold"),
        "clustering_fb": options.get("clustering_fb"),
    }
    pipeline_name = options.get("pipeline_name")
    if pipeline_name:
        kwargs["pipeline_name"] = str(pipeline_name)
    return PyannoteSpeakerDiarizer(
        Device(str(options.get("device", Device.AUTO.value))),
        **kwargs,
    )


def _close(engine: Any) -> None:
    """Мягко освобождает движок воркера (процесс всё равно завершается)."""
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # noqa: BLE001 — освобождение не должно мешать результату
            logger.debug("Воркер: не удалось освободить движок: %s", exc)


def _run_asr(request: WorkerRequest, emit: ProgressCallback) -> dict[str, Any]:
    engine = _build_recognizer(request.options, emit)
    try:
        segments, language, duration = engine.transcribe(
            Path(request.audio_path), language=request.language
        )
    finally:
        _close(engine)
    return asr_result(segments, language, duration)


def _run_diarization(request: WorkerRequest, emit: ProgressCallback) -> dict[str, Any]:
    engine = _build_diarizer(request.options, emit)
    waveform = None
    if request.waveform_path:
        import numpy as np

        waveform = np.load(request.waveform_path)
    try:
        segments = engine.diarize(
            Path(request.audio_path),
            num_speakers=request.options.get("num_speakers"),
            min_speakers=request.options.get("min_speakers"),
            max_speakers=request.options.get("max_speakers"),
            waveform=waveform,
        )
        overlaps = list(engine.overlap_regions())
    finally:
        _close(engine)
    return diarization_result(segments, overlaps)


def execute(request: WorkerRequest, emit: ProgressCallback | None = None) -> dict[str, Any]:
    """Выполняет один запрос и возвращает payload для ответа воркера."""
    callback: ProgressCallback = emit or (lambda _event: None)
    if request.kind == KIND_ASR:
        return _run_asr(request, callback)
    if request.kind == KIND_DIARIZATION:
        return _run_diarization(request, callback)
    raise EngineIsolationError(f"Неизвестный вид воркера: {request.kind!r}")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    """Записывает результат воркера в файл (атомарно в пределах каталога)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Точка входа воркера: ``--request``/``--result`` (пути к JSON)."""
    parser = argparse.ArgumentParser(
        prog="audio_transcriber.isolation.worker",
        description="Воркер изоляции тяжёлых Python-движков (#143).",
    )
    parser.add_argument("--request", required=True, help="путь к JSON-запросу")
    parser.add_argument("--result", required=True, help="путь к JSON-результату")
    args = parser.parse_args(argv)

    result_path = Path(args.result)
    try:
        request = request_from_json(Path(args.request).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _write_result(result_path, {"ok": False, "error": f"Некорректный запрос: {exc}"})
        return 2

    try:
        payload = execute(request, _stderr_progress)
    except Exception as exc:
        logger.debug("Воркер завершился ошибкой", exc_info=True)
        _write_result(
            result_path,
            {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
        )
        return 1

    _write_result(result_path, {"ok": True, "kind": request.kind, "payload": payload})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
