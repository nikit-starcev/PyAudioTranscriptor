"""Интеграционный тест полного конвейера на реальном аудиофайле.

Использует эталонную запись faster-whisper/OpenAI Whisper ``tests_jfk.flac``
(~11 секунд, один говорящий) и лёгкую модель ``tiny``, чтобы тест оставался
быстрым. Диаризация подменена заглушкой с одним говорящим, поскольку
готовые модели pyannote.audio требуют отдельного токена доступа Hugging
Face — это не влияет на проверку связки распознавание -> объединение ->
экспорт.

Не выполняется в обычном прогоне ``pytest`` (см. ``addopts`` в
``pyproject.toml``). Запуск: ``uv run pytest -m integration``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer

pytestmark = pytest.mark.integration


class SingleSpeakerDiarizer:
    """Заглушка диаризации: во всём файле говорит один человек."""

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerSegment]:
        return [SpeakerSegment(start=0.0, end=15.0, speaker_id="SPEAKER_00")]


def test_full_pipeline_on_real_audio(jfk_audio_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = AppConfig(
        input_file=jfk_audio_file,
        output_dir=output_dir,
        model_name="tiny",
        language="en",
        export_formats=(ExportFormat.TXT, ExportFormat.JSON, ExportFormat.SRT, ExportFormat.DOCX),
        speaker_names={"SPEAKER_00": "JFK"},
    )

    result = run_pipeline(
        config,
        device=Device.CPU,
        recognizer=WhisperSpeechRecognizer("tiny", Device.CPU),
        diarizer=SingleSpeakerDiarizer(),
        merger=OverlapSegmentMerger(),
    )

    assert result.entries
    assert any("country" in entry.text.lower() for entry in result.entries)
    assert all(entry.speaker.display_name == "JFK" for entry in result.entries)

    stem = jfk_audio_file.stem
    for export_format in config.export_formats:
        exported_path = output_dir / f"{stem}.{export_format.value}"
        assert exported_path.exists()
        assert exported_path.stat().st_size > 0
