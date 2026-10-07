"""Интеграционный тест распознавания через faster-whisper (реальная модель).

Запуск: ``pytest -m integration``. Модели пробуются от лёгких к тяжёлым; если
ни одна не загрузилась (нет сети и локального кэша), тест пропускается.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.domain.enums import Device
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer
from audio_transcriber.utils.exceptions import TranscriptionError

from ._helpers import (
    DURATION_TOLERANCE,
    SHORT_FIXTURE_SECONDS,
    assert_asr_result_sane,
)

pytestmark = pytest.mark.integration


def test_faster_whisper_transcribes_short_audio(
    short_speech_wav: Path, faster_whisper_models: list[str]
) -> None:
    """Реальная модель faster-whisper распознаёт фрагмент и даёт корректную длительность."""
    last_error: Exception | None = None
    for model_name in faster_whisper_models:
        recognizer = WhisperSpeechRecognizer(model_name, Device.CPU)
        try:
            segments, language, duration = recognizer.transcribe(
                short_speech_wav, language="en"
            )
        except TranscriptionError as exc:
            last_error = exc
            continue
        break
    else:
        pytest.skip(f"faster-whisper не запустился ни с одной моделью: {last_error}")

    assert language.lower().startswith("en")
    assert duration == pytest.approx(SHORT_FIXTURE_SECONDS, abs=DURATION_TOLERANCE)
    assert_asr_result_sane(segments, duration)
