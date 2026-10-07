"""Интеграционный тест распознавания через whisper.cpp (реальный бинарник).

Запуск: ``pytest -m integration``. Если бинарник или модель whisper.cpp
недоступны, тест пропускается (см. фикстуру ``whisper_cpp_paths``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.transcription.whisper_cpp_engine import WhisperCppRecognizer

from ._helpers import (
    DURATION_TOLERANCE,
    SHORT_FIXTURE_SECONDS,
    WhisperCppPaths,
    assert_asr_result_sane,
)

pytestmark = pytest.mark.integration


def test_whisper_cpp_transcribes_short_audio(
    short_speech_wav: Path, whisper_cpp_paths: WhisperCppPaths
) -> None:
    """Реальный whisper-cli распознаёт короткий фрагмент и даёт корректную длительность."""
    recognizer = WhisperCppRecognizer(
        whisper_cpp_paths.model,
        binary=whisper_cpp_paths.binary,
        library_path=whisper_cpp_paths.library_path,
    )

    segments, language, duration = recognizer.transcribe(short_speech_wav, language="en")

    assert language.lower().startswith("en")
    assert duration == pytest.approx(SHORT_FIXTURE_SECONDS, abs=DURATION_TOLERANCE)
    assert_asr_result_sane(segments, duration)
