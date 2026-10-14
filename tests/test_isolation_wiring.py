"""Проверка подключения изоляции к конвейеру (#143)."""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization.base import SpeakerDiarizer
from audio_transcriber.domain.enums import AsrBackend, Device
from audio_transcriber.isolation import SubprocessPyannoteDiarizer, SubprocessWhisperRecognizer
from audio_transcriber.pipeline import _build_single_recognizer
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer


def test_build_recognizer_uses_subprocess_when_isolation_enabled(audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        asr_backend=AsrBackend.FASTER_WHISPER,
        engine_isolation="subprocess",
        model_name="small",
        word_timestamps=False,
    )

    recognizer = _build_single_recognizer(config, config.asr_backend, Device.CPU)

    assert isinstance(recognizer, SubprocessWhisperRecognizer)
    assert recognizer._model_name == "small"
    assert recognizer._word_timestamps is False


def test_build_recognizer_keeps_in_process_when_isolation_off(audio_file: Path) -> None:
    config = AppConfig(
        input_file=audio_file,
        asr_backend=AsrBackend.FASTER_WHISPER,
        engine_isolation="off",
    )

    recognizer = _build_single_recognizer(config, config.asr_backend, Device.CPU)

    assert isinstance(recognizer, WhisperSpeechRecognizer)
    assert not isinstance(recognizer, SubprocessWhisperRecognizer)


def test_subprocess_wrappers_match_protocols() -> None:
    recognizer = SubprocessWhisperRecognizer("small", Device.CPU)
    diarizer = SubprocessPyannoteDiarizer(Device.CPU)

    assert isinstance(recognizer, SpeechRecognizer)
    assert isinstance(diarizer, SpeakerDiarizer)
