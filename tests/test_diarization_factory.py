"""Тесты выбора движка диаризации (#62).

Режим ``auto`` предпочитает nemo-speech, когда его бинарник доступен, иначе
откатывается к pyannote. Явный выбор всегда соблюдается. Существующий путь
pyannote не должен ломаться.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization import factory
from audio_transcriber.diarization.nemo_speech_engine import NemoSpeechSpeakerDiarizer
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device


def _config(audio_file: Path, **overrides: object) -> AppConfig:
    return AppConfig(input_file=audio_file, **overrides)


def test_resolve_explicit_pyannote_ignores_nemo(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "binary_available", lambda _binary: True)
    config = _config(audio_file, diarization_engine="pyannote")

    assert factory.resolve_diarization_engine(config) == "pyannote"


def test_resolve_explicit_nemo_speech(audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "binary_available", lambda _binary: False)
    config = _config(audio_file, diarization_engine="nemo-speech")

    assert factory.resolve_diarization_engine(config) == "nemo-speech"


def test_resolve_auto_prefers_nemo_when_available(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "binary_available", lambda _binary: True)

    assert factory.resolve_diarization_engine(_config(audio_file)) == "nemo-speech"


def test_resolve_auto_falls_back_to_pyannote(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "binary_available", lambda _binary: False)

    assert factory.resolve_diarization_engine(_config(audio_file)) == "pyannote"


def test_create_diarizer_nemo_carries_settings(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        audio_file,
        diarization_engine="nemo-speech",
        nemo_speech_binary="/opt/nemo/nemo-speech",
        nemo_speech_lib_path="/opt/nemo/lib",
        nemo_speech_model="sortformer",
        nemo_speech_device="vulkan",
    )

    diarizer = factory.create_diarizer(config, Device.CPU)

    assert isinstance(diarizer, NemoSpeechSpeakerDiarizer)
    assert diarizer._binary == "/opt/nemo/nemo-speech"
    assert diarizer._lib_path == "/opt/nemo/lib"
    assert diarizer._model == "sortformer"
    assert diarizer._device == "vulkan"
    assert diarizer.supports_enrollment is False


def test_create_diarizer_pyannote_path_intact(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(audio_file, diarization_engine="pyannote")

    diarizer = factory.create_diarizer(config, Device.CPU)

    assert isinstance(diarizer, PyannoteSpeakerDiarizer)
