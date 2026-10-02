"""Тесты выбора движка диаризации (#62) и маршрутизации по числу говорящих (#64).

Явный ``pyannote``/``nemo-speech``/``hybrid`` соблюдается всегда. В режиме
``auto`` движок выбирается по числу говорящих: до cap — nemo-speech (если
доступен), выше — гибрид (оконный EEND + эмбеддинги, если доступен), иначе
pyannote (если доступен); неизвестное число → pyannote (безопасно).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.diarization import factory
from audio_transcriber.diarization.hybrid_engine import HybridSpeakerDiarizer
from audio_transcriber.diarization.nemo_speech_engine import NemoSpeechSpeakerDiarizer
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device


def _config(audio_file: Path, **overrides: object) -> AppConfig:
    return AppConfig(input_file=audio_file, **overrides)


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Управляемые доступности движков и оценщика (по умолчанию доступны все)."""
    state: dict[str, object] = {
        "nemo": True,
        "pyannote": True,
        "hybrid": True,
        "estimate": 3,
        "estimate_calls": 0,
    }
    monkeypatch.setattr(factory, "binary_available", lambda _binary: bool(state["nemo"]))
    monkeypatch.setattr(factory, "pyannote_available", lambda: bool(state["pyannote"]))
    monkeypatch.setattr(
        factory, "hybrid_available", lambda _config: bool(state["hybrid"])
    )

    def fake_estimate(*_args: object, **_kwargs: object) -> int | None:
        state["estimate_calls"] = int(state["estimate_calls"]) + 1
        count = state["estimate"]
        return count if count is None else int(count)

    monkeypatch.setattr(factory, "estimate_speaker_count", fake_estimate)
    return state


# --- явный выбор ------------------------------------------------------------


def test_resolve_explicit_pyannote_ignores_nemo(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch, engines: dict[str, object]
) -> None:
    config = _config(audio_file, diarization_engine="pyannote")

    assert factory.resolve_diarization_engine(config) == "pyannote"
    assert engines["estimate_calls"] == 0


def test_resolve_explicit_nemo_speech(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch, engines: dict[str, object]
) -> None:
    config = _config(audio_file, diarization_engine="nemo-speech")

    assert factory.resolve_diarization_engine(config) == "nemo-speech"
    assert engines["estimate_calls"] == 0


def test_resolve_explicit_hybrid(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch, engines: dict[str, object]
) -> None:
    config = _config(audio_file, diarization_engine="hybrid")

    assert factory.resolve_diarization_engine(config) == "hybrid"
    assert engines["estimate_calls"] == 0


# --- auto: по num_speakers / max_speakers -----------------------------------


def test_auto_num_speakers_within_cap_prefers_nemo(
    audio_file: Path, engines: dict[str, object]
) -> None:
    config = _config(audio_file, num_speakers=3)

    decision = factory.decide_diarization(config)

    assert decision.engine == "nemo-speech"
    assert decision.speaker_count == 3
    assert decision.estimated is False
    assert engines["estimate_calls"] == 0


def test_auto_num_speakers_above_cap_uses_hybrid(
    audio_file: Path, engines: dict[str, object]
) -> None:
    config = _config(audio_file, num_speakers=7)

    decision = factory.decide_diarization(config)

    assert decision.engine == "hybrid"
    assert decision.speaker_count == 7


def test_auto_num_speakers_above_cap_hybrid_disabled_uses_pyannote(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["hybrid"] = False
    config = _config(audio_file, num_speakers=7)

    decision = factory.decide_diarization(config)

    assert decision.engine == "pyannote"
    assert decision.speaker_count == 7


def test_auto_max_speakers_used_when_num_absent(
    audio_file: Path, engines: dict[str, object]
) -> None:
    config = _config(audio_file, max_speakers=4)

    decision = factory.decide_diarization(config)

    assert decision.engine == "nemo-speech"
    assert decision.speaker_count == 4
    assert engines["estimate_calls"] == 0


# --- auto: по оценке --------------------------------------------------------


def test_auto_estimated_within_cap_prefers_nemo(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["estimate"] = 2

    decision = factory.decide_diarization(_config(audio_file))

    assert decision.engine == "nemo-speech"
    assert decision.speaker_count == 2
    assert decision.estimated is True
    assert engines["estimate_calls"] == 1


def test_auto_estimated_above_cap_uses_hybrid(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["estimate"] = 6

    decision = factory.decide_diarization(_config(audio_file))

    assert decision.engine == "hybrid"
    assert decision.speaker_count == 6
    assert decision.estimated is True


def test_auto_estimated_above_cap_hybrid_missing_uses_pyannote(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["estimate"] = 6
    engines["hybrid"] = False

    decision = factory.decide_diarization(_config(audio_file))

    assert decision.engine == "pyannote"
    assert decision.speaker_count == 6


def test_auto_estimate_unknown_uses_pyannote(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["estimate"] = None

    decision = factory.decide_diarization(_config(audio_file))

    assert decision.engine == "pyannote"
    assert decision.speaker_count is None
    assert "неизвестно" in decision.reason


def test_auto_estimate_disabled_uses_pyannote(
    audio_file: Path, engines: dict[str, object]
) -> None:
    config = _config(audio_file, diarization_estimate_enabled=False)

    decision = factory.decide_diarization(config)

    assert decision.engine == "pyannote"
    assert engines["estimate_calls"] == 0


# --- auto: недоступность движков --------------------------------------------


def test_auto_within_cap_falls_back_to_pyannote_when_nemo_missing(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["nemo"] = False
    config = _config(audio_file, num_speakers=2)

    assert factory.resolve_diarization_engine(config) == "pyannote"


def test_auto_above_cap_falls_back_to_nemo_when_pyannote_and_hybrid_missing(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["pyannote"] = False
    engines["hybrid"] = False
    config = _config(audio_file, num_speakers=6)

    assert factory.resolve_diarization_engine(config) == "nemo-speech"


def test_auto_unknown_falls_back_to_nemo_when_pyannote_missing(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["estimate"] = None
    engines["pyannote"] = False

    assert factory.resolve_diarization_engine(_config(audio_file)) == "nemo-speech"


def test_auto_within_cap_both_missing_keeps_nemo(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["nemo"] = False
    engines["pyannote"] = False

    assert factory.resolve_diarization_engine(_config(audio_file, num_speakers=2)) == "nemo-speech"


def test_route_cap_is_configurable(audio_file: Path, engines: dict[str, object]) -> None:
    config = _config(audio_file, num_speakers=5, diarization_route_max_speakers=8)

    assert factory.resolve_diarization_engine(config) == "nemo-speech"


# --- create_diarizer: параметры движков сохраняются -------------------------


def test_create_diarizer_nemo_carries_settings(
    audio_file: Path, engines: dict[str, object]
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


def test_create_diarizer_pyannote_path_intact(
    audio_file: Path, engines: dict[str, object]
) -> None:
    config = _config(audio_file, diarization_engine="pyannote")

    diarizer = factory.create_diarizer(config, Device.CPU)

    assert isinstance(diarizer, PyannoteSpeakerDiarizer)


def test_create_diarizer_hybrid_carries_settings(
    audio_file: Path, engines: dict[str, object]
) -> None:
    config = _config(
        audio_file,
        diarization_engine="hybrid",
        nemo_speech_binary="/opt/nemo/nemo-speech",
        nemo_speech_model="sortformer",
        nemo_speech_device="vulkan",
        num_speakers=6,
        diarization_hybrid_window_seconds=60.0,
        diarization_hybrid_overlap_seconds=1.0,
        diarization_hybrid_overload_split=False,
        diarization_hybrid_subwindow_seconds=20.0,
        diarization_hybrid_max_split_depth=2,
        diarization_estimate_threshold=0.65,
    )

    diarizer = factory.create_diarizer(config, Device.CPU)

    assert isinstance(diarizer, HybridSpeakerDiarizer)
    assert diarizer._binary == "/opt/nemo/nemo-speech"
    assert diarizer._model == "sortformer"
    assert diarizer._device == "vulkan"
    assert diarizer._window_seconds == 60.0
    assert diarizer._overlap_seconds == 1.0
    assert diarizer._threshold == 0.65
    assert diarizer._expected_speakers == 6
    assert diarizer._overload_split is False
    assert diarizer._subwindow_seconds == 20.0
    assert diarizer._max_split_depth == 2


def test_create_diarizer_auto_above_cap_builds_hybrid(
    audio_file: Path, engines: dict[str, object]
) -> None:
    engines["estimate"] = 6

    diarizer = factory.create_diarizer(_config(audio_file), Device.CPU)

    assert isinstance(diarizer, HybridSpeakerDiarizer)
    assert diarizer._expected_speakers == 6


def test_create_diarizer_auto_routes_and_emits_progress(
    audio_file: Path, engines: dict[str, object]
) -> None:
    events: list[object] = []
    config = _config(audio_file, num_speakers=2)

    diarizer = factory.create_diarizer(config, Device.CPU, on_progress=events.append)

    assert isinstance(diarizer, NemoSpeechSpeakerDiarizer)
    assert any(getattr(event, "stage", None) == "diarization" for event in events)
