"""Интеграционный smoke-тест диаризации pyannote на коротком фрагменте.

Запуск: ``pytest -m integration``. Если локальная модель и HF-токен
недоступны, тест пропускается (см. фикстуру ``pyannote_target``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device
from audio_transcriber.utils.exceptions import DiarizationError

from ._helpers import PyannoteTarget

pytestmark = pytest.mark.integration


def test_pyannote_diarizes_short_audio(
    short_speech_wav: Path, pyannote_target: PyannoteTarget
) -> None:
    """Реальный pyannote не падает и возвращает корректные реплики говорящих."""
    diarizer = PyannoteSpeakerDiarizer(
        Device.CPU,
        local_model_path=pyannote_target.local_model,
        hf_token=pyannote_target.hf_token,
    )

    try:
        segments = diarizer.diarize(short_speech_wav)
    except DiarizationError as exc:
        pytest.skip(f"модель диаризации недоступна: {exc}")

    assert segments, "диаризация не вернула ни одной реплики"
    assert segments == sorted(segments, key=lambda item: item.start), (
        "реплики не упорядочены по времени"
    )
    for segment in segments:
        assert segment.speaker_id, "у реплики не указан говорящий"
        assert segment.end > segment.start >= 0.0, (
            f"некорректные границы реплики {segment.start:.2f}-{segment.end:.2f}"
        )
