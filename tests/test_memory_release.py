"""Тесты освобождения моделей/буфера (#140/#141).

Проверяют, что у in-process движков есть ``close``, который действительно
сбрасывает ссылку на модель (RSS не держится до конца прогона), и что
:func:`release_memory` не тянет за собой импорт ``torch`` на CPU-путях.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from audio_transcriber.domain.enums import Device
from audio_transcriber.utils.memory import release_memory


def test_release_memory_without_cuda_does_not_import_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """На путях без torch освобождение не провоцирует его импорт (экономия ~435 МБ)."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    release_memory()

    assert "torch" not in sys.modules


def test_whisper_recognizer_close_drops_model() -> None:
    from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer

    recognizer = WhisperSpeechRecognizer("small", Device.CPU)
    recognizer._model = object()

    recognizer.close()

    assert recognizer._model is None


def test_gigaam_recognizer_close_drops_adapter() -> None:
    from audio_transcriber.transcription.gigaam_engine import GigaAmRecognizer

    recognizer = GigaAmRecognizer()
    recognizer._adapter = object()
    recognizer._vad_active = True

    recognizer.close()

    assert recognizer._adapter is None
    assert recognizer._vad_active is False


def test_hybrid_recognizer_close_closes_both_engines() -> None:
    from audio_transcriber.transcription.hybrid import HybridSpeechRecognizer

    class _Engine:
        def __init__(self) -> None:
            self.closed = 0

        def transcribe(self, *args: object, **kwargs: object):  # pragma: no cover
            return [], "ru", 0.0

        def close(self) -> None:
            self.closed += 1

    primary, fallback = _Engine(), _Engine()
    recognizer = HybridSpeechRecognizer(primary, fallback)  # type: ignore[arg-type]

    recognizer.close()

    assert primary.closed == 1
    assert fallback.closed == 1


def test_pyannote_diarizer_close_drops_pipeline() -> None:
    from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer

    diarizer = PyannoteSpeakerDiarizer(Device.CPU)
    diarizer._pipeline = object()

    diarizer.close()

    assert diarizer._pipeline is None


def test_sherpa_diarizer_close_drops_embedder() -> None:
    from audio_transcriber.diarization.sherpa_engine import SherpaSpeakerDiarizer

    diarizer = SherpaSpeakerDiarizer()
    diarizer._embedder = object()

    diarizer.close()

    assert diarizer._embedder is None


def test_hybrid_diarizer_close_drops_embedder() -> None:
    from audio_transcriber.diarization.hybrid_engine import HybridSpeakerDiarizer

    diarizer = HybridSpeakerDiarizer("cpu")
    diarizer._embedder = object()

    diarizer.close()

    assert diarizer._embedder is None


def test_speaker_embedder_close_drops_extractor() -> None:
    from audio_transcriber.diarization.embeddings import SpeakerEmbedder

    embedder = SpeakerEmbedder(Path("model.onnx"))
    embedder._extractor = object()

    embedder.close()

    assert embedder._extractor is None


def test_pyannote_embedding_engine_close_drops_inference() -> None:
    from audio_transcriber.diarization.enrollment import PyannoteEmbeddingEngine

    engine = PyannoteEmbeddingEngine(device=Device.CPU)
    engine._inference = object()

    engine.close()

    assert engine._inference is None
