"""Изоляция тяжёлых Python-движков в отдельный процесс (#143).

Модуль предоставляет обёртки, повторяющие контракты ``SpeechRecognizer`` и
``SpeakerDiarizer``, но выполняющие модель в процессе-воркере
(:mod:`audio_transcriber.isolation.worker`): после стадии RSS основного процесса
не содержит torch/CTranslate2/ONNX-моделей.
"""

from __future__ import annotations

from audio_transcriber.isolation.diarizer import SubprocessPyannoteDiarizer
from audio_transcriber.isolation.recognizer import SubprocessWhisperRecognizer

__all__ = ["SubprocessPyannoteDiarizer", "SubprocessWhisperRecognizer"]
