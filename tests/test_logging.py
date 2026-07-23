"""Тесты настройки логирования (:mod:`audio_transcriber.utils.logging`)."""

from __future__ import annotations

import logging
import warnings

from audio_transcriber.utils.logging import setup_logging


def test_setup_logging_silences_flop_counter_logger() -> None:
    setup_logging(verbose=False)

    assert logging.getLogger("torch.utils.flop_counter").level == logging.ERROR


def test_setup_logging_silences_pyannote_torchcodec_warning() -> None:
    setup_logging(verbose=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn_explicit(
            "\ntorchcodec is not installed correctly so built-in audio decoding will fail.",
            UserWarning,
            "io.py",
            48,
            module="pyannote.audio.core.io",
        )

    assert caught == []


def test_setup_logging_silences_pyannote_reproducibility_warning() -> None:
    setup_logging(verbose=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn_explicit(
            "TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues.",
            UserWarning,
            "reproducibility.py",
            74,
            module="pyannote.audio.utils.reproducibility",
        )

    assert caught == []
