"""Пакет пост-коррекции текста стенограммы."""

from audio_transcriber.correction.base import TextCorrector
from audio_transcriber.correction.morph_corrector import MorphTextCorrector

__all__ = ["TextCorrector", "MorphTextCorrector"]
