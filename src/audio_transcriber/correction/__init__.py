"""Постобработка стенограммы по пользовательскому словарю терминов."""

from audio_transcriber.correction.base import TextCorrector
from audio_transcriber.correction.symspell_corrector import SymSpellTextCorrector

__all__ = ["TextCorrector", "SymSpellTextCorrector"]
