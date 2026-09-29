"""Пакет очистки стенограммы: артефакты ASR, повторы и нормализация текста."""

from audio_transcriber.cleaning.artifact_filter import ArtifactCleaner
from audio_transcriber.cleaning.base import (
    ArtifactCleanerProtocol,
    RepetitionCleanerProtocol,
    TextNormalizerProtocol,
)
from audio_transcriber.cleaning.repetition_filter import RepetitionCleaner
from audio_transcriber.cleaning.text_normalizer import TextNormalizer

__all__ = [
    "ArtifactCleaner",
    "ArtifactCleanerProtocol",
    "RepetitionCleaner",
    "RepetitionCleanerProtocol",
    "TextNormalizer",
    "TextNormalizerProtocol",
]
