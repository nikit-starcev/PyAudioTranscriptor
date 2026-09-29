"""Доменные модели, перечисления и операции редактирования, не зависящие
от конкретных реализаций движков распознавания речи, диаризации и экспорта.
"""

from audio_transcriber.domain.editing import merge_speakers, rename_speaker

__all__ = ["merge_speakers", "rename_speaker"]
