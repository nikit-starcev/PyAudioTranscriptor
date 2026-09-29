"""Шумоподавление (денойз) аудио перед распознаванием и диаризацией."""

from __future__ import annotations

from audio_transcriber.denoising.base import DenoiserProtocol
from audio_transcriber.denoising.deepfilter import DeepFilterDenoiser

__all__ = ["DeepFilterDenoiser", "DenoiserProtocol"]
