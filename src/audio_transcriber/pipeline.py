"""Сборка конвейера: распознавание -> диаризация -> объединение -> коррекция -> экспорт.

Каждый этап конвейера обращается к своему компоненту только через протокол
(``SpeechRecognizer``, ``SpeakerDiarizer``, ``SegmentMerger``,
``TextCorrector``, ``ResultExporter``), поэтому конкретную реализацию можно
передать снаружи — это используется в тестах для подстановки фиктивных движков.
"""

from __future__ import annotations

import logging

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.base import TextCorrector
from audio_transcriber.correction.symspell_corrector import SymSpellTextCorrector
from audio_transcriber.diarization.base import SpeakerDiarizer
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.factory import create_exporter
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.merging.base import SegmentMerger
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer
from audio_transcriber.utils.device import resolve_device

logger = logging.getLogger(__name__)


def run_pipeline(
    config: AppConfig,
    *,
    device: Device | None = None,
    recognizer: SpeechRecognizer | None = None,
    diarizer: SpeakerDiarizer | None = None,
    merger: SegmentMerger | None = None,
    corrector: TextCorrector | None = None,
) -> TranscriptionResult:
    """Прогоняет входной файл через полный конвейер и экспортирует результат."""

    device = device or resolve_device(config.device)
    recognizer = recognizer or WhisperSpeechRecognizer(
        config.model_name,
        device,
        initial_prompt=config.initial_prompt,
        hotwords=config.hotwords,
    )
    diarizer = diarizer or PyannoteSpeakerDiarizer(device, hf_token=config.hf_token)
    merger = merger or OverlapSegmentMerger()
    if corrector is None and config.correction_terms:
        corrector = SymSpellTextCorrector(list(config.correction_terms))

    logger.info("Распознавание речи...")
    transcription_segments, language, duration = recognizer.transcribe(
        config.input_file, language=config.language
    )

    logger.info("Определение говорящих...")
    speaker_segments = diarizer.diarize(config.input_file, num_speakers=config.num_speakers)

    entries, speakers = merger.merge(
        transcription_segments, speaker_segments, config.speaker_names
    )

    if corrector is not None:
        logger.info("Постобработка текста по словарю (%d терминов)...", len(config.correction_terms))
        entries = corrector.correct(entries)

    result = TranscriptionResult(
        source_path=config.input_file,
        language=language,
        duration=duration,
        entries=entries,
        speakers=speakers,
    )

    for export_format in config.export_formats:
        output_path = config.output_dir / f"{config.input_file.stem}.{export_format.value}"
        logger.info("Экспорт в %s: %s", export_format.value, output_path)
        create_exporter(export_format).export(result, output_path)

    return result
