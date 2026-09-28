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
from audio_transcriber.correction.morph_corrector import MorphTextCorrector
from audio_transcriber.diarization.base import SpeakerDiarizer
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.domain.enums import AsrBackend, Device
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.factory import create_exporter
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.merging.base import SegmentMerger
from audio_transcriber.merging.sentence_merger import SentenceMerger
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.transcription.whisper_cpp_engine import WhisperCppRecognizer
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer
from audio_transcriber.utils.device import resolve_device

logger = logging.getLogger(__name__)


def _build_recognizer(
    config: AppConfig, device: Device, on_progress: ProgressCallback | None = None
) -> SpeechRecognizer:
    """Создаёт движок распознавания речи в зависимости от выбранного бэкенда."""
    if config.asr_backend is AsrBackend.WHISPER_CPP:
        return WhisperCppRecognizer(
            config.whisper_cpp_model,  # type: ignore[arg-type]
            binary=config.whisper_cpp_binary,
            library_path=config.whisper_cpp_lib_path,
            threads=config.whisper_cpp_threads,
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
            on_progress=on_progress,
        )
    return WhisperSpeechRecognizer(
        config.model_name,
        device,
        initial_prompt=config.initial_prompt,
        hotwords=config.hotwords,
    )


def run_pipeline(
    config: AppConfig,
    *,
    device: Device | None = None,
    recognizer: SpeechRecognizer | None = None,
    diarizer: SpeakerDiarizer | None = None,
    merger: SegmentMerger | None = None,
    sentence_merger: SentenceMerger | None = None,
    corrector: TextCorrector | None = None,
    llm_client: LlmClient | None = None,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Прогоняет входной файл через полный конвейер и экспортирует результат.

    Необязательный ``on_progress`` вызывается с событиями :class:`ProgressEvent`
    на каждом этапе — это используется TUI для живого отображения прогресса.
    """

    emit = on_progress or (lambda _event: None)
    config.ensure_output_dir()
    device = device or resolve_device(config.device)
    recognizer = recognizer or _build_recognizer(config, device, on_progress=emit)

    merger = merger or OverlapSegmentMerger()
    sentence_merger = sentence_merger or SentenceMerger()
    if corrector is None and config.enable_correction:
        corrector = MorphTextCorrector(
            min_word_length=config.correction_min_word_length,
            min_similarity=config.correction_min_similarity,
            max_candidates=config.correction_max_candidates,
        )

    logger.info("Распознавание речи...")
    emit(ProgressEvent("asr", "Распознавание речи", fraction=None))
    transcription_segments, language, duration = recognizer.transcribe(
        config.input_file, language=config.language
    )

    if config.diarization_enabled:
        # whisper.cpp сам использует GPU через Vulkan; pyannote.audio в этом
        # гибриде не имеет GPU-бэкенда (ROCm не поддерживает старые AMD-карты),
        # поэтому диаризация всегда выполняется на CPU.
        diarization_device = (
            Device.CPU if config.asr_backend is AsrBackend.WHISPER_CPP else device
        )
        active_diarizer = diarizer or PyannoteSpeakerDiarizer(
            diarization_device,
            hf_token=config.hf_token,
            local_model_path=config.pyannote_local_model,
            on_progress=emit,
        )
        logger.info("Определение говорящих...")
        emit(ProgressEvent("diarization", "Определение говорящих", fraction=None))
        speaker_segments = active_diarizer.diarize(
            config.input_file, num_speakers=config.num_speakers
        )
    else:
        logger.info("Диаризация отключена — все реплики без разметки говорящих")
        emit(ProgressEvent("diarization", "Диаризация отключена", fraction=1.0))
        speaker_segments = []

    emit(ProgressEvent("merge", "Объединение сегментов", fraction=None))
    entries, speakers = merger.merge(transcription_segments, speaker_segments, config.speaker_names)

    # Склеиваем подряд идущие короткие сегменты одного говорящего в реплики-
    # предложения — корректор и LLM должны видеть уже цельный текст.
    merged_entries = sentence_merger.merge(entries)
    if len(merged_entries) != len(entries):
        logger.info(
            "Склейка реплик: %d сегментов → %d реплик",
            len(entries),
            len(merged_entries),
        )
    entries = merged_entries

    if corrector is not None:
        logger.info("Автоисправление опечаток (только неизвестные словоформы)...")
        emit(ProgressEvent("correction", "Автоисправление опечаток", fraction=None))
        entries = corrector.correct(entries)

    participants = None
    if config.llm_enabled:
        from audio_transcriber.llm.postprocess import run_llm_postprocess

        logger.info("LLM-постобработка (имена участников, правка терминов)...")
        entries, speakers, participants = run_llm_postprocess(
            config, entries, speakers, client=llm_client, on_progress=emit
        )

    result = TranscriptionResult(
        source_path=config.input_file,
        language=language,
        duration=duration,
        entries=entries,
        speakers=speakers,
        participants=participants,
    )

    for export_format in config.export_formats:
        output_path = config.output_dir / f"{config.input_file.stem}.{export_format.value}"
        logger.info("Экспорт в %s: %s", export_format.value, output_path)
        emit(ProgressEvent("export", f"Экспорт {export_format.value}", fraction=None))
        create_exporter(export_format).export(result, output_path)

    emit(ProgressEvent("done", "Готово", fraction=1.0))
    return result
