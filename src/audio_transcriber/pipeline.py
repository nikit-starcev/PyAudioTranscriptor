"""Сборка конвейера: денойз -> распознавание -> диаризация -> объединение -> очистка -> коррекция -> экспорт.

Каждый этап конвейера обращается к своему компоненту только через протокол
(``SpeechRecognizer``, ``SpeakerDiarizer``, ``SegmentMerger``,
``ArtifactCleanerProtocol``, ``TextCorrector``, ``ResultExporter``,
``DenoiserProtocol``), поэтому конкретную реализацию можно передать снаружи —
это используется в тестах для подстановки фиктивных движков.
"""

from __future__ import annotations

import logging

from audio_transcriber.cache.denoiser import CachingDenoiser
from audio_transcriber.cache.serialization import (
    asr_from_payload,
    asr_payload,
    diarization_from_payload,
    diarization_payload,
)
from audio_transcriber.cache.store import StageCache
from audio_transcriber.cleaning.artifact_filter import ArtifactCleaner
from audio_transcriber.cleaning.base import (
    ArtifactCleanerProtocol,
    RepetitionCleanerProtocol,
    TextNormalizerProtocol,
)
from audio_transcriber.cleaning.repetition_filter import RepetitionCleaner
from audio_transcriber.cleaning.text_normalizer import TextNormalizer
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.base import TextCorrector
from audio_transcriber.correction.morph_corrector import MorphTextCorrector
from audio_transcriber.denoising.base import DenoiserProtocol
from audio_transcriber.denoising.deepfilter import DeepFilterDenoiser
from audio_transcriber.diarization.base import SpeakerDiarizer
from audio_transcriber.diarization.enrollment import (
    SpeakerEmbeddingEngine,
    assign_speaker_names,
)
from audio_transcriber.diarization.pyannote_engine import (
    DEFAULT_PIPELINE as DIARIZATION_PIPELINE,
)
from audio_transcriber.diarization.pyannote_engine import PyannoteSpeakerDiarizer
from audio_transcriber.diarization.samples import extract_speaker_samples
from audio_transcriber.domain.enums import AsrBackend, Device
from audio_transcriber.domain.models import SpeakerOverlap, TranscriptionResult
from audio_transcriber.export.factory import create_exporter
from audio_transcriber.export.timeline import (
    build_speaker_tracks,
    render_timeline_text,
    write_timeline,
)
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.merging.aligner import OverlapSegmentMerger
from audio_transcriber.merging.base import SegmentMerger
from audio_transcriber.merging.overlap import mark_overlap_entries
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


def _asr_cache_params(
    config: AppConfig, device: Device, recognizer: SpeechRecognizer
) -> dict[str, object]:
    """Релевантные параметры стадии ASR для ключа кэша.

    Включается только то, что влияет на результат: бэкенд и его модель, язык,
    устройство, подсказки и признак денойза (он меняет входное аудио).
    Имя класса движка отсекает кэш при смене реализации.
    """
    params: dict[str, object] = {
        "engine": type(recognizer).__name__,
        "backend": config.asr_backend.value,
        "language": config.language,
        "device": device.value,
        "denoise": config.denoise,
        "initial_prompt": config.initial_prompt,
        "hotwords": config.hotwords,
    }
    if config.asr_backend is AsrBackend.WHISPER_CPP:
        params["whisper_cpp_model"] = (
            str(config.whisper_cpp_model) if config.whisper_cpp_model else None
        )
        params["whisper_cpp_threads"] = config.whisper_cpp_threads
    else:
        params["model"] = config.model_name
    return params


def _diarization_cache_params(
    config: AppConfig, device: Device, diarizer: SpeakerDiarizer
) -> dict[str, object]:
    """Релевантные параметры стадии диаризации для ключа кэша."""
    return {
        "engine": type(diarizer).__name__,
        "device": device.value,
        "denoise": config.denoise,
        "num_speakers": config.num_speakers,
        "pipeline": DIARIZATION_PIPELINE,
        "local_model": (
            str(config.pyannote_local_model) if config.pyannote_local_model else None
        ),
        # Пометка наложения влияет на то, собираются ли зоны перекрытий.
        "mark_overlap": config.mark_overlap,
    }


def run_pipeline(
    config: AppConfig,
    *,
    device: Device | None = None,
    recognizer: SpeechRecognizer | None = None,
    diarizer: SpeakerDiarizer | None = None,
    merger: SegmentMerger | None = None,
    sentence_merger: SentenceMerger | None = None,
    artifact_cleaner: ArtifactCleanerProtocol | None = None,
    repetition_cleaner: RepetitionCleanerProtocol | None = None,
    text_normalizer: TextNormalizerProtocol | None = None,
    corrector: TextCorrector | None = None,
    llm_client: LlmClient | None = None,
    denoiser: DenoiserProtocol | None = None,
    enrollment_engine: SpeakerEmbeddingEngine | None = None,
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
    if artifact_cleaner is None and config.clean_artifacts:
        artifact_cleaner = ArtifactCleaner()
    if repetition_cleaner is None and config.collapse_repeats:
        repetition_cleaner = RepetitionCleaner(
            min_words=config.repeat_min_words,
            similarity=config.repeat_similarity,
        )
    if text_normalizer is None and config.normalize_text:
        text_normalizer = TextNormalizer()
    if corrector is None and config.enable_correction:
        corrector = MorphTextCorrector(
            min_word_length=config.correction_min_word_length,
            min_similarity=config.correction_min_similarity,
            max_candidates=config.correction_max_candidates,
        )

    # Дорогие стадии кэшируются по ключу от исходного файла и параметров.
    cache = StageCache(config.resolved_cache_dir(), enabled=config.use_cache)

    # Шумоподавление идёт первым: и распознавание, и диаризация должны видеть
    # один и тот же очищенный файл, иначе временные метки разъедутся.
    denoiser = denoiser or (DeepFilterDenoiser() if config.denoise else None)
    if denoiser is not None and config.use_cache:
        denoiser = CachingDenoiser(denoiser, cache, source=config.input_file)
    audio_path = config.input_file
    overlaps: list[SpeakerOverlap] = []
    # Имена, сопоставленные говорящим по образцам голоса (enrollment).
    # Приоритетнее переименования по индексу (``--speaker-name``).
    enrollment_names: dict[str, str] = {}
    try:
        if denoiser is not None:
            logger.info("Шумоподавление (DeepFilterNet)...")
            emit(ProgressEvent("denoise", "Шумоподавление", fraction=None))
            audio_path = denoiser.denoise(config.input_file)

        asr_key = cache.key("asr", config.input_file, _asr_cache_params(config, device, recognizer))
        cached_asr = cache.load("asr", asr_key)
        if cached_asr is not None:
            try:
                transcription_segments, language, duration = asr_from_payload(cached_asr)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Кэш ASR повреждён, будет пересчитан: %s", exc)
                cached_asr = None
        if cached_asr is not None:
            logger.info("Кэш ASR: попадание (%s)", asr_key[:12])
            emit(ProgressEvent("asr", "Распознавание речи", fraction=None, detail="из кэша"))
        else:
            logger.info("Кэш ASR: промах — распознавание речи...")
            emit(ProgressEvent("asr", "Распознавание речи", fraction=None))
            transcription_segments, language, duration = recognizer.transcribe(
                audio_path, language=config.language
            )
            cache.save(
                "asr", asr_key, asr_payload(transcription_segments, language, duration)
            )

        # whisper.cpp сам использует GPU через Vulkan; pyannote.audio в этом
        # гибриде не имеет GPU-бэкенда (ROCm не поддерживает старые AMD-карты),
        # поэтому диаризация всегда выполняется на CPU.
        diarization_device = (
            Device.CPU if config.asr_backend is AsrBackend.WHISPER_CPP else device
        )
        if config.diarization_enabled:
            active_diarizer = diarizer or PyannoteSpeakerDiarizer(
                diarization_device,
                hf_token=config.hf_token,
                local_model_path=config.pyannote_local_model,
                on_progress=emit,
            )
            dia_key = cache.key(
                "diarization",
                config.input_file,
                _diarization_cache_params(config, diarization_device, active_diarizer),
            )
            cached_dia = cache.load("diarization", dia_key)
            if cached_dia is not None:
                try:
                    speaker_segments, overlaps = diarization_from_payload(cached_dia)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Кэш диаризации повреждён, будет пересчитан: %s", exc)
                    cached_dia = None
            if cached_dia is not None:
                logger.info("Кэш диаризации: попадание (%s)", dia_key[:12])
                emit(
                    ProgressEvent(
                        "diarization", "Определение говорящих", fraction=None, detail="из кэша"
                    )
                )
            else:
                logger.info("Кэш диаризации: промах — определение говорящих...")
                emit(ProgressEvent("diarization", "Определение говорящих", fraction=None))
                speaker_segments = active_diarizer.diarize(
                    audio_path, num_speakers=config.num_speakers
                )
                # Зоны наложения речи — из обычной (не эксклюзивной) разметки, если
                # движок её умеет. Отсутствие метода — не ошибка (мягкая деградация).
                if config.mark_overlap:
                    overlap_getter = getattr(active_diarizer, "overlap_regions", None)
                    if callable(overlap_getter):
                        overlaps = list(overlap_getter())
                cache.save(
                    "diarization", dia_key, diarization_payload(speaker_segments, overlaps)
                )
        else:
            logger.info("Диаризация отключена — все реплики без разметки говорящих")
            emit(ProgressEvent("diarization", "Диаризация отключена", fraction=1.0))
            speaker_segments = []

        # Enrollment: сопоставляем говорящих с именами по образцам голоса.
        # Делаем это здесь, пока доступно аудио (денойзенный файл закрывается
        # в finally). Приоритет у enrollment-имён выше ``--speaker-name``.
        # К явным образцам добавляются файлы библиотеки ``voices_dir``.
        references = config.resolved_speaker_references()
        if config.diarization_enabled and references and speaker_segments:
            logger.info("Сопоставление говорящих с образцами голоса (enrollment)...")
            emit(ProgressEvent("diarization", "Сопоставление голосов", fraction=None))
            enrollment_names = assign_speaker_names(
                speaker_segments=speaker_segments,
                references=references,
                audio_path=audio_path,
                min_similarity=config.enrollment_min_similarity,
                device=diarization_device,
                local_model_path=config.pyannote_local_model,
                engine=enrollment_engine,
            )
    finally:
        # Временный денойзенный WAV нужен только ASR и диаризации; удаляем его,
        # как только оба этапа завершились (в т.ч. при ошибке).
        close = getattr(denoiser, "close", None)
        if callable(close):
            close()

    emit(ProgressEvent("merge", "Объединение сегментов", fraction=None))
    known_speakers = {**config.speaker_names, **enrollment_names}
    entries, speakers = merger.merge(transcription_segments, speaker_segments, known_speakers)

    # Чистим неречевые пометки Whisper ([СМЕХ], [BLANK_AUDIO], ♪ и т.п.),
    # схлопываем зацикленные повторы и аккуратно нормализуем текст — всё до
    # склейки предложений, чтобы корректор, LLM и экспорт работали с готовым
    # текстом. Реплики, состоящие только из пометок, здесь же отбрасываются.
    if artifact_cleaner is not None:
        logger.info("Очистка неречевых артефактов...")
        emit(ProgressEvent("clean", "Очистка артефактов", fraction=None))
        entries = artifact_cleaner.clean(entries)

    if repetition_cleaner is not None:
        logger.info("Схлопывание повторяющихся реплик...")
        entries = repetition_cleaner.clean(entries)

    if text_normalizer is not None:
        logger.info("Нормализация текста...")
        entries = text_normalizer.normalize(entries)

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
    summary = None
    if config.llm_enabled:
        from audio_transcriber.llm.postprocess import run_llm_postprocess

        logger.info("LLM-постобработка (имена участников, правка терминов, резюме)...")
        entries, speakers, participants, summary = run_llm_postprocess(
            config,
            entries,
            speakers,
            client=llm_client,
            # В режиме «протокол по кнопке» резюме не считается на прогоне:
            # имена и термины правятся, а резюме пересчитывается позже по
            # актуальной (в т.ч. переименованной) стенограмме.
            summarize=config.protocol_auto and config.llm_summary,
            on_progress=emit,
        )

    # Помечаем реплики в зонах наложения речи в самом конце — после всех
    # текстовых правок, чтобы пометка соответствовала финальным репликам.
    if config.mark_overlap and overlaps:
        entries = mark_overlap_entries(entries, overlaps)

    result = TranscriptionResult(
        source_path=config.input_file,
        language=language,
        duration=duration,
        entries=entries,
        speakers=speakers,
        participants=participants,
        summary=summary,
        low_confidence_threshold=config.low_confidence_threshold,
    )

    if config.protocol_auto:
        for export_format in config.export_formats:
            output_path = (
                config.output_dir / f"{config.input_file.stem}.{export_format.value}"
            )
            logger.info("Экспорт в %s: %s", export_format.value, output_path)
            emit(ProgressEvent("export", f"Экспорт {export_format.value}", fraction=None))
            create_exporter(export_format).export(result, output_path)

        if config.timeline:
            if build_speaker_tracks(result):
                # Подробная сводка «кто когда говорил» — в лог (INFO), HTML —
                # рядом с остальными результатами.
                logger.info("Таймлайн говорящих:\n%s", render_timeline_text(result))
                timeline_path = config.output_dir / f"{config.input_file.stem}.timeline.html"
                logger.info("Экспорт таймлайна: %s", timeline_path)
                emit(ProgressEvent("export", "Экспорт таймлайна", fraction=None))
                write_timeline(result, timeline_path)
            else:
                logger.info("Таймлайн пропущен — нет данных о говорящих")
    else:
        logger.info(
            "Протокол отложен (protocol_auto=False) — стенограмма готова, "
            "экспорт и резюме будут выполнены по запросу"
        )

    if config.export_speaker_samples:
        # Образцы голоса извлекаем из того же аудио, что шло в ASR/диаризацию
        # (денойзенный файл), если оно ещё доступно; иначе — из исходного.
        # Любая ошибка здесь не должна ронять конвейер: экспорт уже выполнен.
        sample_audio = audio_path if audio_path.is_file() else config.input_file
        emit(ProgressEvent("export", "Образцы голоса", fraction=None))
        try:
            written = extract_speaker_samples(
                result,
                audio_path=sample_audio,
                output_dir=config.output_dir,
            )
        except Exception as exc:  # noqa: BLE001 — мягкая деградация
            logger.warning("Образцы голоса не сохранены: %s", exc)
            written = {}
        if written:
            logger.info(
                "Образцы голоса сохранены (%d): %s",
                len(written),
                ", ".join(path.name for path in written.values()),
            )
        else:
            logger.info(
                "Образцы голоса пропущены — нет чистой речи говорящих или она тише порога"
            )

    emit(ProgressEvent("done", "Готово", fraction=1.0))
    return result
