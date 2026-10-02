"""Сборка конвейера: денойз -> распознавание -> диаризация -> объединение -> очистка -> коррекция -> экспорт.

Каждый этап конвейера обращается к своему компоненту только через протокол
(``SpeechRecognizer``, ``SpeakerDiarizer``, ``SegmentMerger``,
``ArtifactCleanerProtocol``, ``TextCorrector``, ``ResultExporter``,
``DenoiserProtocol``), поэтому конкретную реализацию можно передать снаружи —
это используется в тестах для подстановки фиктивных движков.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

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
from audio_transcriber.diarization.factory import create_diarizer
from audio_transcriber.diarization.hybrid_engine import (
    DIARIZATION_HYBRID_IMPL_VERSION,
    HybridSpeakerDiarizer,
)
from audio_transcriber.diarization.overlap import DIARIZATION_IMPL_VERSION
from audio_transcriber.diarization.pyannote_engine import (
    DEFAULT_PIPELINE as DIARIZATION_PIPELINE,
)
from audio_transcriber.diarization.reference import ReferencePrepareOptions
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
from audio_transcriber.merging.overlap import apply_overlap_regions
from audio_transcriber.merging.sentence_merger import SentenceMerger
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.transcription.base import SpeechRecognizer
from audio_transcriber.transcription.gigaam_engine import (
    ASR_IMPL_VERSION as GIGAAM_ASR_IMPL_VERSION,
)
from audio_transcriber.transcription.gigaam_engine import GigaAmRecognizer
from audio_transcriber.transcription.hybrid import (
    HYBRID_IMPL_VERSION,
    HybridOptions,
    HybridSpeechRecognizer,
)
from audio_transcriber.transcription.whisper_cpp_engine import (
    ASR_IMPL_VERSION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SECONDS,
    WhisperCppRecognizer,
)
from audio_transcriber.transcription.whisper_engine import WhisperSpeechRecognizer
from audio_transcriber.utils.audio import load_waveform
from audio_transcriber.utils.device import resolve_device
from audio_transcriber.utils.exceptions import ProcessingCancelled
from audio_transcriber.utils.subprocess_registry import terminate_all_processes

logger = logging.getLogger(__name__)


class _SharedWaveform:
    """Однократное декодирование 16-кГц waveform для нескольких стадий.

    Диаризация, enrollment и извлечение образцов нуждаются в одном и том же
    декодированном аудио. Если денойз уже посчитал waveform — отдаётся он;
    иначе аудио декодируется **один раз** при первом обращении и кэшируется,
    а последующие обращения возвращают тот же массив. Ошибка декодирования
    мягко деградирует до ``None``: потребитель сам решит, что делать (как и
    раньше, когда декодировал файл сам).
    """

    def __init__(self, path: Path, waveform: np.ndarray | None = None) -> None:
        self._path = path
        self._waveform = waveform
        self._attempted = waveform is not None

    def get(self) -> np.ndarray | None:
        """Возвращает waveform, декодируя его при первом обращении (не более раз)."""
        if not self._attempted:
            self._attempted = True
            try:
                self._waveform = load_waveform(self._path)
            except Exception as exc:  # noqa: BLE001 — потребитель деградирует сам
                logger.debug(
                    "Общий waveform: не удалось декодировать %s: %s", self._path, exc
                )
                self._waveform = None
        return self._waveform


def _whisper_cpp_vad_model() -> Path | None:
    """Необязательная ggml-модель Silero VAD для whisper.cpp (из окружения)."""
    raw = os.environ.get("WHISPER_CPP_VAD_MODEL", "").strip()
    return Path(raw) if raw else None


def _parse_env_float(name: str, default: float) -> float:
    """Разбирает необязательное число из окружения; при мусоре — значение по умолчанию."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r — не число, использую значение по умолчанию %.1f", name, raw, default
        )
        return default
    if value < 0.0:
        logger.warning(
            "%s=%.1f не может быть отрицательным, использую %.1f", name, value, default
        )
        return default
    return value


def _whisper_cpp_chunk_settings() -> tuple[float, float]:
    """Настройки чанкинга длинных файлов whisper.cpp из окружения.

    ``WHISPER_CPP_CHUNK_SECONDS`` — целевая длина куска (0 — чанкинг выключен,
    прежнее поведение), ``WHISPER_CPP_CHUNK_OVERLAP`` — перекрытие кусков.
    Значения по умолчанию — :data:`DEFAULT_CHUNK_SECONDS` и
    :data:`DEFAULT_CHUNK_OVERLAP`. Некорректные значения мягко заменяются
    значениями по умолчанию, чтобы не ронять распознавание.
    """
    chunk_seconds = _parse_env_float("WHISPER_CPP_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)
    chunk_overlap = _parse_env_float("WHISPER_CPP_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    if chunk_seconds > 0.0 and chunk_overlap >= chunk_seconds:
        logger.warning(
            "WHISPER_CPP_CHUNK_OVERLAP=%.1f >= WHISPER_CPP_CHUNK_SECONDS=%.1f — "
            "ограничиваю перекрытие половиной куска",
            chunk_overlap,
            chunk_seconds,
        )
        chunk_overlap = chunk_seconds / 2.0
    return chunk_seconds, chunk_overlap


def _build_single_recognizer(
    config: AppConfig,
    backend: AsrBackend,
    device: Device,
    on_progress: ProgressCallback | None = None,
) -> SpeechRecognizer:
    """Создаёт один движок распознавания для выбранного бэкенда."""
    if backend is AsrBackend.WHISPER_CPP:
        # Silero-VAD-модель для whisper.cpp (необязательно). Если путь задан,
        # VAD выравнивается с faster-whisper (vad_filter=True) — те же условия
        # отсечения тишины/не-речи.
        chunk_seconds, chunk_overlap = _whisper_cpp_chunk_settings()
        return WhisperCppRecognizer(
            config.whisper_cpp_model,  # type: ignore[arg-type]
            binary=config.whisper_cpp_binary,
            library_path=config.whisper_cpp_lib_path,
            threads=config.whisper_cpp_threads,
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
            on_progress=on_progress,
            vad_filter=True,
            vad_model=_whisper_cpp_vad_model(),
            chunk_seconds=chunk_seconds,
            chunk_overlap=chunk_overlap,
        )
    if backend is AsrBackend.GIGAAM:
        return GigaAmRecognizer(
            config.gigaam_model,
            model_path=config.gigaam_model_path,
            quantization=config.gigaam_quantization,
            device=device,
            use_vad=config.gigaam_vad,
            on_progress=on_progress,
        )
    return WhisperSpeechRecognizer(
        config.model_name,
        device,
        initial_prompt=config.initial_prompt,
        hotwords=config.hotwords,
        on_progress=on_progress,
    )


def _build_recognizer(
    config: AppConfig, device: Device, on_progress: ProgressCallback | None = None
) -> SpeechRecognizer:
    """Создаёт движок распознавания речи в зависимости от выбранного бэкенда.

    При включённом гибридном ASR (#57) основной движок оборачивается в
    :class:`HybridSpeechRecognizer`: «плохие» сегменты дорабатывает резервный
    движок (faster-whisper/whisper.cpp). Резервный создаётся без собственного
    прогресс-колбэка, чтобы его события не перебивали прогресс основного.
    """
    primary = _build_single_recognizer(config, config.asr_backend, device, on_progress)
    if not config.hybrid_asr:
        return primary

    fallback = _build_single_recognizer(
        config, config.hybrid_fallback_backend, device, on_progress=None
    )
    return HybridSpeechRecognizer(
        primary,
        fallback,
        options=HybridOptions(
            low_logprob_threshold=config.hybrid_low_logprob_threshold,
            no_speech_threshold=config.hybrid_no_speech_threshold,
            silence_rms_threshold=config.hybrid_silence_rms_threshold,
            min_segment_seconds=config.hybrid_min_segment_seconds,
            context_seconds=config.hybrid_context_seconds,
        ),
        on_progress=on_progress,
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
        # VAD влияет на сегменты, поэтому смена модели VAD должна сбрасывать
        # кэш ASR (иначе включение VAD не даст эффекта на закэшированном файле).
        vad_model = _whisper_cpp_vad_model()
        params["whisper_cpp_vad_model"] = str(vad_model) if vad_model else None
        # Параметры чанкинга влияют на сегменты: их смена должна сбрасывать кэш
        # (иначе изменение длины куска/перекрытия не подействует на закэшированном
        # файле). Значения берём у самого движка — единственного источника истины.
        if isinstance(recognizer, WhisperCppRecognizer):
            params["whisper_cpp_chunk_seconds"] = recognizer.chunk_seconds
            params["whisper_cpp_chunk_overlap"] = recognizer.chunk_overlap
        # Версия реализации движка: чанкинг длинных файлов меняет результат при
        # тех же параметрах, поэтому старый кэш должен быть пересчитан ровно раз.
        params["asr_impl_version"] = ASR_IMPL_VERSION
    elif config.asr_backend is AsrBackend.GIGAAM:
        # Движок + модель + квантизация — всё, от чего зависит результат.
        params["gigaam_model"] = config.gigaam_model
        params["gigaam_model_path"] = (
            str(config.gigaam_model_path) if config.gigaam_model_path else None
        )
        params["gigaam_quantization"] = config.gigaam_quantization
        params["gigaam_vad"] = config.gigaam_vad
        params["asr_impl_version"] = GIGAAM_ASR_IMPL_VERSION
    else:
        params["model"] = config.model_name

    if config.hybrid_asr:
        fallback_is_cpp = config.hybrid_fallback_backend is AsrBackend.WHISPER_CPP
        params["hybrid"] = {
            # Версия логики отбора/склейки: меняет результат при тех же порогах.
            "impl_version": HYBRID_IMPL_VERSION,
            "fallback_backend": config.hybrid_fallback_backend.value,
            # Модель резервного движка тоже определяет результат и обязана
            # участвовать в ключе (иначе смена модели не пересчитает кэш).
            "fallback_model": (
                str(config.whisper_cpp_model)
                if fallback_is_cpp and config.whisper_cpp_model
                else config.model_name
            ),
            "low_logprob_threshold": config.hybrid_low_logprob_threshold,
            "no_speech_threshold": config.hybrid_no_speech_threshold,
            "silence_rms_threshold": config.hybrid_silence_rms_threshold,
            "min_segment_seconds": config.hybrid_min_segment_seconds,
            "context_seconds": config.hybrid_context_seconds,
        }
    return params


def _diarization_cache_params(
    config: AppConfig, device: Device, diarizer: SpeakerDiarizer
) -> dict[str, object]:
    """Релевантные параметры стадии диаризации для ключа кэша."""
    params: dict[str, object] = {
        "engine": type(diarizer).__name__,
        "device": device.value,
        # Выбранный движок и его параметры: смена nemo-speech -> pyannote (или
        # модели/устройства nemo) меняет результат при тех же входных данных.
        "diarization_engine": config.diarization_engine,
        "nemo_speech_binary": config.nemo_speech_binary,
        "nemo_speech_model": config.nemo_speech_model,
        "nemo_speech_device": config.nemo_speech_device,
        "denoise": config.denoise,
        "num_speakers": config.num_speakers,
        "min_speakers": config.min_speakers,
        "max_speakers": config.max_speakers,
        # Гиперпараметры pyannote, применяемые через ``pipeline.instantiate``.
        # Их смена меняет сегменты при том же входе, поэтому они обязаны
        # участвовать в ключе кэша.
        "min_duration_off": config.diarization_min_duration_off,
        "clustering_threshold": config.diarization_clustering_threshold,
        "clustering_fb": config.diarization_clustering_fb,
        "pipeline": DIARIZATION_PIPELINE,
        # Версия формата/алгоритма диаризации: добавление участников зон
        # наложения (``SpeakerOverlap.speaker_ids``) меняет результат при тех же
        # параметрах, поэтому старый кэш пересчитывается ровно один раз.
        "diarization_impl_version": DIARIZATION_IMPL_VERSION,
        "local_model": (
            str(config.pyannote_local_model) if config.pyannote_local_model else None
        ),
        # Пометка наложения влияет на то, собираются ли зоны перекрытий.
        "mark_overlap": config.mark_overlap,
    }
    if isinstance(diarizer, HybridSpeakerDiarizer):
        # Параметры гибрида (окна/порог/модель эмбеддингов) меняют результат при
        # том же входе, поэтому входят в ключ только для гибридного движка —
        # иначе каждый апгрейд инвалидировал бы кэш и остальных движков.
        params["hybrid_impl_version"] = DIARIZATION_HYBRID_IMPL_VERSION
        params["hybrid_window_seconds"] = config.diarization_hybrid_window_seconds
        params["hybrid_overlap_seconds"] = config.diarization_hybrid_overlap_seconds
        params["hybrid_min_speaker_seconds"] = config.diarization_hybrid_min_speaker_seconds
        params["hybrid_embedding_model"] = config.diarization_estimate_model
        params["hybrid_threshold"] = config.diarization_estimate_threshold
    return params


def _ensure_not_cancelled(
    cancel_event: threading.Event | None, context: str
) -> None:
    """Прерывает конвейер, если запрошена отмена.

    Вызывается в контрольных точках — перед и после каждой стадии. Помимо
    собственно сигнала отмены гасит зарегистрированные дочерние процессы
    (``whisper-cli``/``llama-server``): если стадия уже их запустила, она
    завершится быстро ошибкой, которую верхний обработчик конвертирует в
    :class:`ProcessingCancelled`.
    """
    if cancel_event is None or not cancel_event.is_set():
        return
    terminate_all_processes()
    logger.info("Обработка отменена по запросу пользователя (%s)", context)
    raise ProcessingCancelled(f"Обработка отменена: {context}")


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
    cancel_event: threading.Event | None = None,
) -> TranscriptionResult:
    """Прогоняет входной файл через полный конвейер и экспортирует результат.

    Необязательный ``on_progress`` вызывается с событиями :class:`ProgressEvent`
    на каждом этапе — это используется TUI для живого отображения прогресса.

    ``cancel_event`` — флаг отмены от веб-воркера. Проверяется в контрольных
    точках (перед/после каждой стадии); при срабатывании поднимается
    :class:`ProcessingCancelled`, а дочерние процессы стадии (если есть)
    гасятся. Уже завершённые стадии к этому моменту лежат в стадийном кэше —
    повторный запуск их переиспользует.
    """

    emit = on_progress or (lambda _event: None)
    config.ensure_output_dir()
    device = device or resolve_device(config.device)
    recognizer = recognizer or _build_recognizer(config, device, on_progress=emit)

    # ``mark_overlap`` управляет и пометкой зон наложения, и сбором
    # ``extra_speakers`` в объединителе: при выключенном режиме реплики остаются
    # с одним говорящим, как раньше.
    merger = merger or OverlapSegmentMerger(mark_overlap=config.mark_overlap)
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
    # Единый декодированный 16-кГц waveform для всех стадий, которым нужно
    # аудио: диаризации, enrollment и извлечения образцов. Денойз может отдать
    # его сразу; иначе он декодируется лениво **один раз** при первом обращении
    # и переиспользуется, поэтому повторных декодов одного файла нет.
    shared_waveform: _SharedWaveform | None = None
    overlaps: list[SpeakerOverlap] = []
    # Имена, сопоставленные говорящим по образцам голоса (enrollment).
    # Приоритетнее переименования по индексу (``--speaker-name``).
    enrollment_names: dict[str, str] = {}
    try:
        _ensure_not_cancelled(cancel_event, "перед шумоподавлением")
        if denoiser is not None:
            logger.info("Шумоподавление (DeepFilterNet)...")
            emit(ProgressEvent("denoise", "Шумоподавление", fraction=None))
            audio_path = denoiser.denoise(config.input_file)
            # Явный сигнал попадания в кэш: веб-слой помечает такие стадии как
            # «из кэша» (по ``detail``), не полагаясь на эвристику по времени.
            if getattr(denoiser, "last_hit", False):
                emit(ProgressEvent("denoise", "Шумоподавление", fraction=None, detail="из кэша"))
        # Общий waveform нужен только когда диаризация включена: без неё
        # говорящих нет, поэтому enrollment и извлечение образцов ничего не
        # декодируют. При попадании денойза в кэш ``last_waveform`` = None —
        # тогда waveform декодируется из файла один раз и переиспользуется.
        if config.diarization_enabled:
            shared_waveform = _SharedWaveform(
                audio_path,
                getattr(denoiser, "last_waveform", None) if denoiser is not None else None,
            )
        _ensure_not_cancelled(cancel_event, "после шумоподавления")

        _ensure_not_cancelled(cancel_event, "перед распознаванием речи")
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
            try:
                transcription_segments, language, duration = recognizer.transcribe(
                    audio_path, language=config.language
                )
            except Exception:
                # При отмене whisper-cli убит нами — это не сбой распознавания,
                # а прерывание: поднимаем ProcessingCancelled. Без отмены —
                # пробрасываем исходную ошибку как есть.
                _ensure_not_cancelled(cancel_event, "распознавание речи")
                raise
            cache.save(
                "asr", asr_key, asr_payload(transcription_segments, language, duration)
            )
        _ensure_not_cancelled(cancel_event, "после распознавания речи")

        # whisper.cpp сам использует GPU через Vulkan; pyannote.audio в этом
        # гибриде не имеет GPU-бэкенда (ROCm не поддерживает старые AMD-карты),
        # поэтому диаризация всегда выполняется на CPU.
        diarization_device = (
            Device.CPU if config.asr_backend is AsrBackend.WHISPER_CPP else device
        )
        _ensure_not_cancelled(cancel_event, "перед диаризацией")
        # Enrollment по образцам возможен не у всех движков: у EEND-модели
        # nemo-speech нет per-speaker эмбеддингов. Флаг выставляется при
        # создании активного движка.
        enrollment_supported = True
        if config.diarization_enabled:
            # ``audio_path`` нужен режиму ``auto``: дешёвый оценщик числа
            # говорящих выбирает nemo-speech (<= лимита) или pyannote (#64).
            active_diarizer = diarizer or create_diarizer(
                config,
                diarization_device,
                audio_path=audio_path,
                on_progress=emit,
            )
            enrollment_supported = getattr(active_diarizer, "supports_enrollment", True)
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
                try:
                    speaker_segments = active_diarizer.diarize(
                        audio_path,
                        num_speakers=config.num_speakers,
                        min_speakers=config.min_speakers,
                        max_speakers=config.max_speakers,
                        waveform=(
                            shared_waveform.get() if shared_waveform is not None else None
                        ),
                    )
                except Exception:
                    # Диаризация не порождает дочерних процессов, но отмена во
                    # время неё должна приводить к ProcessingCancelled, а не к
                    # ошибке: проверяем флаг и, если он снят, пробрасываем сбой.
                    _ensure_not_cancelled(cancel_event, "определение говорящих")
                    raise
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
            if not enrollment_supported:
                logger.warning(
                    "Enrollment по образцам голоса недоступен для движка "
                    "%s: у EEND-модели Sortformer нет per-speaker эмбеддингов — "
                    "имена говорящих по образцам не присваиваются. Используйте "
                    "движок pyannote или --speaker-name ИНДЕКС=Имя.",
                    type(active_diarizer).__name__,
                )
            else:
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
                    waveform=shared_waveform.get() if shared_waveform is not None else None,
                    prepare=ReferencePrepareOptions(
                        enabled=config.reference_prepare,
                        min_speech_seconds=config.enrollment_min_sample_seconds,
                        max_seconds=config.enrollment_max_sample_seconds,
                        target_dbfs=config.reference_target_dbfs,
                    ),
                )
        _ensure_not_cancelled(cancel_event, "после сопоставления голосов")
    finally:
        # Временный денойзенный WAV нужен только ASR и диаризации; удаляем его,
        # как только оба этапа завершились (в т.ч. при ошибке).
        close = getattr(denoiser, "close", None)
        if callable(close):
            close()

    _ensure_not_cancelled(cancel_event, "перед объединением сегментов")
    emit(ProgressEvent("merge", "Объединение сегментов", fraction=None))
    known_speakers = {**config.speaker_names, **enrollment_names}
    entries, speakers = merger.merge(transcription_segments, speaker_segments, known_speakers)

    # Чистим неречевые пометки Whisper ([СМЕХ], [BLANK_AUDIO], ♪ и т.п.),
    # схлопываем зацикленные повторы и аккуратно нормализуем текст — всё до
    # склейки предложений, чтобы корректор, LLM и экспорт работали с готовым
    # текстом. Реплики, состоящие только из пометок, здесь же отбрасываются.
    _ensure_not_cancelled(cancel_event, "перед очисткой артефактов")
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

    # Сов-говорящие и признак наложения — из зон перекрытий (они несут
    # участников), вычисляются после склейки: интервалы реплик уже финальные.
    # Дополняет extras, собранные объединителем из перекрывающихся
    # ``speaker_segments`` (для движков, отдающих неэксклюзивную разметку).
    if config.mark_overlap and overlaps:
        entries, speakers = apply_overlap_regions(
            entries, speakers, overlaps, known_speakers=known_speakers
        )

    _ensure_not_cancelled(cancel_event, "перед автоисправлением")
    if corrector is not None:
        logger.info("Автоисправление опечаток (только неизвестные словоформы)...")
        emit(ProgressEvent("correction", "Автоисправление опечаток", fraction=None))
        entries = corrector.correct(entries)

    participants = None
    summary = None
    _ensure_not_cancelled(cancel_event, "перед LLM-постобработкой")
    if config.llm_enabled:
        from audio_transcriber.llm.postprocess import run_llm_postprocess

        logger.info("LLM-постобработка (имена участников, правка терминов, резюме)...")
        try:
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
        except Exception:
            # llama-server мог быть погашен нами при отмене — это прерывание,
            # а не сбой LLM.
            _ensure_not_cancelled(cancel_event, "LLM-постобработка")
            raise

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

    _ensure_not_cancelled(cancel_event, "перед экспортом")
    if config.protocol_auto:
        for export_format in config.export_formats:
            _ensure_not_cancelled(cancel_event, f"экспорт {export_format.value}")
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
        # Образцы голоса извлекаем из того же декодированного аудио, что шло в
        # диаризацию/enrollment (денойзенный waveform), — повторного декода нет.
        # Если общего waveform не было (диаризация отключена), берём файл.
        # Любая ошибка здесь не должна ронять конвейер: экспорт уже выполнен.
        sample_audio = audio_path if audio_path.is_file() else config.input_file
        emit(ProgressEvent("export", "Образцы голоса", fraction=None))
        try:
            written = extract_speaker_samples(
                result,
                audio_path=sample_audio,
                output_dir=config.output_dir,
                waveform=shared_waveform.get() if shared_waveform is not None else None,
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
