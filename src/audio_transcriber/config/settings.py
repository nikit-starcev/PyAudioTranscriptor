"""Конфигурация запуска приложения (:class:`AppConfig`).

Собирает и валидирует параметры, переданные через CLI, прежде чем они
попадут в компоненты конвейера (распознавание, диаризация, объединение,
экспорт).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.config import defaults as config_defaults
from audio_transcriber.config.defaults import (
    DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE,
)
from audio_transcriber.config.defaults import (
    DEFAULT_DEEP_FILTER_BINARY,
    DEFAULT_DIARIZATION_ENGINE,
    DEFAULT_DIARIZATION_ESTIMATE_ENABLED,
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
    DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_ENABLED,
    DEFAULT_DIARIZATION_HYBRID_LINKAGE,
    DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT,
    DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_DIARIZATION_MIN_DURATION_OFF,
    DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS,
    DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS,
    DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS,
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_GIGAAM_MODEL,
    DEFAULT_HYBRID_CONTEXT_SECONDS,
    DEFAULT_HYBRID_LOW_LOGPROB_THRESHOLD,
    DEFAULT_HYBRID_MIN_SEGMENT_SECONDS,
    DEFAULT_HYBRID_NO_SPEECH_THRESHOLD,
    DEFAULT_HYBRID_SILENCE_RMS_THRESHOLD,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_REQUEST_TIMEOUT,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_NEMO_SPEECH_BINARY,
    DEFAULT_NEMO_SPEECH_DEVICE,
    DEFAULT_NEMO_SPEECH_MODEL,
    DEFAULT_REFERENCE_PREPARE,
    DEFAULT_REFERENCE_TARGET_DBFS,
    DEFAULT_SENTENCE_MERGE_MAX_GAP,
    DEFAULT_VOICES_DIR,
    NEMO_SPEECH_MAX_SPEAKERS,
    VALID_DIARIZATION_ENGINES,
    VALID_DIARIZATION_LINKAGES,
    VALID_LLM_PROVIDERS,
    VALID_NEMO_SPEECH_DEVICES,
)
from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.utils.exceptions import ConfigurationError
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple

logger = logging.getLogger(__name__)

_SPEAKER_ID_TEMPLATE = "SPEAKER_{index:02d}"


def _normalize_speaker_references(
    references: Mapping[str, Path | str | Sequence[Path | str]],
) -> dict[str, tuple[Path, ...]]:
    """Приводит образцы голоса к инварианту ``имя -> кортеж путей``.

    Значение может быть одиночным путём/строкой или последовательностью —
    это позволяет задать несколько образцов на одно имя.
    """
    normalized: dict[str, tuple[Path, ...]] = {}
    for raw_name, raw_value in references.items():
        name = raw_name.strip()
        paths: tuple[Path, ...]
        if isinstance(raw_value, (str, Path)):
            paths = (Path(raw_value),)
        else:
            paths = tuple(Path(item) for item in raw_value)
        normalized[name] = paths
    return normalized


@dataclass(slots=True)
class AppConfig:
    """Полная конфигурация одного запуска транскрибации."""

    input_file: Path
    output_dir: Path = Path("output")
    model_name: str = "large-v3-turbo"
    language: str | None = None
    device: Device = Device.AUTO
    export_formats: tuple[ExportFormat, ...] = (ExportFormat.TXT,)
    num_speakers: int | None = None
    # Диапазон числа говорящих для диаризации (аргументы вызова пайплайна
    # pyannote). Заданный ``num_speakers`` приоритетнее: тогда диапазон
    # игнорируется (см. ``_validate``). ``None`` — ограничение не задано.
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Гиперпараметры диаризации pyannote, применяемые через
    # ``pipeline.instantiate(...)`` после загрузки модели. ``min_duration_off``
    # — главный рычаг против дробления реплик (дефолт у нас 0.5, у pyannote 0.0).
    # ``clustering.*`` — грубые рычаги числа говорящих; ``None`` — дефолт модели.
    diarization_min_duration_off: float = DEFAULT_DIARIZATION_MIN_DURATION_OFF
    diarization_clustering_threshold: float | None = None
    diarization_clustering_fb: float | None = None
    # Размечать говорящих (диаризация). При False конвейер идёт без спикеров:
    # локальная модель и токен Hugging Face не нужны.
    diarization_enabled: bool = True
    # Движок диаризации: ``auto`` (pyannote, если nemo-speech не настроен),
    # ``pyannote`` или ``nemo-speech`` (NeMo-Speech.cpp, EEND Sortformer).
    diarization_engine: str = DEFAULT_DIARIZATION_ENGINE
    # --- NeMo-Speech.cpp (#62) ---
    # Путь/имя бинарника ``nemo-speech``, каталог его разделяемых библиотек
    # (``lib/``), модель (имя из каталога, HF-репозиторий или путь к ``.gguf``)
    # и устройство (``auto``/``vulkan``/``cpu``).
    nemo_speech_binary: str = DEFAULT_NEMO_SPEECH_BINARY
    nemo_speech_lib_path: str | None = None
    nemo_speech_model: str = DEFAULT_NEMO_SPEECH_MODEL
    nemo_speech_device: str = DEFAULT_NEMO_SPEECH_DEVICE
    # --- Оценщик числа говорящих и маршрутизация ``auto`` (#64) ---
    # Дешёвый оценщик (sherpa-onnx: silero VAD + эмбеддинги CAM++) выбирает
    # движок для ``auto``: N <= ``diarization_route_max_speakers`` — nemo-speech
    # (быстро), иначе pyannote (точно). ``None``/сбой оценки → pyannote.
    diarization_estimate_enabled: bool = DEFAULT_DIARIZATION_ESTIMATE_ENABLED
    # Сколько секунд речи анализировать (распределённо по записи).
    diarization_estimate_seconds: float = DEFAULT_DIARIZATION_ESTIMATE_SECONDS
    # Порог косинусного расстояния кластеризации эмбеддингов (0; 2).
    diarization_estimate_threshold: float = DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD
    # Имя файла модели в кэше оценщика или путь к локальному .onnx.
    diarization_estimate_model: str = DEFAULT_DIARIZATION_ESTIMATE_MODEL
    # Cap маршрутизации: до него (включительно) ``auto`` берёт nemo-speech.
    diarization_route_max_speakers: int = DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS
    # --- Гибридная диаризация (#64, часть 2) ---
    # Оконный nemo-speech + глобальная склейка говорящих по эмбеддингам.
    # Позволяет обойти лимит Sortformer в 4 спикера; при ``auto`` и N выше cap
    # выбирается гибрид, если доступны бинарник и sherpa-onnx с моделью.
    diarization_hybrid_enabled: bool = DEFAULT_DIARIZATION_HYBRID_ENABLED
    diarization_hybrid_window_seconds: float = DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS
    diarization_hybrid_overlap_seconds: float = DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS
    diarization_hybrid_min_speaker_seconds: float = DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS
    # Linkage и порог пороговой ветки глобальной кластеризации гибрида.
    # ``ward`` (euclidean на L2-нормированных векторах) — лучшее распределение,
    # чем ``complete``/cosine. Порог — в единицах ЕВКЛИДОВА расстояния и НЕ
    # совпадает с косинусным ``diarization_estimate_threshold`` оценщика.
    diarization_hybrid_linkage: str = DEFAULT_DIARIZATION_HYBRID_LINKAGE
    diarization_hybrid_threshold: float = DEFAULT_DIARIZATION_HYBRID_THRESHOLD
    # Переобработка «перегруженных» окон гибрида мелкими окнами (#68).
    diarization_hybrid_overload_split: bool = DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT
    diarization_hybrid_subwindow_seconds: float = DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS
    diarization_hybrid_max_split_depth: int = DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH
    speaker_names: dict[str, str] = field(default_factory=dict)
    # Образцы голоса участников для enrollment-диаризации: имя -> клип(ы).
    # Если заданы и сопоставление уверенное, имя говорящего берётся по голосу
    # и приоритетнее ``speaker_names`` (переименование по индексу). Несколько
    # образцов на одно имя усредняются. Инвариант после нормализации — кортеж.
    speaker_references: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    enrollment_min_similarity: float = DEFAULT_ENROLLMENT_MIN_SIMILARITY
    # Подготовка эталона голоса (#29): VAD-обрезка тишины, ограничение длины
    # 3–10 с речи и лёгкая RMS-нормализация. Одинаково применяется к образцам
    # (при сохранении в библиотеку и при enrollment) и к окнам говорящего.
    reference_prepare: bool = DEFAULT_REFERENCE_PREPARE
    enrollment_min_sample_seconds: float = DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS
    enrollment_max_sample_seconds: float = DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS
    reference_target_dbfs: float = DEFAULT_REFERENCE_TARGET_DBFS
    # Каталог-библиотека образцов голоса: каждый ``<Имя>.wav`` трактуется как
    # образец участника и добавляется к ``speaker_references``. ``None`` —
    # использовать ``./voices`` (если каталог существует). Отсутствие каталога —
    # предупреждение, не ошибка.
    voices_dir: Path | None = None
    # Сохранять по одному образцу голоса на говорящего рядом с результатами
    # (``<output>/<файл>.speakers/<Имя>.wav``) для последующего enrollment.
    export_speaker_samples: bool = True
    hf_token: str | None = None
    pyannote_local_model: Path | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None
    # Удалять неречевые пометки Whisper ([СМЕХ], [BLANK_AUDIO], ♪ и т.п.).
    clean_artifacts: bool = True
    # Схлопывать подряд идущие одинаковые/почти одинаковые реплики
    # (зацикливания Whisper: «Продолжение следует» ×N и т.п.).
    collapse_repeats: bool = True
    repeat_min_words: int = DEFAULT_REPEAT_MIN_WORDS
    repeat_similarity: float = DEFAULT_REPEAT_SIMILARITY
    # Безопасная нормализация текста (пробелы, повторная пунктуация, многоточия).
    normalize_text: bool = True
    # Шумоподавление (DeepFilterNet) перед распознаванием и диаризацией.
    # При отсутствии движка этап мягко пропускается с предупреждением в лог.
    denoise: bool = True
    # Путь/имя внешнего Rust-CLI ``deep-filter`` (DeepFilterNet >= 0.5.6).
    # Пустое значение — ошибка конфигурации; при недоступном бинарнике стадия
    # денойза мягко пропускается.
    deep_filter_binary: str = DEFAULT_DEEP_FILTER_BINARY
    # Помечать реплики, попавшие в зоны наложения речи (говорят >= 2 человек).
    # Требует обычной (не эксклюзивной) разметки pyannote; иначе мягко
    # пропускается без пометок и без падения.
    mark_overlap: bool = True
    # Сводить кластеры, получившие одно и то же уверенное имя (enrollment
    # many-to-one, ручное переименование/``--speaker-name``), в одного
    # говорящего. Безымянные («Спикер N») не сливаются; совпадение имён —
    # точное. По умолчанию включено.
    merge_same_name_speakers: bool = True
    sentence_merge_max_gap: float = DEFAULT_SENTENCE_MERGE_MAX_GAP
    # Постадийный кэш дорогих этапов (шумоподавление, распознавание, диаризация).
    # При повторном запуске на том же файле с теми же параметрами стадии не
    # пересчитываются — это и ускоряет прогоны, и даёт возобновление после сбоя.
    use_cache: bool = True
    # Каталог кэша; None — <output_dir>/.cache.
    cache_dir: Path | None = None
    # Десктоп-уведомление (notify-send) по завершении обработки. Если утилиты
    # нет — тихий no-op (см. audio_transcriber.utils.notifications).
    notifications: bool = True
    # Таймлайн «кто когда говорил»: HTML (<имя>.timeline.html рядом с
    # результатами) плюс подробная текстовая сводка в лог. Строится на этапе
    # экспорта только при наличии данных диаризации; иначе мягко пропускается.
    timeline: bool = True
    # Пословные таймстемпы (#45): собирать слова с временами из токенов ASR
    # (для whisper.cpp — из ``-ojf``; для faster-whisper — из word-режима).
    # По умолчанию включено; при выключении поле ``words`` остаётся пустым.
    word_timestamps: bool = True
    # Порог низкой уверенности ASR: реплики со средним avg_logprob ниже
    # порога помечаются в txt/docx/json. Логвероятности <= 0.
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
    enable_correction: bool = False
    correction_min_word_length: int = DEFAULT_CORRECTION_MIN_WORD_LENGTH
    correction_min_similarity: float = DEFAULT_CORRECTION_MIN_SIMILARITY
    correction_max_candidates: int = DEFAULT_CORRECTION_MAX_CANDIDATES
    verbose: bool = False
    asr_backend: AsrBackend = AsrBackend.FASTER_WHISPER
    whisper_cpp_binary: str = "whisper-cli"
    whisper_cpp_model: Path | None = None
    whisper_cpp_lib_path: str | None = None
    whisper_cpp_threads: int | None = None
    # --- GigaAM v3 (RU) через onnx-asr (#46) ---
    # Имя модели onnx-asr (``gigaam-v3-e2e-rnnt`` и т.п.), локальный каталог
    # модели (``None`` — загрузка с Hugging Face) и квантизация (``int8``/...).
    # Все три входят в ключ кэша ASR.
    gigaam_model: str = DEFAULT_GIGAAM_MODEL
    gigaam_model_path: Path | None = None
    gigaam_quantization: str | None = None
    # Резать длинное аудио встроенным VAD onnx-asr (рекомендуется: у GigaAM
    # ограниченное окно). При недоступной VAD-модели движок мягко деградирует.
    gigaam_vad: bool = True
    # --- Гибридный ASR (#57): «плохие» сегменты основного движка → Whisper ---
    # Включается флагом; дорабатываются только сегменты, не прошедшие пороги.
    hybrid_asr: bool = False
    # Резервный движок для доработки: faster-whisper (модель ``model_name``)
    # или whisper-cpp (модель ``whisper_cpp_model``).
    hybrid_fallback_backend: AsrBackend = AsrBackend.FASTER_WHISPER
    hybrid_low_logprob_threshold: float = DEFAULT_HYBRID_LOW_LOGPROB_THRESHOLD
    hybrid_no_speech_threshold: float = DEFAULT_HYBRID_NO_SPEECH_THRESHOLD
    hybrid_silence_rms_threshold: float = DEFAULT_HYBRID_SILENCE_RMS_THRESHOLD
    hybrid_min_segment_seconds: float = DEFAULT_HYBRID_MIN_SEGMENT_SECONDS
    hybrid_context_seconds: float = DEFAULT_HYBRID_CONTEXT_SECONDS
    llm_enabled: bool = False
    # Провайдер LLM-постобработки: ``llama`` — локальный llama-server
    # (по умолчанию), ``openai`` — внешний OpenAI-совместимый API.
    llm_provider: str = DEFAULT_LLM_PROVIDER
    # Параметры внешнего провайдера (``openai``). Для локального llama-server
    # не используются; ``llm_model`` остаётся путём к GGUF-модели.
    llm_base_url: str | None = None
    llm_model_name: str | None = None
    # API-ключ внешнего провайдера. Секрет: в веб-интерфейсе хранится отдельно
    # (``web-data/secrets.json``) и не попадает в открытом виде в API/логи.
    llm_api_key: str | None = None
    llm_model: Path | None = None
    llm_binary: str = "llama-server"
    llm_lib_path: str | None = None
    llm_gpu: bool = True
    llm_context_size: int = DEFAULT_LLM_CONTEXT_SIZE
    # Таймаут одного HTTP-запроса к llama-server (секунды). Раньше был зашит в
    # клиенте (600 с); теперь настраивается через LLM_REQUEST_TIMEOUT/CLI.
    llm_request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT
    llm_suggest_terms: bool = False
    # Определять имена участников через LLM (независимо от правки терминов).
    llm_extract_names: bool = False
    # Резюме встречи локальной LLM. По умолчанию включено и применяется,
    # только когда включена LLM-постобработка (llm_enabled).
    llm_summary: bool = True
    # Пользовательский шаблон системного промпта резюме (#97). ``None`` —
    # встроенный промпт. Полностью заменяет зашитый формат, когда задан.
    llm_summary_prompt: str | None = None
    # Доп. инструкции пользователя к промптам LLM: инлайн-текст и/или путь к
    # файлу с инструкциями. Подмешиваются в системный промпт каждого этапа.
    llm_prompt_extra: str | None = None
    llm_prompt_file: Path | None = None
    # Инвариант: после нормализации — всегда кортеж Path. Конструктор
    # принимает одиночный путь, строку со списком (через запятую/os.pathsep)
    # или последовательность (см. ``normalize_glossary_paths_tuple``).
    glossary_path: tuple[Path, ...] = ()
    # Локальная SQLite-БД глоссария. ``None`` — путь по умолчанию
    # (``DEFAULT_GLOSSARY_DB`` рядом с рабочим каталогом), см.
    # :meth:`resolved_glossary_db`.
    glossary_db: Path | None = field(
        default_factory=lambda: Path(config_defaults.DEFAULT_GLOSSARY_DB)
    )
    # Использовать ли глоссарий (БД + текстовые файлы) в конвейере.
    glossary_enabled: bool = True
    # Автоматически завершать прогон «протоколом»: считать резюме LLM и
    # экспортировать итоговые документы (txt/docx/...). При ``False`` конвейер
    # останавливается на готовой стенограмме (результат — в памяти вызывающего),
    # а протокол формируется отдельно по запросу (``generate_protocol``).
    # По умолчанию ``True`` — обратная совместимость CLI; TUI ставит ``False``.
    protocol_auto: bool = True

    def __post_init__(self) -> None:
        # Нормализуем пути к глоссариям до проверки (тип поля — tuple[Path, ...],
        # поэтому из кода сюда приходит уже кортеж, но CLI/тесты могут передать
        # любой из поддерживаемых форматов).
        self.glossary_path = normalize_glossary_paths_tuple(self.glossary_path)
        # БД глоссария: CLI может передать строку — приводим к Path.
        if self.glossary_db is not None:
            self.glossary_db = Path(self.glossary_db)
        # Путь к локальной модели GigaAM: CLI/окружение могут передать строку.
        if self.gigaam_model_path is not None:
            self.gigaam_model_path = Path(self.gigaam_model_path)
        # Образцы голоса: допускаем одиночный путь или последовательность на имя.
        self.speaker_references = _normalize_speaker_references(self.speaker_references)
        self._validate()

    def _validate(self) -> None:
        if not self.input_file.exists():
            raise ConfigurationError(f"Входной файл не найден: {self.input_file}")

        if not self.input_file.is_file():
            raise ConfigurationError(f"Указанный путь не является файлом: {self.input_file}")

        if self.num_speakers is not None and self.num_speakers < 1:
            raise ConfigurationError("Количество говорящих должно быть положительным числом")

        self._validate_speaker_range()
        self._validate_diarization_hyperparameters()
        self._validate_diarization_engine()
        self._validate_diarization_routing()

        if not isinstance(self.diarization_enabled, bool):
            raise ConfigurationError("DIARIZATION_ENABLED должно быть true или false")

        if not self.export_formats:
            raise ConfigurationError("Не указан ни один формат экспорта")

        if not isinstance(self.clean_artifacts, bool):
            raise ConfigurationError("CLEAN_ARTIFACTS должно быть true или false")

        if not isinstance(self.glossary_enabled, bool):
            raise ConfigurationError("GLOSSARY_ENABLED должно быть true или false")

        if not isinstance(self.protocol_auto, bool):
            raise ConfigurationError("PROTOCOL_AUTO должно быть true или false")

        if not isinstance(self.denoise, bool):
            raise ConfigurationError("DENOISE должно быть true или false")

        if (
            not isinstance(self.deep_filter_binary, str)
            or not self.deep_filter_binary.strip()
        ):
            raise ConfigurationError("DEEP_FILTER_BINARY должно быть непустой строкой")
        self.deep_filter_binary = self.deep_filter_binary.strip()

        if not isinstance(self.collapse_repeats, bool):
            raise ConfigurationError("COLLAPSE_REPEATS должно быть true или false")

        if not isinstance(self.normalize_text, bool):
            raise ConfigurationError("NORMALIZE_TEXT должно быть true или false")

        if self.repeat_min_words < 1:
            raise ConfigurationError("REPEAT_MIN_WORDS должно быть целым числом >= 1")

        if not (0.0 < self.repeat_similarity <= 1.0):
            raise ConfigurationError("REPEAT_SIMILARITY должно быть числом в диапазоне (0; 1]")

        if not isinstance(self.mark_overlap, bool):
            raise ConfigurationError("MARK_OVERLAP должно быть true или false")

        if not isinstance(self.merge_same_name_speakers, bool):
            raise ConfigurationError(
                "MERGE_SAME_NAME_SPEAKERS должно быть true или false"
            )

        if isinstance(self.sentence_merge_max_gap, bool) or not isinstance(
            self.sentence_merge_max_gap, (int, float)
        ):
            raise ConfigurationError("SENTENCE_MERGE_MAX_GAP должно быть числом")
        if self.sentence_merge_max_gap <= 0.0:
            raise ConfigurationError(
                "SENTENCE_MERGE_MAX_GAP должно быть положительным числом"
            )

        if not isinstance(self.use_cache, bool):
            raise ConfigurationError("USE_CACHE должно быть true или false")

        if not isinstance(self.notifications, bool):
            raise ConfigurationError("NOTIFICATIONS должно быть true или false")

        if not isinstance(self.timeline, bool):
            raise ConfigurationError("TIMELINE должно быть true или false")

        if not isinstance(self.word_timestamps, bool):
            raise ConfigurationError("WORD_TIMESTAMPS должно быть true или false")

        if not isinstance(self.export_speaker_samples, bool):
            raise ConfigurationError("EXPORT_SPEAKER_SAMPLES должно быть true или false")

        if self.voices_dir is not None and not self.voices_dir.is_dir():
            logger.warning(
                "Каталог библиотеки голосов не найден: %s — образцы из него не будут подхвачены",
                self.voices_dir,
            )

        if isinstance(self.low_confidence_threshold, bool) or not isinstance(
            self.low_confidence_threshold, (int, float)
        ):
            raise ConfigurationError("LOW_CONFIDENCE_THRESHOLD должно быть числом")
        if self.low_confidence_threshold > 0.0:
            raise ConfigurationError(
                "LOW_CONFIDENCE_THRESHOLD должно быть числом <= 0 "
                "(логвероятности не превышают нуля)"
            )

        if self.correction_min_word_length < 1:
            raise ConfigurationError("CORRECTION_MIN_WORD_LENGTH должно быть целым числом >= 1")

        if not (0.0 < self.correction_min_similarity <= 1.0):
            raise ConfigurationError(
                "CORRECTION_MIN_SIMILARITY должно быть числом в диапазоне (0; 1]"
            )

        if self.correction_max_candidates < 1:
            raise ConfigurationError("CORRECTION_MAX_CANDIDATES должно быть целым числом >= 1")

        if self.llm_context_size < 128:
            raise ConfigurationError("LLM_CONTEXT должно быть целым числом >= 128")

        if isinstance(self.llm_request_timeout, bool) or not isinstance(
            self.llm_request_timeout, (int, float)
        ):
            raise ConfigurationError("LLM_REQUEST_TIMEOUT должно быть числом")
        if self.llm_request_timeout <= 0.0:
            raise ConfigurationError("LLM_REQUEST_TIMEOUT должно быть положительным числом")

        if not isinstance(self.llm_extract_names, bool):
            raise ConfigurationError("LLM_EXTRACT_NAMES должно быть true или false")

        if not isinstance(self.llm_summary, bool):
            raise ConfigurationError("LLM_SUMMARY должно быть true или false")

        if self.llm_summary_prompt is not None:
            if not isinstance(self.llm_summary_prompt, str):
                raise ConfigurationError("LLM_SUMMARY_PROMPT должно быть текстом")
            # Пустой/пробельный шаблон ничего не меняет — нормализуем в None.
            if not self.llm_summary_prompt.strip():
                self.llm_summary_prompt = None

        if self.llm_prompt_extra is not None:
            if not isinstance(self.llm_prompt_extra, str):
                raise ConfigurationError("LLM_PROMPT_EXTRA должно быть текстом")
            # Пустой/пробельный текст ничего не меняет — нормализуем в None.
            if not self.llm_prompt_extra.strip():
                self.llm_prompt_extra = None

        if self.llm_prompt_file is not None and not self.llm_prompt_file.is_file():
            raise ConfigurationError(
                f"Файл доп. инструкций LLM не найден: {self.llm_prompt_file}"
            )

        if self.asr_backend is AsrBackend.WHISPER_CPP and self.whisper_cpp_model is None:
            raise ConfigurationError(
                "Для бэкенда whisper-cpp необходимо указать путь к ggml-модели "
                "(--whisper-cpp-model)"
            )

        self._validate_gigaam()
        self._validate_hybrid()

        self._validate_llm_provider()

        for glossary_path in self.glossary_path:
            if not glossary_path.is_file():
                raise ConfigurationError(f"Файл глоссария не найден: {glossary_path}")

        if isinstance(self.enrollment_min_similarity, bool) or not isinstance(
            self.enrollment_min_similarity, (int, float)
        ):
            raise ConfigurationError("ENROLLMENT_MIN_SIMILARITY должно быть числом")
        if not (-1.0 <= self.enrollment_min_similarity <= 1.0):
            raise ConfigurationError(
                "ENROLLMENT_MIN_SIMILARITY должно быть числом в диапазоне [-1; 1]"
            )

        self._validate_reference_prepare()

        for name, reference_paths in self.speaker_references.items():
            if not name:
                raise ConfigurationError("В образцах голоса не указано имя участника")
            if not reference_paths:
                raise ConfigurationError(f"Для «{name}» не указан ни один образец голоса")
            for reference_path in reference_paths:
                if not reference_path.is_file():
                    raise ConfigurationError(f"Образец голоса не найден: {reference_path}")

    def _validate_speaker_range(self) -> None:
        """Проверяет диапазон числа говорящих (``min_speakers``/``max_speakers``).

        Границы должны быть положительными, а ``min`` не больше ``max``. Точное
        число говорящих (``num_speakers``) приоритетнее диапазона, поэтому при
        его наличии границы сбрасываются с предупреждением — так поведение
        конвейера и ключ кэша остаются однозначными.
        """
        for name, value in (
            ("MIN_SPEAKERS", self.min_speakers),
            ("MAX_SPEAKERS", self.max_speakers),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"{name} должно быть целым числом")
            if value < 1:
                raise ConfigurationError(f"{name} должно быть положительным числом")

        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ConfigurationError(
                "MIN_SPEAKERS не может быть больше MAX_SPEAKERS "
                f"({self.min_speakers} > {self.max_speakers})"
            )

        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            logger.warning(
                "Задано точное число говорящих (%d) — MIN_SPEAKERS/MAX_SPEAKERS "
                "игнорируются",
                self.num_speakers,
            )
            self.min_speakers = None
            self.max_speakers = None

    def _validate_diarization_hyperparameters(self) -> None:
        """Проверяет гиперпараметры диаризации (pyannote)."""
        min_duration_off = self.diarization_min_duration_off
        if isinstance(min_duration_off, bool) or not isinstance(min_duration_off, (int, float)):
            raise ConfigurationError("DIARIZATION_MIN_DURATION_OFF должно быть числом")
        if min_duration_off < 0.0:
            raise ConfigurationError("DIARIZATION_MIN_DURATION_OFF не может быть отрицательным")

        threshold = self.diarization_clustering_threshold
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ConfigurationError("DIARIZATION_CLUSTERING_THRESHOLD должно быть числом")
            if not (0.0 < threshold <= 1.0):
                raise ConfigurationError(
                    "DIARIZATION_CLUSTERING_THRESHOLD должно быть числом в диапазоне (0; 1]"
                )

        fb = self.diarization_clustering_fb
        if fb is not None:
            if isinstance(fb, bool) or not isinstance(fb, (int, float)):
                raise ConfigurationError("DIARIZATION_CLUSTERING_FB должно быть числом")
            if fb <= 0.0:
                raise ConfigurationError("DIARIZATION_CLUSTERING_FB должно быть положительным числом")

    def _validate_diarization_engine(self) -> None:
        """Проверяет движок диаризации и параметры NeMo-Speech.cpp (#62).

        Движок обязан быть известным (``auto``/``pyannote``/``nemo-speech``),
        устройство — ``auto``/``vulkan``/``cpu``. Лимит Sortformer в 4 спикера
        не является ошибкой конфигурации: превышение лишь предупреждает, что
        верхние границы числа говорящих не будут соблюдены движком.
        """
        engine = (self.diarization_engine or "").strip().casefold()
        if engine not in VALID_DIARIZATION_ENGINES:
            raise ConfigurationError(
                f"Неизвестный движок диаризации: {self.diarization_engine!r} "
                f"(допустимо: {', '.join(VALID_DIARIZATION_ENGINES)})"
            )
        self.diarization_engine = engine

        device = (self.nemo_speech_device or "").strip().casefold()
        if device not in VALID_NEMO_SPEECH_DEVICES:
            raise ConfigurationError(
                f"Неизвестное устройство NeMo-Speech.cpp: {self.nemo_speech_device!r} "
                f"(допустимо: {', '.join(VALID_NEMO_SPEECH_DEVICES)})"
            )
        self.nemo_speech_device = device

        if not isinstance(self.nemo_speech_binary, str) or not self.nemo_speech_binary.strip():
            raise ConfigurationError("NEMO_SPEECH_BINARY должно быть непустой строкой")
        self.nemo_speech_binary = self.nemo_speech_binary.strip()

        if not isinstance(self.nemo_speech_model, str) or not self.nemo_speech_model.strip():
            raise ConfigurationError("NEMO_SPEECH_MODEL должно быть непустой строкой")
        self.nemo_speech_model = self.nemo_speech_model.strip()

        if self.nemo_speech_lib_path is not None:
            # Пустая строка — «не задано»: нормализуем в None.
            self.nemo_speech_lib_path = self.nemo_speech_lib_path.strip() or None

        requested = [
            value
            for value in (self.num_speakers, self.max_speakers)
            if value is not None and value > NEMO_SPEECH_MAX_SPEAKERS
        ]
        # Для ``auto`` предупреждение не нужно: маршрутизация (#64) при числе
        # говорящих выше лимита сама уходит на pyannote. Предупреждаем только
        # при явно выбранном nemo-speech.
        if requested and self.diarization_engine == "nemo-speech":
            logger.warning(
                "Движок диаризации nemo-speech (Sortformer) поддерживает не более "
                "%d спикеров, но запрошено %d — число говорящих будет ограничено "
                "возможностями модели",
                NEMO_SPEECH_MAX_SPEAKERS,
                max(requested),
            )

    def _validate_diarization_routing(self) -> None:
        """Проверяет параметры оценщика числа говорящих и маршрутизации (#64)."""
        if not isinstance(self.diarization_estimate_enabled, bool):
            raise ConfigurationError("DIARIZATION_ESTIMATE_ENABLED должно быть true или false")

        seconds = self.diarization_estimate_seconds
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0.0:
            raise ConfigurationError(
                "DIARIZATION_ESTIMATE_SECONDS должно быть положительным числом"
            )

        threshold = self.diarization_estimate_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not (0.0 < threshold < 2.0)
        ):
            raise ConfigurationError(
                "DIARIZATION_ESTIMATE_THRESHOLD должно быть числом в диапазоне (0; 2)"
            )

        if (
            not isinstance(self.diarization_estimate_model, str)
            or not self.diarization_estimate_model.strip()
        ):
            raise ConfigurationError("DIARIZATION_ESTIMATE_MODEL должно быть непустой строкой")
        self.diarization_estimate_model = self.diarization_estimate_model.strip()

        route_max = self.diarization_route_max_speakers
        if isinstance(route_max, bool) or not isinstance(route_max, int) or route_max < 1:
            raise ConfigurationError(
                "DIARIZATION_ROUTE_MAX_SPEAKERS должно быть целым числом >= 1"
            )

        self._validate_diarization_hybrid()

    def _validate_diarization_hybrid(self) -> None:
        """Проверяет параметры гибридной диаризации (#64, часть 2)."""
        if not isinstance(self.diarization_hybrid_enabled, bool):
            raise ConfigurationError("DIARIZATION_HYBRID_ENABLED должно быть true или false")

        window = self.diarization_hybrid_window_seconds
        if isinstance(window, bool) or not isinstance(window, (int, float)) or window <= 0.0:
            raise ConfigurationError(
                "DIARIZATION_HYBRID_WINDOW_SECONDS должно быть положительным числом"
            )

        overlap = self.diarization_hybrid_overlap_seconds
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)):
            raise ConfigurationError("DIARIZATION_HYBRID_OVERLAP_SECONDS должно быть числом")
        if overlap < 0.0:
            raise ConfigurationError(
                "DIARIZATION_HYBRID_OVERLAP_SECONDS не может быть отрицательным"
            )
        if overlap >= window:
            raise ConfigurationError(
                "DIARIZATION_HYBRID_OVERLAP_SECONDS должно быть меньше "
                "DIARIZATION_HYBRID_WINDOW_SECONDS"
            )

        min_speaker = self.diarization_hybrid_min_speaker_seconds
        if (
            isinstance(min_speaker, bool)
            or not isinstance(min_speaker, (int, float))
            or min_speaker <= 0.0
        ):
            raise ConfigurationError(
                "DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS должно быть положительным числом"
            )

        split = self.diarization_hybrid_overload_split
        if not isinstance(split, bool):
            raise ConfigurationError(
                "DIARIZATION_HYBRID_OVERLOAD_SPLIT должно быть true или false"
            )

        subwindow = self.diarization_hybrid_subwindow_seconds
        if (
            isinstance(subwindow, bool)
            or not isinstance(subwindow, (int, float))
            or subwindow <= 0.0
        ):
            raise ConfigurationError(
                "DIARIZATION_HYBRID_SUBWINDOW_SECONDS должно быть положительным числом"
            )
        if subwindow >= window:
            raise ConfigurationError(
                "DIARIZATION_HYBRID_SUBWINDOW_SECONDS должно быть меньше "
                "DIARIZATION_HYBRID_WINDOW_SECONDS"
            )

        depth = self.diarization_hybrid_max_split_depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ConfigurationError(
                "DIARIZATION_HYBRID_MAX_SPLIT_DEPTH должно быть целым числом >= 0"
            )

        linkage = (self.diarization_hybrid_linkage or "").strip().casefold()
        if linkage not in VALID_DIARIZATION_LINKAGES:
            raise ConfigurationError(
                f"DIARIZATION_HYBRID_LINKAGE должно быть одним из "
                f"{', '.join(sorted(VALID_DIARIZATION_LINKAGES))} (получено "
                f"{self.diarization_hybrid_linkage!r})"
            )
        self.diarization_hybrid_linkage = linkage

        hybrid_threshold = self.diarization_hybrid_threshold
        if (
            isinstance(hybrid_threshold, bool)
            or not isinstance(hybrid_threshold, (int, float))
            or not (0.0 < hybrid_threshold < 2.0)
        ):
            raise ConfigurationError(
                "DIARIZATION_HYBRID_THRESHOLD должно быть числом в диапазоне (0; 2)"
            )

    def _validate_llm_provider(self) -> None:
        """Проверяет провайдера LLM и параметры выбранного режима.

        Провайдер обязан быть известным (``llama``/``openai``) — иначе это
        опечатка, которую лучше поймать сразу. Параметры внешнего провайдера
        (``base_url``/имя модели/ключ) нормализуются: пустые строки — ``None``.
        При включённой LLM нехватка параметров выбранного режима — ошибка
        конфигурации (fail-fast): локальному ``llama`` нужна ``llm_model``,
        внешнему ``openai`` — ``base_url`` и имя модели. Иначе постобработка
        молча пропускалась бы, а пользователь не понимал бы, почему.
        """
        provider = (self.llm_provider or "").strip().casefold()
        if provider not in VALID_LLM_PROVIDERS:
            raise ConfigurationError(
                f"Неизвестный провайдер LLM: {self.llm_provider!r} "
                f"(допустимо: {', '.join(VALID_LLM_PROVIDERS)})"
            )
        self.llm_provider = provider

        if isinstance(self.llm_base_url, str):
            stripped = self.llm_base_url.strip()
            self.llm_base_url = stripped or None
        if isinstance(self.llm_model_name, str):
            stripped = self.llm_model_name.strip()
            self.llm_model_name = stripped or None
        if isinstance(self.llm_api_key, str):
            stripped = self.llm_api_key.strip()
            self.llm_api_key = stripped or None

        if not self.llm_enabled:
            return

        if provider == "llama":
            if self.llm_model is None:
                raise ConfigurationError(
                    "LLM включена (LLM_ENABLED=true), но не задана локальная модель. "
                    "Укажите путь к GGUF-модели (LLM_MODEL/--llm-model) или "
                    "переключитесь на внешний провайдер: LLM_PROVIDER=openai "
                    "с LLM_BASE_URL и LLM_MODEL_NAME."
                )
            return

        missing: list[str] = []
        if not self.llm_base_url:
            missing.append("LLM_BASE_URL/--llm-base-url")
        if not self.llm_model_name:
            missing.append("LLM_MODEL_NAME/--llm-model-name")
        if missing:
            raise ConfigurationError(
                "LLM включена (LLM_ENABLED=true) с провайдером openai, но не "
                f"задано: {', '.join(missing)}. Для локальной модели задайте "
                "LLM_PROVIDER=llama и LLM_MODEL."
            )

        # Внешний провайдер: текст стенограммы уходит за пределы машины —
        # это осознанный выбор пользователя, поэтому фиксируем в логе.
        logger.warning(
            "LLM-провайдер «%s»: стенограмма отправляется на внешний сервер %s — "
            "текст покидает локальную машину (проект заявлен как «100%% локально»).",
            provider,
            self.llm_base_url,
        )

    def _validate_gigaam(self) -> None:
        """Проверяет параметры бэкенда GigaAM (onnx-asr)."""
        if not isinstance(self.gigaam_vad, bool):
            raise ConfigurationError("GIGAAM_VAD должно быть true или false")

        if not isinstance(self.gigaam_model, str) or not self.gigaam_model.strip():
            raise ConfigurationError("GIGAAM_MODEL должно быть непустой строкой")
        self.gigaam_model = self.gigaam_model.strip()

        if self.gigaam_quantization is not None:
            if not isinstance(self.gigaam_quantization, str):
                raise ConfigurationError("GIGAAM_QUANTIZATION должно быть строкой")
            # Пустое значение — «без квантизации»: нормализуем в None, чтобы
            # ключ кэша и вызов onnx-asr были однозначны.
            self.gigaam_quantization = self.gigaam_quantization.strip() or None

    def _validate_hybrid(self) -> None:
        """Проверяет параметры гибридного ASR (#57)."""
        if not isinstance(self.hybrid_asr, bool):
            raise ConfigurationError("HYBRID_ASR должно быть true или false")

        if self.hybrid_low_logprob_threshold > 0.0:
            raise ConfigurationError(
                "HYBRID_LOW_LOGPROB_THRESHOLD должно быть числом <= 0 "
                "(логвероятности не превышают нуля)"
            )
        if not (0.0 <= self.hybrid_no_speech_threshold <= 1.0):
            raise ConfigurationError(
                "HYBRID_NO_SPEECH_THRESHOLD должно быть числом в диапазоне [0; 1]"
            )
        if self.hybrid_silence_rms_threshold < 0.0:
            raise ConfigurationError(
                "HYBRID_SILENCE_RMS_THRESHOLD не может быть отрицательным"
            )
        if self.hybrid_min_segment_seconds <= 0.0:
            raise ConfigurationError(
                "HYBRID_MIN_SEGMENT_SECONDS должно быть положительным числом"
            )
        if self.hybrid_context_seconds < 0.0:
            raise ConfigurationError(
                "HYBRID_CONTEXT_SECONDS не может быть отрицательным"
            )

        if not self.hybrid_asr:
            return

        if self.hybrid_fallback_backend is AsrBackend.WHISPER_CPP and (
            self.whisper_cpp_model is None
        ):
            raise ConfigurationError(
                "Гибридный ASR с резервным бэкендом whisper-cpp требует путь к "
                "ggml-модели (--whisper-cpp-model)"
            )
        if self.hybrid_fallback_backend is self.asr_backend:
            logger.warning(
                "Гибридный ASR включён, но основной и резервный движки совпадают "
                "(%s) — оба прохода выполняет один и тот же движок; "
                "смысл гибрида теряется",
                self.asr_backend.value,
            )

    def _validate_reference_prepare(self) -> None:
        """Проверяет параметры подготовки эталона голоса (#29)."""
        if not isinstance(self.reference_prepare, bool):
            raise ConfigurationError("REFERENCE_PREPARE должно быть true или false")

        for name, value in (
            ("ENROLLMENT_MIN_SAMPLE_SECONDS", self.enrollment_min_sample_seconds),
            ("ENROLLMENT_MAX_SAMPLE_SECONDS", self.enrollment_max_sample_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"{name} должно быть числом")
            if value <= 0.0:
                raise ConfigurationError(f"{name} должно быть положительным числом")

        if self.enrollment_min_sample_seconds > self.enrollment_max_sample_seconds:
            raise ConfigurationError(
                "ENROLLMENT_MIN_SAMPLE_SECONDS не может быть больше "
                "ENROLLMENT_MAX_SAMPLE_SECONDS "
                f"({self.enrollment_min_sample_seconds} > {self.enrollment_max_sample_seconds})"
            )

        if isinstance(self.reference_target_dbfs, bool) or not isinstance(
            self.reference_target_dbfs, (int, float)
        ):
            raise ConfigurationError("REFERENCE_TARGET_DBFS должно быть числом")
        if self.reference_target_dbfs >= 0.0:
            raise ConfigurationError(
                "REFERENCE_TARGET_DBFS должно быть отрицательным (уровень ниже 0 dBFS)"
            )

    def resolved_cache_dir(self) -> Path:
        """Каталог постадийного кэша: ``cache_dir`` или ``<output_dir>/.cache``."""
        return self.cache_dir if self.cache_dir is not None else self.output_dir / ".cache"

    def resolved_glossary_db(self) -> Path:
        """Путь к SQLite-БД глоссария: ``glossary_db`` или значение по умолчанию."""
        if self.glossary_db is not None:
            return self.glossary_db
        return Path(config_defaults.DEFAULT_GLOSSARY_DB)

    def resolved_voices_dir(self) -> Path:
        """Каталог-библиотека образцов голоса: ``voices_dir`` или ``./voices``."""
        return self.voices_dir if self.voices_dir is not None else Path(DEFAULT_VOICES_DIR)

    def resolved_speaker_references(self) -> dict[str, tuple[Path, ...]]:
        """Явные образцы голоса плюс образцы из библиотеки ``voices_dir``.

        Порядок и дедупликация — см.
        :func:`audio_transcriber.diarization.voices.merge_references`.
        """
        from audio_transcriber.diarization.voices import (
            collect_voice_library,
            merge_references,
        )

        return merge_references(
            self.speaker_references,
            collect_voice_library(self.resolved_voices_dir()),
        )

    def ensure_output_dir(self) -> None:
        """Создаёт директорию результатов, если её ещё нет.

        Вызывается перед запуском (а не в конструкторе), чтобы валидация
        конфигурации не выполняла ввод-вывод — в том числе при
        :func:`dataclasses.replace` на каждый файл очереди.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Не удалось создать директорию результатов {self.output_dir}: {exc}"
            ) from exc

    @staticmethod
    def parse_speaker_names(raw_values: list[str]) -> dict[str, str]:
        """Разбирает значения ``--speaker-name`` вида ``"0=Иван"``.

        Возвращает отображение идентификатора говорящего, присваиваемого
        диаризацией (например ``"SPEAKER_00"``), на пользовательское имя.
        """

        mapping: dict[str, str] = {}
        for raw in raw_values:
            if "=" not in raw:
                raise ConfigurationError(
                    f"Некорректный формат --speaker-name: '{raw}'. "
                    "Ожидается ИНДЕКС=Имя, например: --speaker-name 0=Иван"
                )
            index_part, name = (part.strip() for part in raw.split("=", maxsplit=1))
            if not index_part.isdigit():
                raise ConfigurationError(
                    f"Некорректный индекс говорящего в '{raw}': "
                    "ожидалось целое неотрицательное число"
                )
            if not name:
                raise ConfigurationError(f"Не указано имя говорящего в '{raw}'")

            speaker_id = _SPEAKER_ID_TEMPLATE.format(index=int(index_part))
            mapping[speaker_id] = name

        return mapping

    @staticmethod
    def parse_speaker_references(raw_values: list[str]) -> dict[str, tuple[Path, ...]]:
        """Разбирает значения ``--speaker-reference`` вида ``"Иван=путь.wav"``.

        Возвращает отображение имени участника на кортеж путей к образцам
        голоса. Одно и то же имя можно указать несколько раз — тогда у него
        будет несколько образцов (они усредняются при сопоставлении).
        """

        grouped: dict[str, list[Path]] = {}
        for raw in raw_values:
            if "=" not in raw:
                raise ConfigurationError(
                    f"Некорректный формат --speaker-reference: '{raw}'. "
                    "Ожидается Имя=путь.wav, например: --speaker-reference Иван=ivan.wav"
                )
            name_part, path_part = (part.strip() for part in raw.split("=", maxsplit=1))
            if not name_part:
                raise ConfigurationError(f"Не указано имя участника в '{raw}'")
            if not path_part:
                raise ConfigurationError(f"Не указан путь к образцу голоса в '{raw}'")
            grouped.setdefault(name_part, []).append(Path(path_part))

        return {name: tuple(paths) for name, paths in grouped.items()}

