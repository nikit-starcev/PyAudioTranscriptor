"""Конфигурация запуска приложения (:class:`AppConfig`).

Собирает и валидирует параметры, переданные через CLI, прежде чем они
попадут в компоненты конвейера (распознавание, диаризация, объединение,
экспорт).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.config.defaults import (
    DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE,
)
from audio_transcriber.config.defaults import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
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
    # Размечать говорящих (диаризация). При False конвейер идёт без спикеров:
    # локальная модель и токен Hugging Face не нужны.
    diarization_enabled: bool = True
    speaker_names: dict[str, str] = field(default_factory=dict)
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
    # Помечать реплики, попавшие в зоны наложения речи (говорят >= 2 человек).
    # Требует обычной (не эксклюзивной) разметки pyannote; иначе мягко
    # пропускается без пометок и без падения.
    mark_overlap: bool = True
    # Постадийный кэш дорогих этапов (шумоподавление, распознавание, диаризация).
    # При повторном запуске на том же файле с теми же параметрами стадии не
    # пересчитываются — это и ускоряет прогоны, и даёт возобновление после сбоя.
    use_cache: bool = True
    # Каталог кэша; None — <output_dir>/.cache.
    cache_dir: Path | None = None
    # Десктоп-уведомление (notify-send) по завершении обработки. Если утилиты
    # нет — тихий no-op (см. audio_transcriber.utils.notifications).
    notifications: bool = True
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
    llm_enabled: bool = False
    llm_model: Path | None = None
    llm_binary: str = "llama-server"
    llm_lib_path: str | None = None
    llm_gpu: bool = True
    llm_context_size: int = DEFAULT_LLM_CONTEXT_SIZE
    llm_suggest_terms: bool = False
    # Определять имена участников через LLM (независимо от правки терминов).
    llm_extract_names: bool = False
    # Резюме встречи локальной LLM. По умолчанию включено и применяется,
    # только когда включена LLM-постобработка (llm_enabled).
    llm_summary: bool = True
    # Доп. инструкции пользователя к промптам LLM: инлайн-текст и/или путь к
    # файлу с инструкциями. Подмешиваются в системный промпт каждого этапа.
    llm_prompt_extra: str | None = None
    llm_prompt_file: Path | None = None
    # Инвариант: после нормализации — всегда кортеж Path. Конструктор
    # принимает одиночный путь, строку со списком (через запятую/os.pathsep)
    # или последовательность (см. ``normalize_glossary_paths_tuple``).
    glossary_path: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        # Нормализуем пути к глоссариям до проверки (тип поля — tuple[Path, ...],
        # поэтому из кода сюда приходит уже кортеж, но CLI/тесты могут передать
        # любой из поддерживаемых форматов).
        self.glossary_path = normalize_glossary_paths_tuple(self.glossary_path)
        self._validate()

    def _validate(self) -> None:
        if not self.input_file.exists():
            raise ConfigurationError(f"Входной файл не найден: {self.input_file}")

        if not self.input_file.is_file():
            raise ConfigurationError(f"Указанный путь не является файлом: {self.input_file}")

        if self.num_speakers is not None and self.num_speakers < 1:
            raise ConfigurationError("Количество говорящих должно быть положительным числом")

        if not isinstance(self.diarization_enabled, bool):
            raise ConfigurationError("DIARIZATION_ENABLED должно быть true или false")

        if not self.export_formats:
            raise ConfigurationError("Не указан ни один формат экспорта")

        if not isinstance(self.clean_artifacts, bool):
            raise ConfigurationError("CLEAN_ARTIFACTS должно быть true или false")

        if not isinstance(self.denoise, bool):
            raise ConfigurationError("DENOISE должно быть true или false")

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

        if not isinstance(self.use_cache, bool):
            raise ConfigurationError("USE_CACHE должно быть true или false")

        if not isinstance(self.notifications, bool):
            raise ConfigurationError("NOTIFICATIONS должно быть true или false")

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

        if not isinstance(self.llm_extract_names, bool):
            raise ConfigurationError("LLM_EXTRACT_NAMES должно быть true или false")

        if not isinstance(self.llm_summary, bool):
            raise ConfigurationError("LLM_SUMMARY должно быть true или false")

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

        if self.llm_enabled and self.llm_model is None:
            logger.warning(
                "LLM включена, но модель не задана (LLM_MODEL/--llm-model) — "
                "LLM-постобработка будет пропущена"
            )

        for glossary_path in self.glossary_path:
            if not glossary_path.is_file():
                raise ConfigurationError(f"Файл глоссария не найден: {glossary_path}")

    def resolved_cache_dir(self) -> Path:
        """Каталог постадийного кэша: ``cache_dir`` или ``<output_dir>/.cache``."""
        return self.cache_dir if self.cache_dir is not None else self.output_dir / ".cache"

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
