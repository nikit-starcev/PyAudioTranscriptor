"""Конфигурация запуска приложения (:class:`AppConfig`).

Собирает и валидирует параметры, переданные через CLI, прежде чем они
попадут в компоненты конвейера (распознавание, диаризация, объединение,
экспорт).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from audio_transcriber.config.defaults import DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE
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
    llm_extract_names: bool = True
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
