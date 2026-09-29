"""Точка входа CLI-приложения на Typer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from audio_transcriber import __version__
from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_GLOSSARY_DB,
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
)
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.correction.defaults import (
    DEFAULT_CORRECTION_MAX_CANDIDATES,
    DEFAULT_CORRECTION_MIN_SIMILARITY,
    DEFAULT_CORRECTION_MIN_WORD_LENGTH,
)
from audio_transcriber.domain.enums import AsrBackend, Device, ExportFormat
from audio_transcriber.llm.client import DEFAULT_CONTEXT_SIZE as DEFAULT_LLM_CONTEXT_SIZE
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.utils.device import resolve_device
from audio_transcriber.utils.exceptions import AudioTranscriberError
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple
from audio_transcriber.utils.hotwords import build_hotwords
from audio_transcriber.utils.logging import setup_logging
from audio_transcriber.utils.notifications import notify

if TYPE_CHECKING:
    from audio_transcriber.storage.glossary_db import ImportReport

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="audio-transcriber",
    help=(
        "Локальная CLI-утилита для транскрибации аудиозаписей телефонных "
        "разговоров с разделением говорящих (speaker diarization)."
    ),
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(f"audio-transcriber {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Показать версию приложения и выйти.",
    ),
) -> None:
    """AudioTranscriptor — локальная транскрибация разговоров с разметкой говорящих."""


@app.command()
def transcribe(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Путь к аудио- или видеофайлу для транскрибации (mp3, wav, mp4, webm, ...).",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        "-o",
        help="Директория для сохранения результатов (будет создана при отсутствии).",
    ),
    model: str = typer.Option(
        "large-v3-turbo",
        "--model",
        "-m",
        help=(
            "Модель faster-whisper (tiny, base, small, medium, large-v3-turbo, "
            "large-v3 и т.д.). large-v3-turbo — оптимальный баланс точности и "
            "скорости для большинства видеокарт."
        ),
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Код языка речи (ru, en, ...). По умолчанию — автоопределение.",
    ),
    device: Device = typer.Option(
        Device.AUTO.value,
        "--device",
        "-d",
        case_sensitive=False,
        help="Устройство для вычислений: cuda, cpu или auto.",
    ),
    export_format: list[ExportFormat] = typer.Option(
        [ExportFormat.TXT.value],
        "--format",
        "-f",
        case_sensitive=False,
        help="Формат(ы) экспорта результата. Можно указать несколько раз.",
    ),
    num_speakers: int | None = typer.Option(
        None,
        "--num-speakers",
        "-n",
        min=1,
        help="Точное количество говорящих, если оно известно заранее.",
    ),
    diarization: bool = typer.Option(
        True,
        "--diarization/--no-diarization",
        help=(
            "Размечать говорящих (диаризация). При --no-diarization конвейер "
            "идёт без спикеров: локальная модель диаризации и токен Hugging Face "
            "не нужны."
        ),
    ),
    speaker_name: list[str] = typer.Option(
        [],
        "--speaker-name",
        help=(
            "Пользовательское имя говорящего в формате ИНДЕКС=Имя "
            "(например: --speaker-name 0=Иван). Можно указать несколько раз."
        ),
    ),
    speaker_reference: list[str] = typer.Option(
        [],
        "--speaker-reference",
        help=(
            "Образец голоса участника в формате Имя=путь.wav. Говорящие "
            "сопоставляются с именами по голосу (косинусное сходство "
            "эмбеддингов), а не по индексу. Можно указать несколько раз, в том "
            "числе несколько образцов на одно имя. Приоритетнее --speaker-name."
        ),
    ),
    enrollment_min_similarity: float = typer.Option(
        DEFAULT_ENROLLMENT_MIN_SIMILARITY,
        "--enrollment-min-similarity",
        help=(
            "Порог косинусного сходства [-1; 1] для присвоения имени по образцу "
            f"голоса. Ниже порога говорящий остаётся «Спикер N». "
            f"По умолчанию {DEFAULT_ENROLLMENT_MIN_SIMILARITY}."
        ),
    ),
    voices_dir: Path | None = typer.Option(
        None,
        "--voices-dir",
        help=(
            "Каталог-библиотека образцов голоса: каждый <Имя>.wav считается "
            "образцом участника и добавляется к --speaker-reference. "
            "По умолчанию ./voices (если каталог существует)."
        ),
    ),
    speaker_samples: bool = typer.Option(
        True,
        "--speaker-samples/--no-speaker-samples",
        help=(
            "Сохранять по одному образцу голоса на говорящего рядом с "
            "результатами (<файл>.speakers/<Имя>.wav) для последующего "
            "enrollment. По умолчанию включено."
        ),
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        help=(
            "Токен доступа Hugging Face для модели диаризации. "
            "По умолчанию берётся из переменной окружения HF_TOKEN."
        ),
    ),
    pyannote_local_model: Path | None = typer.Option(
        None,
        "--pyannote-local-model",
        help=(
            "Путь к локальной копии модели диаризации (директория с config.yaml). "
            "Позволяет работать полностью офлайн, без токена и сети."
        ),
    ),
    initial_prompt: str | None = typer.Option(
        None,
        "--initial-prompt",
        help=(
            "Подсказка для распознавания речи: имена участников, термины, "
            "написание которых модель часто путает с похожими по звучанию словами."
        ),
    ),
    hotwords: str | None = typer.Option(
        None,
        "--hotwords",
        help=(
            "Короткий список слов для подсказки ASR во время распознавания "
            "(ограничен ~100 токенами)."
        ),
    ),
    clean_artifacts: bool = typer.Option(
        True,
        "--clean/--no-clean",
        help=(
            "Удалять неречевые пометки Whisper ([СМЕХ], [АПЛОДИСМЕНТЫ], "
            "[BLANK_AUDIO], (аплодисменты), музыкальные символы ♪ и т.п.). "
            "По умолчанию включено."
        ),
    ),
    denoise: bool = typer.Option(
        True,
        "--denoise/--no-denoise",
        help=(
            "Шумоподавление (DeepFilterNet) перед распознаванием и диаризацией. "
            "По умолчанию включено; при отсутствии DeepFilterNet этап "
            "пропускается без ошибки."
        ),
    ),
    collapse_repeats: bool = typer.Option(
        True,
        "--collapse-repeats/--no-collapse-repeats",
        help=(
            "Схлопывать подряд идущие одинаковые/почти одинаковые реплики "
            "(зацикливания Whisper: «Продолжение следует» ×N и т.п.). "
            "По умолчанию включено."
        ),
    ),
    repeat_min_words: int = typer.Option(
        DEFAULT_REPEAT_MIN_WORDS,
        "--repeat-min-words",
        min=1,
        help=(
            "Минимальная длина (в словах) повторяющейся реплики для схлопывания. "
            f"Защищает короткую осмысленную речь («да, да»). "
            f"По умолчанию {DEFAULT_REPEAT_MIN_WORDS}."
        ),
    ),
    repeat_similarity: float = typer.Option(
        DEFAULT_REPEAT_SIMILARITY,
        "--repeat-similarity",
        help=(
            "Порог сходства «почти одинаковых» реплик (0; 1] "
            f"(по умолчанию {DEFAULT_REPEAT_SIMILARITY})."
        ),
    ),
    normalize_text: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help=(
            "Безопасная нормализация текста: повторная пунктуация, лишние "
            "многоточия, пробелы. Слова не переписываются. По умолчанию включена."
        ),
    ),
    mark_overlap: bool = typer.Option(
        True,
        "--overlap/--no-overlap",
        help=(
            "Помечать реплики, попавшие в зоны наложения речи (одновременно "
            "говорили несколько человек). По умолчанию включено; при "
            "недоступности данных о перекрытиях пометок не будет."
        ),
    ),
    cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help=(
            "Постадийный кэш дорогих этапов (денойз/ASR/диаризация): повторный "
            "запуск на том же файле с теми же параметрами не пересчитывает их. "
            "Это же даёт возобновление после сбоя. По умолчанию включён."
        ),
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Очистить каталог кэша результатов перед запуском.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Каталог постадийного кэша. По умолчанию <output_dir>/.cache.",
    ),
    notifications: bool = typer.Option(
        True,
        "--notify/--no-notify",
        help=(
            "Показать десктоп-уведомление (notify-send) по завершении обработки. "
            "Если утилиты нет — уведомление тихо пропускается. По умолчанию включено."
        ),
    ),
    timeline: bool = typer.Option(
        True,
        "--timeline/--no-timeline",
        help=(
            "Строить таймлайн «кто когда говорил»: HTML (<имя>.timeline.html рядом "
            "с результатами) и подробную текстовую сводку в лог. По умолчанию "
            "включено; без данных диаризации таймлайн мягко пропускается."
        ),
    ),
    low_confidence_threshold: float = typer.Option(
        DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        "--low-confidence-threshold",
        help=(
            "Порог низкой уверенности ASR (<= 0): реплики со средним "
            "avg_logprob ниже порога помечаются в экспорте. "
            f"По умолчанию {DEFAULT_LOW_CONFIDENCE_THRESHOLD}."
        ),
    ),
    enable_correction: bool = typer.Option(
        False,
        "--enable-correction",
        help=(
            "Включить автоисправление опечаток ASR. Исправляются только слова, "
            "неизвестные морфологическому анализатору русского языка. "
            "По умолчанию выключено."
        ),
    ),
    correction_min_word_length: int = typer.Option(
        DEFAULT_CORRECTION_MIN_WORD_LENGTH,
        "--correction-min-word-length",
        min=1,
        help=(
            "Минимальная длина слова для автоисправления "
            f"(по умолчанию {DEFAULT_CORRECTION_MIN_WORD_LENGTH})."
        ),
    ),
    correction_min_similarity: float = typer.Option(
        DEFAULT_CORRECTION_MIN_SIMILARITY,
        "--correction-min-similarity",
        help=(
            "Минимальное сходство опечатки и кандидата (0; 1] "
            f"(по умолчанию {DEFAULT_CORRECTION_MIN_SIMILARITY})."
        ),
    ),
    correction_max_candidates: int = typer.Option(
        DEFAULT_CORRECTION_MAX_CANDIDATES,
        "--correction-max-candidates",
        min=1,
        help=(
            "Максимум кандидатов OpenCorpora на один префикс "
            f"(по умолчанию {DEFAULT_CORRECTION_MAX_CANDIDATES})."
        ),
    ),
    asr_backend: AsrBackend = typer.Option(
        AsrBackend.FASTER_WHISPER.value,
        "--asr-backend",
        case_sensitive=False,
        help=(
            "Движок распознавания: faster-whisper (CUDA/CPU через PyTorch) "
            "или whisper-cpp (GPU через Vulkan — для AMD-карт без ROCm)."
        ),
    ),
    whisper_cpp_model: Path | None = typer.Option(
        None,
        "--whisper-cpp-model",
        help="Путь к ggml-модели для бэкенда whisper-cpp (обязателен при whisper-cpp).",
    ),
    whisper_cpp_binary: str = typer.Option(
        "whisper-cli",
        "--whisper-cpp-binary",
        help="Путь или имя бинарника whisper-cli (по умолчанию whisper-cli).",
    ),
    whisper_cpp_lib_path: str | None = typer.Option(
        None,
        "--whisper-cpp-lib-path",
        help=(
            "Каталог с библиотеками whisper.cpp (задаётся как LD_LIBRARY_PATH "
            "при запуске бинарника)."
        ),
    ),
    whisper_cpp_threads: int | None = typer.Option(
        None,
        "--whisper-cpp-threads",
        min=1,
        help="Число потоков для whisper.cpp (по умолчанию — значение самого бинарника).",
    ),
    llm: bool = typer.Option(
        False,
        "--llm",
        help=(
            "Включить LLM-постобработку: извлечение имён участников и правка "
            "терминов по глоссарию. Требует локальную модель llama.cpp (см. "
            "--llm-model). По умолчанию выключено."
        ),
    ),
    llm_model: Path | None = typer.Option(
        None,
        "--llm-model",
        help="Путь к GGUF-модели LLM (например, Qwen2.5-7B-Instruct Q4_K_M).",
    ),
    llm_binary: str = typer.Option(
        "llama-server",
        "--llm-binary",
        help="Путь или имя бинарника llama-server (по умолчанию llama-server).",
    ),
    llm_lib_path: str | None = typer.Option(
        None,
        "--llm-lib-path",
        help=(
            "Каталог с библиотеками llama.cpp (задаётся как LD_LIBRARY_PATH при запуске бинарника)."
        ),
    ),
    llm_gpu: bool = typer.Option(
        True,
        "--llm-gpu/--llm-cpu",
        help=(
            "Использовать GPU (Vulkan) для LLM. При --llm-cpu инференс идёт "
            "на CPU. При нехватке VRAM клиент сам деградирует до CPU."
        ),
    ),
    llm_context: int = typer.Option(
        DEFAULT_LLM_CONTEXT_SIZE,
        "--llm-context",
        min=128,
        help=(f"Размер контекста LLM в токенах (по умолчанию {DEFAULT_LLM_CONTEXT_SIZE})."),
    ),
    llm_suggest_terms: bool = typer.Option(
        False,
        "--llm-suggest-terms",
        help=(
            "После обработки собрать термины-кандидаты, которых нет в "
            "глоссарии, и записать их в файл <глоссарий>.suggested.txt."
        ),
    ),
    llm_extract_names: bool = typer.Option(
        False,
        "--llm-names/--llm-no-names",
        help=(
            "ЭКСПЕРИМЕНТАЛЬНО: определять имена участников через LLM "
            "(по умолчанию выключено). Функция ещё нестабильна. "
            "Правка терминов по глоссарию работает независимо."
        ),
    ),
    llm_summary: bool = typer.Option(
        True,
        "--llm-summary/--no-llm-summary",
        help=(
            "Строить резюме встречи локальной LLM (тема, участники, решения, "
            "открытые вопросы, задачи). По умолчанию включено; применяется, "
            "только когда включена LLM-постобработка (--llm)."
        ),
    ),
    llm_prompt_extra: str | None = typer.Option(
        None,
        "--llm-prompt-extra",
        help=(
            "Доп. инструкции к промптам LLM (текстом). Подмешиваются в "
            "системный промпт каждого этапа (резюме/термины/имена)."
        ),
    ),
    llm_prompt_file: Path | None = typer.Option(
        None,
        "--llm-prompt-file",
        help=(
            "Путь к файлу с доп. инструкциями к промптам LLM. Содержимое "
            "добавляется к системному промпту каждого этапа."
        ),
    ),
    glossary: list[str] = typer.Option(
        [],
        "--glossary",
        help=(
            "Путь к файлу(ам) глоссария терминов. Можно указать несколько раз "
            "или перечислить пути через запятую. Термины объединяются."
        ),
    ),
    glossary_db: Path | None = typer.Option(
        None,
        "--glossary-db",
        help=(
            "Путь к SQLite-БД глоссария. По умолчанию glossary.db рядом с "
            "рабочим каталогом (переменная окружения GLOSSARY_DB)."
        ),
    ),
    glossary_enabled: bool = typer.Option(
        True,
        "--glossary-enabled/--no-glossary",
        help="Использовать глоссарий (БД и текстовые файлы). По умолчанию включён.",
    ),
    protocol: bool = typer.Option(
        True,
        "--protocol/--no-protocol",
        help=(
            "Автоматически завершать прогон протоколом: считать резюме LLM и "
            "экспортировать итоговые документы. При --no-protocol прогон "
            "останавливается на готовой стенограмме — файлы не пишутся "
            "(протокол можно собрать отдельно). По умолчанию включено."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Подробный режим логирования (уровень DEBUG).",
    ),
) -> None:
    """Распознать речь в аудиозаписи и экспортировать стенограмму с разметкой говорящих."""

    log_file = setup_logging(verbose=verbose, log_dir=output_dir / "logs")
    if log_file is not None:
        logger.info("Подробные логи сохраняются в файл: %s", log_file)

    try:
        if hotwords:
            hotwords, dropped_terms = build_hotwords(hotwords)
            if dropped_terms:
                logger.warning(
                    "Не поместилось в лимит hotwords ASR и не будет учтено: %s.",
                    ", ".join(dropped_terms),
                )

        config = AppConfig(
            input_file=input_file,
            output_dir=output_dir,
            model_name=model,
            language=language,
            device=device,
            export_formats=tuple(dict.fromkeys(export_format)),
            num_speakers=num_speakers,
            diarization_enabled=diarization,
            speaker_names=AppConfig.parse_speaker_names(speaker_name),
            speaker_references=AppConfig.parse_speaker_references(speaker_reference),
            enrollment_min_similarity=enrollment_min_similarity,
            voices_dir=voices_dir,
            export_speaker_samples=speaker_samples,
            hf_token=hf_token,
            pyannote_local_model=pyannote_local_model,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            clean_artifacts=clean_artifacts,
            collapse_repeats=collapse_repeats,
            repeat_min_words=repeat_min_words,
            repeat_similarity=repeat_similarity,
            normalize_text=normalize_text,
            denoise=denoise,
            mark_overlap=mark_overlap,
            use_cache=cache,
            cache_dir=cache_dir,
            notifications=notifications,
            timeline=timeline,
            low_confidence_threshold=low_confidence_threshold,
            enable_correction=enable_correction,
            correction_min_word_length=correction_min_word_length,
            correction_min_similarity=correction_min_similarity,
            correction_max_candidates=correction_max_candidates,
            verbose=verbose,
            asr_backend=asr_backend,
            whisper_cpp_model=whisper_cpp_model,
            whisper_cpp_binary=whisper_cpp_binary,
            whisper_cpp_lib_path=whisper_cpp_lib_path,
            whisper_cpp_threads=whisper_cpp_threads,
            llm_enabled=llm,
            llm_model=llm_model,
            llm_binary=llm_binary,
            llm_lib_path=llm_lib_path,
            llm_gpu=llm_gpu,
            llm_context_size=llm_context,
            llm_suggest_terms=llm_suggest_terms,
            llm_extract_names=llm_extract_names,
            llm_summary=llm_summary,
            llm_prompt_extra=llm_prompt_extra,
            llm_prompt_file=llm_prompt_file,
            glossary_path=normalize_glossary_paths_tuple(glossary),
            glossary_db=glossary_db,
            glossary_enabled=glossary_enabled,
            protocol_auto=protocol,
        )
        config.ensure_output_dir()
        if clear_cache:
            from audio_transcriber.cache.store import StageCache

            removed = StageCache(config.resolved_cache_dir()).clear()
            logger.info("Кэш очищен: удалено %d файл(ов)", removed)
        resolved_device = resolve_device(config.device)

        logger.info("Входной файл: %s", config.input_file)
        logger.info("Директория результатов: %s", config.output_dir)
        logger.info("Бэкенд распознавания: %s", config.asr_backend.value)
        logger.info("Модель распознавания: %s", config.model_name)
        logger.info("Язык: %s", config.language or "автоопределение")
        logger.info("Устройство: %s (запрошено: %s)", resolved_device.value, config.device.value)
        logger.info("Форматы экспорта: %s", ", ".join(fmt.value for fmt in config.export_formats))
        logger.info(
            "Протокол по завершении: %s",
            "включён" if config.protocol_auto else "выключен (--no-protocol)",
        )
        logger.info("Диаризация: %s", "включена" if config.diarization_enabled else "выключена")
        if config.diarization_enabled:
            logger.info("Количество говорящих: %s", config.num_speakers or "автоопределение")
        if config.speaker_names:
            logger.info("Пользовательские имена говорящих: %s", config.speaker_names)
        resolved_references = config.resolved_speaker_references()
        if resolved_references:
            logger.info(
                "Образцы голоса (enrollment): %s",
                ", ".join(
                    f"{name} ({len(paths)})" for name, paths in resolved_references.items()
                ),
            )
            logger.info("Порог сопоставления по голосу: %.2f", config.enrollment_min_similarity)
        if config.voices_dir is not None or config.resolved_voices_dir().is_dir():
            logger.info("Библиотека голосов: %s", config.resolved_voices_dir())
        logger.info(
            "Сохранение образцов голоса: %s",
            "включено" if config.export_speaker_samples else "выключено",
        )
        logger.info(
            "Очистка неречевых артефактов: %s",
            "включена" if config.clean_artifacts else "выключена",
        )
        logger.info(
            "Схлопывание повторяющихся реплик: %s",
            "включено" if config.collapse_repeats else "выключено",
        )
        logger.info(
            "Нормализация текста: %s",
            "включена" if config.normalize_text else "выключена",
        )
        logger.info(
            "Пометка наложения речи: %s",
            "включена" if config.mark_overlap else "выключена",
        )
        logger.info(
            "Постадийный кэш: %s (%s)",
            "включён" if config.use_cache else "выключен",
            config.resolved_cache_dir(),
        )
        logger.info(
            "Порог низкой уверенности ASR: %.2f",
            config.low_confidence_threshold,
        )
        logger.info(
            "Таймлайн говорящих: %s",
            "включён" if config.timeline else "выключен",
        )
        logger.info(
            "Шумоподавление (DeepFilterNet): %s",
            "включено" if config.denoise else "выключено",
        )
        logger.info(
            "Автоисправление опечаток: %s",
            "включено" if config.enable_correction else "выключено",
        )
        if config.enable_correction:
            logger.info(
                "Параметры автоисправления: min_word_length=%d, "
                "min_similarity=%.2f, max_candidates=%d",
                config.correction_min_word_length,
                config.correction_min_similarity,
                config.correction_max_candidates,
            )
        logger.info(
            "LLM-постобработка: %s",
            "включена" if config.llm_enabled else "выключена",
        )
        if config.llm_enabled:
            logger.info(
                "Параметры LLM: контекст=%d токенов, GPU=%s, "
                "резюме=%s, определение имён=%s, предложения терминов=%s",
                config.llm_context_size,
                "да" if config.llm_gpu else "нет",
                "включено" if config.llm_summary else "выключено",
                "включено" if config.llm_extract_names else "выключено",
                "включены" if config.llm_suggest_terms else "выключены",
            )
            if config.llm_prompt_extra or config.llm_prompt_file:
                logger.info(
                    "Доп. инструкции к промптам LLM: %s",
                    config.llm_prompt_file or "заданы текстом",
                )

        result = run_pipeline(config, device=resolved_device)
    except KeyboardInterrupt:
        logger.warning("Обработка прервана пользователем (Ctrl+C)")
        raise typer.Exit(code=130) from None
    except AudioTranscriberError as exc:
        logger.error(str(exc))
        if notifications:
            notify("Транскрибация не удалась", f"{input_file.name}: {exc}")
        raise typer.Exit(code=1) from exc

    logger.info("Готово: %d реплик(и), %d говорящих", len(result.entries), len(result.speakers))
    if notifications:
        notify(
            "Транскрибация завершена",
            f"{config.input_file.name}: {len(result.entries)} реплик(и), "
            f"{len(result.speakers)} говорящих",
        )


@app.command()
def doctor() -> None:
    """Проверить окружение и показать отчёт со статусами ✓/✗.

    Код возврата 0, если всё критичное в порядке, иначе 1. Команда ничего не
    считает и не загружает моделей — только быстрые проверки.
    """
    import os

    from audio_transcriber import doctor as doctor_module

    config_path, file_env = doctor_module.load_config_env()
    # Переменные окружения процесса могут дополнять/переопределять config.env
    # (например, HF_TOKEN без записи в файл); приоритет — у config.env.
    env = dict(os.environ)
    env.update(file_env)
    checks = doctor_module.run_doctor(config_path, env)
    typer.echo(doctor_module.format_report(checks))
    if doctor_module.has_critical_failures(checks):
        raise typer.Exit(code=1)


@app.command()
def tui() -> None:
    """Открыть интерактивный интерфейс (htop-стиль) для запуска транскрибации."""
    import logging
    import warnings

    # В TUI прогресс показывается в панели, а не в консоли, поэтому
    # приглушаем INFO-логи и шумные предупреждения pyannote/torch,
    # которые иначе портят полноэкранный вывод.
    logging.getLogger().setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", module=r"pyannote\..*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch.*")

    from audio_transcriber.tui.app import TranscriberApp

    TranscriberApp().run()


# --- Подкоманда glossary: локальная БД глоссария ---------------------------

glossary_app = typer.Typer(
    name="glossary",
    help="Управление локальной SQLite-БД глоссария (импорт, список, источники).",
    add_completion=False,
    no_args_is_help=True,
)


def _resolve_db_path(db: Path | None) -> Path:
    """Путь к БД: явный ``--db``, иначе ``GLOSSARY_DB`` или значение по умолчанию."""
    import os

    if db is not None:
        return db
    return Path(os.environ.get("GLOSSARY_DB", DEFAULT_GLOSSARY_DB))


def _print_import_report(report: ImportReport) -> None:
    action = "перезаписан" if report.replaced else "обновлён"
    typer.echo(
        f"Источник «{report.source}» ({report.kind}) {action}: "
        f"добавлено {report.added}, пропущено {report.skipped}, всего {report.total}."
    )


@glossary_app.command("import")
def glossary_import(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Файл глоссария."),
    source: str | None = typer.Option(None, "--source", help="Имя источника (по умолчанию — имя файла)."),
    kind: str | None = typer.Option(
        None,
        "--kind",
        case_sensitive=False,
        help="Тип файла: txt или csv. По умолчанию определяется по расширению.",
    ),
    replace: bool = typer.Option(
        True, "--replace/--no-replace", help="Заменять существующий одноимённый источник."
    ),
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Импортировать глоссарий из .txt или .csv в локальную БД."""
    from audio_transcriber.storage.glossary_db import GlossaryDB

    resolved_kind = (kind or "").strip().casefold()
    if not resolved_kind:
        resolved_kind = "csv" if path.suffix.casefold() == ".csv" else "txt"
    if resolved_kind not in {"txt", "csv"}:
        typer.echo(f"Ошибка: неизвестный тип «{kind}». Допустимо: txt или csv.", err=True)
        raise typer.Exit(code=1)

    try:
        with GlossaryDB(_resolve_db_path(db)) as glossary_db:
            if resolved_kind == "csv":
                report = glossary_db.import_csv(path, source=source, replace=replace)
            else:
                report = glossary_db.import_txt(path, source=source, replace=replace)
    except (OSError, UnicodeError, ValueError) as exc:
        typer.echo(f"Ошибка импорта: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_import_report(report)


@glossary_app.command("list")
def glossary_list(
    source: str | None = typer.Option(None, "--source", help="Фильтр по имени источника."),
    search: str | None = typer.Option(None, "--search", help="Поиск по канону/варианту/заметке."),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Ограничить число строк."),
    enabled_only: bool = typer.Option(
        False, "--enabled-only", help="Показывать только включённые записи."
    ),
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Показать записи глоссария с фильтрами."""
    from audio_transcriber.storage.glossary_db import GlossaryDB

    try:
        with GlossaryDB(_resolve_db_path(db)) as glossary_db:
            entries = glossary_db.list_entries(
                source=source, enabled_only=enabled_only, search=search, limit=limit
            )
            source_names = {src.id: src.name for src in glossary_db.list_sources()}
    except OSError as exc:
        typer.echo(f"Ошибка чтения БД: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not entries:
        typer.echo("Записи не найдены.")
        return

    header = f"{'ID':>5}  {'Канон':<28} {'Ошибочная форма':<22} {'Источник':<20} Вкл"
    typer.echo(header)
    typer.echo("-" * len(header))
    for entry in entries:
        variant = entry.variant or ""
        source_name = source_names.get(entry.source_id or -1, "—")
        enabled = "да" if entry.enabled else "нет"
        typer.echo(
            f"{entry.id:>5}  {entry.canonical:<28} {variant:<22} {source_name:<20} {enabled}"
        )
    typer.echo(f"\nВсего показано: {len(entries)}.")


@glossary_app.command("add")
def glossary_add(
    term: str = typer.Argument(..., help='Термин или пара в виде "ошибочная форма = канон".'),
    source: str = typer.Option("manual", "--source", help="Источник записи."),
    note: str | None = typer.Option(None, "--note", help="Заметка к записи."),
    category: str | None = typer.Option(None, "--category", help="Категория записи."),
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Добавить в глоссарий термин или явную пару."""
    from audio_transcriber.storage.glossary_db import GlossaryDB

    value = term.strip()
    if not value:
        typer.echo("Ошибка: пустой термин.", err=True)
        raise typer.Exit(code=1)
    variant: str | None = None
    canonical = value
    if "=" in value:
        wrong, _, canon = value.partition("=")
        wrong, canon = wrong.strip(), canon.strip()
        if not wrong or not canon:
            typer.echo(
                "Ошибка: некорректная пара. Ожидается «ошибочная форма = канон».",
                err=True,
            )
            raise typer.Exit(code=1)
        variant, canonical = wrong, canon

    try:
        with GlossaryDB(_resolve_db_path(db)) as glossary_db:
            entry_id = glossary_db.add_entry(
                canonical, variant=variant, source=source, note=note, category=category
            )
    except (OSError, ValueError) as exc:
        typer.echo(f"Ошибка добавления: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if variant:
        typer.echo(f"Добавлено (id {entry_id}): «{variant} = {canonical}» (источник «{source}»).")
    else:
        typer.echo(f"Добавлено (id {entry_id}): «{canonical}» (источник «{source}»).")


@glossary_app.command("sources")
def glossary_sources(
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Показать источники глоссария и число записей в них."""
    from audio_transcriber.storage.glossary_db import GlossaryDB

    try:
        with GlossaryDB(_resolve_db_path(db)) as glossary_db:
            sources = glossary_db.list_sources()
            counts = glossary_db.entry_counts()
    except OSError as exc:
        typer.echo(f"Ошибка чтения БД: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not sources:
        typer.echo("Источников нет. Импортируйте глоссарий: glossary import <файл>.")
        return

    header = f"{'Имя':<32} {'Тип':<6} {'Вкл':<5} {'Записей':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for src in sources:
        enabled = "да" if src.enabled else "нет"
        count = counts.get(src.name, 0)
        typer.echo(f"{src.name:<32} {src.kind:<6} {enabled:<5} {count:>8}")


@glossary_app.command("enable")
def glossary_enable(
    source: str = typer.Argument(..., help="Имя источника."),
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Включить источник глоссария."""
    _set_source_enabled(source, True, db)


@glossary_app.command("disable")
def glossary_disable(
    source: str = typer.Argument(..., help="Имя источника."),
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Отключить источник глоссария (термины остаются в БД)."""
    _set_source_enabled(source, False, db)


def _set_source_enabled(source: str, enabled: bool, db: Path | None) -> None:
    """Общий обработчик команд enable/disable."""
    from audio_transcriber.storage.glossary_db import GlossaryDB

    try:
        with GlossaryDB(_resolve_db_path(db)) as glossary_db:
            found = glossary_db.set_source_enabled(source, enabled)
    except OSError as exc:
        typer.echo(f"Ошибка чтения БД: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not found:
        typer.echo(f"Источник «{source}» не найден.", err=True)
        raise typer.Exit(code=1)
    state = "включён" if enabled else "отключён"
    typer.echo(f"Источник «{source}» {state}.")


@glossary_app.command("remove")
def glossary_remove(
    target: str = typer.Argument(..., help="Имя источника или числовой id записи."),
    cascade: bool = typer.Option(
        True,
        "--cascade/--keep-entries",
        help="При удалении источника удалять его записи (иначе — открепить).",
    ),
    db: Path | None = typer.Option(None, "--db", help="Путь к БД глоссария."),
) -> None:
    """Удалить источник глоссария или отдельную запись по id."""
    from audio_transcriber.storage.glossary_db import GlossaryDB

    try:
        with GlossaryDB(_resolve_db_path(db)) as glossary_db:
            if target.strip().isdigit():
                removed = glossary_db.delete_entry(int(target))
                if not removed:
                    typer.echo(f"Запись с id {target} не найдена.", err=True)
                    raise typer.Exit(code=1)
                typer.echo(f"Запись с id {target} удалена.")
                return
            if target not in {src.name for src in glossary_db.list_sources()}:
                typer.echo(f"Источник «{target}» не найден.", err=True)
                raise typer.Exit(code=1)
            affected = glossary_db.delete_source(target, cascade=cascade)
    except OSError as exc:
        typer.echo(f"Ошибка чтения БД: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    suffix = "вместе с записями" if cascade else "(записи сохранены без источника)"
    typer.echo(f"Источник «{target}» удалён: затронуто записей — {affected} {suffix}.")


app.add_typer(glossary_app, name="glossary")


if __name__ == "__main__":
    app()
