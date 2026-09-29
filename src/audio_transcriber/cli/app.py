"""Точка входа CLI-приложения на Typer."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from audio_transcriber import __version__
from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
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
        logger.info("Диаризация: %s", "включена" if config.diarization_enabled else "выключена")
        if config.diarization_enabled:
            logger.info("Количество говорящих: %s", config.num_speakers or "автоопределение")
        if config.speaker_names:
            logger.info("Пользовательские имена говорящих: %s", config.speaker_names)
        if config.speaker_references:
            logger.info(
                "Образцы голоса (enrollment): %s",
                ", ".join(
                    f"{name} ({len(paths)})" for name, paths in config.speaker_references.items()
                ),
            )
            logger.info("Порог сопоставления по голосу: %.2f", config.enrollment_min_similarity)
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


if __name__ == "__main__":
    app()
