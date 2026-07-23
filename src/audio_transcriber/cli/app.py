"""Точка входа CLI-приложения на Typer."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from audio_transcriber import __version__
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.enums import Device, ExportFormat
from audio_transcriber.pipeline import run_pipeline
from audio_transcriber.utils.device import resolve_device
from audio_transcriber.utils.exceptions import AudioTranscriberError
from audio_transcriber.utils.hotwords import build_hotwords
from audio_transcriber.utils.logging import setup_logging

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
        help="Путь к аудиофайлу для транскрибации (например, звонок в .mp3).",
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
    speaker_name: list[str] = typer.Option(
        [],
        "--speaker-name",
        help=(
            "Пользовательское имя говорящего в формате ИНДЕКС=Имя "
            "(например: --speaker-name 0=Иван). Можно указать несколько раз."
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
    enable_correction: bool = typer.Option(
        False,
        "--enable-correction",
        help=(
            "Включить автоисправление опечаток ASR. Исправляются только слова, "
            "неизвестные морфологическому анализатору русского языка. "
            "По умолчанию выключено."
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
            speaker_names=AppConfig.parse_speaker_names(speaker_name),
            hf_token=hf_token,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            enable_correction=enable_correction,
            verbose=verbose,
        )
        resolved_device = resolve_device(config.device)

        logger.info("Входной файл: %s", config.input_file)
        logger.info("Директория результатов: %s", config.output_dir)
        logger.info("Модель распознавания: %s", config.model_name)
        logger.info("Язык: %s", config.language or "автоопределение")
        logger.info(
            "Устройство: %s (запрошено: %s)", resolved_device.value, config.device.value
        )
        logger.info(
            "Форматы экспорта: %s", ", ".join(fmt.value for fmt in config.export_formats)
        )
        logger.info(
            "Количество говорящих: %s", config.num_speakers or "автоопределение"
        )
        if config.speaker_names:
            logger.info("Пользовательские имена говорящих: %s", config.speaker_names)
        logger.info(
            "Автоисправление опечаток: %s",
            "включено" if config.enable_correction else "выключено",
        )

        result = run_pipeline(config, device=resolved_device)
    except KeyboardInterrupt:
        logger.warning("Обработка прервана пользователем (Ctrl+C)")
        raise typer.Exit(code=130) from None
    except AudioTranscriberError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info(
        "Готово: %d реплик(и), %d говорящих", len(result.entries), len(result.speakers)
    )


if __name__ == "__main__":
    app()
