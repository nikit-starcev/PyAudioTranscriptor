"""Точка входа CLI-приложения на Typer."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from audio_transcriber import __version__
from audio_transcriber.cleaning.repetition_filter import (
    DEFAULT_REPEAT_MIN_WORDS,
    DEFAULT_REPEAT_SIMILARITY,
)
from audio_transcriber.cli.env_config import as_bool, collect_env_kwargs
from audio_transcriber.config.defaults import (
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
    DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    DEFAULT_GIGAAM_MODEL,
    DEFAULT_GLOSSARY_DB,
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
    VALID_DIARIZATION_ENGINES,
    VALID_NEMO_SPEECH_DEVICES,
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
from audio_transcriber.utils.config_env import load_config_env
from audio_transcriber.utils.device import resolve_device
from audio_transcriber.utils.exceptions import AudioTranscriberError
from audio_transcriber.utils.glossary_paths import normalize_glossary_paths_tuple
from audio_transcriber.utils.hotwords import build_hotwords
from audio_transcriber.utils.logging import setup_logging
from audio_transcriber.utils.net import (
    DEFAULT_WEB_PORT,
    WEB_PORT_SCAN_LIMIT,
    find_available_port,
    is_port_available,
)
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


#: Соответствие имени CLI-опции имени поля :class:`AppConfig`, если они
#: различаются. Для остальных опций имя поля совпадает с именем параметра.
_PARAM_TO_FIELD: dict[str, str] = {
    "model": "model_name",
    "export_format": "export_formats",
    "diarization": "diarization_enabled",
    "min_duration_off": "diarization_min_duration_off",
    "clustering_threshold": "diarization_clustering_threshold",
    "clustering_fb": "diarization_clustering_fb",
    "speaker_name": "speaker_names",
    "speaker_reference": "speaker_references",
    "speaker_samples": "export_speaker_samples",
    "diarization_estimate": "diarization_estimate_enabled",
    "diarization_hybrid": "diarization_hybrid_enabled",
    "cache": "use_cache",
    "llm": "llm_enabled",
    "llm_context": "llm_context_size",
    "glossary": "glossary_path",
    "protocol": "protocol_auto",
}

#: Параметры, которые не переносятся в :class:`AppConfig` напрямую и
#: обрабатываются отдельно (ввод/вывод, логирование, hotwords, очистка кэша).
_NON_FIELD_PARAMS = frozenset(
    {"input_file", "output_dir", "verbose", "hotwords", "clear_cache"}
)


def _is_explicit(ctx: typer.Context, name: str) -> bool:
    """Пользователь явно задал опцию командной строки (или через envvar)?

    Значения из ``config.env`` опциям не передаются: их подставляет
    :func:`collect_env_kwargs`. Поэтому «явным» считаем всё, кроме дефолта.
    Источник сравниваем по имени: Typer использует собственный форк click, и
    ``ParameterSource`` из обычного ``click`` с ним не совпадает.
    """
    source = ctx.get_parameter_source(name)
    return source is not None and source.name not in {"DEFAULT", "DEFAULT_MAP"}


def _explicit_overrides(
    ctx: typer.Context, values: Mapping[str, object]
) -> dict[str, object]:
    """Явно заданные CLI-опции в виде kwargs для :class:`AppConfig`.

    Только поля, для которых пользователь указал флаг (или envvar), —
    остальные придут из ``config.env``. ``values`` — уже приведённые значения
    параметров (``locals()`` внутри команды: Typer хранит в ``ctx.params``
    сырые строки, а сконвертированные передаёт в callback).
    """
    overrides: dict[str, object] = {}
    for name in ctx.params:
        if name in _NON_FIELD_PARAMS or not _is_explicit(ctx, name):
            continue
        field = _PARAM_TO_FIELD.get(name, name)
        value = values[name]
        if name == "export_format":
            overrides[field] = tuple(dict.fromkeys(value))  # type: ignore[call-overload]
        elif name == "speaker_name":
            overrides[field] = AppConfig.parse_speaker_names(value)  # type: ignore[arg-type]
        elif name == "speaker_reference":
            overrides[field] = AppConfig.parse_speaker_references(value)  # type: ignore[arg-type]
        elif name == "glossary":
            overrides[field] = normalize_glossary_paths_tuple(value)  # type: ignore[arg-type]
        else:
            overrides[field] = value
    return overrides


def _resolve_output_dir(
    default: Path, ctx: typer.Context, env_defaults: dict[str, str]
) -> Path:
    """Каталог результатов: явный ``--output-dir``, иначе ``OUTPUT_DIR``, иначе дефолт."""
    if _is_explicit(ctx, "output_dir"):
        return default
    raw = (env_defaults.get("OUTPUT_DIR") or "").strip()
    return Path(raw) if raw else default


def _resolve_verbose(
    default: bool, ctx: typer.Context, env_defaults: dict[str, str]
) -> bool:
    """Подробное логирование: явный ``--verbose``, иначе ``VERBOSE`` из config.env."""
    if _is_explicit(ctx, "verbose"):
        return default
    return as_bool(env_defaults.get("VERBOSE"), default)


@app.command()
def transcribe(
    ctx: typer.Context,
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
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        min=1,
        help=(
            "Нижняя граница числа говорящих для диаризации (включительно). "
            "Игнорируется, если задано --num-speakers."
        ),
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        min=1,
        help=(
            "Верхняя граница числа говорящих для диаризации (включительно). "
            "Игнорируется, если задано --num-speakers."
        ),
    ),
    min_duration_off: float = typer.Option(
        DEFAULT_DIARIZATION_MIN_DURATION_OFF,
        "--min-duration-off",
        min=0.0,
        help=(
            "Гиперпараметр pyannote segmentation.min_duration_off: паузы короче "
            "этого значения склеиваются внутри реплики. Главный рычаг против "
            "дробления говорящих. "
            f"По умолчанию {DEFAULT_DIARIZATION_MIN_DURATION_OFF} (у pyannote 0.0)."
        ),
    ),
    clustering_threshold: float | None = typer.Option(
        None,
        "--clustering-threshold",
        help=(
            "Гиперпараметр pyannote clustering.threshold (диапазон (0; 1]): порог "
            "решения «один и тот же говорящий». Пусто — значение модели по умолчанию."
        ),
    ),
    clustering_fb: float | None = typer.Option(
        None,
        "--clustering-fb",
        help=(
            "Гиперпараметр pyannote clustering.Fb (> 0): регуляризация "
            "кластеризации; выше — меньше «Спикеров». "
            "Пусто — значение модели по умолчанию."
        ),
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
    diarization_engine: str = typer.Option(
        DEFAULT_DIARIZATION_ENGINE,
        "--diarization-engine",
        envvar="DIARIZATION_ENGINE",
        case_sensitive=False,
        help=(
            "Движок диаризации: auto (pyannote, если nemo-speech не настроен), "
            "pyannote или nemo-speech (NeMo-Speech.cpp, GPU через Vulkan). "
            f"По умолчанию {DEFAULT_DIARIZATION_ENGINE}. "
            f"Допустимо: {', '.join(VALID_DIARIZATION_ENGINES)}."
        ),
    ),
    nemo_speech_binary: str = typer.Option(
        DEFAULT_NEMO_SPEECH_BINARY,
        "--nemo-speech-binary",
        envvar="NEMO_SPEECH_BINARY",
        help=(
            "Путь или имя бинарника nemo-speech (NeMo-Speech.cpp). "
            f"По умолчанию {DEFAULT_NEMO_SPEECH_BINARY} (ищется в PATH)."
        ),
    ),
    nemo_speech_lib_path: str | None = typer.Option(
        None,
        "--nemo-speech-lib-path",
        envvar="NEMO_SPEECH_LIB_PATH",
        help=(
            "Каталог lib/ бандла nemo-speech (подмешивается в LD_LIBRARY_PATH, "
            "если не содержит libstdc++.so.6/libgcc_s.so.1)."
        ),
    ),
    nemo_speech_model: str = typer.Option(
        DEFAULT_NEMO_SPEECH_MODEL,
        "--nemo-speech-model",
        envvar="NEMO_SPEECH_MODEL",
        help=(
            "Модель диаризации: имя из каталога nemo-speech, HF-репозиторий или "
            f"путь к .gguf. По умолчанию {DEFAULT_NEMO_SPEECH_MODEL} "
            "(Sortformer, 4 спикера)."
        ),
    ),
    nemo_speech_device: str = typer.Option(
        DEFAULT_NEMO_SPEECH_DEVICE,
        "--nemo-speech-device",
        envvar="NEMO_SPEECH_DEVICE",
        case_sensitive=False,
        help=(
            "Устройство nemo-speech: auto (по умолчанию), vulkan (GPU AMD/Intel) "
            f"или cpu. Допустимо: {', '.join(VALID_NEMO_SPEECH_DEVICES)}."
        ),
    ),
    diarization_estimate: bool = typer.Option(
        DEFAULT_DIARIZATION_ESTIMATE_ENABLED,
        "--diarization-estimate/--no-diarization-estimate",
        envvar="DIARIZATION_ESTIMATE_ENABLED",
        help=(
            "Дешёвая оценка числа говорящих (sherpa-onnx) для режима auto: до "
            "лимита — nemo-speech (быстро), выше — pyannote (точно). При сбое "
            "оценки auto безопасно выбирает pyannote."
        ),
    ),
    diarization_estimate_seconds: float = typer.Option(
        DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
        "--diarization-estimate-seconds",
        envvar="DIARIZATION_ESTIMATE_SECONDS",
        min=0.1,
        help=(
            "Сколько секунд речи анализировать оценщику (распределённо по всей "
            f"записи). По умолчанию {DEFAULT_DIARIZATION_ESTIMATE_SECONDS}."
        ),
    ),
    diarization_estimate_threshold: float = typer.Option(
        DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
        "--diarization-estimate-threshold",
        envvar="DIARIZATION_ESTIMATE_THRESHOLD",
        min=0.01,
        max=1.99,
        help=(
            "Порог косинусного расстояния кластеризации эмбеддингов оценщика "
            f"(0; 2). По умолчанию {DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD}."
        ),
    ),
    diarization_estimate_model: str = typer.Option(
        DEFAULT_DIARIZATION_ESTIMATE_MODEL,
        "--diarization-estimate-model",
        envvar="DIARIZATION_ESTIMATE_MODEL",
        help=(
            "Модель эмбеддингов оценщика: имя файла в кэше (скачается с релиза "
            "sherpa-onnx) или путь к локальному .onnx."
        ),
    ),
    diarization_route_max_speakers: int = typer.Option(
        DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS,
        "--diarization-route-max-speakers",
        envvar="DIARIZATION_ROUTE_MAX_SPEAKERS",
        min=1,
        help=(
            "Cap маршрутизации auto: до этого числа говорящих выбирается быстрый "
            f"nemo-speech. По умолчанию {DEFAULT_DIARIZATION_ROUTE_MAX_SPEAKERS}."
        ),
    ),
    diarization_hybrid: bool = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_ENABLED,
        "--diarization-hybrid/--no-diarization-hybrid",
        envvar="DIARIZATION_HYBRID_ENABLED",
        help=(
            "Гибридная диаризация для >4 говорящих: оконный nemo-speech + "
            "глобальная склейка говорящих по эмбеддингам (sherpa-onnx). В auto "
            "выбирается при N выше cap, если доступны бинарник и модель."
        ),
    ),
    diarization_hybrid_window_seconds: float = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
        "--diarization-hybrid-window-seconds",
        envvar="DIARIZATION_HYBRID_WINDOW_SECONDS",
        min=1.0,
        help=(
            "Длительность окна гибридной диаризации (секунды). Короче — выше шанс "
            f"уложиться в 4 говорящих, но больше запусков. "
            f"По умолчанию {DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS}."
        ),
    ),
    diarization_hybrid_overlap_seconds: float = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
        "--diarization-hybrid-overlap-seconds",
        envvar="DIARIZATION_HYBRID_OVERLAP_SECONDS",
        min=0.0,
        help=(
            "Перекрытие соседних окон гибрида (секунды): речь на границе не "
            f"теряется. По умолчанию {DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS}."
        ),
    ),
    diarization_hybrid_min_speaker_seconds: float = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
        "--diarization-hybrid-min-speaker-seconds",
        envvar="DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS",
        min=0.1,
        help=(
            "Минимальная суммарная речь локального говорящего в окне (секунды) "
            "для эмбеддинга. Короче — вектор ненадёжен. По умолчанию "
            f"{DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS}."
        ),
    ),
    diarization_hybrid_linkage: str = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_LINKAGE,
        "--diarization-hybrid-linkage",
        envvar="DIARIZATION_HYBRID_LINKAGE",
        help=(
            "Linkage пороговой ветки кластеризации гибрида: ward (euclidean, по "
            "умолчанию — лучшее распределение), complete или average (cosine). "
            f"По умолчанию {DEFAULT_DIARIZATION_HYBRID_LINKAGE}."
        ),
    ),
    diarization_hybrid_threshold: float = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
        "--diarization-hybrid-threshold",
        envvar="DIARIZATION_HYBRID_THRESHOLD",
        min=0.01,
        max=1.99,
        help=(
            "Порог кластеризации гибрида. Для ward — в единицах евклидова "
            "расстояния (0; 2), НЕ совпадает с порогом оценщика. "
            f"По умолчанию {DEFAULT_DIARIZATION_HYBRID_THRESHOLD}."
        ),
    ),
    diarization_hybrid_overload_split: bool = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT,
        "--diarization-hybrid-overload-split/--no-diarization-hybrid-overload-split",
        envvar="DIARIZATION_HYBRID_OVERLOAD_SPLIT",
        help=(
            "Переобрабатывать «перегруженные» окна гибрида (все 4 головы "
            "Sortformer и частая смена говорящего) более мелкими окнами, чтобы "
            "не терять говорящих при >4 в окне."
        ),
    ),
    diarization_hybrid_subwindow_seconds: float = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
        "--diarization-hybrid-subwindow-seconds",
        envvar="DIARIZATION_HYBRID_SUBWINDOW_SECONDS",
        min=1.0,
        help=(
            "Длительность мелкого окна при переобработке перегруженного окна "
            f"(секунды). По умолчанию {DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS}."
        ),
    ),
    diarization_hybrid_max_split_depth: int = typer.Option(
        DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH,
        "--diarization-hybrid-max-split-depth",
        envvar="DIARIZATION_HYBRID_MAX_SPLIT_DEPTH",
        min=0,
        help=(
            "Максимальная глубина рекурсивной нарезки перегруженного окна "
            f"гибрида. По умолчанию {DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH}."
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
    merge_same_name_speakers: bool = typer.Option(
        True,
        "--merge-same-names/--no-merge-same-names",
        help=(
            "Сводить кластеры диаризации с одинаковым уверенным именем "
            "(enrollment/--speaker-name) в одного говорящего. Безымянные "
            "«Спикер N» не сливаются. По умолчанию включено."
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
    word_timestamps: bool = typer.Option(
        True,
        "--word-timestamps/--no-word-timestamps",
        help=(
            "Пословные таймстемпы: собирать слова с временами из токенов ASR "
            "(whisper.cpp — из -ojf; faster-whisper — из word-режима). "
            "По умолчанию включено; при выключении поле words остаётся пустым."
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
            "Движок распознавания: faster-whisper (CUDA/CPU через PyTorch), "
            "whisper-cpp (GPU через Vulkan — для AMD-карт без ROCm) или "
            "gigaam (GigaAM v3 RU через onnx-asr, без torch; опциональная "
            "зависимость onnx-asr[cpu,hub])."
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
    gigaam_model: str = typer.Option(
        DEFAULT_GIGAAM_MODEL,
        "--gigaam-model",
        envvar="GIGAAM_MODEL",
        help=(
            "Имя модели onnx-asr для бэкенда gigaam "
            "(gigaam-v3-e2e-rnnt, gigaam-v3-ctc, gigaam-v3-e2e-ctc, ...). "
            f"По умолчанию {DEFAULT_GIGAAM_MODEL} (пунктуация и заглавные буквы)."
        ),
    ),
    gigaam_model_path: Path | None = typer.Option(
        None,
        "--gigaam-model-path",
        envvar="GIGAAM_MODEL_PATH",
        help=(
            "Локальный каталог с ONNX-моделью GigaAM. Если не задан — модель "
            "скачивается с Hugging Face при первом запуске (нужен onnx-asr[hub])."
        ),
    ),
    gigaam_quantization: str | None = typer.Option(
        None,
        "--gigaam-quantization",
        envvar="GIGAAM_QUANTIZATION",
        help="Квантизация ONNX-модели GigaAM (например, int8). По умолчанию без неё.",
    ),
    gigaam_vad: bool = typer.Option(
        True,
        "--gigaam-vad/--no-gigaam-vad",
        help=(
            "Резать длинное аудио встроенным VAD onnx-asr для GigaAM "
            "(рекомендуется: окно модели 20–30 с). При недоступной VAD-модели "
            "распознавание идёт без VAD. По умолчанию включено."
        ),
    ),
    hybrid_asr: bool = typer.Option(
        False,
        "--hybrid-asr",
        help=(
            "Гибридный ASR: после основного движка «плохие» сегменты (низкая "
            "уверенность, тишина, обрывки) повторно распознаются резервным "
            "движком Whisper. По умолчанию выключено."
        ),
    ),
    hybrid_fallback_backend: AsrBackend = typer.Option(
        AsrBackend.FASTER_WHISPER.value,
        "--hybrid-fallback-backend",
        case_sensitive=False,
        help=(
            "Резервный движок для доработки «плохих» сегментов: faster-whisper "
            "(модель --model) или whisper-cpp (модель --whisper-cpp-model)."
        ),
    ),
    hybrid_low_logprob_threshold: float = typer.Option(
        DEFAULT_HYBRID_LOW_LOGPROB_THRESHOLD,
        "--hybrid-low-logprob-threshold",
        help=(
            "Порог средней логвероятности для доработки (<= 0): ниже — сегмент "
            f"считается «плохим». По умолчанию {DEFAULT_HYBRID_LOW_LOGPROB_THRESHOLD}."
        ),
    ),
    hybrid_no_speech_threshold: float = typer.Option(
        DEFAULT_HYBRID_NO_SPEECH_THRESHOLD,
        "--hybrid-no-speech-threshold",
        help=(
            "Порог вероятности отсутствия речи (Whisper, 0..1): выше — сегмент "
            f"дорабатывается. По умолчанию {DEFAULT_HYBRID_NO_SPEECH_THRESHOLD}."
        ),
    ),
    hybrid_silence_rms_threshold: float = typer.Option(
        DEFAULT_HYBRID_SILENCE_RMS_THRESHOLD,
        "--hybrid-silence-rms-threshold",
        help=(
            "Порог RMS-энергии сегмента (шкала [-1, 1]): ниже — тишина/шум. "
            f"По умолчанию {DEFAULT_HYBRID_SILENCE_RMS_THRESHOLD}."
        ),
    ),
    hybrid_min_segment_seconds: float = typer.Option(
        DEFAULT_HYBRID_MIN_SEGMENT_SECONDS,
        "--hybrid-min-segment-seconds",
        help=(
            "Сегменты короче этого значения дорабатываются резервным движком. "
            f"По умолчанию {DEFAULT_HYBRID_MIN_SEGMENT_SECONDS} с."
        ),
    ),
    hybrid_context_seconds: float = typer.Option(
        DEFAULT_HYBRID_CONTEXT_SECONDS,
        "--hybrid-context-seconds",
        help=(
            "Контекст вокруг «плохого» сегмента при нарезке для Whisper (с); "
            f"результат обрезается границами сегмента. По умолчанию "
            f"{DEFAULT_HYBRID_CONTEXT_SECONDS} с."
        ),
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
    llm_request_timeout: float = typer.Option(
        DEFAULT_LLM_REQUEST_TIMEOUT,
        "--llm-request-timeout",
        help=(
            "Таймаут одного запроса к llama-server в секундах "
            f"(по умолчанию {DEFAULT_LLM_REQUEST_TIMEOUT:g})."
        ),
    ),
    llm_provider: str = typer.Option(
        DEFAULT_LLM_PROVIDER,
        "--llm-provider",
        envvar="LLM_PROVIDER",
        help=(
            "Провайдер LLM-постобработки: llama (локальный llama.cpp, по "
            "умолчанию) или openai (внешний OpenAI-совместимый API — OpenAI, "
            "Ollama, vLLM, LM Studio, OpenRouter). При openai текст уходит за "
            "пределы машины."
        ),
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        envvar="LLM_BASE_URL",
        help=(
            "Базовый URL внешней OpenAI-совместимой LLM (например, "
            "https://api.openai.com/v1 или http://localhost:11434/v1). "
            "Только для --llm-provider openai."
        ),
    ),
    llm_model_name: str | None = typer.Option(
        None,
        "--llm-model-name",
        envvar="LLM_MODEL_NAME",
        help=(
            "Имя модели внешней LLM (например, gpt-4o-mini или llama3.1). "
            "Только для --llm-provider openai."
        ),
    ),
    llm_api_key: str | None = typer.Option(
        None,
        "--llm-api-key",
        envvar="LLM_API_KEY",
        help=(
            "API-ключ внешней LLM (передаётся заголовком Authorization: Bearer). "
            "Необязателен для локальных серверов. По умолчанию — из окружения LLM_API_KEY."
        ),
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
    """Распознать речь в аудиозаписи и экспортировать стенограмму с разметкой говорящих.

    Значения по умолчанию берутся из ``config.env`` (как в веб-интерфейсе и TUI);
    явно указанный флаг перебивает соответствующую настройку из файла.
    """

    # Значения по умолчанию берём из config.env (единый источник настроек, как
    # у веба), а CLI-флаги применяем как явный override — только когда заданы
    # (issue #76). Без флагов поведение CLI совпадает с config.env.
    _, env_defaults = load_config_env()
    resolved_output_dir = _resolve_output_dir(output_dir, ctx, env_defaults)
    resolved_verbose = _resolve_verbose(verbose, ctx, env_defaults)
    # Уведомления могут понадобиться и до сборки AppConfig (ошибка конфигурации).
    resolved_notifications = (
        notifications
        if _is_explicit(ctx, "notifications")
        else as_bool(env_defaults.get("NOTIFICATIONS"), True)
    )

    log_file = setup_logging(
        verbose=resolved_verbose, log_dir=resolved_output_dir / "logs"
    )
    if log_file is not None:
        logger.info("Подробные логи сохраняются в файл: %s", log_file)

    try:
        config_kwargs = collect_env_kwargs(
            env_defaults,
            input_file=input_file,
            output_dir=resolved_output_dir,
        )
        config_kwargs.update(_explicit_overrides(ctx, locals()))
        config_kwargs["verbose"] = resolved_verbose

        # hotwords обрабатываются отдельно (обрезка по лимиту токенов): берём
        # явный флаг, иначе HOTWORDS из config.env, иначе ничего.
        if _is_explicit(ctx, "hotwords"):
            raw_hotwords = hotwords
        else:
            raw_hotwords = (env_defaults.get("HOTWORDS") or "").strip() or None
        if raw_hotwords:
            raw_hotwords, dropped_terms = build_hotwords(raw_hotwords)
            if dropped_terms:
                logger.warning(
                    "Не поместилось в лимит hotwords ASR и не будет учтено: %s.",
                    ", ".join(dropped_terms),
                )
        config_kwargs["hotwords"] = raw_hotwords

        config = AppConfig(**config_kwargs)  # type: ignore[arg-type]
        config.ensure_output_dir()
        if clear_cache:
            from audio_transcriber.cache.store import StageCache

            removed = StageCache(config.resolved_cache_dir()).clear()
            logger.info("Кэш очищен: удалено %d файл(ов)", removed)
        resolved_device = resolve_device(config.device)

        logger.info("Входной файл: %s", config.input_file)
        logger.info("Директория результатов: %s", config.output_dir)
        logger.info("Бэкенд распознавания: %s", config.asr_backend.value)
        if config.asr_backend is AsrBackend.GIGAAM:
            logger.info(
                "Модель GigaAM: %s (квантизация: %s, VAD: %s)",
                config.gigaam_model,
                config.gigaam_quantization or "нет",
                "вкл" if config.gigaam_vad else "выкл",
            )
        else:
            logger.info("Модель распознавания: %s", config.model_name)
        if config.hybrid_asr:
            logger.info(
                "Гибридный ASR: резервный движок %s (модель %s)",
                config.hybrid_fallback_backend.value,
                (
                    config.whisper_cpp_model
                    if config.hybrid_fallback_backend is AsrBackend.WHISPER_CPP
                    else config.model_name
                ),
            )
        logger.info("Язык: %s", config.language or "автоопределение")
        logger.info("Устройство: %s (запрошено: %s)", resolved_device.value, config.device.value)
        logger.info("Форматы экспорта: %s", ", ".join(fmt.value for fmt in config.export_formats))
        logger.info(
            "Протокол по завершении: %s",
            "включён" if config.protocol_auto else "выключен (--no-protocol)",
        )
        logger.info(
            "Пословные таймстемпы: %s",
            "включены" if config.word_timestamps else "выключены (--no-word-timestamps)",
        )
        logger.info("Диаризация: %s", "включена" if config.diarization_enabled else "выключена")
        if config.diarization_enabled:
            logger.info("Движок диаризации: %s", config.diarization_engine)
            if config.diarization_engine == "auto":
                logger.info(
                    "Маршрутизация auto: оценщик %s, анализ %s с речи, cap %d говорящих",
                    "включён" if config.diarization_estimate_enabled else "выключен",
                    config.diarization_estimate_seconds,
                    config.diarization_route_max_speakers,
                )
            if config.diarization_engine in ("auto", "nemo-speech"):
                logger.info(
                    "NeMo-Speech.cpp: бинарник %s, модель %s, устройство %s%s",
                    config.nemo_speech_binary,
                    config.nemo_speech_model,
                    config.nemo_speech_device,
                    (
                        f", библиотеки {config.nemo_speech_lib_path}"
                        if config.nemo_speech_lib_path
                        else ""
                    ),
                )
            logger.info("Количество говорящих: %s", config.num_speakers or "автоопределение")
            if config.min_speakers is not None or config.max_speakers is not None:
                logger.info(
                    "Диапазон числа говорящих: %s–%s",
                    config.min_speakers if config.min_speakers is not None else "—",
                    config.max_speakers if config.max_speakers is not None else "—",
                )
            logger.info(
                "Гиперпараметры диаризации: min_duration_off=%s%s%s",
                config.diarization_min_duration_off,
                (
                    f", clustering.threshold={config.diarization_clustering_threshold}"
                    if config.diarization_clustering_threshold is not None
                    else ""
                ),
                (
                    f", clustering.Fb={config.diarization_clustering_fb}"
                    if config.diarization_clustering_fb is not None
                    else ""
                ),
            )
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
        if resolved_notifications:
            notify("Транскрибация не удалась", f"{input_file.name}: {exc}")
        raise typer.Exit(code=1) from exc

    logger.info("Готово: %d реплик(и), %d говорящих", len(result.entries), len(result.speakers))
    if resolved_notifications:
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


def _resolve_web_port(ctx: typer.Context, host: str, port: int | None) -> int:
    """Определяет фактический порт веб-сервера (issue #27).

    * ``--port`` задан явно: порт используется как есть, а если занят — ошибка
      без молчаливой подмены.
    * ``--port`` не задан: начиная с :data:`DEFAULT_WEB_PORT` подбирается первый
      свободный порт в пределах :data:`WEB_PORT_SCAN_LIMIT`.
    """
    from click.core import ParameterSource

    source = ctx.get_parameter_source("port")
    explicit = port is not None and source is not ParameterSource.DEFAULT

    if explicit:
        assert port is not None  # для mypy: explicit ⇒ port задан
        if not is_port_available(host, port):
            typer.echo(
                f"Порт {port} уже занят. Освободите его или укажите другой "
                "порт через --port.",
                err=True,
            )
            raise typer.Exit(code=1)
        return port

    base = DEFAULT_WEB_PORT
    if is_port_available(host, base):
        return base
    selected = find_available_port(host, start=base, limit=WEB_PORT_SCAN_LIMIT)
    if selected is None:
        typer.echo(
            f"Порт {base} занят, а свободных портов в диапазоне "
            f"{base}–{base + WEB_PORT_SCAN_LIMIT} не нашлось. "
            "Укажите свободный порт через --port.",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"Порт {base} занят, использую свободный {selected}.")
    return selected


@app.command()
def web(
    ctx: typer.Context,
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Адрес прослушивания (по умолчанию только локальный интерфейс).",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help=(
            "Порт локального сервера веб-интерфейса. Если не задан, "
            f"берётся {DEFAULT_WEB_PORT}, а при занятости подбирается свободный."
        ),
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Не открывать браузер автоматически.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Автоперезапуск сервера при изменении кода (для разработки).",
    ),
) -> None:
    """Запустить локальный веб-интерфейс транскрибации (127.0.0.1)."""
    actual_port = _resolve_web_port(ctx, host, port)

    try:
        from audio_transcriber.web.app import serve
    except ImportError as exc:  # веб-зависимости не установлены
        typer.echo(
            "Веб-интерфейс недоступен: не установлены зависимости. "
            'Установите их: uv pip install --python .venv/bin/python ".[web]"',
            err=True,
        )
        raise typer.Exit(code=1) from exc

    typer.echo(f"Веб-интерфейс: http://{host}:{actual_port}/")
    serve(host=host, port=actual_port, open_browser=not no_browser, reload=reload)


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
