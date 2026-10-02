"""Дешёвая оценка числа говорящих (#64).

Режим ``DIARIZATION_ENGINE=auto`` выбирает движок по числу говорящих: до лимита
— быстрый ``nemo-speech`` (Sortformer, 4 спикера), выше — точный ``pyannote``.
Чтобы это решение не стоило минут диаризации, число говорящих оценивается
заранее и дёшево:

1. **VAD** (silero через ``sherpa-onnx``) режет аудио на речевые фрагменты.
2. Из фрагментов берётся выборка до ``max_seconds`` секунд речи, **равномерно
   распределённая по записи** (не только начало): цели идут по накопленному
   времени речи, окно берётся из нужного фрагмента.
3. **Эмбеддинги** говорящего считает ``SpeakerEmbeddingExtractor``
   (модель 3D-Speaker CAM++, ONNX Runtime; PyTorch не нужен).
4. **Агломеративная кластеризация** (косинусная метрика, average-linkage) с
   порогом даёт число кластеров — оценку числа говорящих.

Мягкая деградация — ключевой контракт: любая ошибка (нет ``sherpa-onnx``,
модель не скачалась, битое аудио, сбой кластеризации) возвращает ``None``.
Вызывающий код (маршрутизация движка) тогда безопасно выбирает pyannote —
конвейер не падает.

Модели скачиваются с официальных релизов ``k2-fsa/sherpa-onnx`` в локальный
кэш (по умолчанию ``~/.cache/audio-transcriber/sherpa``), проверяются на
непустоту, запись атомарна (временный файл → ``replace``). Путь можно
переопределить (``model_dir`` или явный путь к ``.onnx``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
    DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
)
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ESTIMATE_MODEL_URL",
    "SILERO_VAD_FILENAME",
    "SILERO_VAD_URL",
    "default_model_cache_dir",
    "estimate_speaker_count",
]

#: Тип речевого фрагмента: (начало, конец) в сэмплах.
SpeechSpan = tuple[int, int]

#: Имя ONNX-модели silero VAD (та же, что использует сам sherpa-onnx).
SILERO_VAD_FILENAME = "silero_vad.onnx"

#: URL модели эмбеддингов по умолчанию (release ``speaker-recongition-models``
#: — да, с опечаткой в имени релиза у k2-fsa, это исторически так).
ESTIMATE_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{DEFAULT_DIARIZATION_ESTIMATE_MODEL}"
)

#: URL silero VAD из релиза ``asr-models``.
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{SILERO_VAD_FILENAME}"
)

#: Длительность одного окна эмбеддинга (секунды). Короткие окна быстрее, но
#: менее стабильны; 1.5 с — компромисс для CAM++ на 16 кГц.
ESTIMATE_WINDOW_SECONDS = 1.5

#: Ансамбль конфигураций выборки: (доля от ``max_seconds``, длительность окна).
#: Оценка одного набора окон нестабильна (эмбеддинги на реальном материале
#: шумные), поэтому число говорящих считается для нескольких конфигураций, а
#: итог — медиана. Это устойчиво к выбросам и всё ещё занимает секунды.
_ENSEMBLE_CONFIGS: tuple[tuple[float, float], ...] = (
    (0.5, 1.5),
    (2.0 / 3.0, 1.5),
    (1.0, 1.5),
    (1.0, 2.0),
    (1.5, 2.0),
)

#: Минимальная доля речи для анализа (секунды): при крошечном ``max_seconds``
#: конфигурация ансамбля не должна схлопываться в одно окно.
_MIN_ANALYZED_SECONDS = 2.0

#: Размер буфера VAD (секунды). Нужен, чтобы вместить непрерывный монолог;
#: при превышении sherpa-onnx мягко расширяет буфер без потери данных.
_VAD_BUFFER_SECONDS = 300.0

#: Таймаут загрузки одной модели (секунды).
_DOWNLOAD_TIMEOUT = 120.0


def _sherpa_available() -> bool:
    """Установлен ли ``sherpa-onnx`` (без импорта — ``find_spec``)."""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):
        return False


def default_model_cache_dir() -> Path:
    """Каталог кэша моделей оценщика (``XDG_CACHE_HOME`` или ``~/.cache``)."""
    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(raw).expanduser() if raw else Path.home() / ".cache"
    return base / "audio-transcriber" / "sherpa"


def _emit(on_progress: ProgressCallback | None, message: str, fraction: float | None) -> None:
    if on_progress is None:
        return
    on_progress(ProgressEvent("diarization", message=message, fraction=fraction))


def _is_valid_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _download_file(
    url: str,
    target: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path | None:
    """Скачивает ``url`` в ``target`` атомарно; ``None`` при любой ошибке.

    Пишет во временный файл рядом с целевым и делает ``replace`` — недокачанный
    файл никогда не выглядит как готовая модель.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp_path = Path(raw_tmp)
    _emit(on_progress, f"Загрузка модели оценки говорящих: {target.name}", 0.0)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "audio-transcriber"}
        )
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else 0
            received = 0
            with open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if total > 0:
                        _emit(
                            on_progress,
                            f"Загрузка модели оценки говорящих: {target.name}",
                            min(1.0, received / total),
                        )
        if received == 0:
            raise OSError("получен пустой ответ")
        tmp_path.replace(target)
    except Exception as exc:  # noqa: BLE001 — мягкая деградация: оценка не критична
        logger.warning(
            "Не удалось скачать модель оценки говорящих %s: %s — оценка пропущена",
            url,
            exc,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Не удалось удалить временный файл %s", tmp_path)
        return None
    _emit(on_progress, f"Модель оценки говорящих готова: {target.name}", 1.0)
    return target


def _resolve_named_model(
    model: str,
    cache_dir: Path,
    *,
    default_url: str,
    on_progress: ProgressCallback | None,
) -> Path | None:
    """Находит модель ``model``: существующий путь или файл в каталоге кэша.

    Для модели по умолчанию недостающий файл скачивается. Неизвестное имя без
    файла на диске не скачивается — модель не критична, вернём ``None``.
    """
    candidate = Path(model).expanduser()
    if _is_valid_file(candidate):
        return candidate

    in_cache = cache_dir / model
    if _is_valid_file(in_cache):
        return in_cache

    if not candidate.is_absolute() and model == DEFAULT_DIARIZATION_ESTIMATE_MODEL:
        return _download_file(default_url, in_cache, on_progress=on_progress)

    logger.info(
        "Модель оценки говорящих %r не найдена и не будет скачана (ожидается путь "
        "к .onnx или имя модели по умолчанию)",
        model,
    )
    return None


def _resolve_models(
    model: str,
    cache_dir: Path | None,
    *,
    on_progress: ProgressCallback | None,
) -> tuple[Path, Path] | None:
    """Возвращает пути ``(модель эмбеддингов, модель VAD)`` или ``None``."""
    resolved_cache = cache_dir or default_model_cache_dir()
    embedding = _resolve_named_model(
        model,
        resolved_cache,
        default_url=ESTIMATE_MODEL_URL,
        on_progress=on_progress,
    )
    if embedding is None:
        return None

    candidate_vad = resolved_cache / SILERO_VAD_FILENAME
    vad_path: Path | None
    if _is_valid_file(candidate_vad):
        vad_path = candidate_vad
    else:
        vad_path = _download_file(SILERO_VAD_URL, candidate_vad, on_progress=on_progress)
    if vad_path is None:
        return None
    return embedding, vad_path


def _load_samples(
    audio_path: str | Path | None,
    waveform: np.ndarray | None,
) -> np.ndarray | None:
    """Декодирует или переиспользует моно waveform 16 кГц float32."""
    if waveform is not None:
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return samples if samples.size > 0 else None
    if audio_path is None:
        return None
    samples = load_waveform(Path(audio_path))
    return samples if samples.size > 0 else None


def _detect_speech(samples: np.ndarray, vad_model: Path) -> list[SpeechSpan]:
    """Возвращает речевые фрагменты ``(начало, конец)`` в сэмплах (silero VAD)."""
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.min_speech_duration = 0.25
    config.sample_rate = SAMPLE_RATE
    window_size = int(config.silero_vad.window_size) or 512

    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=_VAD_BUFFER_SECONDS)
    spans: list[SpeechSpan] = []

    def drain() -> None:
        while not detector.empty():
            segment = detector.front
            start = int(segment.start)
            end = start + len(segment.samples)
            spans.append((start, end))
            detector.pop()

    total = samples.shape[0]
    for offset in range(0, total, window_size):
        detector.accept_waveform(samples[offset : offset + window_size])
        drain()
    detector.flush()
    drain()
    return spans


def _start_at(
    cumulative: Sequence[tuple[float, float, int, int]],
    target_seconds: float,
) -> int | None:
    """Начало речевого фрагмента в сэмплах для целевого времени речи (секунды)."""
    for speech_start, speech_end, start_sample, _end_sample in cumulative:
        if speech_start <= target_seconds <= speech_end:
            offset = int((target_seconds - speech_start) * SAMPLE_RATE)
            return start_sample + offset
    return None


def _sample_windows(
    samples: np.ndarray,
    spans: Sequence[SpeechSpan],
    *,
    max_seconds: float,
    window_seconds: float = ESTIMATE_WINDOW_SECONDS,
) -> list[np.ndarray]:
    """Берёт окна речи, равномерно распределённые по записи.

    Цели идут по накопленному времени речи, поэтому выборка охватывает всю
    запись, а не только её начало. Окно выравнивается внутри содержащего его
    фрагмента и обрезается по границам массива.
    """
    usable = [(start, end) for start, end in spans if end > start]
    if not usable:
        return []

    cumulative: list[tuple[float, float, int, int]] = []
    accumulated = 0.0
    for start, end in usable:
        duration = (end - start) / SAMPLE_RATE
        cumulative.append((accumulated, accumulated + duration, start, end))
        accumulated += duration
    if accumulated <= 0.0:
        return []

    window_samples = int(window_seconds * SAMPLE_RATE)
    if window_samples <= 0 or window_samples > samples.shape[0]:
        return []
    speech_seconds = min(float(max_seconds), accumulated)
    window_count = max(1, round(speech_seconds / window_seconds))

    windows: list[np.ndarray] = []
    for index in range(window_count):
        target = (index + 0.5) / window_count * accumulated
        start_sample = _start_at(cumulative, target)
        if start_sample is None:
            continue
        start_sample = max(0, min(start_sample, samples.shape[0] - window_samples))
        windows.append(samples[start_sample : start_sample + window_samples])
    return windows


def _compute_embeddings(windows: Sequence[np.ndarray], model: Path) -> np.ndarray:
    """L2-нормированные эмбеддинги говорящего для каждого окна."""
    import sherpa_onnx

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
    config.model = str(model)
    config.provider = "cpu"
    config.num_threads = max(1, min(4, os.cpu_count() or 1))
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    vectors: list[np.ndarray] = []
    for window in windows:
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, window)
        stream.input_finished()
        vector = np.asarray(extractor.compute(stream), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        vectors.append(vector)
    return np.stack(vectors)


def _cluster_embeddings(embeddings: np.ndarray, *, threshold: float) -> int:
    """Агломеративная кластеризация по косинусному расстоянию → число кластеров."""
    from sklearn.cluster import AgglomerativeClustering

    count = int(embeddings.shape[0])
    if count <= 1:
        return max(count, 1)

    distance = 1.0 - embeddings @ embeddings.T
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(threshold),
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)
    return max(1, len(set(labels.tolist())))


def _ensemble_speaker_count(
    samples: np.ndarray,
    spans: Sequence[SpeechSpan],
    model: Path,
    *,
    max_seconds: float,
    threshold: float,
) -> int | None:
    """Медиана оценок по ансамблю конфигураций выборки и окон.

    Одиночный запуск на шумных эмбеддингах даёт нестабильное число; медиана по
    нескольким выборкам/окнам устойчива к выбросам. ``None`` — если ни одна
    конфигурация не дала данных.
    """
    counts: list[int] = []
    for factor, window_seconds in _ENSEMBLE_CONFIGS:
        analyzed_seconds = max(_MIN_ANALYZED_SECONDS, max_seconds * factor)
        windows = _sample_windows(
            samples,
            spans,
            max_seconds=analyzed_seconds,
            window_seconds=window_seconds,
        )
        if not windows:
            continue
        if len(windows) < 2:
            counts.append(1)
            continue
        embeddings = _compute_embeddings(windows, model)
        if embeddings.shape[0] < 2:
            counts.append(1)
            continue
        counts.append(_cluster_embeddings(embeddings, threshold=threshold))
    if not counts:
        return None
    counts.sort()
    middle = len(counts) // 2
    if len(counts) % 2 == 1:
        return counts[middle]
    return round((counts[middle - 1] + counts[middle]) / 2)


def estimate_speaker_count(
    audio_path: str | Path | None = None,
    *,
    waveform: np.ndarray | None = None,
    max_seconds: float = DEFAULT_DIARIZATION_ESTIMATE_SECONDS,
    threshold: float = DEFAULT_DIARIZATION_ESTIMATE_THRESHOLD,
    model: str = DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    model_dir: Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> int | None:
    """Оценивает число говорящих по аудио (или готовому waveform).

    Возвращает целое ``>= 1`` либо ``None``, если оценка невозможна (нет
    ``sherpa-onnx``/моделей, битое аудио, нет речи, сбой кластеризации).
    ``None`` — безопасный сигнал «число неизвестно»: маршрутизация движка
    выберет pyannote. Функция не поднимает исключений.
    """
    if not _sherpa_available():
        logger.info(
            "Оценка числа говорящих недоступна: не установлен sherpa-onnx "
            "(uv pip install sherpa-onnx)"
        )
        return None

    started = time.monotonic()
    try:
        samples = _load_samples(audio_path, waveform)
        if samples is None:
            logger.info("Оценка числа говорящих: нет аудио для анализа")
            return None

        models = _resolve_models(model, model_dir, on_progress=on_progress)
        if models is None:
            return None
        embedding_model, vad_model = models

        spans = _detect_speech(samples, vad_model)
        if not spans:
            logger.info("Оценка числа говорящих: VAD не нашёл речи")
            return None

        count = _ensemble_speaker_count(
            samples,
            spans,
            embedding_model,
            max_seconds=max_seconds,
            threshold=threshold,
        )
        if count is None:
            logger.info("Оценка числа говорящих: нет пригодных речевых окон")
            return None
        logger.info(
            "Оценка числа говорящих: ≈%d (%.2f с, речь %.0f с из %d фрагментов)",
            count,
            time.monotonic() - started,
            sum(end - start for start, end in spans) / SAMPLE_RATE,
            len(spans),
        )
        return count
    except Exception as exc:  # noqa: BLE001 — мягкая деградация: оценка не критична
        logger.warning(
            "Оценка числа говорящих не удалась (%s: %s) — будет выбран pyannote",
            type(exc).__name__,
            exc,
        )
        return None
