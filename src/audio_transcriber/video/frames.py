"""Декодирование кадров видео через PyAV и отбор стабильных кадров слайдов.

Модуль решает три задачи:

* декодирует видеопоток и сэмплирует кадры с заданным интервалом (память
  ограничена: держится только текущий кадр и представитель текущего прогона);
* детектирует смену сцен через libavfilter (``select=gt(scene,…)``) прямо в
  процессе декодирования PyAV — без внешнего ffmpeg и новых зависимостей;
* отбирает «стабильные» кадры (слайд считается показанным, если удерживается
  не короче порога) и сравнивает кадры по сетке ячеек, чтобы замечать локальные
  изменения на шаблонных слайдах (заголовок/текст), которые не видны по
  глобальному различию.

Все функции отбора (``analysis_difference``, ``same_frame``,
``iter_occurrences``) чистые и тестируются на синтетических кадрах без видео.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np

logger = logging.getLogger(__name__)

#: Интервал сэмплирования кадров (секунды). Компромисс слабого CPU: реже —
#: дешевле декод, но можно пропустить короткий слайд.
DEFAULT_SAMPLE_INTERVAL_SECONDS = 2.0

#: Порог смены сцены для libavfilter ``select=gt(scene,…)`` (0..1).
DEFAULT_SCENE_THRESHOLD = 0.3

#: Размер аналитического изображения (квадрат) для сравнения кадров.
DEFAULT_ANALYSIS_SIZE = 64

#: Число ячеек сетки по стороне при сравнении кадров (ловит локальные правки).
DEFAULT_GRID = 8

#: Ширина сохраняемого ключевого кадра (для OCR и sidecar). Высокое разрешение
#: не держится в памяти пачкой: пишется на диск сразу при закрытии прогона.
DEFAULT_PREVIEW_WIDTH = 1280

#: Порог различия кадров (доля 0..1): выше — это уже другой слайд. Небольшой,
#: чтобы ловить смену заголовка на шаблонном слайде (изменяется малая доля
#: кадра), но выше шума сжатия идентичных кадров.
DEFAULT_CHANGE_THRESHOLD = 0.06

#: Минимальная длительность удержания слайда (секунды): короче — переход/анимация.
DEFAULT_STABILITY_SECONDS = 1.5


class VideoFileError(RuntimeError):
    """Видеофайл нельзя открыть или в нём нет видеопотока."""


@dataclass(frozen=True, slots=True)
class FrameSample:
    """Один сэмплированный кадр: таймкод, аналитика и полноразмерный кадр.

    ``analysis`` — небольшое grayscale-изображение для сравнения; ``image`` —
    RGB-кадр для сохранения/OCR; ``scene_cut`` — смена сцены произошла перед
    этим кадром (принудительная граница слайда).
    """

    timestamp: float
    analysis: np.ndarray
    image: np.ndarray
    scene_cut: bool = False


@dataclass(frozen=True, slots=True)
class DetectedFrame:
    """Стабильный кадр слайда: интервал ``[start, end)`` и представитель."""

    start: float
    end: float | None
    timestamp: float
    analysis: np.ndarray
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class ScannedFrame:
    """Стабильный кадр с уже сохранённым на диск представителем."""

    start: float
    end: float | None
    timestamp: float
    analysis: np.ndarray
    image_path: Path


@dataclass(frozen=True, slots=True)
class VideoScan:
    """Результат сканирования: стабильные кадры и длительность видео."""

    frames: tuple[ScannedFrame, ...]
    duration: float | None


def has_video_stream(path: Path) -> bool:
    """Есть ли в файле видеопоток (открывается ли контейнер вообще)."""
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.video)
    except Exception as exc:  # noqa: BLE001 — битый/недоступный файл = нет потока
        logger.debug("Не удалось проверить видеопоток %s: %s", path, exc)
        return False


def to_analysis_image(image: np.ndarray, *, size: int = DEFAULT_ANALYSIS_SIZE) -> np.ndarray:
    """Приводит кадр к аналитическому изображению ``size × size`` (float32).

    Уменьшение — усреднением по блокам (area average), а не выборкой пикселей:
    тонкие штрихи текста на шаблонных слайдах иначе теряются и слайды кажутся
    одинаковыми. Цвет сохраняется (каналы не усредняются). Для изображений
    меньше целевого размера используется ближайшая выборка.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] > 4:
        array = array[..., :3]
    if array.ndim not in (2, 3):
        raise ValueError(f"Ожидалось 2D/3D изображение, получено {array.shape}")
    height, width = array.shape[:2]
    if height < 1 or width < 1 or size < 1:
        raise ValueError("Некорректный размер изображения или анализа")
    if height >= size and width >= size:
        return _box_average(array, size)
    rows = np.arange(size) * height // size
    cols = np.arange(size) * width // size
    return array[np.ix_(rows, cols)]


def _box_average(array: np.ndarray, size: int) -> np.ndarray:
    """Уменьшает изображение до ``size × size`` усреднением по блокам."""
    height, width = array.shape[0], array.shape[1]
    rows = np.linspace(0, height, size + 1).astype(np.intp)
    cols = np.linspace(0, width, size + 1).astype(np.intp)
    row_counts = np.diff(rows).astype(np.float32)
    col_counts = np.diff(cols).astype(np.float32)
    row_sums = np.add.reduceat(array, rows[:-1], axis=0)
    col_sums = np.add.reduceat(row_sums, cols[:-1], axis=1)
    counts = row_counts[:, None] * col_counts[None, :]
    if array.ndim == 3:
        counts = counts[..., None]
    return col_sums / counts


def analysis_difference(
    first: np.ndarray, second: np.ndarray, *, grid: int = DEFAULT_GRID
) -> float:
    """Максимальное различие по ячейкам сетки (0..1) между аналитическими кадрами.

    Глобальное среднее различие «размывает» локальные правки, поэтому берётся
    максимум по сетке: изменение заголовка на шаблонном слайде попадает в
    отдельную ячейку и даёт высокий балл. Каналы цвета берутся максимумом, чтобы
    изменение одного канала не «размывалось» усреднением.
    """
    if first.shape != second.shape:
        return 1.0
    diff = np.abs(first - second) / 255.0
    size = min(first.shape[0], first.shape[1])
    cells = max(1, min(grid, size))
    step = max(1, size // cells)
    trimmed = diff[: step * cells, : step * cells]
    if trimmed.ndim == 3:
        channels = trimmed.shape[2]
        per_channel = trimmed.reshape(cells, step, cells, step, channels).mean(axis=(1, 3))
        reshaped = per_channel.max(axis=2)
    else:
        reshaped = trimmed.reshape(cells, step, cells, step).mean(axis=(1, 3))
    return float(reshaped.max())


def frame_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    size: int = DEFAULT_ANALYSIS_SIZE,
    grid: int = DEFAULT_GRID,
) -> float:
    """Различие двух произвольных кадров (0..1) через аналитическое приведение."""
    return analysis_difference(
        to_analysis_image(first, size=size),
        to_analysis_image(second, size=size),
        grid=grid,
    )


def same_frame(
    first: np.ndarray,
    second: np.ndarray,
    threshold: float = DEFAULT_CHANGE_THRESHOLD,
    *,
    size: int = DEFAULT_ANALYSIS_SIZE,
    grid: int = DEFAULT_GRID,
) -> bool:
    """Считаются ли два кадра одним и тем же слайдом (различие не выше порога)."""
    return frame_difference(first, second, size=size, grid=grid) <= threshold


def iter_occurrences(
    samples: Iterable[FrameSample],
    *,
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    stability_seconds: float = DEFAULT_STABILITY_SECONDS,
    duration: float | None = None,
) -> Iterator[DetectedFrame]:
    """Отбирает стабильные кадры слайдов из потока сэмплов.

    Кадры группируются в прогоны «похожих»; прогон закрывается при смене сцены
    или заметном различии кадра. Слайдом считается прогон длительностью не
    короче ``stability_seconds``; представитель — последний кадр прогона (после
    доигрывания анимаций). Память O(1) относительно числа сэмплов: держится
    только представитель текущего прогона.
    """
    iterator = iter(samples)
    try:
        first = next(iterator)
    except StopIteration:
        return
    run_start = first.timestamp
    run_last = first
    for sample in iterator:
        boundary = sample.scene_cut or analysis_difference(
            run_last.analysis, sample.analysis
        ) > change_threshold
        if not boundary:
            run_last = sample
            continue
        emitted = _finalize_run(run_start, sample.timestamp, run_last, stability_seconds)
        if emitted is not None:
            yield emitted
        run_start = sample.timestamp
        run_last = sample
    emitted = _finalize_run(run_start, duration, run_last, stability_seconds)
    if emitted is not None:
        yield emitted


def _finalize_run(
    start: float, end: float | None, run_last: FrameSample, stability_seconds: float
) -> DetectedFrame | None:
    """Закрывает прогон: возвращает слайд, если он удерживался достаточно долго."""
    if run_last.timestamp - start < stability_seconds:
        return None
    return DetectedFrame(
        start=start,
        end=end,
        timestamp=run_last.timestamp,
        analysis=run_last.analysis,
        image=run_last.image,
    )


class _SceneFilter:
    """Обёртка libavfilter-графа смены сцен (``select=gt(scene,…)``)."""

    def __init__(self, graph: Any, source: Any, sink: Any) -> None:
        #: Граф держится живым: без ссылки на него фильтры становятся «висячими».
        self._graph = graph
        self._source = source
        self._sink = sink

    def push(self, frame: Any, time_base: Any) -> list[float]:
        """Проталкивает кадр в граф и возвращает таймкоды прошедших кадров."""
        try:
            self._source.push(frame)
        except Exception as exc:  # noqa: BLE001 — граф не критичен для сэмплинга
            logger.debug("Сбой подачи кадра в фильтр сцен: %s", exc)
            return []
        cuts: list[float] = []
        while True:
            try:
                output = self._sink.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                break
            except Exception as exc:  # noqa: BLE001 — конец потока/ошибка графа
                logger.debug("Сбой выборки кадра из фильтра сцен: %s", exc)
                break
            timestamp = _frame_time(output, time_base)
            if timestamp is not None:
                cuts.append(timestamp)
        return cuts


def _build_scene_filter(stream: Any, threshold: float) -> _SceneFilter | None:
    """Строит фильтр смены сцен или ``None``, если libavfilter недоступен."""
    try:
        graph = av.filter.Graph()
        source = graph.add_buffer(template=stream)
        selector = graph.add("select", f"gt(scene\\,{threshold})")
        sink = graph.add("buffersink")
        source.link_to(selector)
        selector.link_to(sink)
        graph.configure()
    except Exception as exc:  # noqa: BLE001 — мягкая деградация до сравнения кадров
        logger.debug("Детект смены сцен через libavfilter недоступен: %s", exc)
        return None
    return _SceneFilter(graph, source, sink)


def _frame_time(frame: Any, time_base: Any) -> float | None:
    """Таймкод кадра в секундах (``pts·time_base``) или ``None``."""
    pts = getattr(frame, "pts", None)
    if pts is not None and time_base is not None:
        try:
            return float(pts * time_base)
        except (TypeError, ValueError):
            return None
    time = getattr(frame, "time", None)
    if time is not None:
        try:
            return float(time)
        except (TypeError, ValueError):
            return None
    return None


def _frame_to_array(frame: Any, width: int) -> np.ndarray:
    """Приводит кадр PyAV к RGB ndarray, при необходимости уменьшая ширину."""
    frame_width = int(frame.width)
    frame_height = int(frame.height)
    if width > 0 and frame_width > width:
        height = max(1, round(frame_height * width / frame_width))
        frame = frame.reformat(width=width, height=height)
    array = frame.to_ndarray(format="rgb24")
    return np.ascontiguousarray(array)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Кодирует RGB-кадр в PNG средствами PyAV (без PIL)."""
    height, width = image.shape[:2]
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image, dtype=np.uint8), format="rgb24")
    container = av.open(str(path), "w", format="image2")
    try:
        stream = container.add_stream("png", rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "rgb24"
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


def scan_video(
    path: Path,
    frames_dir: Path,
    *,
    interval: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    stability_seconds: float = DEFAULT_STABILITY_SECONDS,
    preview_width: int = DEFAULT_PREVIEW_WIDTH,
    analysis_size: int = DEFAULT_ANALYSIS_SIZE,
) -> VideoScan:
    """Сканирует видео: сэмплирует кадры, детектит сцены, отбирает слайды.

    Представители стабильных кадров сразу пишутся в ``frames_dir`` (PNG), в
    памяти держатся только аналитические изображения — это важно для слабого
    железа. Ошибка открытия/отсутствие видеопотока — :class:`VideoFileError`.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        container = av.open(str(path))
    except Exception as exc:
        raise VideoFileError(f"Не удалось открыть видео {path}: {exc}") from exc
    try:
        try:
            stream = container.streams.video[0]
        except IndexError as exc:
            raise VideoFileError(f"В файле {path} не найдена видеодорожка") from exc
        time_base = stream.time_base
        duration = _stream_duration(container, stream, time_base)
        scene_filter = _build_scene_filter(stream, scene_threshold)
        frames: list[ScannedFrame] = []
        for index, detected in enumerate(
            iter_occurrences(
                _iter_samples(
                    container, stream, time_base, scene_filter, interval, analysis_size, preview_width
                ),
                change_threshold=change_threshold,
                stability_seconds=stability_seconds,
                duration=duration,
            )
        ):
            image_path = frames_dir / f"candidate-{index:04d}.png"
            _write_png(image_path, detected.image)
            frames.append(
                ScannedFrame(
                    start=detected.start,
                    end=detected.end,
                    timestamp=detected.timestamp,
                    analysis=detected.analysis,
                    image_path=image_path,
                )
            )
        return VideoScan(frames=tuple(frames), duration=duration)
    finally:
        container.close()


def _iter_samples(
    container: Any,
    stream: Any,
    time_base: Any,
    scene_filter: _SceneFilter | None,
    interval: float,
    analysis_size: int,
    preview_width: int,
) -> Iterator[FrameSample]:
    """Декодирует поток и отдаёт сэмплы с периодом ``interval`` секунд."""
    last_sample: float | None = None
    pending_cuts: list[float] = []
    for frame in container.decode(stream):
        timestamp = _frame_time(frame, time_base)
        if scene_filter is not None:
            pending_cuts.extend(scene_filter.push(frame, time_base))
        if timestamp is None:
            continue
        if last_sample is not None and timestamp < last_sample + interval:
            continue
        image = _frame_to_array(frame, preview_width)
        analysis = to_analysis_image(image, size=analysis_size)
        scene_cut = last_sample is not None and any(
            last_sample < cut <= timestamp for cut in pending_cuts
        )
        pending_cuts = [cut for cut in pending_cuts if cut > timestamp]
        yield FrameSample(
            timestamp=timestamp, analysis=analysis, image=image, scene_cut=scene_cut
        )
        last_sample = timestamp


def _stream_duration(container: Any, stream: Any, time_base: Any) -> float | None:
    """Длительность видео: из потока, иначе из контейнера (секунды)."""
    if stream.duration is not None and time_base is not None:
        return float(stream.duration * time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    return None
