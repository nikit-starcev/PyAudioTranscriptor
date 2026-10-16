"""Извлечение слайдов из видео и sidecar-хранилище.

Опциональный пост-шаг по кнопке (аналог «протокола»), а не стадия конвейера:
считается по запросу поверх готового результата задачи и не раздувает основной
``result.json``. Хранилище — sidecar ``<stem>.slides/`` рядом с результатом (по
образцу ``*.speakers/``): PNG ключевых кадров и ``slides.json``.

Цепочка: декод/детект сцен (PyAV + libavfilter) → отбор стабильных кадров →
OCR представителей → кластеризация по тексту → упорядоченный список слайдов с
таймкодами. Дорогой результат кэшируется стадией ``slides`` (метаданные и
картинки), чтобы повторное нажатие кнопки не пересчитывало OCR.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.cache.store import StageCache
from audio_transcriber.config.settings import AppConfig
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.video.cluster import (
    DEFAULT_IMAGE_DIFF_THRESHOLD,
    DEFAULT_TEXT_JACCARD_THRESHOLD,
    SlideCandidate,
    SlideGroup,
    cluster_candidates,
)
from audio_transcriber.video.frames import (
    DEFAULT_CHANGE_THRESHOLD,
    DEFAULT_PREVIEW_WIDTH,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_STABILITY_SECONDS,
    ScannedFrame,
    VideoScan,
    scan_video,
)
from audio_transcriber.video.ocr import OcrEngine, find_ocr_engine

logger = logging.getLogger(__name__)

#: Суффикс каталога слайдов рядом с результатом: ``<файл>.slides``.
SLIDES_SUFFIX = ".slides"

#: Имя файла метаданных слайдов в sidecar-каталоге.
METADATA_NAME = "slides.json"

#: Версия формата ``slides.json`` (стороннее совместимое представление).
METADATA_VERSION = 1

#: Версия реализации извлечения (входит в ключ кэша — при изменении алгоритма
#: старый кэш игнорируется).
IMPL_VERSION = 1

#: Расширение картинок слайдов (учитывается эвикцией кэша через extra_suffixes).
IMAGE_SUFFIX = ".png"

#: Кэшируемая стадия.
STAGE_SLIDES = "slides"


def _no_progress(_event: ProgressEvent) -> None:
    """Пустой публикатор прогресса (когда колбэк не передан)."""
    return None


@dataclass(frozen=True, slots=True)
class Slide:
    """Слайд: таймкод, OCR-текст, движок и файл ключевого кадра (в sidecar)."""

    index: int
    start: float
    end: float | None
    text: str
    engine: str
    image: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "engine": self.engine,
            "image": self.image,
        }


@dataclass(frozen=True, slots=True)
class SlidesArtifacts:
    """Результат извлечения: слайды и каталог sidecar."""

    slides: tuple[Slide, ...]
    directory: Path


def slides_directory(output_dir: Path, source_path: Path) -> Path:
    """Каталог слайдов для записи: ``<output_dir>/<stem>.slides``."""
    return output_dir / f"{source_path.stem}{SLIDES_SUFFIX}"


def slide_image_name(index: int) -> str:
    """Имя файла ключевого кадра слайда в sidecar-каталоге."""
    return f"slide-{index:03d}{IMAGE_SUFFIX}"


def read_slides_metadata(directory: Path) -> dict[str, object] | None:
    """Читает ``slides.json`` из sidecar-каталога (``None`` — нет/повреждён)."""
    try:
        payload = json.loads((directory / METADATA_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def build_slides_payload(metadata: Mapping[str, object], *, job_id: str) -> dict[str, object]:
    """Собирает контракт ответа API из метаданных ``slides.json``.

    ``image`` — URL получения картинки кадра; ``total`` — число слайдов.
    """
    raw = metadata.get("slides")
    slides: list[dict[str, object]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            slides.append(
                {
                    "index": index,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "text": item.get("text", ""),
                    "engine": item.get("engine", ""),
                    "image": f"/api/jobs/{job_id}/slides/{index}/image",
                }
            )
    return {"slides": slides, "total": len(slides)}


def _cache_params(
    *,
    engine: str,
    interval: float,
    scene_threshold: float,
    change_threshold: float,
    stability_seconds: float,
    text_threshold: float,
    image_threshold: float,
    preview_width: int,
) -> dict[str, object]:
    """Параметры стадии ``slides`` для ключа кэша (всё, что влияет на результат)."""
    return {
        "impl_version": IMPL_VERSION,
        "engine": engine,
        "interval": interval,
        "scene_threshold": scene_threshold,
        "change_threshold": change_threshold,
        "stability_seconds": stability_seconds,
        "text_threshold": text_threshold,
        "image_threshold": image_threshold,
        "preview_width": preview_width,
    }


def _group_representative(candidates: list[SlideCandidate], members: tuple[int, ...]) -> int:
    """Индекс представителя группы: самый длинный текст, тай-брейк — раньше."""
    return max(members, key=lambda index: (len(candidates[index].text.strip()), -index))


def _group_bounds(
    candidates: list[SlideCandidate], members: tuple[int, ...]
) -> tuple[float, float | None]:
    """Границы слайда: минимум начал и максимум концов группы (``None`` при пропуске)."""
    start = min(candidates[index].start for index in members)
    ends = [candidates[index].end for index in members]
    if any(end is None for end in ends):
        return round(start, 3), None
    values = [end for end in ends if end is not None]
    return round(start, 3), round(max(values), 3)


def _ordered_groups(
    candidates: list[SlideCandidate], groups: list[SlideGroup]
) -> list[tuple[int, ...]]:
    """Упорядочивает группы по времени первого появления слайда."""
    return sorted(
        (group.members for group in groups),
        key=lambda members: min(candidates[index].start for index in members),
    )


def _write_metadata(directory: Path, metadata: Mapping[str, object]) -> None:
    """Пишет ``slides.json`` в sidecar атомарно (через временный файл)."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / METADATA_NAME
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(target)


def _build_metadata(
    source: Path, engine: str, duration: float | None, slides: list[Slide]
) -> dict[str, object]:
    return {
        "version": METADATA_VERSION,
        "engine": engine,
        "source": str(source),
        "duration": duration,
        "slides": [slide.as_dict() for slide in slides],
    }


def _extract(
    source: Path,
    directory: Path,
    engine: OcrEngine,
    *,
    interval: float,
    scene_threshold: float,
    change_threshold: float,
    stability_seconds: float,
    text_threshold: float,
    image_threshold: float,
    preview_width: int,
    cache: StageCache,
    key: str,
    emit: ProgressCallback,
) -> SlidesArtifacts:
    """Полный расчёт: сцены → стабильные кадры → OCR → кластеризация → sidecar."""
    emit(ProgressEvent("slides", "Детект сцен и стабильных кадров", fraction=None))
    with tempfile.TemporaryDirectory(prefix="slides-") as tmp:
        scan = scan_video(
            source,
            Path(tmp),
            interval=interval,
            scene_threshold=scene_threshold,
            change_threshold=change_threshold,
            stability_seconds=stability_seconds,
            preview_width=preview_width,
        )
        frames = scan.frames
        emit(ProgressEvent("slides", f"Распознавание кадров: {len(frames)}", fraction=None))
        candidates: list[SlideCandidate] = []
        for frame in frames:
            text = engine.recognize(frame.image_path)
            candidates.append(
                SlideCandidate(
                    start=frame.start, end=frame.end, analysis=frame.analysis, text=text
                )
            )
        groups = cluster_candidates(
            candidates,
            text_threshold=text_threshold,
            image_diff_threshold=image_threshold,
        )
        ordered = _ordered_groups(candidates, groups)
        slides = _store_sidecar(directory, source, scan, frames, candidates, ordered, engine.name)
    _cache_result(cache, key, source, engine.name, scan.duration, slides, directory)
    emit(ProgressEvent("slides", f"Слайдов: {len(slides)}", fraction=1.0))
    return SlidesArtifacts(slides=tuple(slides), directory=directory)


def _store_sidecar(
    directory: Path,
    source: Path,
    scan: VideoScan,
    frames: tuple[ScannedFrame, ...],
    candidates: list[SlideCandidate],
    ordered: list[tuple[int, ...]],
    engine: str,
) -> list[Slide]:
    """Пишет картинки и ``slides.json`` в sidecar; возвращает список слайдов."""
    directory.mkdir(parents=True, exist_ok=True)
    slides: list[Slide] = []
    for index, members in enumerate(ordered):
        representative = _group_representative(candidates, members)
        start, end = _group_bounds(candidates, members)
        image_name = slide_image_name(index)
        (directory / image_name).write_bytes(frames[representative].image_path.read_bytes())
        slides.append(
            Slide(
                index=index,
                start=start,
                end=end,
                text=candidates[representative].text,
                engine=engine,
                image=image_name,
            )
        )
    _write_metadata(directory, _build_metadata(source, engine, scan.duration, slides))
    return slides


def _cache_result(
    cache: StageCache,
    key: str,
    source: Path,
    engine: str,
    duration: float | None,
    slides: list[Slide],
    directory: Path,
) -> None:
    """Кэширует метаданные и картинки стадии ``slides`` (мягкая деградация)."""
    if not cache.enabled:
        return
    cache.save(STAGE_SLIDES, key, _build_metadata(source, engine, duration, slides))
    for slide in slides:
        try:
            data = (directory / slide.image).read_bytes()
        except OSError as exc:
            logger.warning("Не удалось прочитать кадр слайда %s для кэша: %s", slide.image, exc)
            continue
        cache.save_blob(STAGE_SLIDES, key, f"-{slide.index:03d}{IMAGE_SUFFIX}", data)


def _materialize_from_cache(
    cache: StageCache, key: str, metadata: Mapping[str, object], directory: Path, engine: str
) -> tuple[Slide, ...] | None:
    """Восстанавливает sidecar из кэша: ``slides.json`` и недостающие картинки."""
    raw = metadata.get("slides")
    if not isinstance(raw, list):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    slides: list[Slide] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        index = item.get("index")
        image = item.get("image")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(image, str):
            return None
        destination = directory / image
        if not destination.is_file():
            cached = cache.load_blob(STAGE_SLIDES, key, f"-{index:03d}{IMAGE_SUFFIX}")
            if cached is None:
                return None
            destination.write_bytes(cached.read_bytes())
        slides.append(
            Slide(
                index=index,
                start=_as_float(item.get("start"), 0.0),
                end=_as_optional_float(item.get("end")),
                text=_as_str(item.get("text"), ""),
                engine=_as_str(item.get("engine"), engine),
                image=image,
            )
        )
    _write_metadata(directory, metadata)
    return tuple(slides)


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_str(value: object, default: str) -> str:
    """Строка из значения метаданных или ``default`` (терпимость к формату)."""
    return value if isinstance(value, str) else default


#: Тип фабрики извлечения слайдов (совместим с :func:`extract_slides`).
SlidesExtractor = Callable[..., SlidesArtifacts]


def extract_slides(
    config: AppConfig,
    *,
    on_progress: ProgressCallback | None = None,
    ocr: OcrEngine | None = None,
    interval: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    stability_seconds: float = DEFAULT_STABILITY_SECONDS,
    text_threshold: float = DEFAULT_TEXT_JACCARD_THRESHOLD,
    image_threshold: float = DEFAULT_IMAGE_DIFF_THRESHOLD,
    preview_width: int = DEFAULT_PREVIEW_WIDTH,
) -> SlidesArtifacts:
    """Извлекает слайды из видео задачи и сохраняет sidecar ``*.slides/``.

    ``ocr`` — готовый движок (в тестах); ``None`` — движок выбирается по
    доступности и при отсутствии бросается
    :class:`~audio_transcriber.video.ocr.OcrUnavailableError` (вызывающий код
    отвечает 503). Дорогой результат кэшируется стадией ``slides``: повторное
    нажатие кнопки не пересчитывает OCR.
    """
    emit = on_progress or _no_progress
    source = Path(config.input_file)
    engine = ocr if ocr is not None else find_ocr_engine()
    directory = slides_directory(config.output_dir, source)
    cache = StageCache(
        config.resolved_cache_dir(),
        enabled=config.use_cache,
        max_bytes=config.cache_max_bytes,
        ttl_seconds=config.cache_ttl_seconds,
        extra_suffixes=(IMAGE_SUFFIX,),
    )
    params = _cache_params(
        engine=engine.name,
        interval=interval,
        scene_threshold=scene_threshold,
        change_threshold=change_threshold,
        stability_seconds=stability_seconds,
        text_threshold=text_threshold,
        image_threshold=image_threshold,
        preview_width=preview_width,
    )
    key = cache.key(STAGE_SLIDES, source, params)
    if cache.enabled:
        cached = cache.load(STAGE_SLIDES, key)
        if cached is not None:
            slides = _materialize_from_cache(cache, key, cached, directory, engine.name)
            if slides is not None:
                emit(ProgressEvent("slides", "Слайды из кэша", fraction=1.0))
                return SlidesArtifacts(slides=slides, directory=directory)
    return _extract(
        source,
        directory,
        engine,
        interval=interval,
        scene_threshold=scene_threshold,
        change_threshold=change_threshold,
        stability_seconds=stability_seconds,
        text_threshold=text_threshold,
        image_threshold=image_threshold,
        preview_width=preview_width,
        cache=cache,
        key=key,
        emit=emit,
    )
