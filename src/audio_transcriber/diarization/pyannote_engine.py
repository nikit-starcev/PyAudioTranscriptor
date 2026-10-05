"""Реализация диаризации на pyannote.audio."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from audio_transcriber.config.defaults import DEFAULT_DIARIZATION_MIN_DURATION_OFF
from audio_transcriber.diarization.overlap import compute_overlap_regions
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.exceptions import DiarizationError

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"

#: Таймаут сетевого обращения к Hugging Face (секунды) при загрузке модели без
#: локальной копии. Ограничивает ожидание на 401/недоступности: иначе клиент
#: hf_hub_download может надолго «зависнуть» на ретраях (CLOSE-WAIT).
DEFAULT_HF_TIMEOUT_SECONDS = 15


def _import_pipeline_class() -> Any:
    """Импортирует ``pyannote.audio.Pipeline`` (отдельная точка для тестов)."""
    from pyannote.audio import Pipeline

    return Pipeline


@contextmanager
def _huggingface_offline() -> Iterator[None]:
    """Переводит huggingface_hub в offline-режим на время блока.

    При заданной локальной копии модели обращаться в сеть за подмоделями
    нельзя: если чего-то не хватает локально, ошибка будет быстрой и понятной,
    а не зависанием на ретраях gated-репозитория. Константа читается
    huggingface_hub при каждом запросе, поэтому подмена действует сразу.
    """
    try:
        import huggingface_hub.constants as hf_constants
    except Exception:  # noqa: BLE001 — huggingface_hub может отсутствовать
        yield
        return
    previous = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = previous


@contextmanager
def _huggingface_bounded_timeout(seconds: int) -> Iterator[None]:
    """Ограничивает сетевые таймауты HF, чтобы 401/сбой падали быстро.

    ``hf_hub_download`` читает эти константы при каждом вызове, поэтому
    подмена модульных значений действует на текущую загрузку. Ретраи на
    сетевых ошибках остаются, но каждый запрос ограничен ``seconds`` секундами.
    """
    try:
        import huggingface_hub.constants as hf_constants
    except Exception:  # noqa: BLE001 — huggingface_hub может отсутствовать
        yield
        return
    previous_etag = hf_constants.HF_HUB_ETAG_TIMEOUT
    previous_download = hf_constants.HF_HUB_DOWNLOAD_TIMEOUT
    hf_constants.HF_HUB_ETAG_TIMEOUT = seconds
    hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = seconds
    try:
        yield
    finally:
        hf_constants.HF_HUB_ETAG_TIMEOUT = previous_etag
        hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = previous_download


class PyannoteSpeakerDiarizer:
    """Определяет говорящих через pyannote.audio. Реализует протокол ``SpeakerDiarizer``."""

    def __init__(
        self,
        device: Device,
        *,
        pipeline_name: str = DEFAULT_PIPELINE,
        hf_token: str | None = None,
        local_model_path: Path | str | None = None,
        on_progress: ProgressCallback | None = None,
        min_duration_off: float = DEFAULT_DIARIZATION_MIN_DURATION_OFF,
        clustering_threshold: float | None = None,
        clustering_fb: float | None = None,
    ) -> None:
        self._device = device
        self._pipeline_name = pipeline_name
        self._local_model_path = Path(local_model_path) if local_model_path else None
        self._on_progress = on_progress
        self._hf_token = (
            hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self._min_duration_off = min_duration_off
        self._clustering_threshold = clustering_threshold
        self._clustering_fb = clustering_fb
        self._pipeline: Any = None
        self._overlaps: list[SpeakerOverlap] = []

    def _hyperparameters(self) -> dict[str, Any]:
        """Собирает гиперпараметры диаризации для ``pipeline.instantiate``.

        Возвращает только явно заданные параметры: ``min_duration_off``
        передаётся всегда (наш дефолт 0.5), а ``clustering.threshold`` и
        ``clustering.Fb`` — лишь когда заданы, чтобы остальные значения
        остались дефолтами модели.
        """
        params: dict[str, Any] = {}
        segmentation: dict[str, Any] = {}
        if self._min_duration_off is not None:
            segmentation["min_duration_off"] = self._min_duration_off
        if segmentation:
            params["segmentation"] = segmentation
        clustering: dict[str, Any] = {}
        if self._clustering_threshold is not None:
            clustering["threshold"] = self._clustering_threshold
        if self._clustering_fb is not None:
            clustering["Fb"] = self._clustering_fb
        if clustering:
            params["clustering"] = clustering
        return params

    def _apply_hyperparameters(self, pipeline: Any) -> None:
        """Применяет гиперпараметры к загруженному пайплайну.

        ``instantiate`` в текущих версиях pyannote бросает исключение на
        неизвестный параметр, а не только предупреждает. Чтобы смена набора
        параметров между версиями не роняла диаризацию, ошибку логируем и
        оставляем дефолты модели.
        """
        params = self._hyperparameters()
        if not params:
            return
        try:
            pipeline.instantiate(params)
        except Exception as exc:  # noqa: BLE001 — версионная совместимость параметров
            logger.warning(
                "Не удалось применить гиперпараметры диаризации %s: %s — "
                "использую значения модели по умолчанию",
                params,
                exc,
            )
            return
        logger.debug("Гиперпараметры диаризации применены: %s", params)

    def _local_checkpoint(self) -> str | None:
        """Путь к локальной копии модели.

        ``None`` — локальная копия не задана (загрузка с Hugging Face). Если
        путь задан, но не существует, — быстрый понятный фейл вместо
        молчаливого похода в сеть за gated-моделью.
        """
        if self._local_model_path is None:
            return None
        if not self._local_model_path.exists():
            raise DiarizationError(
                f"Локальная модель диаризации не найдена: {self._local_model_path}. "
                "Проверьте путь PYANNOTE_LOCAL_MODEL/--pyannote-local-model."
            )
        return str(self._local_model_path)

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch

        pipeline_cls = _import_pipeline_class()
        local_checkpoint = self._local_checkpoint()

        try:
            if local_checkpoint is not None:
                # Локальная копия модели: грузится напрямую с диска, без
                # обращения к Hugging Face (токен и сеть не нужны). Offline-режим
                # гарантирует, что и подмодели не уйдут в сеть.
                logger.debug("Загрузка локальной модели диаризации из '%s'", local_checkpoint)
                with _huggingface_offline():
                    pipeline = pipeline_cls.from_pretrained(local_checkpoint)
            else:
                # Без локальной копии загрузка идёт с Hugging Face. Ограничиваем
                # сетевые таймауты, чтобы 401/недоступность падали за секунды,
                # а не зависали на долгих ретраях.
                logger.debug("Загрузка модели диаризации '%s'", self._pipeline_name)
                with _huggingface_bounded_timeout(DEFAULT_HF_TIMEOUT_SECONDS):
                    pipeline = pipeline_cls.from_pretrained(
                        self._pipeline_name, token=self._hf_token
                    )
        except Exception as exc:
            if local_checkpoint is not None:
                raise DiarizationError(
                    f"Не удалось загрузить локальную модель диаризации "
                    f"'{local_checkpoint}': {exc}"
                ) from exc
            raise DiarizationError(
                f"Не удалось загрузить модель диаризации '{self._pipeline_name}'. "
                "Если модель «gated», примите её условия на huggingface.co и укажите "
                "действующий токен (--hf-token или переменная окружения HF_TOKEN) — "
                "либо задайте локальную копию модели "
                "(--pyannote-local-model / PYANNOTE_LOCAL_MODEL) для работы офлайн. "
                f"Исходная ошибка: {exc}"
            ) from exc

        if pipeline is None:
            raise DiarizationError(
                f"Модель диаризации '{self._pipeline_name}' не найдена или недоступна"
            )

        # Гиперпараметры применяем сразу после загрузки, до переноса на
        # устройство (``instantiate`` вызывает повторную инициализацию пайплайна).
        self._apply_hyperparameters(pipeline)

        pipeline.to(torch.device(self._device.value))
        self._pipeline = pipeline
        return self._pipeline

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: np.ndarray | None = None,
    ) -> list[SpeakerSegment]:
        pipeline = self._load_pipeline()

        import torch
        from pyannote.audio.pipelines.utils.hook import ProgressHook

        # Переданный waveform (например, результат денойза) позволяет не
        # декодировать тот же файл повторно. Иначе декодируем сами — как раньше.
        if waveform is None:
            waveform = load_waveform(audio_path)
        audio_input = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }

        try:
            # Точное число говорящих приоритетнее диапазона: pyannote сам
            # разрешает приоритет ``num_speakers`` над min/max, но мы не
            # передаём лишние аргументы, чтобы не зависеть от версии.
            speaker_kwargs: dict[str, int] = {}
            if num_speakers is not None:
                speaker_kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers is not None:
                    speaker_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    speaker_kwargs["max_speakers"] = max_speakers

            emit = self._on_progress
            if emit is not None:

                def _hook(
                    step_name: object,
                    _step_artifact: object,
                    file: object = None,
                    total: float | None = None,
                    completed: float | None = None,
                ) -> None:
                    # pyannote передаёт completed/total в разных комбинациях:
                    # completed=None — одноразовый шаг (без прогресса),
                    # completed может превышать total (batch_size > num_chunks),
                    # поэтому прогресс нормализуется в диапазон [0; 1].
                    del file  # параметр обязателен (pyannote зовёт hook(file=...)), не используется
                    if completed is None or total is None:
                        fraction = None
                    elif completed >= total:
                        fraction = 1.0
                    else:
                        fraction = max(0.0, min(1.0, completed / total))
                    emit(
                        ProgressEvent(
                            "diarization",
                            message=str(step_name),
                            fraction=fraction,
                            detail=str(step_name),
                        )
                    )

                output = pipeline(audio_input, hook=_hook, **speaker_kwargs)
            else:
                # ProgressHook показывает прогресс внутренних этапов
                # (сегментация, эмбеддинги, кластеризация) в консоли.
                with ProgressHook() as hook:
                    output = pipeline(audio_input, hook=hook, **speaker_kwargs)
        except Exception as exc:
            raise DiarizationError(
                f"Ошибка при определении говорящих в файле {audio_path}: {exc}"
            ) from exc

        # Эксклюзивная разметка: в каждый момент один говорящий — её и отдаём
        # на объединение с ASR.
        exclusive = getattr(output, "exclusive_speaker_diarization", output)
        segments = [
            SpeakerSegment(start=turn.start, end=turn.end, speaker_id=speaker)
            for turn, _, speaker in exclusive.itertracks(yield_label=True)
        ]

        # Интервалы наложения речи берём из обычной (не эксклюзивной) разметки,
        # если она есть. В старых версиях pyannote её нет — тогда перекрытий
        # не будет (пустой список), конвейер не падает.
        overlap_source = getattr(output, "speaker_diarization", None)
        source = overlap_source if overlap_source is not None else exclusive
        source_segments = [
            SpeakerSegment(start=turn.start, end=turn.end, speaker_id=speaker)
            for turn, _, speaker in source.itertracks(yield_label=True)
        ]
        self._overlaps = compute_overlap_regions(source_segments)

        return segments

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """Интервалы наложения речи из последнего вызова :meth:`diarize`."""
        return list(self._overlaps)
