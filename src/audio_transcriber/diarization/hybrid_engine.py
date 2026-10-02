"""Гибридная диаризация: оконный EEND + глобальная склейка говорящих (#64).

Модель NeMo-Speech.cpp (Sortformer, EEND) имеет фиксированную голову на 4
спикера: на записи с большим числом участников она «схлопывает» говорящих.
Гибридный движок обходит лимит, комбинируя быстрый локальный EEND с глобальной
кластеризацией эмбеддингов:

1. Аудио режется на окна :data:`~DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS` с
   небольшим перекрытием. По возможности границы сдвигаются на паузы (минимум
   энергии), чтобы не разрезать реплику.
2. В каждом окне ``nemo-speech diarize`` даёт локальных говорящих (<=4).
3. Для каждого локального говорящего в окне собирается его речь; при
   достаточной длительности считается speaker-эмбеддинг (общий модуль
   :mod:`audio_transcriber.diarization.embeddings`, модель CAM++).
4. Все эмбеддинги кластеризуются глобально (агломеративно, косинус,
   average-linkage). Каждому кластеру — глобальный ``SPEAKER_XX``.
5. Локальные сегменты перекладываются в глобальные ID и склеиваются;
   перекрытие окон учтено зонами владения (без дублей и пропусков).

Мягкая деградация: сбой отдельного окна — пропуск; недоступность эмбеддера/
модели или слишком мало эмбеддингов — понятная
:class:`HybridDiarizationError`, по которой маршрутизация не выбирает гибрид
(доступность проверяется заранее в фабрике).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_HYBRID_MAX_EMBEDDING_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_NEMO_SPEECH_BINARY,
    DEFAULT_NEMO_SPEECH_DEVICE,
    DEFAULT_NEMO_SPEECH_MODEL,
)
from audio_transcriber.diarization import embeddings as embedding_utils
from audio_transcriber.diarization.nemo_speech_engine import (
    DEFAULT_NEMO_SPEECH_TIMEOUT,
    diarize_audio,
)
from audio_transcriber.diarization.overlap import compute_overlap_regions
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.env import binary_available
from audio_transcriber.utils.exceptions import DiarizationError

logger = logging.getLogger(__name__)

#: Версия алгоритма гибрида. Изменение окон/склейки/кластеризации меняет
#: результат при тех же входах — участвует в ключе кэша диаризации.
DIARIZATION_HYBRID_IMPL_VERSION = 1

#: Радиус поиска паузы при сдвиге границы окна (секунды).
DEFAULT_SNAP_RADIUS_SECONDS = 3.0

#: Длина кадра оценки энергии при поиске паузы (секунды).
_ENERGY_FRAME_SECONDS = 0.05

#: Склейка соседних сегментов одного говорящего с зазором не больше (секунды).
_MERGE_GAP_SECONDS = 0.02

#: Минимальная длина сегмента после обрезки по зоне владения окна (секунды).
_MIN_STITCH_SECONDS = 1e-3


class HybridDiarizationError(DiarizationError):
    """Гибридная диаризация недоступна или не дала результата.

    Отдельный подкласс — чтобы вызывающий код (и тесты) могли отличить
    ожидаемую деградацию гибрида от прочих ошибок диаризации.
    """


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    """Окно анализа: границы и «зона владения» в сэмплах.

    ``start``/``end`` — диапазон, отдаваемый в ``nemo-speech`` (с перекрытием).
    ``own_start``/``own_end`` — непересекающийся интервал, сегменты из которого
    попадают в итог: так перекрытие соседних окон не даёт дублей.
    """

    start: int
    end: int
    own_start: int
    own_end: int

    @property
    def duration(self) -> float:
        """Длительность окна в секундах."""
        return (self.end - self.start) / SAMPLE_RATE


def _quietest_offset(
    samples: np.ndarray,
    target: int,
    radius: int,
    *,
    frame: int,
) -> int:
    """Смещение ближайшего кадра с минимальной энергией около ``target``.

    Используется для сдвига границы окна на паузу. Возвращает ``target``, если
    искать негде (край записи, слишком короткий диапазон).
    """
    if frame <= 0 or radius <= 0:
        return target
    low = max(0, target - radius)
    high = min(samples.shape[0], target + radius)
    span = high - low
    if span < frame:
        return target
    count = span // frame
    if count < 1:
        return target
    trimmed = samples[low : low + count * frame].reshape(count, frame)
    energy = np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1))
    best = int(np.argmin(energy))
    return low + best * frame + frame // 2


def plan_windows(
    total_samples: int,
    *,
    window_seconds: float = DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    samples: np.ndarray | None = None,
    snap_radius_seconds: float = DEFAULT_SNAP_RADIUS_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> list[AnalysisWindow]:
    """Планирует окна анализа и их непересекающиеся зоны владения.

    Окна идут с шагом ``window - overlap`` и покрывают всю запись. Если передан
    ``samples``, внутренние границы сдвигаются на ближайшую паузу в пределах
    ``snap_radius_seconds`` (не ближе половины перекрытия к соседу).
    """
    if total_samples <= 0:
        return []
    window_samples = max(1, int(window_seconds * sample_rate))
    overlap_samples = max(0, min(int(overlap_seconds * sample_rate), window_samples - 1))
    hop = max(1, window_samples - overlap_samples)

    starts: list[int] = []
    position = 0
    while position < total_samples:
        starts.append(position)
        if position + window_samples >= total_samples:
            break
        position += hop

    if samples is not None and len(starts) > 1 and snap_radius_seconds > 0.0:
        frame = max(1, int(_ENERGY_FRAME_SECONDS * sample_rate))
        radius = int(snap_radius_seconds * sample_rate)
        min_step = max(1, overlap_samples // 2)
        for index in range(1, len(starts)):
            snapped = _quietest_offset(samples, starts[index], radius, frame=frame)
            lower = starts[index - 1] + min_step
            upper = total_samples - 1
            starts[index] = max(lower, min(snapped, upper))

    spans: list[tuple[int, int]] = [
        (start, min(start + window_samples, total_samples)) for start in starts
    ]

    windows: list[AnalysisWindow] = []
    last = len(spans) - 1
    for index, (start, end) in enumerate(spans):
        own_start = (
            0 if index == 0 else (spans[index - 1][1] + start) // 2
        )
        own_end = (
            total_samples if index == last else (end + spans[index + 1][0]) // 2
        )
        own_start = max(start, min(own_start, end - 1))
        own_end = min(end, max(own_end, own_start + 1))
        windows.append(
            AnalysisWindow(start=start, end=end, own_start=own_start, own_end=own_end)
        )
    return windows


def _group_by_speaker(segments: Sequence[SpeakerSegment]) -> dict[str, list[SpeakerSegment]]:
    """Группирует сегменты окна по локальному идентификатору говорящего."""
    grouped: dict[str, list[SpeakerSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.speaker_id, []).append(segment)
    return grouped


def _collect_speaker_samples(
    samples: np.ndarray,
    window: AnalysisWindow,
    segments: Sequence[SpeakerSegment],
    *,
    max_seconds: float,
) -> np.ndarray:
    """Собирает речь локального говорящего в окне в один клип (для эмбеддинга)."""
    budget = int(max_seconds * SAMPLE_RATE)
    if budget <= 0:
        return np.empty(0, dtype=np.float32)
    pieces: list[np.ndarray] = []
    collected = 0
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        start = max(0, window.start + int(segment.start * SAMPLE_RATE))
        end = min(samples.shape[0], window.start + int(segment.end * SAMPLE_RATE))
        if end <= start:
            continue
        piece = samples[start:end]
        pieces.append(piece)
        collected += piece.shape[0]
        if collected >= budget:
            break
    if not pieces:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(pieces)[:budget]


def _assign_global_clusters(
    observations: Sequence[tuple[int, str, np.ndarray]],
    *,
    threshold: float,
    expected_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> np.ndarray:
    """Глобально кластеризует эмбеддинги локальных говорящих → метки.

    ``expected_speakers`` (оценка числа говорящих или явный ``num_speakers``)
    используется как ориентир — точное число кластеров. Иначе — порог; при
    заданных ``min/max`` число кластеров при необходимости доводится до границ.
    """
    matrix = np.stack([vector for _window, _speaker, vector in observations]).astype(np.float32)
    count = int(matrix.shape[0])
    if count <= 1:
        return np.zeros(count, dtype=int)

    n_clusters: int | None = None
    if expected_speakers is not None:
        n_clusters = max(1, min(int(expected_speakers), count))

    labels = embedding_utils.cluster_embeddings(
        matrix, threshold=threshold, n_clusters=n_clusters
    )
    if n_clusters is None and (min_speakers is not None or max_speakers is not None):
        distinct = len(set(labels.tolist()))
        target: int | None = None
        if max_speakers is not None and distinct > max_speakers:
            target = max(1, min(int(max_speakers), count))
        elif min_speakers is not None and distinct < min_speakers:
            target = max(1, min(int(min_speakers), count))
        if target is not None and target != distinct:
            labels = embedding_utils.cluster_embeddings(
                matrix, threshold=threshold, n_clusters=target
            )
    return labels


def _merge_adjacent(
    segments: Sequence[SpeakerSegment], *, gap: float = _MERGE_GAP_SECONDS
) -> list[SpeakerSegment]:
    """Склеивает соседние сегменты одного говорящего (в т.ч. через мелкий зазор)."""
    merged: list[SpeakerSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start, item.end, item.speaker_id)):
        if not merged:
            merged.append(segment)
            continue
        last = merged[-1]
        if segment.speaker_id == last.speaker_id and segment.start <= last.end + gap:
            merged[-1] = SpeakerSegment(
                start=last.start,
                end=max(last.end, segment.end),
                speaker_id=last.speaker_id,
            )
        else:
            merged.append(segment)
    return merged


class HybridSpeakerDiarizer:
    """Диаризация: оконный nemo-speech + глобальная склейка по эмбеддингам.

    Реализует протокол ``SpeakerDiarizer``. Enrollment по образцам для этого
    движка не предусмотрен (как и у чистого EEND nemo-speech).

    :param device: устройство ``nemo-speech`` (``auto``/``vulkan``/``cpu``).
    :param binary: путь или имя бинарника ``nemo-speech``.
    :param lib_path: каталог разделяемых библиотек бандла (``lib/``).
    :param model: модель ``nemo-speech`` (имя, HF-репозиторий или ``.gguf``).
    :param window_seconds: длительность окна (секунды).
    :param overlap_seconds: перекрытие соседних окон (секунды).
    :param min_speaker_seconds: порог речи локального говорящего для эмбеддинга.
    :param max_embedding_seconds: максимум аудио говорящего для одного эмбеддинга.
    :param embedding_model: имя/путь ONNX-модели эмбеддингов (CAM++).
    :param threshold: порог косинусного расстояния глобальной кластеризации.
    :param expected_speakers: ориентир числа говорящих (оценка или ``num_speakers``).
    :param embedder: готовый эмбеддер (для тестов; по умолчанию создаётся сам).
    """

    supports_enrollment = False

    def __init__(
        self,
        device: str = DEFAULT_NEMO_SPEECH_DEVICE,
        *,
        binary: str = DEFAULT_NEMO_SPEECH_BINARY,
        lib_path: str | None = None,
        model: str = DEFAULT_NEMO_SPEECH_MODEL,
        on_progress: ProgressCallback | None = None,
        timeout: float = DEFAULT_NEMO_SPEECH_TIMEOUT,
        window_seconds: float = DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
        overlap_seconds: float = DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
        min_speaker_seconds: float = DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
        max_embedding_seconds: float = DEFAULT_DIARIZATION_HYBRID_MAX_EMBEDDING_SECONDS,
        embedding_model: str = DEFAULT_DIARIZATION_ESTIMATE_MODEL,
        embedding_model_dir: Path | None = None,
        threshold: float = DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
        expected_speakers: int | None = None,
        embedder: object | None = None,
    ) -> None:
        self._device = device
        self._binary = binary
        self._lib_path = lib_path
        self._model = model
        self._on_progress = on_progress
        self._timeout = timeout
        self._window_seconds = window_seconds
        self._overlap_seconds = overlap_seconds
        self._min_speaker_seconds = min_speaker_seconds
        self._max_embedding_seconds = max_embedding_seconds
        self._embedding_model = embedding_model
        self._embedding_model_dir = embedding_model_dir
        self._threshold = threshold
        self._expected_speakers = expected_speakers
        self._embedder = embedder
        self._overlaps: list[SpeakerOverlap] = []

    def _emit(self, message: str, *, fraction: float | None, detail: str = "") -> None:
        emit = self._on_progress
        if emit is not None:
            emit(
                ProgressEvent(
                    "diarization",
                    message=message,
                    fraction=fraction,
                    detail=detail,
                )
            )

    def _resolve_embedder(self) -> object:
        if self._embedder is not None:
            return self._embedder
        if not embedding_utils.sherpa_available():
            raise HybridDiarizationError(
                "sherpa-onnx не установлен — гибридная диаризация недоступна "
                "(uv pip install sherpa-onnx)"
            )
        model_path = embedding_utils.resolve_embedding_model(
            self._embedding_model,
            self._embedding_model_dir,
            on_progress=self._on_progress,
        )
        if model_path is None:
            raise HybridDiarizationError(
                f"Модель эмбеддингов {self._embedding_model!r} недоступна — "
                "гибридная диаризация невозможна (укажите путь к .onnx или "
                "разрешите загрузку)"
            )
        self._embedder = embedding_utils.SpeakerEmbedder(model_path)
        return self._embedder

    def _collect_observations(
        self,
        audio_path: Path,
        samples: np.ndarray,
        windows: Sequence[AnalysisWindow],
        embedder: object,
    ) -> tuple[list[list[SpeakerSegment]], list[tuple[int, str, np.ndarray]]]:
        """Диаризует окна и считает эмбеддинги локальных говорящих."""
        window_segments: list[list[SpeakerSegment]] = []
        observations: list[tuple[int, str, np.ndarray]] = []
        total = len(windows)
        for index, window in enumerate(windows):
            self._emit(
                "Гибридная диаризация",
                fraction=index / total,
                detail=f"окно {index + 1}/{total}",
            )
            window_wave = samples[window.start : window.end]
            segments = diarize_audio(
                audio_path,
                binary=self._binary,
                device=self._device,
                model=self._model,
                lib_path=self._lib_path,
                timeout=self._timeout,
                waveform=window_wave,
            )
            if segments is None:
                logger.warning(
                    "Гибридная диаризация: окно %d/%d не обработано — пропущено",
                    index + 1,
                    total,
                )
                window_segments.append([])
                continue
            window_segments.append(segments)

            for speaker, speaker_segments in _group_by_speaker(segments).items():
                speech_seconds = sum(
                    max(0.0, segment.end - segment.start) for segment in speaker_segments
                )
                if speech_seconds < self._min_speaker_seconds:
                    continue
                clip = _collect_speaker_samples(
                    samples,
                    window,
                    speaker_segments,
                    max_seconds=self._max_embedding_seconds,
                )
                if clip.size == 0:
                    continue
                try:
                    vector = embedder.embed(clip)  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001 — мягкая деградация окна
                    logger.warning(
                        "Гибридная диаризация: не удалось посчитать эмбеддинг "
                        "говорящего %s в окне %d (%s) — говорящий пропущен",
                        speaker,
                        index + 1,
                        exc,
                    )
                    continue
                observations.append((index, speaker, np.asarray(vector, dtype=np.float32)))
        return window_segments, observations

    def _stitch(
        self,
        windows: Sequence[AnalysisWindow],
        window_segments: Sequence[Sequence[SpeakerSegment]],
        observations: Sequence[tuple[int, str, np.ndarray]],
        labels: np.ndarray,
    ) -> list[SpeakerSegment]:
        """Перекладывает локальные сегменты в глобальные ID и склеивает их."""
        key_to_cluster = {
            (window_index, speaker): int(label)
            for (window_index, speaker, _vector), label in zip(observations, labels, strict=True)
        }

        # Глобальные ID — по первому появлению кластера в записи (детерминированно).
        cluster_to_global: dict[int, str] = {}
        for index, segments in enumerate(window_segments):
            for segment in sorted(segments, key=lambda item: (item.start, item.end)):
                cluster = key_to_cluster.get((index, segment.speaker_id))
                if cluster is None or cluster in cluster_to_global:
                    continue
                cluster_to_global[cluster] = f"SPEAKER_{len(cluster_to_global):02d}"

        stitched: list[SpeakerSegment] = []
        for index, window in enumerate(windows):
            own_start = window.own_start / SAMPLE_RATE
            own_end = window.own_end / SAMPLE_RATE
            offset = window.start / SAMPLE_RATE
            for segment in window_segments[index]:
                cluster = key_to_cluster.get((index, segment.speaker_id))
                if cluster is None:
                    continue
                start = max(segment.start + offset, own_start)
                end = min(segment.end + offset, own_end)
                if end - start <= _MIN_STITCH_SECONDS:
                    continue
                stitched.append(
                    SpeakerSegment(
                        start=start,
                        end=end,
                        speaker_id=cluster_to_global[cluster],
                    )
                )
        return _merge_adjacent(stitched)

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        waveform: np.ndarray | None = None,
    ) -> list[SpeakerSegment]:
        self._overlaps = []

        if not binary_available(self._binary):
            raise HybridDiarizationError(
                f"Бинарник nemo-speech не найден ({self._binary!r}) — гибридная "
                "диаризация невозможна. Задайте NEMO_SPEECH_BINARY."
            )

        if waveform is None:
            waveform = load_waveform(audio_path)
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise HybridDiarizationError("Пустой waveform — гибридная диаризация невозможна")

        embedder = self._resolve_embedder()
        windows = plan_windows(
            samples.shape[0],
            window_seconds=self._window_seconds,
            overlap_seconds=self._overlap_seconds,
            samples=samples,
        )
        if not windows:
            raise HybridDiarizationError("Не удалось спланировать окна анализа")

        window_segments, observations = self._collect_observations(
            audio_path, samples, windows, embedder
        )
        if not observations:
            raise HybridDiarizationError(
                "Не удалось получить ни одного эмбеддинга говорящего "
                "(слишком короткие окна/речь или сбой эмбеддера) — гибридная "
                "диаризация не дала результата"
            )

        expected = num_speakers if num_speakers is not None else self._expected_speakers
        labels = _assign_global_clusters(
            observations,
            threshold=self._threshold,
            expected_speakers=expected,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        segments = self._stitch(windows, window_segments, observations, labels)
        self._overlaps = compute_overlap_regions(segments)
        self._emit("Гибридная диаризация", fraction=1.0, detail="готово")
        return segments

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """Интервалы наложения речи из последнего вызова :meth:`diarize`."""
        return list(self._overlaps)
