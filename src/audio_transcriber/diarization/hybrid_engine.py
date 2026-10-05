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
4. Все эмбеддинги кластеризуются глобально (агломеративно) **по порогу**.
   По умолчанию — ``ward``/euclidean на L2-нормированных векторах: на реальных
   данных он даёт меньше хвостовых кластеров и лучшее распределение, чем
   косинусный ``complete`` (TV к долям реплик протокола ≈0.12 против ≈0.19).
   Каждому кластеру — глобальный ``SPEAKER_XX``. Явное ``num_speakers``
   (пользователь задал точно) кластеризует ровно в это число — там linkage
   ``ward``/euclidean (лучшее распределение при фиксированном ``k``); дешёвая
   **оценка** N (нужна движку ``auto`` для маршрутизации) — только мягкий
   ориентир и не форсирует число кластеров, иначе недооценка N склеивала бы
   участников.
5. Локальные сегменты перекладываются в глобальные ID и склеиваются;
   перекрытие окон учтено зонами владения (без дублей и пропусков).

Обход лимита 4 внутри окна (#68): если в окне заняты все 4 головы Sortformer и
говорящий меняется почти на каждом соседнем сегменте, не исключено, что реальных
участников было больше 4 и лишние «прилипли» к занятым головам. Такое
«перегруженное» окно повторно обрабатывается более мелкими окнами (рекурсивно,
до ``DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH``), а локальные говорящие
затем склеиваются глобально. Детектор консервативен: обычные записи с ≤4
говорящими не дробятся, митигация отключается флагом
``DIARIZATION_HYBRID_OVERLOAD_SPLIT``.

Мягкая деградация: сбой отдельного окна — пропуск; недоступность эмбеддера/
модели или слишком мало эмбеддингов — понятная
:class:`HybridDiarizationError`, по которой маршрутизация не выбирает гибрид
(доступность проверяется заранее в фабрике).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_HYBRID_LINKAGE,
    DEFAULT_DIARIZATION_HYBRID_MAX_EMBEDDING_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH,
    DEFAULT_DIARIZATION_HYBRID_MIN_SPEAKER_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLAP_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_OVERLOAD_CHANGE_RATE,
    DEFAULT_DIARIZATION_HYBRID_OVERLOAD_MIN_SEGMENTS,
    DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT,
    DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
    DEFAULT_DIARIZATION_HYBRID_THRESHOLD,
    DEFAULT_DIARIZATION_HYBRID_WINDOW_SECONDS,
    DEFAULT_NEMO_SPEECH_BINARY,
    DEFAULT_NEMO_SPEECH_DEVICE,
    DEFAULT_NEMO_SPEECH_MODEL,
    NEMO_SPEECH_MAX_SPEAKERS,
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
#: 3 — кластеризация по порогу вместо форсирования числа кластеров по оценке N.
#: 4 — linkage complete/cosine (порог) и ward/euclidean (форсированный N)
#:     вместо average: average перемерживал кластеры через chaining.
#: 5 — linkage по порогу переключён на ward/euclidean и отдельный euclidean-порог
#:     (гибрид больше не переиспользует косинусный порог оценщика); порог
#:     min_speaker_seconds 1.5 → 3.0. Замер: complete/0.50 → k≈16, топ ~48%,
#:     TV≈0.19; ward/1.30 → k≈10, топ ~36%, TV≈0.12.
#: 6 — в ключ кэша диаризации добавлены фактический денойз и все параметры
#:     гибрида/оценщика (#83); версия поднята как сигнал смены семантики ключа.
DIARIZATION_HYBRID_IMPL_VERSION = 6

#: Запас (в говорящих) к мягкой оценке числа говорящих при кластеризации.
#: Оценка ``expected_speakers`` никогда не задаёт точное число кластеров: она
#: лишь ограничивает их сверху значением ``оценка + запас``. Запас нужен,
#: потому что дешёвая оценка заметно недооценивает N, а также шумна; порог
#: кластеризации — основной механизм, а оценка — только страховка от
#: лавинного переразбиения.
_ESTIMATE_HEADROOM = 4

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


def window_is_overloaded(
    segments: Sequence[SpeakerSegment],
    *,
    speaker_cap: int = NEMO_SPEECH_MAX_SPEAKERS,
    change_rate: float = DEFAULT_DIARIZATION_HYBRID_OVERLOAD_CHANGE_RATE,
    min_segments: int = DEFAULT_DIARIZATION_HYBRID_OVERLOAD_MIN_SEGMENTS,
) -> bool:
    """Похоже ли, что в окне реально говорили больше ``speaker_cap`` человек.

    EEND-модель Sortformer держит фиксированную голову на ``speaker_cap``
    говорящих: если в окне их больше, лишние «прилипают» к занятым головам.
    Прямо определить это по выходу нельзя, поэтому сигнал косвенный: заняты
    **все** головы и сегменты сильно «путаются» — говорящий меняется почти на
    каждом соседнем сегменте. Проверка заведомо консервативна (обычная
    запись с ≤4 говорящими не дробится зря).
    """
    if speaker_cap < 1:
        return False
    distinct = {segment.speaker_id for segment in segments}
    if len(distinct) < speaker_cap:
        return False
    ordered = sorted(segments, key=lambda item: (item.start, item.end, item.speaker_id))
    if len(ordered) < max(2, min_segments):
        return False
    transitions = sum(
        1 for previous, current in pairwise(ordered) if previous.speaker_id != current.speaker_id
    )
    return transitions / (len(ordered) - 1) >= change_rate


def _plan_subwindows(
    parent: AnalysisWindow,
    samples: np.ndarray,
    *,
    subwindow_seconds: float,
    overlap_seconds: float,
    snap_radius_seconds: float = DEFAULT_SNAP_RADIUS_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> list[AnalysisWindow]:
    """Планирует мелкие окна внутри «перегруженного» ``parent``.

    Границы привязаны к паузам, а зоны владения покрывают **зону владения
    родителя** без пропусков и дублей с соседними окнами: крайние окна
    «ужимаются» до ``parent.own_start``/``parent.own_end``. Если родитель и так
    короче мелкого окна, возвращается он сам (без дробления).
    """
    parent_length = parent.end - parent.start
    if parent_length <= 0:
        return [parent]
    subwindow_samples = max(1, int(subwindow_seconds * sample_rate))
    if subwindow_samples >= parent_length:
        return [parent]

    overlap = max(0.0, min(overlap_seconds, subwindow_seconds / 2.0))
    local = samples[parent.start : parent.end]
    planned = plan_windows(
        parent_length,
        window_seconds=subwindow_seconds,
        overlap_seconds=overlap,
        samples=local,
        snap_radius_seconds=snap_radius_seconds,
        sample_rate=sample_rate,
    )
    if len(planned) <= 1:
        return [parent]

    result: list[AnalysisWindow] = []
    last = len(planned) - 1
    for index, window in enumerate(planned):
        own_start = parent.start + window.own_start
        own_end = parent.start + window.own_end
        if index == 0:
            own_start = parent.own_start
        if index == last:
            own_end = parent.own_end
        own_start = max(parent.start, min(own_start, parent.end - 1))
        own_end = min(parent.end, max(own_end, own_start + 1))
        result.append(
            AnalysisWindow(
                start=parent.start + window.start,
                end=parent.start + window.end,
                own_start=own_start,
                own_end=own_end,
            )
        )
    return result


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
    num_speakers: int | None,
    expected_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    linkage: str = DEFAULT_DIARIZATION_HYBRID_LINKAGE,
) -> np.ndarray:
    """Глобально кластеризует эмбеддинги локальных говорящих → метки.

    Различаются два источника числа говорящих:

    * **явное** ``num_speakers`` (пользователь задал точно) — кластеризация
      идёт ровно в это число кластеров (в пределах числа наблюдений);
    * **оценка** ``expected_speakers`` (от оценщика, нужна в первую очередь для
      маршрутизации движка) — лишь мягкий ориентир: кластеризация идёт **по
      порогу** (``n_clusters=None``), а оценка ограничивает число кластеров
      сверху значением ``оценка + :data:`_ESTIMATE_HEADROOM```. Это страховка
      от лавинного переразбиения, но **не** жёсткое равенство ``N`` (дешёвая
      оценка недооценивает N, и форсирование по ней склеивало говорящих).

    ``min_speakers``/``max_speakers`` — явные границы пользователя; применяются
    только при кластеризации по порогу и приоритетнее мягкой оценки.

    ``linkage`` задаёт метод связи **пороговой** ветки: по умолчанию ``ward``
    (метрика ``euclidean`` на L2-нормированных векторах) — на реальных данных он
    даёт наименьший TV и меньше хвостовых кластеров, чем ``complete``/cosine.
    Любой **форсированный** ``n_clusters`` (явный ``num_speakers``, а также
    доводка до ``min/max``/мягкого потолка) всегда идёт с ``ward``/euclidean —
    при фиксированном ``k`` он даёт лучшее распределение. Чтобы вернуть прежнее
    поведение (порог по косинусу), передайте ``linkage="complete"``.
    """
    matrix = np.stack([vector for _window, _speaker, vector in observations]).astype(np.float32)
    count = int(matrix.shape[0])
    if count <= 1:
        return np.zeros(count, dtype=int)

    metric = "cosine" if linkage in {"complete", "average", "single"} else "euclidean"

    if num_speakers is not None:
        exact = max(1, min(int(num_speakers), count))
        return embedding_utils.cluster_embeddings(
            matrix,
            threshold=threshold,
            n_clusters=exact,
            linkage="ward",
            metric="euclidean",
        )

    labels = embedding_utils.cluster_embeddings(
        matrix,
        threshold=threshold,
        n_clusters=None,
        linkage=linkage,
        metric=metric,
    )
    distinct = len(set(labels.tolist()))

    target: int | None = None
    if max_speakers is not None and distinct > max_speakers:
        target = max(1, min(int(max_speakers), count))
    elif min_speakers is not None and distinct < min_speakers:
        target = max(1, min(int(min_speakers), count))
    elif expected_speakers is not None:
        soft_cap = max(1, min(int(expected_speakers) + _ESTIMATE_HEADROOM, count))
        if distinct > soft_cap:
            target = soft_cap
    if target is not None and target != distinct:
        labels = embedding_utils.cluster_embeddings(
            matrix,
            threshold=threshold,
            n_clusters=target,
            linkage="ward",
            metric="euclidean",
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
    :param threshold: порог расстояния глобальной кластеризации (для ``ward`` —
        в единицах евклидова расстояния на L2-нормированных векторах).
    :param linkage: метод связи пороговой ветки кластеризации (``ward`` по
        умолчанию; ``complete`` — прежнее косинусное поведение).
    :param expected_speakers: **мягкая оценка** числа говорящих (обычно от
        оценщика для маршрутизации ``auto``). Не форсирует число кластеров: см.
        :func:`_assign_global_clusters`. Явное число задаётся аргументом
        ``num_speakers`` метода :meth:`diarize` — только оно кластеризует ровно
        в это число.
    :param overload_split: переобрабатывать ли «перегруженные» окна мелкими.
    :param subwindow_seconds: длительность мелкого окна при переобработке.
    :param max_split_depth: максимальная глубина рекурсивной нарезки.
    :param embedder: готовый эмбеддер (для тестов; по умолчанию создаётся сам).
    """

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
        linkage: str = DEFAULT_DIARIZATION_HYBRID_LINKAGE,
        expected_speakers: int | None = None,
        overload_split: bool = DEFAULT_DIARIZATION_HYBRID_OVERLOAD_SPLIT,
        subwindow_seconds: float = DEFAULT_DIARIZATION_HYBRID_SUBWINDOW_SECONDS,
        max_split_depth: int = DEFAULT_DIARIZATION_HYBRID_MAX_SPLIT_DEPTH,
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
        self._linkage = linkage
        self._expected_speakers = expected_speakers
        self._overload_split = overload_split
        self._subwindow_seconds = subwindow_seconds
        self._max_split_depth = max_split_depth
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

    def enrollment_engine(self) -> object | None:
        """Эмбеддер гибрида (CAM++) для enrollment; ``None`` при недоступности.

        Enrollment сопоставляет образцы голоса и кластеры **в одном**
        пространстве эмбеддингов. Гибрид кластеризует говорящих CAM++
        (sherpa-onnx), поэтому для enrollment нужно использовать тот же
        эмбеддер, а не pyannote-WeSpeaker: иначе «образец × кластер» сравниваются
        разными моделями и уверенные совпадения теряются. Ошибка (нет
        sherpa/model) не поднимается — вызывающий код откатывается на движок по
        умолчанию (pyannote).
        """
        try:
            return self._resolve_embedder()
        except HybridDiarizationError as exc:
            logger.warning(
                "Гибрид: эмбеддер CAM++ недоступен для enrollment (%s) — "
                "сопоставление пойдёт движком по умолчанию",
                exc,
            )
            return None

    def _diarize_window(
        self, audio_path: Path, samples: np.ndarray, window: AnalysisWindow
    ) -> list[SpeakerSegment] | None:
        """Диаризует один диапазон аудио через ``nemo-speech``."""
        window_wave = samples[window.start : window.end]
        return diarize_audio(
            audio_path,
            binary=self._binary,
            device=self._device,
            model=self._model,
            lib_path=self._lib_path,
            timeout=self._timeout,
            waveform=window_wave,
        )

    def _should_split(self, segments: Sequence[SpeakerSegment], *, depth: int) -> bool:
        """Нужно ли переобработать окно более мелкими (см. #68)."""
        if not self._overload_split or depth >= self._max_split_depth:
            return False
        return window_is_overloaded(segments)

    def _resolve_units(
        self,
        audio_path: Path,
        samples: np.ndarray,
        windows: Sequence[AnalysisWindow],
        *,
        depth: int = 0,
    ) -> tuple[list[AnalysisWindow], list[list[SpeakerSegment]]]:
        """Диаризует окна и разбивает «перегруженные» на более мелкие.

        Возвращает выровненные списки ``(окна, сегменты)``: у «перегруженного»
        окна вместо одного элемента появляются его мелкие подокна (рекурсивно,
        до ``max_split_depth``), локальные говорящие которых затем
        склеиваются глобально по эмбеддингам. Сбой окна — пустой список сегментов.
        """
        units: list[AnalysisWindow] = []
        unit_segments: list[list[SpeakerSegment]] = []
        for window in windows:
            segments = self._diarize_window(audio_path, samples, window)
            if segments is None:
                logger.warning(
                    "Гибридная диаризация: окно %.1f–%.1f с не обработано — пропущено",
                    window.start / SAMPLE_RATE,
                    window.end / SAMPLE_RATE,
                )
                units.append(window)
                unit_segments.append([])
                continue
            if self._should_split(segments, depth=depth):
                subwindows = _plan_subwindows(
                    window,
                    samples,
                    subwindow_seconds=self._subwindow_seconds,
                    overlap_seconds=self._overlap_seconds,
                )
                if len(subwindows) > 1:
                    logger.info(
                        "Гибрид: перегруженное окно %.1f–%.1f с (%d лок. говорящих, "
                        "%d сегментов) → переобработка %d мелкими окнами (глубина %d)",
                        window.start / SAMPLE_RATE,
                        window.end / SAMPLE_RATE,
                        len({segment.speaker_id for segment in segments}),
                        len(segments),
                        len(subwindows),
                        depth + 1,
                    )
                    sub_units, sub_segments = self._resolve_units(
                        audio_path, samples, subwindows, depth=depth + 1
                    )
                    units.extend(sub_units)
                    unit_segments.extend(sub_segments)
                    continue
            units.append(window)
            unit_segments.append(segments)
        return units, unit_segments

    def _collect_embeddings(
        self,
        samples: np.ndarray,
        units: Sequence[AnalysisWindow],
        unit_segments: Sequence[Sequence[SpeakerSegment]],
        embedder: object,
    ) -> list[tuple[int, str, np.ndarray]]:
        """Считает эмбеддинги локальных говорящих каждой единицы анализа."""
        observations: list[tuple[int, str, np.ndarray]] = []
        total = len(units)
        for index, (window, segments) in enumerate(
            zip(units, unit_segments, strict=True)
        ):
            self._emit(
                "Гибридная диаризация",
                fraction=index / total,
                detail=f"окно {index + 1}/{total}",
            )
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
        return observations

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

        units, unit_segments = self._resolve_units(audio_path, samples, windows)
        observations = self._collect_embeddings(samples, units, unit_segments, embedder)
        if not observations:
            raise HybridDiarizationError(
                "Не удалось получить ни одного эмбеддинга говорящего "
                "(слишком короткие окна/речь или сбой эмбеддера) — гибридная "
                "диаризация не дала результата"
            )

        labels = _assign_global_clusters(
            observations,
            threshold=self._threshold,
            num_speakers=num_speakers,
            expected_speakers=self._expected_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            linkage=self._linkage,
        )
        segments = self._stitch(units, unit_segments, observations, labels)
        self._overlaps = compute_overlap_regions(segments)
        self._emit("Гибридная диаризация", fraction=1.0, detail="готово")
        return segments

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """Интервалы наложения речи из последнего вызова :meth:`diarize`."""
        return list(self._overlaps)
