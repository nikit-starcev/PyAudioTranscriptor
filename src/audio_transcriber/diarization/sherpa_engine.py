"""Альтернативная диаризация на sherpa-onnx (#63).

Движок ``sherpa`` полностью работает на ONNX Runtime (CPU), без PyTorch и без
внешних бинарников. Конвейер простой и переиспользует уже имеющиеся компоненты:

1. **Сегментация речи** — silero VAD через ``sherpa-onnx`` (общий помощник
   :func:`~audio_transcriber.diarization.speaker_count.detect_speech_spans`,
   тот же, что у оценщика числа говорящих) режет запись на речевые фрагменты.
2. **Эмбеддинги говорящего** — :class:`~audio_transcriber.diarization.embeddings.SpeakerEmbedder`
   (3D-Speaker CAM++, та же модель, что у оценщика и гибрида). Каждый речевой
   фрагмент режется на окна фиксированной длины, для каждого считается
   L2-нормированный эмбеддинг.
3. **Агломеративная кластеризация** — та же глобальная склейка, что у гибрида
   (:func:`~audio_transcriber.diarization.hybrid_engine._assign_global_clusters`):
   по косинусному порогу (``complete``-linkage) либо в точное число кластеров
   при явном ``num_speakers``.
4. **Сегменты** — каждому окну достаётся кластер, соседние окна одного
   говорящего склеиваются; на выходе обычные ``(speaker_id, start, end)``.

Enrollment (имена по образцам) сохраняется: движок отдаёт свой CAM++-эмбеддер
через :meth:`SherpaSpeakerDiarizer.enrollment_engine`, поэтому образцы голоса и
кластеры сравниваются в одном пространстве эмбеддингов.

Ограничения (не «продавать» лишнее): VAD даёт эксклюзивную разметку, поэтому
**перекрытия речи не детектируются** (``overlap_regions`` всегда пуст) — по
этому качеству движок не лучше pyannote. GPU-ONNX на AMD не используется —
счёт идёт на CPU. Движок **не** выбирается режимом ``auto``: он альтернативный
и включается явно (``DIARIZATION_ENGINE=sherpa``).

Мягкая деградация: при отсутствии ``sherpa-onnx``/моделей поднимается понятная
:class:`SherpaDiarizationError` с подсказкой, а не падение по необработанному
исключению.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_DIARIZATION_ESTIMATE_MODEL,
    DEFAULT_DIARIZATION_SHERPA_LINKAGE,
    DEFAULT_DIARIZATION_SHERPA_THRESHOLD,
)
from audio_transcriber.diarization import embeddings as embedding_utils
from audio_transcriber.diarization.hybrid_engine import (
    _assign_global_clusters,
    _merge_adjacent,
)
from audio_transcriber.diarization.speaker_count import (
    detect_speech_spans,
    resolve_vad_model,
)
from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.exceptions import DiarizationError

logger = logging.getLogger(__name__)

#: Версия алгоритма движка ``sherpa``. Смена окон/склейки/кластеризации меняет
#: результат при тех же входах — участвует в ключе кэша диаризации.
#: 2 — собственные параметры кластеризации движка (порог/линкаж) вместо
#:     унаследованного порога оценщика числа говорящих (#133).
SHERPA_DIARIZATION_IMPL_VERSION = 2

#: Длина окна анализа (секунды). Короткие окна точнее по границам реплик, но
#: дороже (больше инференсов CAM++); 1.5 с — компромисс, согласованный с
#: окном оценщика числа говорящих.
SHERPA_WINDOW_SECONDS = 1.5

#: Минимальная длина куска, которым заканчивается речевой фрагмент (секунды).
#: Короче — приклеивается к предыдущему окну того же фрагмента, а не отдаётся
#: на эмбеддинг отдельно (вектор был бы шумным).
SHERPA_MIN_TAIL_SECONDS = 0.5

#: Linkage пороговой ветки кластеризации по умолчанию (``complete``/cosine).
#: Собственный параметр движка (``DIARIZATION_SHERPA_LINKAGE``), а не
#: переиспользование параметров оценщика числа говорящих или гибрида:
#: ``ward``/euclidean на непрерывных окнах sherpa даёт лавинное переразбиение
#: (см. :data:`DEFAULT_DIARIZATION_SHERPA_LINKAGE`). ``ward``/euclidean
#: применяется автоматически при явном ``num_speakers`` (форсированный k).
SHERPA_LINKAGE = DEFAULT_DIARIZATION_SHERPA_LINKAGE


class SherpaDiarizationError(DiarizationError):
    """Диаризация на ``sherpa`` недоступна или не дала результата."""


def plan_speech_windows(
    spans: Sequence[tuple[int, int]],
    *,
    window_seconds: float = SHERPA_WINDOW_SECONDS,
    min_tail_seconds: float = SHERPA_MIN_TAIL_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[int, int]]:
    """Режет речевые фрагменты на окна анализа фиксированной длины.

    Каждый фрагмент ``(start, end)`` из ``spans`` (в сэмплах) покрывается
    последовательными окнами по ``window_seconds``. Слишком короткий «хвост»
    (< ``min_tail_seconds``) не выделяется в отдельное окно, а приклеивается к
    предыдущему окну того же фрагмента — так эмбеддинг короткого обрывка не
    попадает в кластеризацию.
    """
    window_samples = max(1, int(window_seconds * sample_rate))
    min_tail_samples = max(0, int(min_tail_seconds * sample_rate))
    windows: list[tuple[int, int]] = []
    for start, end in spans:
        if end <= start:
            continue
        position = start
        while position < end:
            chunk_end = min(position + window_samples, end)
            is_last = chunk_end >= end
            if is_last and end - position < min_tail_samples and windows:
                # Короткий хвост: расширяем предыдущее окно до конца фрагмента,
                # если оно принадлежит этому же фрагменту.
                prev_start, prev_end = windows[-1]
                if prev_end == position:
                    windows[-1] = (prev_start, end)
                    break
            windows.append((position, chunk_end))
            position = chunk_end
    return windows


class SherpaSpeakerDiarizer:
    """Диаризация на ``sherpa-onnx`` (VAD + CAM++ + агломеративная кластеризация).

    Реализует протокол ``SpeakerDiarizer``. Enrollment по образцам сохраняется:
    :meth:`enrollment_engine` отдаёт CAM++-эмбеддер, совместимый с образцами.

    :param embedding_model: имя/путь ONNX-модели эмбеддингов (CAM++, по
        умолчанию та же, что у оценщика и гибрида).
    :param threshold: косинусный порог кластеризации (``complete``-linkage по
        умолчанию), используется при отсутствии явного ``num_speakers``.
        Собственный откалиброванный порог движка
        (``DIARIZATION_SHERPA_THRESHOLD``), а не порог оценщика числа говорящих.
    :param linkage: linkage пороговой ветки кластеризации
        (``DIARIZATION_SHERPA_LINKAGE``; ``complete``/cosine по умолчанию).
    :param embedding_model_dir: каталог кэша моделей (``None`` — по умолчанию).
    :param window_seconds: длина окна анализа (секунды).
    :param on_progress: необязательный колбэк этапов.
    :param embedder: готовый эмбеддер (для тестов; по умолчанию создаётся сам).
    """

    def __init__(
        self,
        *,
        embedding_model: str = DEFAULT_DIARIZATION_ESTIMATE_MODEL,
        threshold: float = DEFAULT_DIARIZATION_SHERPA_THRESHOLD,
        linkage: str = SHERPA_LINKAGE,
        embedding_model_dir: Path | None = None,
        window_seconds: float = SHERPA_WINDOW_SECONDS,
        on_progress: ProgressCallback | None = None,
        embedder: object | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._threshold = threshold
        self._linkage = linkage
        self._embedding_model_dir = embedding_model_dir
        self._window_seconds = window_seconds
        self._on_progress = on_progress
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
            raise SherpaDiarizationError(
                "sherpa-onnx не установлен — движок диаризации 'sherpa' "
                "недоступен (uv pip install sherpa-onnx)"
            )
        model_path = embedding_utils.resolve_embedding_model(
            self._embedding_model,
            self._embedding_model_dir,
            on_progress=self._on_progress,
        )
        if model_path is None:
            raise SherpaDiarizationError(
                f"Модель эмбеддингов {self._embedding_model!r} недоступна — "
                "движок 'sherpa' не может работать (укажите путь к .onnx или "
                "разрешите загрузку)"
            )
        self._embedder = embedding_utils.SpeakerEmbedder(model_path)
        return self._embedder

    def _resolve_vad(self) -> Path:
        vad_model = resolve_vad_model(
            self._embedding_model_dir, on_progress=self._on_progress
        )
        if vad_model is None:
            raise SherpaDiarizationError(
                "Модель silero VAD недоступна — движок 'sherpa' не может "
                "сегментировать речь (проверьте сеть/кэш моделей)"
            )
        return vad_model

    def enrollment_engine(self) -> object | None:
        """CAM++-эмбеддер для enrollment; ``None`` при недоступности.

        Enrollment сопоставляет образцы голоса и кластеры **в одном**
        пространстве эмбеддингов. Движок кластеризует говорящих CAM++
        (sherpa-onnx), поэтому для enrollment используется тот же эмбеддер.
        Ошибка (нет ``sherpa``/модели) не поднимается — вызывающий код
        откатывается на движок enrollment по умолчанию.
        """
        try:
            return self._resolve_embedder()
        except SherpaDiarizationError as exc:
            logger.warning(
                "sherpa: эмбеддер CAM++ недоступен для enrollment (%s) — "
                "сопоставление пойдёт движком по умолчанию",
                exc,
            )
            return None

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

        if waveform is None:
            waveform = load_waveform(audio_path)
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise SherpaDiarizationError("Пустой waveform — диаризация 'sherpa' невозможна")

        embedder = self._resolve_embedder()
        vad_model = self._resolve_vad()

        self._emit("sherpa: сегментация речи", fraction=0.0, detail="silero VAD")
        spans = detect_speech_spans(samples, vad_model)
        if not spans:
            logger.info("sherpa: VAD не нашёл речи — пустая разметка")
            self._emit("sherpa: нет речи", fraction=1.0)
            return []

        windows = plan_speech_windows(spans, window_seconds=self._window_seconds)
        if not windows:
            logger.info("sherpa: нет пригодных речевых окон — пустая разметка")
            self._emit("sherpa: нет речи", fraction=1.0)
            return []

        vectors: list[np.ndarray] = []
        used_windows: list[tuple[int, int]] = []
        total = len(windows)
        for index, (start, end) in enumerate(windows):
            clip = samples[start:end]
            if clip.size == 0:
                continue
            try:
                vector = embedder.embed(clip)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — мягкая деградация окна
                logger.warning(
                    "sherpa: не удалось посчитать эмбеддинг окна %d (%.1f–%.1f с): %s",
                    index + 1,
                    start / SAMPLE_RATE,
                    end / SAMPLE_RATE,
                    exc,
                )
                continue
            vectors.append(np.asarray(vector, dtype=np.float32))
            used_windows.append((start, end))
            self._emit(
                "sherpa: эмбеддинги говорящих",
                fraction=(index + 1) / total,
                detail=f"окно {index + 1}/{total}",
            )

        if not vectors:
            raise SherpaDiarizationError(
                "Не удалось получить ни одного эмбеддинга говорящего — "
                "диаризация 'sherpa' не дала результата"
            )

        observations = [
            (index, "spk", vector) for index, vector in enumerate(vectors)
        ]
        labels = _assign_global_clusters(
            observations,
            threshold=self._threshold,
            num_speakers=num_speakers,
            expected_speakers=None,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            linkage=self._linkage,
        )

        cluster_to_global: dict[int, int] = {}
        segments: list[SpeakerSegment] = []
        for (start, end), label in zip(used_windows, labels.tolist(), strict=True):
            cluster = int(label)
            if cluster not in cluster_to_global:
                cluster_to_global[cluster] = len(cluster_to_global)
            segments.append(
                SpeakerSegment(
                    start=start / SAMPLE_RATE,
                    end=end / SAMPLE_RATE,
                    speaker_id=f"SPEAKER_{cluster_to_global[cluster]:02d}",
                )
            )

        merged = _merge_adjacent(segments)
        self._overlaps = []
        logger.info(
            "sherpa: %d говорящих, %d сегментов (%d окон)",
            len(cluster_to_global),
            len(merged),
            len(used_windows),
        )
        self._emit("sherpa: готово", fraction=1.0)
        return merged

    def overlap_regions(self) -> list[SpeakerOverlap]:
        """VAD даёт эксклюзивную разметку — перекрытия не детектируются."""
        return list(self._overlaps)
