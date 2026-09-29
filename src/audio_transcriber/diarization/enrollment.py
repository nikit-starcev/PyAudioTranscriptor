"""Присвоение говорящим имён по образцам голоса (speaker enrollment).

Идея: пользователь даёт короткие клипы-образцы («Имя → путь к файлу»), а
движок сам сопоставляет диаризованных говорящих с именами по голосу, а не по
индексу (``--speaker-name 0=Иван``). Сопоставление идёт по косинусному сходству
speaker-эмбеддингов: у каждого образца и у каждого говорящего считается один
усреднённый вектор, после чего выполняется жадный мэтчинг один-к-одному с
порогом.

Компонент устроен так, чтобы любая проблема (нет образцов, модель недоступна,
битый файл, ошибка инференса) приводила лишь к предупреждению в лог: имена не
присваиваются, конвейер продолжает работу как раньше.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from audio_transcriber.config.defaults import DEFAULT_ENROLLMENT_MIN_SIMILARITY
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform

logger = logging.getLogger(__name__)

#: Длина окна эмбеддинга (секунды). WeSpeakerResNet34 обучен на чанках 5 c.
DEFAULT_EMBEDDING_WINDOW_SECONDS = 5.0

#: Сегменты короче этого порога дают ненадёжный эмбеддинг и пропускаются.
MIN_SEGMENT_SECONDS = 0.5

#: Сколько самых длинных сегментов говорящего усреднять в его эмбеддинг.
MAX_REPRESENTATIVE_SEGMENTS = 3

#: Окно меньше этого числа сэмплов не несёт полезного сигнала.
MIN_WINDOW_SAMPLES = 160


@runtime_checkable
class SpeakerEmbeddingEngine(Protocol):
    """Контракт движка speaker-эмбеддингов (позволяет подменять его в тестах)."""

    @property
    def window_seconds(self) -> float:
        """Рекомендуемая длина окна аудио для одного эмбеддинга."""
        ...

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        """Возвращает эмбеддинг (1D вектор) моно waveform float32."""
        ...


class PyannoteEmbeddingEngine:
    """Реальный движок эмбеддингов на pyannote.audio (локальная модель, офлайн).

    Локальный каталог модели диаризации содержит подкаталог ``embedding`` с
    весами WeSpeakerResNet34. Если локальной модели нет, используется
    ``pipeline_name`` (как и в диаризации, это может потребовать сети);
    недоступность модели приводит к мягкой деградации у вызывающего кода.
    """

    def __init__(
        self,
        *,
        local_model_path: Path | str | None = None,
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        device: Device = Device.CPU,
    ) -> None:
        self._local_model_path = Path(local_model_path) if local_model_path else None
        self._pipeline_name = pipeline_name
        self._device = device
        self._inference: object | None = None

    def _resolve_source(self) -> str | Path:
        if self._local_model_path is not None:
            embedding_dir = self._local_model_path / "embedding"
            if embedding_dir.is_dir():
                return embedding_dir
            if self._local_model_path.is_dir():
                return self._local_model_path
        return self._pipeline_name

    def _load(self) -> object:
        if self._inference is not None:
            return self._inference

        import torch
        from pyannote.audio import Inference, Model

        source = self._resolve_source()
        logger.debug("Загрузка модели эмбеддингов голоса из '%s'", source)
        model = Model.from_pretrained(source)
        if model is None:
            raise RuntimeError(f"Модель эмбеддингов '{source}' недоступна")
        model.to(torch.device(self._device.value))
        self._inference = Inference(model, window="whole")
        return self._inference

    @property
    def window_seconds(self) -> float:
        inference = self._load()
        model = getattr(inference, "model", None)
        specifications = getattr(model, "specifications", None)
        duration = getattr(specifications, "duration", None)
        return float(duration) if duration else DEFAULT_EMBEDDING_WINDOW_SECONDS

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        import torch

        inference = self._load()
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        inputs = {
            "waveform": torch.from_numpy(samples).unsqueeze(0),
            "sample_rate": SAMPLE_RATE,
        }
        vector = inference(inputs)  # type: ignore[operator]
        return np.asarray(vector, dtype=np.float32).reshape(-1)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-нормирует вектор; нулевой вектор возвращается без изменений."""
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array
    return array / norm


def average_embeddings(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    """Усредняет эмбеддинги и нормирует результат; ``None`` для пустого входа."""
    usable = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    usable = [vector for vector in usable if vector.size]
    if not usable:
        return None
    return l2_normalize(np.mean(np.vstack(usable), axis=0))


def cosine_similarities(
    speaker_embeddings: Mapping[str, np.ndarray],
    reference_embeddings: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Косинусное сходство «говорящий × имя» (оба входа L2-нормируются)."""
    matrix: dict[str, dict[str, float]] = {}
    for speaker_id, speaker_vector in speaker_embeddings.items():
        normalized_speaker = l2_normalize(speaker_vector)
        row: dict[str, float] = {}
        for name, reference_vector in reference_embeddings.items():
            normalized_reference = l2_normalize(reference_vector)
            row[name] = float(np.dot(normalized_speaker, normalized_reference))
        matrix[speaker_id] = row
    return matrix


def match_speakers(
    similarities: Mapping[str, Mapping[str, float]], min_similarity: float
) -> dict[str, str]:
    """Жадный мэтчинг один-к-одному по убыванию сходства и с порогом.

    Каждый говорящий и каждое имя используются не более одного раза. Пары с
    сходством ниже ``min_similarity`` не рассматриваются — лучше оставить
    «Спикер N», чем присвоить чужое имя.
    """
    candidates: list[tuple[float, str, str]] = [
        (float(score), speaker_id, name)
        for speaker_id, row in similarities.items()
        for name, score in row.items()
    ]
    # Детерминированный порядок: сходство по убыванию, затем id/имя.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched: dict[str, str] = {}
    used_names: set[str] = set()
    for score, speaker_id, name in candidates:
        if score < min_similarity:
            break
        if speaker_id in matched or name in used_names:
            continue
        matched[speaker_id] = name
        used_names.add(name)
    return matched


def _extract_window(waveform: np.ndarray, start: float, end: float) -> np.ndarray:
    """Возвращает срез waveform [start, end) в сэмплах с обрезкой по границам."""
    total = len(waveform)
    first = max(0, round(start * SAMPLE_RATE))
    last = min(total, round(end * SAMPLE_RATE))
    if last <= first:
        return waveform[:0]
    return waveform[first:last]


def _segments_by_speaker(
    speaker_segments: Sequence[SpeakerSegment],
) -> dict[str, list[SpeakerSegment]]:
    grouped: dict[str, list[SpeakerSegment]] = {}
    for segment in speaker_segments:
        grouped.setdefault(segment.speaker_id, []).append(segment)
    return grouped


def _representative_windows(
    segments: Sequence[SpeakerSegment], window_seconds: float
) -> list[tuple[float, float]]:
    """Самые длинные сегменты говорящего → окна эмбеддинга.

    Окно не длиннее ``window_seconds`` и не выходит за границы сегмента — иначе
    в эмбеддинг попадёт речь соседнего говорящего.
    """
    usable = [segment for segment in segments if segment.end - segment.start >= MIN_SEGMENT_SECONDS]
    usable.sort(key=lambda segment: segment.end - segment.start, reverse=True)
    return [
        (segment.start, min(segment.end, segment.start + window_seconds))
        for segment in usable[:MAX_REPRESENTATIVE_SEGMENTS]
    ]


def _log_similarity_matrix(matrix: Mapping[str, Mapping[str, float]]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for speaker_id, row in matrix.items():
        scores = ", ".join(f"{name}={score:.3f}" for name, score in sorted(row.items()))
        logger.debug("Enrollment: сходство %s: %s", speaker_id, scores)


def _clean_references(
    references: Mapping[str, Sequence[Path]],
) -> dict[str, list[Path]]:
    cleaned: dict[str, list[Path]] = {}
    for name, paths in references.items():
        name = name.strip()
        usable = [Path(path) for path in paths]
        if name and usable:
            cleaned[name] = usable
    return cleaned


def assign_speaker_names(
    *,
    speaker_segments: Sequence[SpeakerSegment],
    references: Mapping[str, Sequence[Path]],
    audio_path: Path,
    min_similarity: float = DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    device: Device = Device.CPU,
    local_model_path: Path | str | None = None,
    engine: SpeakerEmbeddingEngine | None = None,
) -> dict[str, str]:
    """Сопоставляет говорящих с именами по образцам голоса.

    Возвращает ``speaker_id -> имя`` только для уверенных совпадений (сходство
    не ниже ``min_similarity``). Любая ошибка (нет образцов, модель недоступна,
    битый файл) обрабатывается мягко: пишется предупреждение, возвращается
    пустой словарь.
    """
    cleaned = _clean_references(references)
    if not cleaned:
        logger.debug("Enrollment: образцы голоса не заданы — пропуск")
        return {}
    if not speaker_segments:
        logger.info("Enrollment: нет диаризованных говорящих — пропуск")
        return {}

    try:
        active_engine = engine or PyannoteEmbeddingEngine(
            local_model_path=local_model_path, device=device
        )
        window_seconds = float(active_engine.window_seconds)
    except Exception as exc:  # noqa: BLE001 — мягкая деградация
        logger.warning(
            "Enrollment недоступен (модель эмбеддингов не загрузилась), "
            "имена по образцам не применены: %s",
            exc,
        )
        return {}

    reference_embeddings: dict[str, np.ndarray] = {}
    for name, paths in cleaned.items():
        vectors: list[np.ndarray] = []
        for path in paths:
            try:
                waveform = load_waveform(path)
            except Exception as exc:  # noqa: BLE001 — один битый образец не роняет всё
                logger.warning("Enrollment: не удалось прочитать образец %s: %s", path, exc)
                continue
            window = _extract_window(waveform, 0.0, window_seconds)
            if window.size < MIN_WINDOW_SAMPLES:
                logger.warning("Enrollment: образец %s слишком короткий — пропускаю", path)
                continue
            try:
                vectors.append(active_engine.embed(window))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Enrollment: ошибка эмбеддинга образца %s: %s", path, exc)
        averaged = average_embeddings(vectors)
        if averaged is not None:
            reference_embeddings[name] = averaged

    if not reference_embeddings:
        logger.warning("Enrollment: ни один образец не обработан — имена не применены")
        return {}

    try:
        audio = load_waveform(audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Enrollment: не удалось прочитать аудио %s: %s", audio_path, exc)
        return {}

    speaker_embeddings: dict[str, np.ndarray] = {}
    for speaker_id, segments in _segments_by_speaker(speaker_segments).items():
        vectors = []
        for start, end in _representative_windows(segments, window_seconds):
            window = _extract_window(audio, start, end)
            if window.size < MIN_WINDOW_SAMPLES:
                continue
            try:
                vectors.append(active_engine.embed(window))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Enrollment: ошибка эмбеддинга говорящего %s: %s", speaker_id, exc
                )
        averaged = average_embeddings(vectors)
        if averaged is not None:
            speaker_embeddings[speaker_id] = averaged

    if not speaker_embeddings:
        logger.warning("Enrollment: эмбеддинги говорящих не построены — имена не применены")
        return {}

    similarities = cosine_similarities(speaker_embeddings, reference_embeddings)
    _log_similarity_matrix(similarities)
    mapping = match_speakers(similarities, min_similarity)
    if mapping:
        logger.info("Enrollment: сопоставлены говорящие и имена по голосу — %s", mapping)
    else:
        logger.info(
            "Enrollment: уверенных совпадений нет (порог %.2f) — остаются прежние метки",
            min_similarity,
        )
    return mapping
