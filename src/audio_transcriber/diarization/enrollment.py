"""Присвоение говорящим имён по образцам голоса (speaker enrollment).

Идея: пользователь даёт короткие клипы-образцы («Имя → путь к файлу»), а
движок сам сопоставляет диаризованных говорящих с именами по голосу, а не по
индексу (``--speaker-name 0=Иван``). Сопоставление идёт по косинусному сходству
speaker-эмбеддингов: у каждого образца и у каждого говорящего считается один
усреднённый вектор, после чего выполняется мэтчинг с порогом.

**Одна модель на обе стороны.** Эталон и кластеры обязаны сравниваться в одном
пространстве эмбеддингов. Гибридная диаризация кластеризует говорящих CAM++
(sherpa-onnx), поэтому для неё enrollment должен использовать тот же эмбеддер —
гибрид отдаёт его через ``HybridSpeakerDiarizer.enrollment_engine()``, а конвейер
передаёт сюда. Иначе «образец × кластер» сравнивались бы разными моделями
(pyannote-WeSpeaker против CAM++) и уверенные совпадения терялись бы.

**Много-к-одному.** Один реальный участник часто дробится на несколько
кластеров (разные окна/условия записи). Мэтчинг по умолчанию — many-to-one:
каждый кластер независимо берёт своё лучшее имя, если сходство не ниже порога,
так что имя достаётся и крупному кластеру, и его фрагментам (при один-к-одному
имя получал лишь самый «чистый» фрагмент, а крупнейший кластер оставался
безымянным). Ложные срабатывания сдерживает порог. Флаг ``allow_shared_names``
возвращает прежнее поведение один-к-одному.

Качество эмбеддинга сильно зависит от того, что попало в окно: паузы «размывают»
вектор говорящего. Поэтому и у образцов, и у сегментов говорящих выбирается окно
с **наибольшей энергией** (речь), а почти-тихие окна пропускаются с
предупреждением. Матрица сходства и лучшие недобранные кандидаты пишутся в лог
на уровне INFO, чтобы было видно, почему имя не присвоено.

Компонент устроен так, чтобы любая проблема (нет образцов, модель недоступна,
битый файл, ошибка инференса) приводила лишь к предупреждению в лог: имена не
присваиваются, конвейер продолжает работу как раньше.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from audio_transcriber.config.defaults import DEFAULT_ENROLLMENT_MIN_SIMILARITY
from audio_transcriber.diarization import energy
from audio_transcriber.diarization.embeddings import l2_normalize
from audio_transcriber.diarization.reference import (
    ReferencePrepareOptions,
    prepare_reference,
)
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import SpeakerSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform

logger = logging.getLogger(__name__)

#: Длина окна эмбеддинга (секунды). WeSpeakerResNet34 обучен на чанках 5 c.
DEFAULT_EMBEDDING_WINDOW_SECONDS = 5.0

#: Сегменты короче этого порога дают ненадёжный эмбеддинг и пропускаются.
MIN_SEGMENT_SECONDS = 0.5

#: Сколько самых длинных сегментов говорящего усреднять в его эмбеддинг.
#: 10 (было 3): на реальной записи при агрегации лишь 3 окон в эмбеддинг
#: попадает мало речи, и длинные кластеры (напр. «Александр Матвеев») получают
#: заниженное сходство (0.55 — ниже порога), тогда как 10 окон поднимают его до
#: 0.66. Стоимость инференса CAM++ при этом остаётся небольшой.
MAX_REPRESENTATIVE_SEGMENTS = 10

#: Окно меньше этого числа сэмплов не несёт полезного сигнала.
MIN_WINDOW_SAMPLES = 160

@dataclass(frozen=True, slots=True)
class EnrollmentOutcome:
    """Итог enrollment: применённые имена и лучшие недобранные кандидаты.

    ``mapping`` — уверенные совпадения (сходство не ниже порога).
    ``best_candidates`` — для каждого НЕсопоставленного говорящего лучшая пара
    ``(имя, сходство)`` (ниже порога) — для диагностики в логе/TUI.
    ``speaker_count`` — сколько говорящих вообще получили эмбеддинг.
    """

    mapping: dict[str, str] = field(default_factory=dict)
    best_candidates: dict[str, tuple[str, float]] = field(default_factory=dict)
    speaker_count: int = 0



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
    similarities: Mapping[str, Mapping[str, float]],
    min_similarity: float,
    *,
    allow_shared_names: bool = False,
) -> dict[str, str]:
    """Мэтчинг говорящих и имён по косинусному сходству и порогу.

    По умолчанию — **жадный один-к-одному**: каждый говорящий и каждое имя
    используются не более одного раза. Пары ниже ``min_similarity`` не
    рассматриваются. Это консервативно и годится, когда кластеров не больше,
    чем людей.

    ``allow_shared_names=True`` — **много-к-одному**: каждый говорящий
    независимо берёт своё лучшее имя, если сходство не ниже порога, а одно имя
    может достаться нескольким говорящим. Нужно, потому что один реальный
    участник часто дробится на несколько кластеров (разные окна/условия записи):
    при «один-к-одному» имя достаётся лишь одному — обычно небольшому чистому
    фрагменту, — а крупнейший кластер того же человека остаётся безымянным.
    Ложные срабатывания сдерживает порог: назначение всё равно идёт только в
    «лучшее» имя, и оно должно быть не ниже ``min_similarity``. На реальной
    записи это подняло долю верно названного времени с 47% до 84% при нуле
    ложных имён (порог 0.60).
    """
    if allow_shared_names:
        mapping: dict[str, str] = {}
        for speaker_id, row in similarities.items():
            if not row:
                continue
            name = max(row, key=lambda candidate: row[candidate])
            if float(row[name]) >= min_similarity:
                mapping[speaker_id] = name
        return mapping

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


def _best_reference_window(
    waveform: np.ndarray, window_seconds: float, *, sample_rate: int = SAMPLE_RATE
) -> tuple[float, float] | None:
    """Окно образца с наибольшей энергией (речь), либо ``None`` если тишина.

    Берётся окно ``window_seconds`` с наибольшей энергией по всему образцу (а не
    первые секунды, как раньше). Точная обрезка пауз внутри выбранного окна и
    нормализация выполняются единым :func:`prepare_reference` — одинаково для
    эталонов и окон говорящего.
    """
    total = waveform.size / sample_rate
    if total <= 0:
        return None
    prefix = energy.prefix_squares(waveform)
    threshold = energy.energy_threshold(
        energy.median_energy(prefix, sample_rate=sample_rate)
    )
    best = energy.best_energy_window(
        prefix, [(0.0, total)], duration=window_seconds, sample_rate=sample_rate
    )
    if best is None or best[0] < threshold:
        return None
    _, start, end = best
    return start, end


def _prepare_window(
    waveform: np.ndarray,
    *,
    window_seconds: float,
    options: ReferencePrepareOptions | None,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[str]]:
    """Общая подготовка окна (эталона или говорящего): обрезка + нормализация.

    Единая точка обработки эталона и теста — обязательное условие отсутствия
    domain mismatch. Возвращает очищенный waveform и список предупреждений
    качества (для лога).
    """
    prepared = prepare_reference(
        waveform, sample_rate, options=options, max_seconds=window_seconds
    )
    return prepared.waveform, prepared.quality.warnings()


def _representative_windows(
    segments: Sequence[SpeakerSegment],
    prefix: np.ndarray,
    *,
    window_seconds: float,
    threshold: float,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """Окна эмбеддинга: самые длинные сегменты, окно внутри — по энергии.

    Кандидаты — до ``MAX_REPRESENTATIVE_SEGMENTS`` самых длинных чистых сегментов
    говорящего (длинные сегменты надёжнее для эмбеддинга). Внутри каждого берётся
    подокно до ``window_seconds`` с **наибольшей энергией** — то есть речь, а не
    первые секунды, которые могут оказаться паузой. Сегменты почти тише
    ``threshold`` пропускаются с предупреждением; окно не выходит за границы
    сегмента (иначе в эмбеддинг попадёт речь соседнего говорящего).

    Замеры на реальной записи показали, что ранжирование сегментов *только* по
    энергии ухудшает сопоставление (в топ попадают короткие громкие вставки), а
    сохранение самых длинных сегментов с энергетическим выбором окна внутри не
    уступает прежнему поведению и убирает паузы из эмбеддинга.
    """
    usable = [segment for segment in segments if segment.end - segment.start >= MIN_SEGMENT_SECONDS]
    usable.sort(key=lambda segment: (-(segment.end - segment.start), segment.start))

    windows: list[tuple[float, float]] = []
    for segment in usable:
        if len(windows) >= MAX_REPRESENTATIVE_SEGMENTS:
            break
        best = energy.best_energy_window(
            prefix,
            [(segment.start, segment.end)],
            duration=window_seconds,
            sample_rate=sample_rate,
        )
        if best is None:
            continue
        value, start, end = best
        if value < threshold:
            logger.warning(
                "Enrollment: сегмент говорящего %s почти тихий (энергия %.2e < %.2e) — пропуск",
                segment.speaker_id,
                value,
                threshold,
            )
            continue
        windows.append((start, end))
    return windows


def _log_similarity_matrix(matrix: Mapping[str, Mapping[str, float]]) -> None:
    """Пишет матрицу сходства «говорящий × имя» на уровне INFO (для диагностики)."""
    for speaker_id, row in matrix.items():
        scores = ", ".join(f"{name}={score:.3f}" for name, score in sorted(row.items()))
        logger.info("Enrollment: сходство %s: %s", speaker_id, scores)


def _best_candidate(row: Mapping[str, float]) -> tuple[str, float] | None:
    """Лучшая пара ``(имя, сходство)`` в строке матрицы (или ``None``)."""
    if not row:
        return None
    name = max(row, key=lambda candidate: row[candidate])
    return name, float(row[name])


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


def _empty_outcome() -> EnrollmentOutcome:
    return EnrollmentOutcome()


def assign_speaker_names(
    *,
    speaker_segments: Sequence[SpeakerSegment],
    references: Mapping[str, Sequence[Path]],
    audio_path: Path,
    min_similarity: float = DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    device: Device = Device.CPU,
    local_model_path: Path | str | None = None,
    engine: SpeakerEmbeddingEngine | None = None,
    waveform: np.ndarray | None = None,
    prepare: ReferencePrepareOptions | None = None,
    on_progress: ProgressCallback | None = None,
    allow_shared_names: bool = True,
) -> dict[str, str]:
    """Сопоставляет говорящих с именами по образцам голоса.

    Возвращает ``speaker_id -> имя`` только для уверенных совпадений (сходство
    не ниже ``min_similarity``). Любая ошибка (нет образцов, модель недоступна,
    битый файл) обрабатывается мягко: пишется предупреждение, возвращается
    пустой словарь. Для диагностики используйте :func:`enroll_speakers`.

    ``waveform`` — уже декодированное моно аудио задачи (16 кГц float32),
    например результат денойза. Если он передан, файл ``audio_path`` не
    декодируется повторно; ``None`` — декодировать самому, как раньше.
    Образцы голоса всегда читаются со своих путей.

    ``prepare`` — параметры подготовки эталона/окон (VAD-обрезка + RMS-
    нормализация); ``None`` — значения по умолчанию из конфигурации.
    """
    return enroll_speakers(
        speaker_segments=speaker_segments,
        references=references,
        audio_path=audio_path,
        min_similarity=min_similarity,
        device=device,
        local_model_path=local_model_path,
        engine=engine,
        waveform=waveform,
        prepare=prepare,
        on_progress=on_progress,
        allow_shared_names=allow_shared_names,
    ).mapping


def enroll_speakers(
    *,
    speaker_segments: Sequence[SpeakerSegment],
    references: Mapping[str, Sequence[Path]],
    audio_path: Path,
    min_similarity: float = DEFAULT_ENROLLMENT_MIN_SIMILARITY,
    device: Device = Device.CPU,
    local_model_path: Path | str | None = None,
    engine: SpeakerEmbeddingEngine | None = None,
    waveform: np.ndarray | None = None,
    prepare: ReferencePrepareOptions | None = None,
    on_progress: ProgressCallback | None = None,
    allow_shared_names: bool = True,
) -> EnrollmentOutcome:
    """Сопоставляет говорящих с именами и возвращает подробный итог.

    Как :func:`assign_speaker_names`, но вместе с применёнными именами отдаёт
    лучших недобранных кандидатов по каждому говорящему (для статуса TUI).

    ``waveform`` — необязательное уже декодированное моно аудио задачи
    (16 кГц float32): позволяет переиспользовать результат денойза и не
    декодировать ``audio_path`` повторно.

    ``prepare`` — параметры подготовки (VAD-обрезка + RMS-нормализация),
    применяемые **одинаково** к эталонам и к окнам говорящего. ``None`` —
    значения по умолчанию (:class:`ReferencePrepareOptions`).

    ``on_progress`` — необязательный колбэк этапов (порядок: образцы →
    эмбеддинги → сопоставление). Веб-интерфейс использует его для показа
    прогресса «Применить имена» (см. ``web/actions.py``).
    """
    emit = on_progress or (lambda _event: None)
    cleaned = _clean_references(references)
    if not cleaned:
        logger.debug("Enrollment: образцы голоса не заданы — пропуск")
        return _empty_outcome()
    if not speaker_segments:
        logger.info("Enrollment: нет диаризованных говорящих — пропуск")
        return _empty_outcome()

    total_names = len(cleaned)
    emit(ProgressEvent("samples", "Загрузка образцов голоса", 0.0))
    try:
        active_engine = engine or PyannoteEmbeddingEngine(
            local_model_path=local_model_path, device=device
        )
        emit(ProgressEvent("embeddings", "Загрузка модели эмбеддингов", 0.05))
        window_seconds = float(active_engine.window_seconds)
    except Exception as exc:  # noqa: BLE001 — мягкая деградация
        logger.warning(
            "Enrollment недоступен (модель эмбеддингов не загрузилась), "
            "имена по образцам не применены: %s",
            exc,
        )
        return _empty_outcome()

    reference_embeddings: dict[str, np.ndarray] = {}
    for position, (name, paths) in enumerate(cleaned.items(), start=1):
        vectors: list[np.ndarray] = []
        for path in paths:
            try:
                reference_waveform = load_waveform(path)
            except Exception as exc:  # noqa: BLE001 — один битый образец не роняет всё
                logger.warning("Enrollment: не удалось прочитать образец %s: %s", path, exc)
                continue
            bounds = _best_reference_window(reference_waveform, window_seconds)
            if bounds is None:
                logger.warning(
                    "Enrollment: образец %s почти тихий (нет речи) — пропускаю", path
                )
                continue
            window = _extract_window(reference_waveform, *bounds)
            window, warnings = _prepare_window(
                window,
                window_seconds=window_seconds,
                options=prepare,
            )
            if warnings:
                logger.warning(
                    "Enrollment: качество образца %s — %s", path, "; ".join(warnings)
                )
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
        emit(
            ProgressEvent(
                "embeddings",
                f"Эмбеддинги образцов: {position}/{total_names}",
                0.05 + 0.4 * position / total_names,
            )
        )

    if not reference_embeddings:
        logger.warning("Enrollment: ни один образец не обработан — имена не применены")
        return _empty_outcome()

    emit(ProgressEvent("samples", "Загрузка аудиозаписи", 0.5))
    if waveform is not None:
        # Аудио уже декодировано предыдущей стадией (например, денойзом) —
        # повторное чтение файла не нужно.
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    else:
        try:
            audio = load_waveform(audio_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrollment: не удалось прочитать аудио %s: %s", audio_path, exc)
            return _empty_outcome()

    audio_prefix = energy.prefix_squares(audio)
    audio_threshold = energy.energy_threshold(
        energy.median_energy(audio_prefix, sample_rate=SAMPLE_RATE)
    )

    grouped = _segments_by_speaker(speaker_segments)
    total_speakers = len(grouped)
    speaker_embeddings: dict[str, np.ndarray] = {}
    for position, (speaker_id, segments) in enumerate(grouped.items(), start=1):
        vectors = []
        windows = _representative_windows(
            segments,
            audio_prefix,
            window_seconds=window_seconds,
            threshold=audio_threshold,
        )
        for start, end in windows:
            window = _extract_window(audio, start, end)
            # Та же подготовка, что и для эталона (обрезка+нормализация) —
            # иначе домены «эталон» и «тест» расходятся (domain mismatch).
            window, _ = _prepare_window(
                window,
                window_seconds=window_seconds,
                options=prepare,
            )
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
        emit(
            ProgressEvent(
                "embeddings",
                f"Эмбеддинги говорящих: {position}/{total_speakers}",
                0.5 + 0.45 * position / total_speakers,
            )
        )

    if not speaker_embeddings:
        logger.warning("Enrollment: эмбеддинги говорящих не построены — имена не применены")
        return _empty_outcome()

    emit(ProgressEvent("matching", "Сопоставление говорящих и имён", 0.98))
    similarities = cosine_similarities(speaker_embeddings, reference_embeddings)
    _log_similarity_matrix(similarities)
    mapping = match_speakers(
        similarities, min_similarity, allow_shared_names=allow_shared_names
    )
    if mapping:
        logger.info("Enrollment: сопоставлены говорящие и имена по голосу — %s", mapping)
    else:
        logger.info(
            "Enrollment: уверенных совпадений нет (порог %.2f) — остаются прежние метки",
            min_similarity,
        )

    # Понятный построчный лог «кластер → имя/без имени (similarity)»: сразу
    # видно и применённые имена, и почему имя не присвоено.
    best_candidates: dict[str, tuple[str, float]] = {}
    for speaker_id, row in similarities.items():
        if speaker_id in mapping:
            logger.info(
                "Enrollment: кластер %s → %s (%.3f)",
                speaker_id,
                mapping[speaker_id],
                float(row.get(mapping[speaker_id], 0.0)),
            )
            continue
        candidate = _best_candidate(row)
        if candidate is None:
            logger.info("Enrollment: кластер %s → без имени (нет кандидатов)", speaker_id)
            continue
        best_candidates[speaker_id] = candidate
        name, score = candidate
        logger.info(
            "Enrollment: кластер %s → без имени (лучший «%s» %.3f < %.2f)",
            speaker_id,
            name,
            score,
            min_similarity,
        )

    emit(ProgressEvent("matching", "Сопоставление завершено", 1.0))
    return EnrollmentOutcome(
        mapping=mapping,
        best_candidates=best_candidates,
        speaker_count=len(speaker_embeddings),
    )
