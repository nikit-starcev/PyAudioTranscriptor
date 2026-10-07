"""Автоматическое извлечение образцов голоса говорящих из текущей записи.

После диаризации у каждого говорящего есть реплики с временными метками.
Компонент выбирает у говорящего непрерывный чистый участок (без наложений
речи), находит в нём пик энергии и **растит окно от него, пока энергия держится
выше порога** — из-за пауз эмбеддинг говорящего «размывается», поэтому образец
максимально обрезается до речи (а не просто берётся фиксированное окно).
Готовый фрагмент нормализуется по пику и сохраняется отдельным WAV (16 кГц,
моно) рядом с результатами. Эти файлы можно потом переиспользовать как образцы
для enrollment-диаризации (``--speaker-reference`` или библиотека ``voices/``).

Компонент проектируется для мягкой деградации: отсутствие речи у говорящего,
нечитаемое аудио или ошибка записи приводят лишь к предупреждению в лог —
остальные говорящие обрабатываются, а конвейер не падает.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_transcriber.diarization import energy
from audio_transcriber.diarization.energy import (
    DEFAULT_ENERGY_FRAME_SECONDS,
    DEFAULT_MIN_SPEECH_SECONDS,
    energy_threshold,
    median_energy,
    prefix_squares,
    speech_window_around_peak,
    window_energy,
)
from audio_transcriber.domain.models import TranscriptEntry, TranscriptionResult
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform, write_wav
from audio_transcriber.utils.text import sanitize_filename

logger = logging.getLogger(__name__)

#: Максимальная длительность одного образца (секунды). WeSpeaker и enrollment
#: усредняют окна ~5 c, поэтому дольше брать смысла нет, а 8 c — с запасом.
DEFAULT_MAX_SAMPLE_SECONDS = 8.0

#: Желаемая минимальная длительность образца (секунды). Если чистой речи у
#: говорящего меньше — берём лучший доступный участок (без ошибки).
DEFAULT_MIN_SAMPLE_SECONDS = 3.0

#: Длина кадра анализа энергии при поиске участка речи (секунды).
DEFAULT_WINDOW_STEP_SECONDS = DEFAULT_ENERGY_FRAME_SECONDS

#: Псевдонимы для обратной совместимости (канонические значения — в energy).
DEFAULT_SILENCE_ENERGY = energy.DEFAULT_SILENCE_ENERGY
DEFAULT_ENERGY_THRESHOLD_RATIO = energy.DEFAULT_ENERGY_THRESHOLD_RATIO

#: Пиковая амплитуда, к которой нормализуется образец (запас до клиппинга).
DEFAULT_TARGET_PEAK = 0.97

#: Максимальное усиление при нормализации — защита от разгона шума в тишине.
DEFAULT_MAX_NORMALIZE_GAIN = 10.0

#: Суффикс каталога образцов рядом с результатами: ``<файл>.speakers``.
SPEAKER_SAMPLES_SUFFIX = ".speakers"

#: Сколько вариантов прослушивания говорящего отдавать по умолчанию.
DEFAULT_VARIANT_COUNT = 5

#: Минимальный разнос между выбранными вариантами (секунды). Защищает от выдачи
#: почти одинаковых окон, сдвинутых на кадр анализа.
DEFAULT_VARIANT_MIN_SEPARATION_SECONDS = 0.5

#: Временной интервал в секундах.
Interval = tuple[float, float]


@dataclass(frozen=True, slots=True)
class SampleVariant:
    """Один вариант прослушивания говорящего: окно исходного аудио ``[start, end)``.

    ``score`` — средняя энергия окна (для ранжирования и показа в UI); для
    вырожденного пути без waveform (только реплики) равен длительности, чтобы
    варианты тоже были упорядочены по «полезности».
    """

    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        """Длительность окна в секундах."""
        return self.end - self.start

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API (секунды, округлённые до миллисекунд)."""
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "score": round(self.score, 6),
        }


def samples_directory(output_dir: Path, source_path: Path) -> Path:
    """Каталог образцов для записи: ``<output_dir>/<stem>.speakers``."""
    return output_dir / f"{source_path.stem}{SPEAKER_SAMPLES_SUFFIX}"


def _clean_intervals(entries: Sequence[TranscriptEntry], speaker_id: str) -> list[Interval]:
    """Интервалы реплик говорящего без наложения речи (кандидаты на образец)."""
    return [
        (entry.start, entry.end)
        for entry in entries
        if entry.speaker is not None
        and entry.speaker.id == speaker_id
        and not entry.overlap
        and entry.end > entry.start
    ]


def _forbidden_intervals(
    entries: Sequence[TranscriptEntry], speaker_id: str
) -> list[Interval]:
    """Интервалы, в которые образцу нельзя «залезать» (чужая/наложенная речь)."""
    return [
        (entry.start, entry.end)
        for entry in entries
        if entry.end > entry.start
        and not (
            entry.speaker is not None
            and entry.speaker.id == speaker_id
            and not entry.overlap
        )
    ]


def _clean_intervals_by_speaker(
    entries: Sequence[TranscriptEntry],
) -> dict[str, list[Interval]]:
    """Чистые реплики, сгруппированные по говорящему, за один проход.

    Заменяет повторные линейные вызовы :func:`_clean_intervals` (по одному на
    говорящего) словарём: при ``S`` говорящих и ``E`` репликах это ``O(E)``
    вместо ``O(S·E)``.
    """
    grouped: dict[str, list[Interval]] = {}
    for entry in entries:
        if entry.speaker is None or entry.overlap or entry.end <= entry.start:
            continue
        grouped.setdefault(entry.speaker.id, []).append((entry.start, entry.end))
    return grouped


def _all_speech_intervals(entries: Sequence[TranscriptEntry]) -> list[Interval]:
    """Все интервалы реплик положительной длины — общий пул для запретных зон."""
    return [(entry.start, entry.end) for entry in entries if entry.end > entry.start]


def _forbidden_intervals_excluding(
    all_intervals: Sequence[Interval], own: Sequence[Interval]
) -> list[Interval]:
    """Все интервалы, кроме собственных чистых реплик говорящего.

    Точный мультимножественный вычет: из общего пула убирается ровно столько
    вхождений каждого интервала, сколько их среди ``own`` (посторонние реплики с
    тем же интервалом сохраняются). Результат совпадает с
    :func:`_forbidden_intervals` без повторного перебора всех реплик.
    """
    if not own:
        return list(all_intervals)
    remaining = Counter(own)
    result: list[Interval] = []
    for interval in all_intervals:
        if remaining.get(interval, 0) > 0:
            remaining[interval] -= 1
            continue
        result.append(interval)
    return result


def _merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    """Склеивает пересекающиеся/соприкасающиеся интервалы в непересекающиеся."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(base: Sequence[Interval], holes: Sequence[Interval]) -> list[Interval]:
    """Вычитает ``holes`` из ``base``, возвращая оставшиеся куски."""
    remaining: list[Interval] = []
    for base_start, base_end in base:
        pieces: list[Interval] = [(base_start, base_end)]
        for hole_start, hole_end in holes:
            next_pieces: list[Interval] = []
            for piece_start, piece_end in pieces:
                if hole_end <= piece_start or hole_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if hole_start > piece_start:
                    next_pieces.append((piece_start, min(hole_start, piece_end)))
                if hole_end < piece_end:
                    next_pieces.append((max(hole_end, piece_start), piece_end))
            pieces = next_pieces
            if not pieces:
                break
        remaining.extend(pieces)
    return [(start, end) for start, end in remaining if end > start]


def _select_longest_clean_segment(
    entries: Sequence[TranscriptEntry],
    speaker_id: str,
    max_duration: float,
) -> Interval | None:
    """Совместимый со старым поведением выбор самой длинной чистой реплики.

    Используется, когда waveform недоступен (прямые вызовы без аудио).
    """
    candidates = [
        entry
        for entry in entries
        if entry.speaker is not None
        and entry.speaker.id == speaker_id
        and not entry.overlap
        and entry.end > entry.start
    ]
    if not candidates:
        return None

    # Детерминированный выбор: длиннее — лучше, при равенстве — раньше.
    best = max(candidates, key=lambda entry: (entry.end - entry.start, -entry.start))
    start = best.start
    end = min(best.end, start + max_duration)
    # Если внутри выбранной реплики начинается другая (наложение), не заходим в неё.
    next_starts = [
        entry.start
        for entry in entries
        if entry is not best and start < entry.start < best.end
    ]
    if next_starts:
        end = min(end, min(next_starts))
    if end <= start:
        return None
    return start, end


def select_sample_segment(
    entries: Sequence[TranscriptEntry],
    speaker_id: str,
    *,
    max_duration: float = DEFAULT_MAX_SAMPLE_SECONDS,
    waveform: np.ndarray | None = None,
    sample_rate: int = SAMPLE_RATE,
    step: float = DEFAULT_WINDOW_STEP_SECONDS,
    own: Sequence[Interval] | None = None,
    forbidden: Sequence[Interval] | None = None,
) -> Interval | None:
    """Выбирает участок речи говорящего под образец голоса.

    При переданном ``waveform`` внутри чистых реплик говорящего (без наложений и
    чужих интервалов) находится пик энергии и окно растёт от него, пока энергия
    держится выше порога (``DEFAULT_ENERGY_THRESHOLD_RATIO`` от медианы записи,
    но не ниже ``DEFAULT_SILENCE_ENERGY``). Длина ограничена ``max_duration`` и
    снизу — ``DEFAULT_MIN_SPEECH_SECONDS``; короткие паузы внутри речи
    склеиваются. Так образец почти целиком состоит из речи, а не пауз. Если пик
    энергии ниже порога (тишина), образец не создаётся. Без ``waveform``
    работает прежняя эвристика по самой длинной чистой реплике.

    ``own``/``forbidden`` — заранее посчитанные интервалы чистых реплик
    говорящего и запретных зон (см. :func:`_clean_intervals_by_speaker`): их
    можно переиспользовать, чтобы не перебирать все реплики повторно.
    Возвращает ``(start, end)`` или ``None``. ``step`` — длина кадра анализа.
    """
    if waveform is None:
        return _select_longest_clean_segment(entries, speaker_id, max_duration)

    if own is None:
        own = _clean_intervals(entries, speaker_id)
    if not own:
        return None
    if forbidden is None:
        forbidden = _forbidden_intervals(entries, speaker_id)

    allowed = _subtract_intervals(
        _merge_intervals(own), _merge_intervals(forbidden)
    )
    if not allowed:
        return None

    prefix = prefix_squares(waveform)
    threshold = energy_threshold(median_energy(prefix, sample_rate=sample_rate))
    return speech_window_around_peak(
        prefix,
        allowed,
        sample_rate=sample_rate,
        threshold=threshold,
        max_duration=max_duration,
        min_duration=DEFAULT_MIN_SPEECH_SECONDS,
        frame_seconds=step,
    )


def select_sample_variants(
    entries: Sequence[TranscriptEntry],
    speaker_id: str,
    *,
    waveform: np.ndarray | None = None,
    count: int = DEFAULT_VARIANT_COUNT,
    max_duration: float = DEFAULT_MAX_SAMPLE_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    step: float = DEFAULT_WINDOW_STEP_SECONDS,
    min_separation: float = DEFAULT_VARIANT_MIN_SEPARATION_SECONDS,
    own: Sequence[Interval] | None = None,
    forbidden: Sequence[Interval] | None = None,
) -> list[SampleVariant]:
    """Выбирает несколько неперекрывающихся вариантов прослушивания говорящего.

    Возвращает до ``count`` окон ``[start, end)`` исходного аудио, отсортированных
    по убыванию «полезности». С ``waveform`` варианты — это лучшие по энергии
    участки чистой речи говорящего: каждый следующий ищется вне уже выбранных
    (и с разносом ``min_separation``), поэтому окна не пересекаются и не
    дублируются, а их длина/энергия естественно различаются. Без ``waveform``
    (аудио недоступно) берутся самые длинные чистые реплики — как запасной путь.

    Непересекаемость гарантируется и по чужим/наложенным интервалам: варианты
    выбираются только из чистой речи говорящего. ``own``/``forbidden`` — те же
    предпосчитанные интервалы, что и у :func:`select_sample_segment`.
    """
    if count <= 0:
        return []
    if own is None:
        own = _clean_intervals(entries, speaker_id)
    if not own:
        return []
    if forbidden is None:
        forbidden = _forbidden_intervals(entries, speaker_id)
    allowed = _subtract_intervals(
        _merge_intervals(own), _merge_intervals(forbidden)
    )
    if not allowed:
        return []
    if waveform is None:
        return _variants_from_intervals(allowed, count=count, max_duration=max_duration)
    return _variants_by_energy(
        allowed,
        waveform,
        count=count,
        max_duration=max_duration,
        sample_rate=sample_rate,
        step=step,
        min_separation=min_separation,
    )


def _variants_from_intervals(
    allowed: Sequence[Interval], *, count: int, max_duration: float
) -> list[SampleVariant]:
    """Запасной путь без waveform: самые длинные чистые интервалы, обрезанные по max."""
    candidates: list[tuple[float, float, float]] = []
    for start, end in allowed:
        window_end = min(end, start + max_duration)
        if window_end <= start:
            continue
        candidates.append((end - start, start, window_end))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        SampleVariant(start=start, end=end, score=end - start)
        for _, start, end in candidates[:count]
    ]


def _variants_by_energy(
    allowed: Sequence[Interval],
    waveform: np.ndarray,
    *,
    count: int,
    max_duration: float,
    sample_rate: int,
    step: float,
    min_separation: float,
) -> list[SampleVariant]:
    """Итеративно выбирает лучшие по энергии окна вне уже выбранных."""
    prefix = prefix_squares(waveform)
    threshold = energy_threshold(median_energy(prefix, sample_rate=sample_rate))
    remaining = list(allowed)
    variants: list[SampleVariant] = []
    while remaining and len(variants) < count:
        segment = speech_window_around_peak(
            prefix,
            remaining,
            sample_rate=sample_rate,
            threshold=threshold,
            max_duration=max_duration,
            min_duration=DEFAULT_MIN_SPEECH_SECONDS,
            frame_seconds=step,
        )
        if segment is None:
            break
        start, end = segment
        if end <= start:
            break
        score = window_energy(prefix, round(start * sample_rate), round(end * sample_rate))
        variants.append(SampleVariant(start=start, end=end, score=score))
        # Убираем выбранное окно вместе с окрестностью: следующий вариант ищется
        # не ближе ``min_separation``, поэтому почти одинаковые окна не попадают.
        hole = [(max(0.0, start - min_separation), end + min_separation)]
        updated = _subtract_intervals(remaining, hole)
        if not updated or updated == remaining:
            break
        remaining = updated
    return variants


def normalize_sample(
    waveform: np.ndarray, *, target_peak: float = DEFAULT_TARGET_PEAK
) -> np.ndarray:
    """Нормализует образец по пиковой амплитуде, не допуская клиппинга.

    Тихие записи усиливаются (с ограничением :data:`DEFAULT_MAX_NORMALIZE_GAIN`),
    громкие — ослабляются. Пик после нормализации не превышает ``target_peak``,
    поэтому ограничение ``write_wav`` не приводит к срезанию сигнала.
    """
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if not np.isfinite(peak) or peak <= 0.0:
        return samples
    gain = min(DEFAULT_MAX_NORMALIZE_GAIN, target_peak / peak)
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


def slice_waveform(waveform: np.ndarray, start: float, end: float) -> np.ndarray:
    """Возвращает срез waveform ``[start, end)`` в сэмплах с обрезкой по границам."""
    total = len(waveform)
    first = max(0, round(start * SAMPLE_RATE))
    last = min(total, round(end * SAMPLE_RATE))
    if last <= first:
        return waveform[:0]
    return waveform[first:last]


def _unique_stem(base: str, used: set[str]) -> str:
    """Подбирает свободный (без учёта регистра) stem файла в каталоге."""
    stem = base
    counter = 2
    while stem.casefold() in used:
        stem = f"{base} {counter}"
        counter += 1
    used.add(stem.casefold())
    return stem


def extract_speaker_samples(
    result: TranscriptionResult,
    *,
    audio_path: Path,
    output_dir: Path,
    max_duration: float = DEFAULT_MAX_SAMPLE_SECONDS,
    min_duration: float = DEFAULT_MIN_SAMPLE_SECONDS,
    waveform: np.ndarray | None = None,
) -> dict[str, Path]:
    """Сохраняет по одному образцу голоса на говорящего (16 кГц моно WAV).

    Возвращает отображение ``speaker_id -> путь`` к записанным файлам. Говорящие
    без чистой речи или без участков с достаточной энергией пропускаются (причина
    логируется); любая ошибка чтения/записи не прерывает обработку остальных.

    ``waveform`` — уже декодированное моно аудио (16 кГц float32), например
    результат денойза или общий waveform конвейера. Если он передан, аудиофайл
    не декодируется повторно; ``None`` — читать ``audio_path``, как раньше.
    """
    clean_by_speaker = _clean_intervals_by_speaker(result.entries)
    all_intervals = _all_speech_intervals(result.entries)
    speakers_with_speech = [
        speaker for speaker in result.speakers if clean_by_speaker.get(speaker.id)
    ]
    for speaker in result.speakers:
        if speaker not in speakers_with_speech:
            logger.info(
                "Образец голоса: у говорящего %s нет чистых реплик (без наложения) — пропуск",
                speaker.id,
            )
    if not speakers_with_speech:
        return {}

    if waveform is None:
        try:
            waveform = load_waveform(audio_path)
        except Exception as exc:  # noqa: BLE001 — мягкая деградация
            logger.warning("Образцы голоса: не удалось прочитать аудио %s: %s", audio_path, exc)
            return {}
    else:
        # Аудио уже декодировано (денойзом или общей стадией конвейера) —
        # повторное чтение файла не нужно.
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

    planned: list[tuple[str, str, float, float]] = []
    for speaker in speakers_with_speech:
        own = clean_by_speaker[speaker.id]
        forbidden = _forbidden_intervals_excluding(all_intervals, own)
        segment = select_sample_segment(
            result.entries,
            speaker.id,
            max_duration=max_duration,
            waveform=waveform,
            own=own,
            forbidden=forbidden,
        )
        if segment is None:
            logger.info(
                "Образец голоса: у говорящего %s речь тише порога (вероятно, пауза) — пропуск",
                speaker.id,
            )
            continue
        start, end = segment
        if end - start < min_duration:
            logger.info(
                "Образец голоса: у говорящего %s чистый участок короче %.1f с (%.1f с) — берём как есть",
                speaker.id,
                min_duration,
                end - start,
            )
        planned.append((speaker.id, speaker.display_name, start, end))

    if not planned:
        return {}

    directory = samples_directory(output_dir, result.source_path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Образцы голоса: не удалось создать каталог %s: %s", directory, exc)
        return {}

    written: dict[str, Path] = {}
    used: set[str] = set()
    for speaker_id, display_name, start, end in planned:
        window = normalize_sample(slice_waveform(waveform, start, end))
        if window.size == 0:
            logger.warning("Образец голоса: пустой участок у говорящего %s — пропуск", speaker_id)
            continue
        path = directory / f"{_unique_stem(sanitize_filename(display_name), used)}.wav"
        try:
            write_wav(path, window)
        except OSError as exc:
            logger.warning("Образцы голоса: не удалось записать %s: %s", path, exc)
            continue
        written[speaker_id] = path
        logger.info("Образец голоса: %s → %s (%.1f с)", speaker_id, path.name, end - start)
    return written


def find_speaker_samples(
    result: TranscriptionResult, directory: Path
) -> dict[str, Path]:
    """Находит уже сохранённые образцы говорящих в каталоге (для TUI).

    Имя файла — санитизированное ``display_name`` говорящего (так их пишет
    :func:`extract_speaker_samples`). Отсутствие каталога — не ошибка.
    """
    try:
        if not directory.is_dir():
            return {}
    except OSError:
        return {}

    found: dict[str, Path] = {}
    for speaker in result.speakers:
        candidate = directory / f"{sanitize_filename(speaker.display_name)}.wav"
        try:
            if candidate.is_file():
                found[speaker.id] = candidate
        except OSError:
            continue
    return found
