"""Автоматическое извлечение образцов голоса говорящих из текущей записи.

После диаризации у каждого говорящего есть реплики с временными метками.
Компонент выбирает у говорящего непрерывный чистый участок (без наложений
речи) и вырезает из него окно с **наибольшей средней энергией** (RMS) — то есть
участок реальной речи, а не паузы/тишину внутри длинной текстовой реплики.
Готовый фрагмент нормализуется по пику и сохраняется отдельным WAV (16 кГц,
моно) рядом с результатами. Эти файлы можно потом переиспользовать как образцы
для enrollment-диаризации (``--speaker-reference`` или библиотека ``voices/``).

Компонент проектируется для мягкой деградации: отсутствие речи у говорящего,
нечитаемое аудио или ошибка записи приводят лишь к предупреждению в лог —
остальные говорящие обрабатываются, а конвейер не падает.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

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

#: Шаг скользящего окна при поиске самого «громкого» участка (секунды).
DEFAULT_WINDOW_STEP_SECONDS = 0.25

#: Длина кадра, по которому оценивается «типичная» (медианная) энергия записи.
DEFAULT_ENERGY_FRAME_SECONDS = 0.05

#: Абсолютный порог энергии (средний квадрат): ниже — считаем цифровой тишиной.
#: Соответствует RMS ~1e-4 (≈ −80 dBFS).
DEFAULT_SILENCE_ENERGY = 1e-8

#: Какая доля от медианной энергии записи считается «слишком тихой» речью.
#: Окно тише этого порога не сохраняем: это почти наверняка пауза/шум, а не
#: голос. Доля (а не сама медиана) выбрана сознательно: реальная речь почти
#: всегда громче, поэтому порог не отсекает тихие, но валидные образцы.
DEFAULT_ENERGY_THRESHOLD_RATIO = 0.25

#: Пиковая амплитуда, к которой нормализуется образец (запас до клиппинга).
DEFAULT_TARGET_PEAK = 0.97

#: Максимальное усиление при нормализации — защита от разгона шума в тишине.
DEFAULT_MAX_NORMALIZE_GAIN = 10.0

#: Суффикс каталога образцов рядом с результатами: ``<файл>.speakers``.
SPEAKER_SAMPLES_SUFFIX = ".speakers"

#: Временной интервал в секундах.
Interval = tuple[float, float]


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


def _prefix_squares(waveform: np.ndarray) -> np.ndarray:
    """Префиксная сумма квадратов сэмплов — для быстрой оценки энергии окон."""
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(np.square(values), out=prefix[1:])
    return prefix


def _window_energy(prefix: np.ndarray, first: int, last: int) -> float:
    """Средний квадрат сэмплов ``[first, last)`` по префиксной сумме."""
    first = max(0, min(first, prefix.size - 1))
    last = max(0, min(last, prefix.size - 1))
    if last <= first:
        return 0.0
    return float((prefix[last] - prefix[first]) / (last - first))


def _median_energy(prefix: np.ndarray, *, sample_rate: int) -> float:
    """Медианная энергия коротких кадров по всей записи (типичный уровень)."""
    total = prefix.size - 1
    if total <= 0:
        return 0.0
    frame = max(1, round(DEFAULT_ENERGY_FRAME_SECONDS * sample_rate))
    if total < frame:
        return _window_energy(prefix, 0, total)
    starts = np.arange(0, total - frame + 1, frame)
    energies = (prefix[starts + frame] - prefix[starts]) / frame
    return float(np.median(energies))


def _best_energy_window(
    prefix: np.ndarray,
    allowed: Sequence[Interval],
    *,
    duration: float,
    sample_rate: int,
    step: float,
) -> tuple[float, float, float] | None:
    """Окно ``duration`` с наибольшей энергией внутри ``allowed``.

    Возвращает ``(energy, start, end)``; при равенстве энергий выбирается самое
    раннее окно (детерминированность). ``None``, если валидных окон нет.
    """
    if duration <= 0 or step <= 0:
        return None

    best: tuple[float, float, float] | None = None
    for allowed_start, allowed_end in allowed:
        length = allowed_end - allowed_start
        if length <= 0:
            continue
        window = min(duration, length)
        last_start = allowed_end - window
        starts = [allowed_start]
        moment = allowed_start + step
        while moment < last_start - 1e-9:
            starts.append(round(moment, 6))
            moment += step
        if last_start > allowed_start + 1e-9:
            starts.append(round(last_start, 6))

        for start in starts:
            end = start + window
            first = round(start * sample_rate)
            last = round(end * sample_rate)
            energy = _window_energy(prefix, first, last)
            if best is None or energy > best[0]:
                best = (energy, start, end)
    return best


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
) -> Interval | None:
    """Выбирает участок речи говорящего под образец голоса.

    При переданном ``waveform`` выбирается окно (до ``max_duration`` секунд) с
    наибольшей средней энергией внутри чистых реплик говорящего, не заходящее в
    чужие интервалы. Окно тише ``DEFAULT_ENERGY_THRESHOLD_RATIO`` от медианной
    энергии записи (или ниже абсолютного порога тишины) отбрасывается — вместо
    тишины образец не создаётся. Без ``waveform`` работает прежняя эвристика по
    самой длинной чистой реплике. Возвращает ``(start, end)`` или ``None``.
    """
    if waveform is None:
        return _select_longest_clean_segment(entries, speaker_id, max_duration)

    own = _clean_intervals(entries, speaker_id)
    if not own:
        return None

    allowed = _subtract_intervals(
        _merge_intervals(own), _merge_intervals(_forbidden_intervals(entries, speaker_id))
    )
    if not allowed:
        return None

    prefix = _prefix_squares(waveform)
    median = _median_energy(prefix, sample_rate=sample_rate)
    threshold = max(DEFAULT_SILENCE_ENERGY, DEFAULT_ENERGY_THRESHOLD_RATIO * median)
    best = _best_energy_window(
        prefix, allowed, duration=max_duration, sample_rate=sample_rate, step=step
    )
    if best is None or best[0] < threshold:
        return None
    return best[1], best[2]


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
) -> dict[str, Path]:
    """Сохраняет по одному образцу голоса на говорящего (16 кГц моно WAV).

    Возвращает отображение ``speaker_id -> путь`` к записанным файлам. Говорящие
    без чистой речи или без участков с достаточной энергией пропускаются (причина
    логируется); любая ошибка чтения/записи не прерывает обработку остальных.
    """
    speakers_with_speech = [
        speaker for speaker in result.speakers if _clean_intervals(result.entries, speaker.id)
    ]
    for speaker in result.speakers:
        if speaker not in speakers_with_speech:
            logger.info(
                "Образец голоса: у говорящего %s нет чистых реплик (без наложения) — пропуск",
                speaker.id,
            )
    if not speakers_with_speech:
        return {}

    try:
        waveform = load_waveform(audio_path)
    except Exception as exc:  # noqa: BLE001 — мягкая деградация
        logger.warning("Образцы голоса: не удалось прочитать аудио %s: %s", audio_path, exc)
        return {}

    planned: list[tuple[str, str, float, float]] = []
    for speaker in speakers_with_speech:
        segment = select_sample_segment(
            result.entries, speaker.id, max_duration=max_duration, waveform=waveform
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
