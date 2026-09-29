"""Общие энергетические помощники для выбора участков речи.

Примитивы используются и авто-извлечением образцов голоса
(:mod:`audio_transcriber.diarization.samples`), и enrollment-диаризацией
(:mod:`audio_transcriber.diarization.enrollment`): префиксная сумма квадратов,
энергия окна, медианная энергия записи, порог «тишины», поиск окна с
наибольшей энергией и «шлифовка» образца до речи (рост окна от пика энергии).
Вынесены в отдельный модуль, чтобы не дублировать расчёты и держать единое
определение «громкости» во всём конвейере.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Длина кадра анализа энергии (секунды). 50 мс — компромисс между точностью
#: границ речи и стоимостью (энергия считается по префиксной сумме).
DEFAULT_ENERGY_FRAME_SECONDS = 0.05

#: Абсолютный порог энергии (средний квадрат): ниже — цифровая тишина.
#: Соответствует RMS ~1e-4 (≈ −80 dBFS).
DEFAULT_SILENCE_ENERGY = 1e-8

#: Какая доля от медианной энергии записи считается «слишком тихой» речью.
#: Окно тише этого порога — почти наверняка пауза/шум, а не голос. Доля (а не
#: сама медиана) выбрана сознательно: реальная речь почти всегда громче.
DEFAULT_ENERGY_THRESHOLD_RATIO = 0.25

#: Шаг скользящего окна при поиске окна заданной длины с максимальной энергией.
DEFAULT_ENERGY_WINDOW_STEP_SECONDS = 0.25

#: Допустимая длительность почти-тихой «щели» внутри речи при росте окна от
#: пика энергии. Короткие провали между слогами/словами склеиваются, а длинные
#: паузы остаются за границей образца.
DEFAULT_SPEECH_GAP_SECONDS = 0.3

#: Минимальная длительность обрезанного до речи образца (секунды). Короче брать
#: не имеет смысла: эмбеддинг говорящего становится неустойчивым.
DEFAULT_MIN_SPEECH_SECONDS = 1.5

#: Временной интервал в секундах.
Interval = tuple[float, float]


def prefix_squares(waveform: np.ndarray) -> np.ndarray:
    """Префиксная сумма квадратов сэмплов — для быстрой оценки энергии окон."""
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(np.square(values), out=prefix[1:])
    return prefix


def window_energy(prefix: np.ndarray, first: int, last: int) -> float:
    """Средний квадрат сэмплов ``[first, last)`` по префиксной сумме."""
    first = max(0, min(first, prefix.size - 1))
    last = max(0, min(last, prefix.size - 1))
    if last <= first:
        return 0.0
    return float((prefix[last] - prefix[first]) / (last - first))


def median_energy(prefix: np.ndarray, *, sample_rate: int) -> float:
    """Медианная энергия коротких кадров по всей записи (типичный уровень)."""
    total = prefix.size - 1
    if total <= 0:
        return 0.0
    frame = max(1, round(DEFAULT_ENERGY_FRAME_SECONDS * sample_rate))
    if total < frame:
        return window_energy(prefix, 0, total)
    starts = np.arange(0, total - frame + 1, frame)
    energies = (prefix[starts + frame] - prefix[starts]) / frame
    return float(np.median(energies))


def energy_threshold(
    median: float,
    *,
    ratio: float = DEFAULT_ENERGY_THRESHOLD_RATIO,
    floor: float = DEFAULT_SILENCE_ENERGY,
) -> float:
    """Порог «слишком тихо»: доля от медианы, но не ниже абсолютной тишины."""
    return max(floor, ratio * median)


def speech_fraction(
    prefix: np.ndarray,
    start: float,
    end: float,
    *,
    sample_rate: int,
    threshold: float,
    frame_seconds: float = DEFAULT_ENERGY_FRAME_SECONDS,
) -> float:
    """Доля кадров ``[start, end)`` с энергией не ниже ``threshold`` (0..1)."""
    frame = max(1, round(frame_seconds * sample_rate))
    first = max(0, round(start * sample_rate))
    last = max(0, min(round(end * sample_rate), prefix.size - 1))
    if last <= first:
        return 0.0
    loud = 0
    total = 0
    position = first
    while position < last:
        frame_end = min(position + frame, last)
        if window_energy(prefix, position, frame_end) >= threshold:
            loud += 1
        total += 1
        position += frame
    return loud / total if total else 0.0


def best_energy_window(
    prefix: np.ndarray,
    allowed: Sequence[Interval],
    *,
    duration: float,
    sample_rate: int,
    step: float = DEFAULT_ENERGY_WINDOW_STEP_SECONDS,
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
            value = window_energy(prefix, first, last)
            if best is None or value > best[0]:
                best = (value, start, end)
    return best


def _window_around(
    focus: float, duration: float, lower: float, upper: float
) -> Interval:
    """Окно длины ``duration`` с центром около ``focus``, целиком внутри границ."""
    duration = min(duration, upper - lower)
    start = max(lower, min(focus - duration / 2.0, upper - duration))
    return start, start + duration


def speech_window_around_peak(
    prefix: np.ndarray,
    allowed: Sequence[Interval],
    *,
    sample_rate: int,
    threshold: float,
    max_duration: float,
    min_duration: float,
    frame_seconds: float = DEFAULT_ENERGY_FRAME_SECONDS,
    gap_seconds: float = DEFAULT_SPEECH_GAP_SECONDS,
) -> Interval | None:
    """Обрезает образец до речи, растирая окно от пика энергии.

    Внутри разрешённых интервалов находится кадр с максимальной энергией, после
    чего окно растёт влево и вправо, пока энергия держится выше ``threshold``;
    короткие провалы (до ``gap_seconds``) внутри речи склеиваются, длинные паузы
    остаются за границей. Длина ограничивается ``max_duration`` и снизу —
    ``min_duration`` (в пределах разрешённого интервала). ``None``, если пик
    энергии ниже порога (в записи нет речи).
    """
    if max_duration <= 0 or frame_seconds <= 0:
        return None

    frame = max(1, round(frame_seconds * sample_rate))
    gap_frames = max(1, round(gap_seconds / frame_seconds))
    total = prefix.size - 1

    # 1. Кадр с максимальной энергией среди разрешённых интервалов.
    peak: tuple[float, int] | None = None  # (energy, first_sample)
    for allowed_start, allowed_end in allowed:
        first = max(0, min(round(allowed_start * sample_rate), total))
        last = max(0, min(round(allowed_end * sample_rate), total))
        position = first
        while position < last:
            frame_end = min(position + frame, last)
            value = window_energy(prefix, position, frame_end)
            if peak is None or value > peak[0]:
                peak = (value, position)
            position += frame
    if peak is None or peak[0] < threshold:
        return None

    # 2. Интервал, в котором найден пик, — рост не выходит за его границы.
    peak_first = peak[1]
    bounds: Interval | None = None
    for allowed_start, allowed_end in allowed:
        first = max(0, min(round(allowed_start * sample_rate), total))
        last = max(0, min(round(allowed_end * sample_rate), total))
        if first <= peak_first < last:
            bounds = (first / sample_rate, last / sample_rate)
            break
    if bounds is None:
        return None

    # 3. Кадровые энергии разрешённого интервала.
    starts: list[int] = []
    energies: list[float] = []
    position = round(bounds[0] * sample_rate)
    last = round(bounds[1] * sample_rate)
    while position < last:
        frame_end = min(position + frame, last)
        starts.append(position)
        energies.append(window_energy(prefix, position, frame_end))
        position += frame
    if not energies:
        return None

    peak_index = max(range(len(energies)), key=lambda index: energies[index])

    # 4. Рост окна влево/вправо с терпимостью к коротким провалам.
    left = _grow(energies, peak_index, -1, threshold, gap_frames)
    right = _grow(energies, peak_index, +1, threshold, gap_frames)
    start_sample = starts[left]
    end_sample = min(starts[right] + frame, last)
    peak_frame_end = min(starts[peak_index] + frame, last)
    focus = ((starts[peak_index] + peak_frame_end) / 2.0) / sample_rate

    start, end = start_sample / sample_rate, end_sample / sample_rate
    if end - start > max_duration:
        start, end = _window_around(focus, max_duration, *bounds)
    if end - start < min_duration:
        start, end = _window_around(focus, min_duration, *bounds)
    return start, end


def _grow(
    energies: Sequence[float],
    origin: int,
    direction: int,
    threshold: float,
    gap_frames: int,
) -> int:
    """Крайняя позиция роста от ``origin`` в ``direction`` с допуском ``gap_frames``."""
    edge = origin
    quiet = 0
    index = origin
    while 0 <= index + direction < len(energies):
        index += direction
        if energies[index] >= threshold:
            edge = index
            quiet = 0
        else:
            quiet += 1
            if quiet > gap_frames:
                break
    return edge
