"""Подготовка эталонных образцов голоса и оценка их качества.

Эталон (образец голоса) — слабое звено enrollment: тишина, шум и слишком
короткая длительность «размывают» speaker-эмбеддинг, а несогласованная
обработка эталона и тестовых окон даёт domain mismatch и падение косинусных
сходств. Модуль приводит образец к «чистому» виду **на примитивах проекта**
(:mod:`audio_transcriber.diarization.energy`, без новых зависимостей):

1. **VAD-обрезка тишины.** Находится пик энергии и окно растёт от него, пока
   энергия держится выше порога; короткие паузы внутри речи склеиваются, а
   длинные остаются за границей. К окну добавляется небольшой запас
   (:data:`DEFAULT_SPEECH_PAD_SECONDS`), чтобы не срезать атаку/затухание.
2. **Длительность 3–10 с.** Длинные эталоны обрезаются до ``max_seconds``;
   короткие (< :data:`DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS`) сохраняются как
   есть, но помечаются флагом ``too_short``.
3. **Лёгкая RMS-нормализация** до ``target_dbfs`` (по умолчанию −30 dBFS) в
   режиме ``increase_only``: тихие усиливаются (с ограничением усиления),
   громкие не ослабляются. Пик при этом не выходит за пределы ±1.

Та же функция обработки (:func:`prepare_reference`) применяется и к эталону, и
к окнам говорящего при enrollment — эталон и тест оказываются в одном домене.

Денойз здесь сознательно **не делается**: по исследованию он искажает
speaker-признаки; если денойз включён в конвейере, он применяется к аудио задачи
раньше и к эталонам не добавляется.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audio_transcriber.config.defaults import (
    DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS,
    DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS,
    DEFAULT_REFERENCE_PREPARE,
    DEFAULT_REFERENCE_TARGET_DBFS,
)
from audio_transcriber.diarization import energy

#: Дополнительный запас вокруг вырезанной речи (секунды). Небольшой — чтобы не
#: срезать атаку/затухание и не вернуть паузы.
DEFAULT_SPEECH_PAD_SECONDS = 0.1

#: Максимальное усиление при RMS-нормализации — защита от разгона шума/тишины.
DEFAULT_REFERENCE_MAX_GAIN = 10.0

#: Порог клиппинга по абсолютной амплитуде: сэмпл с ``|x| >= порога`` считается
#: срезанным (цифровой потолок WAV 16 бит — почти 1.0).
DEFAULT_CLIP_THRESHOLD = 0.99

#: Порог «низкой энергии» по RMS (dBFS): ниже — образец подозрительно тихий.
DEFAULT_LOW_ENERGY_DBFS = -50.0

#: Доля речи в исходном образце ниже этого значения → флаг ``mostly_non_speech``.
DEFAULT_MOSTLY_NON_SPEECH_RATIO = 0.3


def _dbfs(value: float) -> float:
    """Амплитуда (RMS/peak) → dBFS; ноль/NaN даёт ``-inf``."""
    if not np.isfinite(value) or value <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(value))


def _rms(waveform: np.ndarray) -> float:
    samples = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _peak(waveform: np.ndarray) -> float:
    samples = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


@dataclass(frozen=True, slots=True)
class ReferencePrepareOptions:
    """Параметры подготовки эталона (значения — из конфигурации проекта)."""

    enabled: bool = DEFAULT_REFERENCE_PREPARE
    min_speech_seconds: float = DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS
    max_seconds: float = DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS
    target_dbfs: float = DEFAULT_REFERENCE_TARGET_DBFS
    increase_only: bool = True
    max_gain: float = DEFAULT_REFERENCE_MAX_GAIN
    pad_seconds: float = DEFAULT_SPEECH_PAD_SECONDS

    def with_max_seconds(self, max_seconds: float) -> ReferencePrepareOptions:
        """Копия с другим пределом длительности (окно эмбеддера на enrollment)."""
        return ReferencePrepareOptions(
            enabled=self.enabled,
            min_speech_seconds=self.min_speech_seconds,
            max_seconds=max_seconds,
            target_dbfs=self.target_dbfs,
            increase_only=self.increase_only,
            max_gain=self.max_gain,
            pad_seconds=self.pad_seconds,
        )


@dataclass(frozen=True, slots=True)
class ReferenceQuality:
    """Метрики и флаги качества образца голоса.

    ``speech_seconds`` — суммарная длительность речи в исходном образце (доля
    речи × длительность); ``speech_ratio`` — какая часть образца занята речью.
    ``trimmed_seconds`` — длительность речи после VAD-обрезки (столько реально
    уйдёт в эмбеддинг, с учётом предела длительности).
    """

    duration_seconds: float = 0.0
    speech_seconds: float = 0.0
    trimmed_seconds: float = 0.0
    speech_ratio: float = 0.0
    rms_dbfs: float = float("-inf")
    peak: float = 0.0
    too_short: bool = False
    clipped: bool = False
    low_energy: bool = False
    mostly_non_speech: bool = False

    @property
    def ok(self) -> bool:
        """Нет ни одного тревожного флага."""
        return not (self.too_short or self.clipped or self.low_energy or self.mostly_non_speech)

    def warnings(self) -> list[str]:
        """Человекочитаемые предупреждения (пусто, если образец хороший)."""
        messages: list[str] = []
        if self.too_short:
            messages.append(
                f"короткая речь: {self.trimmed_seconds:.1f} с "
                f"(нужно ≥ {DEFAULT_ENROLLMENT_MIN_SAMPLE_SECONDS:.0f} с)"
            )
        if self.clipped:
            messages.append("есть клиппинг (перегрузка сигнала)")
        if self.low_energy:
            messages.append(f"очень тихий образец (RMS {self.rms_dbfs:.0f} dBFS)")
        if self.mostly_non_speech:
            messages.append(
                f"преимущественно не речь (речь {self.speech_ratio * 100:.0f}%)"
            )
        return messages

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API (метрики + флаги + предупреждения)."""
        return {
            "duration": round(self.duration_seconds, 3),
            "speech_seconds": round(self.speech_seconds, 3),
            "trimmed_seconds": round(self.trimmed_seconds, 3),
            "speech_ratio": round(self.speech_ratio, 3),
            "rms_dbfs": None if self.rms_dbfs == float("-inf") else round(self.rms_dbfs, 2),
            "peak": round(self.peak, 4),
            "too_short": self.too_short,
            "clipped": self.clipped,
            "low_energy": self.low_energy,
            "mostly_non_speech": self.mostly_non_speech,
            "ok": self.ok,
            "warnings": self.warnings(),
        }


@dataclass(frozen=True, slots=True)
class PreparedReference:
    """Результат подготовки: очищенный waveform и его качество/координаты."""

    waveform: np.ndarray = field(repr=False)
    quality: ReferenceQuality
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Качество образца плюс границы вырезанного участка (для API)."""
        payload = dict(self.quality.as_dict())
        payload["start"] = round(self.start_seconds, 3)
        payload["end"] = round(self.end_seconds, 3)
        return payload


def _default_options(options: ReferencePrepareOptions | None) -> ReferencePrepareOptions:
    return options if options is not None else ReferencePrepareOptions()


def _frame_energies(
    prefix: np.ndarray, *, sample_rate: int, frame_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Кадровые энергии и их начала (сэмплы) по префиксной сумме."""
    total = prefix.size - 1
    frame = max(1, round(frame_seconds * sample_rate))
    starts = list(range(0, total, frame))
    if not starts:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    energies = np.array(
        [energy.window_energy(prefix, start, min(start + frame, total)) for start in starts],
        dtype=np.float64,
    )
    return energies, np.asarray(starts, dtype=np.int64)


def trim_to_speech(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    max_seconds: float = DEFAULT_ENROLLMENT_MAX_SAMPLE_SECONDS,
    pad_seconds: float = DEFAULT_SPEECH_PAD_SECONDS,
    frame_seconds: float = energy.DEFAULT_ENERGY_FRAME_SECONDS,
) -> tuple[float, float] | None:
    """Возвращает ``(start, end)`` участка речи с запасом ``pad_seconds``.

    Берётся окно ``max_seconds`` с наибольшей энергией (самый «речевой» участок
    записи, как и раньше), после чего внутри него срезаются **ведущие и
    замыкающие** кадры ниже порога (доля от медианной энергии). Так из образца
    уходят длинные паузы по краям окна, но не теряется речь, если она разбита
    на несколько кусков — обрезка до одного пика могла бы выкинуть основную
    часть. Длина ограничена ``max_seconds``, к границам добавляется запас.
    ``None``, если речи (энергии выше порога) нет.
    """
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    total = samples.size / sample_rate
    if samples.size == 0 or total <= 0 or max_seconds <= 0:
        return None

    prefix = energy.prefix_squares(samples)
    threshold = energy.energy_threshold(
        energy.median_energy(prefix, sample_rate=sample_rate)
    )
    window_seconds = min(max_seconds, total)
    best = energy.best_energy_window(
        prefix, [(0.0, total)], duration=window_seconds, sample_rate=sample_rate
    )
    if best is None:
        return None
    _, start, end = best

    energies, starts = _frame_energies(prefix, sample_rate=sample_rate, frame_seconds=frame_seconds)
    frame = max(1, round(frame_seconds * sample_rate))
    first_sample = round(start * sample_rate)
    last_sample = round(end * sample_rate)
    voiced = [
        index
        for index, position in enumerate(starts)
        if first_sample <= int(position) < last_sample and energies[index] >= threshold
    ]
    if not voiced:
        # В самом «громком» окне речи нет — значит, её нет во всей записи.
        return None
    # Срезаем ведущие/замыкающие тихие кадры, оставляя речь внутри окна.
    start = int(starts[voiced[0]]) / sample_rate
    end = min(int(starts[voiced[-1]]) + frame, samples.size) / sample_rate

    if pad_seconds > 0.0:
        start = max(0.0, start - pad_seconds)
        end = min(total, end + pad_seconds)
    if end <= start:
        return None
    return start, end


def normalize_rms(
    waveform: np.ndarray,
    *,
    target_dbfs: float = DEFAULT_REFERENCE_TARGET_DBFS,
    increase_only: bool = True,
    max_gain: float = DEFAULT_REFERENCE_MAX_GAIN,
) -> np.ndarray:
    """Лёгкая RMS-нормализация до ``target_dbfs`` без клиппинга.

    В режиме ``increase_only`` сигнал только усиливается (громкие не глушатся).
    Усиление ограничено ``max_gain``, результат приводится к ``[-1, 1]``.
    """
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    rms = _rms(samples)
    if rms <= 0.0 or not np.isfinite(rms):
        return samples
    current_dbfs = _dbfs(rms)
    gain = 10.0 ** ((target_dbfs - current_dbfs) / 20.0)
    if increase_only:
        gain = max(1.0, gain)
    gain = min(gain, max_gain)
    if not np.isfinite(gain) or gain <= 0.0:
        return samples
    return np.clip(samples * gain, -1.0, 1.0).astype(np.float32)


def assess_reference(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    *,
    options: ReferencePrepareOptions | None = None,
) -> ReferenceQuality:
    """Оценивает качество образца: длительность речи, RMS, пик, доля речи, флаги."""
    opts = _default_options(options)
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration = samples.size / sample_rate if sample_rate > 0 else 0.0
    peak = _peak(samples)
    rms = _rms(samples)

    speech_ratio = 0.0
    if samples.size and duration > 0:
        prefix = energy.prefix_squares(samples)
        threshold = energy.energy_threshold(
            energy.median_energy(prefix, sample_rate=sample_rate)
        )
        speech_ratio = energy.speech_fraction(
            prefix, 0.0, duration, sample_rate=sample_rate, threshold=threshold
        )
    speech_seconds = duration * speech_ratio

    window = trim_to_speech(
        samples,
        sample_rate,
        max_seconds=opts.max_seconds,
        pad_seconds=opts.pad_seconds,
    )
    trimmed_seconds = (window[1] - window[0]) if window is not None else 0.0

    return ReferenceQuality(
        duration_seconds=duration,
        speech_seconds=speech_seconds,
        trimmed_seconds=trimmed_seconds,
        speech_ratio=speech_ratio,
        rms_dbfs=_dbfs(rms),
        peak=peak,
        too_short=trimmed_seconds < opts.min_speech_seconds,
        clipped=peak >= DEFAULT_CLIP_THRESHOLD,
        low_energy=_dbfs(rms) < DEFAULT_LOW_ENERGY_DBFS,
        mostly_non_speech=speech_ratio < DEFAULT_MOSTLY_NON_SPEECH_RATIO,
    )


def prepare_reference(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    *,
    options: ReferencePrepareOptions | None = None,
    max_seconds: float | None = None,
) -> PreparedReference:
    """Готовит эталон: VAD-обрезка речи, ограничение длины, RMS-нормализация.

    ``max_seconds`` переопределяет предел длительности (на enrollment берётся
    окно эмбеддера, чтобы эталон и окна говорящего обрабатывались одинаково).
    Если речи в записи нет, waveform возвращается без обрезки (только
    нормализация), а качество помечается флагами — решение принимает вызывающий.
    """
    opts = _default_options(options)
    if max_seconds is not None:
        opts = opts.with_max_seconds(max_seconds)
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)

    quality = assess_reference(samples, sample_rate, options=opts)
    if samples.size == 0:
        return PreparedReference(waveform=samples, quality=quality)

    if not opts.enabled:
        # Подготовка выключена — отдаём сигнал как есть (прежнее поведение),
        # но качество всё равно оцениваем, чтобы вызывающий мог предупредить.
        return PreparedReference(waveform=samples, quality=quality)

    window = trim_to_speech(
        samples,
        sample_rate,
        max_seconds=opts.max_seconds,
        pad_seconds=opts.pad_seconds,
    )
    if window is None:
        # Речи не нашли — не режем (вернём как есть), но качество уже помечено.
        prepared = normalize_rms(
            samples,
            target_dbfs=opts.target_dbfs,
            increase_only=opts.increase_only,
            max_gain=opts.max_gain,
        )
        return PreparedReference(waveform=prepared, quality=quality)

    start, end = window
    first = max(0, round(start * sample_rate))
    last = min(samples.size, round(end * sample_rate))
    trimmed = samples[first:last]
    prepared = normalize_rms(
        trimmed,
        target_dbfs=opts.target_dbfs,
        increase_only=opts.increase_only,
        max_gain=opts.max_gain,
    )
    return PreparedReference(
        waveform=prepared,
        quality=quality,
        start_seconds=start,
        end_seconds=end,
    )
