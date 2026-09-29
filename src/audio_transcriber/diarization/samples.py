"""Автоматическое извлечение образцов голоса говорящих из текущей записи.

После диаризации у каждого говорящего есть реплики с временными метками.
Компонент выбирает у каждого говорящего самый длинный непрерывный чистый
участок (без наложений речи), вырезает его из рабочего аудио и сохраняет как
отдельный WAV (16 кГц, моно) рядом с результатами. Эти файлы можно потом
переиспользовать как образцы для enrollment-диаризации (``--speaker-reference``
или библиотека ``voices/``).

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

#: Суффикс каталога образцов рядом с результатами: ``<файл>.speakers``.
SPEAKER_SAMPLES_SUFFIX = ".speakers"


def samples_directory(output_dir: Path, source_path: Path) -> Path:
    """Каталог образцов для записи: ``<output_dir>/<stem>.speakers``."""
    return output_dir / f"{source_path.stem}{SPEAKER_SAMPLES_SUFFIX}"


def select_sample_segment(
    entries: Sequence[TranscriptEntry],
    speaker_id: str,
    *,
    max_duration: float = DEFAULT_MAX_SAMPLE_SECONDS,
) -> tuple[float, float] | None:
    """Самый длинный непрерывный чистый участок речи говорящего.

    Среди реплик говорящего без пометки наложения (``overlap=False``)
    выбирается самая длинная; ``end`` ограничивается ``max_duration`` и
    началом следующей реплики, чтобы образец не «залезал» в соседнюю речь.
    Возвращает ``(start, end)`` или ``None``, если чистой речи нет.
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
    без чистой речи пропускаются; любая ошибка чтения/записи логируется и не
    прерывает обработку остальных говорящих.
    """
    planned: list[tuple[str, str, float, float]] = []
    for speaker in result.speakers:
        segment = select_sample_segment(result.entries, speaker.id, max_duration=max_duration)
        if segment is None:
            logger.info("Образец голоса: у говорящего %s нет чистой речи — пропуск", speaker.id)
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

    try:
        waveform = load_waveform(audio_path)
    except Exception as exc:  # noqa: BLE001 — мягкая деградация
        logger.warning("Образцы голоса: не удалось прочитать аудио %s: %s", audio_path, exc)
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
        window = slice_waveform(waveform, start, end)
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
