"""Доменные модели, описывающие данные, которыми обмениваются компоненты
конвейера: распознавание речи -> диаризация -> объединение -> экспорт.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """Сегмент речи, полученный от движка распознавания (ASR).

    Не содержит информации о говорящем — это добавляется на этапе
    объединения с результатами диаризации.
    """

    start: float
    end: float
    text: str
    avg_logprob: float | None = None


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """Временной интервал, отнесённый диаризацией к конкретному говорящему."""

    start: float
    end: float
    speaker_id: str


@dataclass(frozen=True, slots=True)
class Speaker:
    """Говорящий, участвующий в разговоре."""

    id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class SpeakerOverlap:
    """Интервал, в котором одновременно активны два и более говорящих.

    Используется, чтобы пометить реплики, попавшие в зону наложения речи, и
    назвать её участников. ``speaker_ids`` — идентификаторы говорящих, активных
    в этой зоне (устойчивый порядок). Пустой кортеж означает, что источник зоны
    не сохранил участников (например, старый кэш диаризации): зона пригодна для
    пометки ``overlap``, но имён сов-говорящих из неё не извлечь.
    """

    start: float
    end: float
    speaker_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """Финальная реплика стенограммы: текст, привязанный к говорящему.

    Кроме основного говорящего ``speaker`` реплика может содержать
    ``extra_speakers`` — прочих участников, которые **одновременно** говорили
    вместе с основным (непустое пересечение интервалов диаризации) внутри
    интервала реплики. Смена говорящего (один закончил — другой начал) не
    является наложением и в ``extra_speakers`` не попадает. Порядок
    ``extra_speakers`` — по убыванию суммарной одновременной длительности.
    """

    start: float
    end: float
    text: str
    speaker: Speaker | None = None
    # Средняя логвероятность распознавания реплики (уверенность ASR).
    # ``None`` — движок не предоставил значение (например, старый whisper.cpp).
    avg_logprob: float | None = None
    # Реплика попала в зону наложения речи (одновременно говорят >= 2 человек).
    overlap: bool = False
    # Прочие говорящие, кроме ``speaker``, одновременно говорившие вместе с
    # основным внутри интервала реплики. Пустой список — наложения нет.
    extra_speakers: list[Speaker] = field(default_factory=list)
    # Уверенность привязки основного говорящего: доля интервала реплики,
    # покрытая сегментами диаризации этого говорящего (0..1). ``None``, если
    # диаризации не было и оценить нечего.
    speaker_confidence: float | None = None

    @property
    def speaker_label(self) -> str:
        """Человекочитаемая подпись всех говорящих реплики.

        Основной говорящий первый, затем дополнительные в их порядке (по
        убыванию одновременной длительности): ``"Имя1 + Имя2"``. Если говорящих
        нет — ``"?"``,
        как и раньше в экспортёрах.
        """
        speakers = list(self.extra_speakers)
        if self.speaker is not None:
            speakers.insert(0, self.speaker)
        if not speakers:
            return "?"
        return " + ".join(speaker.display_name for speaker in speakers)


@dataclass(slots=True)
class TranscriptionResult:
    """Итоговый результат работы конвейера для одного аудиофайла."""

    source_path: Path
    language: str | None
    duration: float
    entries: list[TranscriptEntry] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)
    participants: list[str] | None = None
    # Резюме встречи (локальная LLM): участники, тема, решения, открытые
    # вопросы, action items. ``None`` — резюме не запрашивалось/не получено.
    summary: str | None = None
    # Порог «низкой уверенности» для экспортёров: реплики со средним
    # avg_logprob ниже порога помечаются. ``None`` — не помечать.
    low_confidence_threshold: float | None = None
