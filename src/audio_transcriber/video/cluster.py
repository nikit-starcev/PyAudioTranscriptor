"""Кластеризация слайдов по OCR-тексту (token-Jaccard) и изображению.

На шаблонных слайдах pHash/SSIM ненадёжны: смена заголовка почти не меняет
общую картинку. Поэтому основной признак — **пересечение токенов OCR-текста**
(Jaccard); сравнение изображений применяется только как запасной сигнал, когда
у кадров нет текста (пустые/переходные слайды).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from audio_transcriber.video.frames import analysis_difference

#: Порог Jaccard токенов, выше которого два слайда считаются одним. Достаточно
#: высокий, чтобы разные слайды с общей лексикой не сливались; одинаковые
#: слайды с небольшими различиями OCR обычно дают Jaccard > 0.9.
DEFAULT_TEXT_JACCARD_THRESHOLD = 0.6

#: Порог различия изображений для слайдов без текста.
DEFAULT_IMAGE_DIFF_THRESHOLD = 0.12

#: Разделитель токенов: всё, кроме букв/цифр/подчёркивания (Unicode).
_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(text: str) -> frozenset[str]:
    """Множество токенов текста: в нижнем регистре, без пунктуации и пустышек."""
    return frozenset(
        token for token in (part.strip().casefold() for part in _TOKEN_SPLIT.split(text)) if token
    )


def text_jaccard(first: str, second: str) -> float:
    """Jaccard-сходство токенов двух текстов (0..1).

    Оба текста пустые (нет текста на обоих слайдах) — ``1.0``; пустой только
    один — ``0.0`` (текст не подтверждает совпадение).
    """
    left = tokenize(first)
    right = tokenize(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True, slots=True)
class SlideCandidate:
    """Кандидат в слайд: интервал, аналитическое изображение и OCR-текст."""

    start: float
    end: float | None
    analysis: np.ndarray
    text: str


@dataclass(frozen=True, slots=True)
class SlideGroup:
    """Группа индексов кандидатов, отнесённых к одному слайду."""

    members: tuple[int, ...]

    @property
    def first(self) -> int:
        """Индекс самого раннего кандидата группы."""
        return self.members[0]


def same_slide(
    first: SlideCandidate,
    second: SlideCandidate,
    *,
    text_threshold: float = DEFAULT_TEXT_JACCARD_THRESHOLD,
    image_diff_threshold: float = DEFAULT_IMAGE_DIFF_THRESHOLD,
) -> bool:
    """Относятся ли два кандидата к одному слайду.

    Если у обоих есть текст — решает Jaccard токенов (изображение ненадёжно на
    шаблонах). Иначе (хотя бы у одного текста нет) — сравнение изображений.
    """
    if first.text.strip() and second.text.strip():
        return text_jaccard(first.text, second.text) >= text_threshold
    return analysis_difference(first.analysis, second.analysis) <= image_diff_threshold


def _representative_index(candidates: list[SlideCandidate], group: list[int]) -> int:
    """Индекс представителя группы: самый богатый по тексту (тай-брейк — раньше)."""
    return max(group, key=lambda index: (len(candidates[index].text.strip()), -index))


def cluster_candidates(
    candidates: list[SlideCandidate],
    *,
    text_threshold: float = DEFAULT_TEXT_JACCARD_THRESHOLD,
    image_diff_threshold: float = DEFAULT_IMAGE_DIFF_THRESHOLD,
) -> list[SlideGroup]:
    """Greedy-кластеризация кандидатов в уникальные слайды (порядок сохраняется).

    Каждый кандидат сравнивается с представителем уже существующих групп; при
    совпадении добавляется в группу, иначе открывает новую. Представитель —
    кандидат с самым длинным текстом: он даёт наиболее надёжное сравнение.
    """
    groups: list[list[int]] = []
    for index, candidate in enumerate(candidates):
        placed = False
        for group in groups:
            representative = candidates[_representative_index(candidates, group)]
            if same_slide(
                representative,
                candidate,
                text_threshold=text_threshold,
                image_diff_threshold=image_diff_threshold,
            ):
                group.append(index)
                placed = True
                break
        if not placed:
            groups.append([index])
    return [SlideGroup(members=tuple(group)) for group in groups]
