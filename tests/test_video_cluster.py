"""Юнит-тесты кластеризации слайдов (``video.cluster``)."""

from __future__ import annotations

import numpy as np

from audio_transcriber.video.cluster import (
    SlideCandidate,
    cluster_candidates,
    same_slide,
    text_jaccard,
    tokenize,
)


def _candidate(start: float, text: str, *, value: int = 0, end: float | None = None) -> SlideCandidate:
    analysis = np.full((8, 8, 3), float(value), dtype=np.float32)
    return SlideCandidate(start=start, end=end, analysis=analysis, text=text)


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Привет, МИР! 42") == {"привет", "мир", "42"}


def test_text_jaccard_identical_and_disjoint() -> None:
    assert text_jaccard("один два три", "один два три") == 1.0
    assert text_jaccard("один два", "три четыре") == 0.0


def test_text_jaccard_empty_texts() -> None:
    assert text_jaccard("", "") == 1.0
    assert text_jaccard("текст", "") == 0.0


def test_same_slide_uses_text_when_present() -> None:
    first = _candidate(0.0, "проект бюджет отчёт", value=0)
    second = _candidate(1.0, "проект бюджет отчёт", value=255)

    assert same_slide(first, second)


def test_same_slide_text_is_primary_even_when_images_differ() -> None:
    first = _candidate(0.0, "уникальный заголовок alpha", value=0)
    second = _candidate(1.0, "совсем другой заголовок beta", value=255)

    assert not same_slide(first, second)


def test_same_slide_falls_back_to_image_without_text() -> None:
    first = _candidate(0.0, "", value=10)
    same_image = _candidate(1.0, "", value=10)
    other_image = _candidate(2.0, "", value=200)

    assert same_slide(first, same_image)
    assert not same_slide(first, other_image)


def test_cluster_merges_repeated_slides() -> None:
    candidates = [
        _candidate(0.0, "введение в курс", end=5.0),
        _candidate(5.0, "цели занятия", end=10.0),
        _candidate(10.0, "введение в курс", end=15.0),
    ]

    groups = cluster_candidates(candidates)

    assert [group.members for group in groups] == [(0, 2), (1,)]


def test_cluster_keeps_distinct_slides_separate() -> None:
    candidates = [
        _candidate(0.0, "первый раздел", end=5.0),
        _candidate(5.0, "совершенно иное содержание", end=10.0),
    ]

    groups = cluster_candidates(candidates)

    assert [group.members for group in groups] == [(0,), (1,)]
