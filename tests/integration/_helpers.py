"""Общие константы, типы и проверки интеграционных тестов.

Вынесено в обычный модуль (а не в ``conftest``) намеренно: ``conftest`` не
импортируют из тестов, а константы допусков, типы разрешённых целей и проверка
консистентности результата ASR должны переиспользоваться несколькими тестами.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.domain.models import TranscriptionSegment

#: Длительность короткого тестового фрагмента (с). Достаточно для устойчивого
#: распознавания, но мало, чтобы интеграционные тесты оставались быстрыми.
SHORT_FIXTURE_SECONDS = 5.0

#: Допуск по длительности аудио (с) при проверке результата ASR.
DURATION_TOLERANCE = 0.25

#: Допуск (с) на выход сегмента за пределы длительности аудио. ASR может
#: немного «перетянуть» границу последней реплики из-за округления/паддинга.
SEGMENT_END_TOLERANCE = 0.5

#: Корень репозитория: ``<repo>/tests/integration/_helpers.py`` -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Эталонная запись речи (JFK, ~11 с) — источник короткого фрагмента.
REFERENCE_AUDIO = PROJECT_ROOT / "tests" / "tests_jfk.flac"


@dataclass(frozen=True)
class WhisperCppPaths:
    """Разрешённые пути whisper.cpp: бинарник, модель и (опц.) каталог библиотек."""

    binary: str
    model: Path
    library_path: str | None


@dataclass(frozen=True)
class PyannoteTarget:
    """Цель диаризации pyannote: локальная модель и/или токен Hugging Face."""

    local_model: Path | None
    hf_token: str | None


def assert_asr_result_sane(
    segments: Sequence[TranscriptionSegment], duration: float
) -> None:
    """Проверяет, что ASR вернул непустой, упорядоченный и согласованный результат.

    :param segments: сегменты распознавания.
    :param duration: длительность аудио в секундах (ожидается в пределах допуска
        от :data:`SHORT_FIXTURE_SECONDS` — это проверяет вызывающий тест).
    """
    assert segments, "ASR не вернул ни одного сегмента"
    assert any(segment.text.strip() for segment in segments), "все сегменты пусты"
    assert segments == sorted(segments, key=lambda item: item.start), (
        "сегменты не упорядочены по времени"
    )
    for segment in segments:
        assert 0.0 <= segment.start <= segment.end <= duration + SEGMENT_END_TOLERANCE, (
            f"некорректные границы сегмента {segment.start:.2f}-{segment.end:.2f} "
            f"при длительности {duration:.2f}"
        )
