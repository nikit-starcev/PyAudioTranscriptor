"""Протокол обмена между основным процессом и воркером изоляции (#143).

Тяжёлые Python-движки (faster-whisper/faster-whisper и pyannote.audio)
загружают ONNX/CTranslate2/torch-модели в адресное пространство процесса.
На Windows/macOS ``malloc_trim`` (см. #141) недоступен, поэтому освобождённые
аренды часто остаются в RSS до завершения процесса. Изоляция выносит модель в
отдельный процесс-воркер, чей RSS целиком возвращается ОС после стадии.

Протокол намеренно простой: воркер получает файл-запрос с JSON, выполняет
ровно один проход движка и пишет файл-результат с JSON. Прогресс передаётся
отдельным потоком stderr с маркером :data:`PROGRESS_PREFIX`; остальные строки
stderr считаются логом. Результаты сериализуются теми же структурами, что и
стадийный кэш (см. :mod:`audio_transcriber.cache.serialization`), поэтому
несовместимость форматов ловится теми же проверками.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from audio_transcriber.domain.models import SpeakerOverlap, SpeakerSegment, TranscriptionSegment
from audio_transcriber.progress import ProgressEvent

#: Вид воркера: распознавание речи.
KIND_ASR = "asr"
#: Вид воркера: диаризация.
KIND_DIARIZATION = "diarization"
#: Допустимые виды воркеров.
VALID_KINDS = (KIND_ASR, KIND_DIARIZATION)

#: Префикс строк прогресса в stderr воркера. Всё, что не начинается с него, —
#: лог воркера (печатается в основной лог на уровне debug).
PROGRESS_PREFIX = "@@isolate-progress@@ "

#: Таймаут одного прохода воркера по умолчанию (секунды). Как у nemo-speech:
#: с запасом покрывает длинные записи, но не даёт процессу висеть вечно.
DEFAULT_ISOLATION_TIMEOUT_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """Запрос к воркеру: что и с какими параметрами обработать.

    ``options`` — специфичные для вида движка параметры (модель, устройство,
    токен и т.п.). Для диаризации сюда же входят границы числа говорящих
    (``num_speakers``/``min_speakers``/``max_speakers``).
    """

    kind: str
    audio_path: str
    options: dict[str, Any] = field(default_factory=dict)
    language: str | None = None
    waveform_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"неизвестный вид воркера: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """Представление запроса для JSON."""
        return {
            "kind": self.kind,
            "audio_path": self.audio_path,
            "language": self.language,
            "waveform_path": self.waveform_path,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerRequest:
        """Восстанавливает запрос из JSON; битые поля — :class:`ValueError`."""
        kind = data.get("kind")
        if not isinstance(kind, str) or kind not in VALID_KINDS:
            raise ValueError(f"неизвестный вид воркера: {kind!r}")
        audio_path = data.get("audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("не задан audio_path")
        language = data.get("language")
        if language is not None and not isinstance(language, str):
            raise ValueError("language должен быть строкой или null")
        waveform_path = data.get("waveform_path")
        if waveform_path is not None and not isinstance(waveform_path, str):
            raise ValueError("waveform_path должен быть строкой или null")
        options = data.get("options")
        if not isinstance(options, dict):
            raise ValueError("options должен быть объектом")
        return cls(
            kind=kind,
            audio_path=audio_path,
            options=options,
            language=language,
            waveform_path=waveform_path,
        )


def request_to_json(request: WorkerRequest) -> str:
    """Сериализует запрос в JSON-строку."""
    return json.dumps(request.to_dict(), ensure_ascii=False)


def request_from_json(text: str) -> WorkerRequest:
    """Разбирает JSON-строку запроса; битый JSON — :class:`ValueError`."""
    return WorkerRequest.from_dict(json.loads(text))


def asr_result(
    segments: list[TranscriptionSegment], language: str | None, duration: float
) -> dict[str, Any]:
    """Упаковывает результат ASR для ответа воркера."""
    # Импорт локальный: protocol лёгкий, а serialization тянет доменные модели.
    from audio_transcriber.cache.serialization import asr_payload

    return asr_payload(segments, language, duration)


def diarization_result(
    segments: list[SpeakerSegment], overlaps: list[SpeakerOverlap]
) -> dict[str, Any]:
    """Упаковывает результат диаризации (сегменты и зоны наложения)."""
    from audio_transcriber.cache.serialization import diarization_payload

    return diarization_payload(segments, overlaps)


def progress_line(event: ProgressEvent) -> str:
    """Кодирует событие прогресса в строку stderr с маркером."""
    payload = {
        "stage": event.stage,
        "message": event.message,
        "fraction": event.fraction,
        "detail": event.detail,
    }
    return PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False)


def parse_progress_line(line: str) -> ProgressEvent | None:
    """Разбирает строку прогресса; не-прогресс или битая строка — ``None``."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        data = json.loads(line[len(PROGRESS_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    stage = data.get("stage")
    if not isinstance(stage, str):
        return None
    fraction = data.get("fraction")
    if fraction is not None and not isinstance(fraction, (int, float)):
        fraction = None
    message = data.get("message")
    detail = data.get("detail")
    return ProgressEvent(
        stage=stage,
        message=message if isinstance(message, str) else "",
        fraction=float(fraction) if fraction is not None else None,
        detail=detail if isinstance(detail, str) else "",
    )
