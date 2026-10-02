"""Реализация распознавания речи на GigaAM v3 (RU) через ``onnx-asr``.

``onnx-asr`` запускает ONNX-модели на ONNX Runtime без PyTorch: это лёгкая
альтернатива faster-whisper для русского языка (GigaAM v3 показывает высокое
качество на русской речи и обычно быстрее на CPU). Пакет — **опциональная**
зависимость (``pip install 'onnx-asr[cpu,hub]'``): если его нет или модель не
загрузилась, движок поднимает понятную ошибку, а конвейер не меняется для
остальных бэкендов.

Модели GigaAM принимают ограниченное окно (20–30 с), поэтому длинное аудио
обрабатывается через встроенный VAD ``onnx-asr``: он режет запись на речевые
сегменты и распознаёт каждый. VAD-путь даёт и границы сегментов, и потаксенные
логвероятности — из них считается средняя уверенность ``avg_logprob`` (аналог
faster-whisper), которую использует гибридный ASR (#57). При отключённом VAD
(или недоступной VAD-модели) распознаётся один результат по всему файлу.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from audio_transcriber.config.defaults import (
    DEFAULT_GIGAAM_MODEL,
    DEFAULT_GIGAAM_VAD_MODEL,
)
from audio_transcriber.domain.enums import Device
from audio_transcriber.domain.models import TranscriptionSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform
from audio_transcriber.utils.exceptions import TranscriptionError

logger = logging.getLogger(__name__)

#: Версия реализации движка. Участвует в ключе кэша ASR: изменение логики,
#: влияющей на результат при тех же параметрах, инвалидирует старый кэш.
ASR_IMPL_VERSION = 1

# Параметры VAD по умолчанию — как в onnx-asr (SegmentResult VAD options),
# чтобы длинные записи резались предсказуемо. Имя VAD-модели —
# ``DEFAULT_GIGAAM_VAD_MODEL`` (импортируется из config.defaults).
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SPEECH_DURATION_MS = 250.0
DEFAULT_VAD_MAX_SPEECH_DURATION_S = 20.0
DEFAULT_VAD_MIN_SILENCE_DURATION_MS = 100.0
DEFAULT_VAD_SPEECH_PAD_MS = 30.0


def resolve_onnx_providers(device: Device) -> list[str] | None:
    """Провайдеры ONNX Runtime под запрошенное устройство.

    ``onnx-asr[cpu]`` ставит CPU-сборку onnxruntime, в которой CUDA-провайдера
    нет. Передавать несуществующий провайдер нельзя — onnxruntime упадёт,
    поэтому CUDA подключается только если он реально доступен; иначе тихо
    используется CPU (мягкая деградация).
    """
    if device is not Device.CUDA:
        return None

    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001 — onnxruntime необязателен здесь
        logger.debug("onnxruntime недоступен для выбора провайдера: %s", exc)
        return None

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.warning(
        "Запрошено устройство CUDA, но в onnxruntime нет CUDAExecutionProvider "
        "(установлена CPU-сборка) — GigaAM работает на CPU."
    )
    return None


def _mean_logprob(logprobs: Iterable[float] | None) -> float | None:
    """Среднее потаксенных логвероятностей; ``None``, если их нет.

    onnx-asr для RNN-T отдаёт ``logprobs=None`` (нужны ``need_logprobs``) —
    тогда уверенность неизвестна, и детектор гибрида опирается на другие
    признаки (длительность, энергия, no-speech).
    """
    if logprobs is None:
        return None
    values = [float(value) for value in logprobs]
    if not values:
        return None
    return sum(values) / len(values)


class GigaAmRecognizer:
    """Распознаёт русскую речь через GigaAM v3 и ``onnx-asr``.

    Реализует протокол :class:`~audio_transcriber.transcription.base.SpeechRecognizer`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GIGAAM_MODEL,
        *,
        model_path: Path | None = None,
        quantization: str | None = None,
        device: Device = Device.AUTO,
        use_vad: bool = True,
        vad_model: str = DEFAULT_GIGAAM_VAD_MODEL,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        vad_min_speech_duration_ms: float = DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
        vad_max_speech_duration_s: float = DEFAULT_VAD_MAX_SPEECH_DURATION_S,
        vad_min_silence_duration_ms: float = DEFAULT_VAD_MIN_SILENCE_DURATION_MS,
        vad_speech_pad_ms: float = DEFAULT_VAD_SPEECH_PAD_MS,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_path = Path(model_path) if model_path is not None else None
        self._quantization = quantization
        self._device = device
        self._use_vad = use_vad
        self._vad_model = vad_model
        self._vad_threshold = vad_threshold
        self._vad_min_speech_duration_ms = vad_min_speech_duration_ms
        self._vad_max_speech_duration_s = vad_max_speech_duration_s
        self._vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self._vad_speech_pad_ms = vad_speech_pad_ms
        self._on_progress = on_progress
        self._adapter: Any = None
        #: VAD реально подключён (а не просто запрошен): при недоступной
        #: VAD-модели движок деградирует к распознаванию одного результата.
        self._vad_active = False

    @property
    def model_name(self) -> str:
        """Имя модели (участвует в ключе кэша ASR)."""
        return self._model_name

    @property
    def quantization(self) -> str | None:
        """Квантизация модели (участвует в ключе кэша ASR)."""
        return self._quantization

    @property
    def model_path(self) -> Path | None:
        """Локальный каталог модели (``None`` — загрузка с Hugging Face)."""
        return self._model_path

    @property
    def use_vad(self) -> bool:
        """Используется ли VAD-сегментация длинного аудио."""
        return self._use_vad

    def _emit(self, fraction: float | None = None, detail: str = "") -> None:
        if self._on_progress is not None:
            self._on_progress(
                ProgressEvent("asr", "Распознавание речи", fraction=fraction, detail=detail)
            )

    def _load_adapter(self) -> Any:
        """Ленивая загрузка модели (и VAD) через ``onnx-asr``.

        Импорт и загрузка модели откладываются до первого вызова: при
        ``--asr-backend whisper-cpp``/``faster-whisper`` пакет onnx-asr не нужен.
        """
        if self._adapter is not None:
            return self._adapter

        try:
            import onnx_asr
        except ImportError as exc:
            raise TranscriptionError(
                "Бэкенд gigaam требует пакет onnx-asr. Установите его: "
                "uv pip install --python .venv/bin/python 'onnx-asr[cpu,hub]' "
                "(модель GigaAM будет загружена с Hugging Face при первом запуске)."
            ) from exc

        providers = resolve_onnx_providers(self._device)
        model_path = str(self._model_path) if self._model_path is not None else None
        try:
            adapter = onnx_asr.load_model(
                self._model_name,
                model_path,
                quantization=self._quantization,
                providers=providers,
            )
            adapter, self._vad_active = self._with_vad_if_enabled(
                onnx_asr, adapter, providers
            )
            # Таймстемпы дают границы сегментов; logprobs — уверенность для
            # детектора «плохих» сегментов гибрида.
            self._adapter = adapter.with_timestamps()
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Не удалось загрузить модель GigaAM '{self._model_name}': {exc}"
            ) from exc
        return self._adapter

    def _with_vad_if_enabled(
        self, onnx_asr: Any, adapter: Any, providers: Any
    ) -> tuple[Any, bool]:
        """Подключает VAD, если он включён; при сбое мягко деградирует.

        Возвращает ``(адаптер, подключён_ли_VAD)``: флаг нужен, чтобы
        ``transcribe`` выбрал правильную форму результата (сегменты VAD против
        единственного результата по всему файлу).
        """
        if not self._use_vad:
            return adapter, False

        try:
            vad = onnx_asr.load_vad(self._vad_model, providers=providers)
        except Exception as exc:  # noqa: BLE001 — VAD необязателен
            logger.warning(
                "Не удалось загрузить VAD '%s' для GigaAM (%s) — длинное аудио "
                "может распознаваться хуже; работаю без VAD",
                self._vad_model,
                exc,
            )
            return adapter, False

        adapter = adapter.with_vad(
            vad,
            threshold=self._vad_threshold,
            min_speech_duration_ms=self._vad_min_speech_duration_ms,
            max_speech_duration_s=self._vad_max_speech_duration_s,
            min_silence_duration_ms=self._vad_min_silence_duration_ms,
            speech_pad_ms=self._vad_speech_pad_ms,
        )
        return adapter, True

    @staticmethod
    def _make_segment(
        result: Any, start: float, end: float
    ) -> TranscriptionSegment | None:
        """Собирает сегмент стенограммы из результата ``onnx-asr``."""
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            return None
        return TranscriptionSegment(
            start=start,
            end=end,
            text=text,
            avg_logprob=_mean_logprob(getattr(result, "logprobs", None)),
        )

    @staticmethod
    def _result_bounds(result: Any, duration: float) -> tuple[float, float]:
        """Границы единственного результата без VAD.

        Берём самое раннее начало и самый поздний конец из потаксенных
        таймстемпов; если их нет — весь файл.
        """
        timestamps = getattr(result, "timestamps", None)
        if not timestamps:
            return 0.0, duration
        start = float(min(timestamps))
        # Таймстемп в onnx-asr — начало токена; конец последнего неизвестен,
        # поэтому тянем его на среднюю длительность токена (ограничение сверху).
        last = float(max(timestamps))
        step = (last - start) / (len(timestamps) - 1) if len(timestamps) >= 2 else 0.0
        return start, min(duration, last + step)

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        adapter = self._load_adapter()

        if language and not language.casefold().startswith("ru"):
            logger.debug(
                "GigaAM v3 — русскоязычная модель, параметр language=%r "
                "игнорируется (распознаётся русская речь)",
                language,
            )

        # onnx-asr принимает NumPy-массив 16 кГц моно; декодируем сами, чтобы
        # не зависеть от формата контейнера (mp3/mp4/webm) и совпадать с
        # остальным конвейером.
        waveform = load_waveform(audio_path)
        duration = len(waveform) / SAMPLE_RATE
        if duration <= 0.0:
            return [], "ru", 0.0

        self._emit(0.0)
        try:
            results = adapter.recognize(waveform, sample_rate=SAMPLE_RATE)
        except Exception as exc:
            raise TranscriptionError(
                f"Ошибка при распознавании речи (GigaAM) в файле {audio_path}: {exc}"
            ) from exc

        segments: list[TranscriptionSegment] = []
        if self._vad_active:
            # С VAD ``recognize`` — итератор сегментов (TimestampedSegmentResult).
            iterator: Iterator[Any] = iter(results)
            for item in iterator:
                segment = self._make_segment(
                    item, float(item.start or 0.0), float(item.end or 0.0)
                )
                if segment is not None:
                    segments.append(segment)
                    end = segment.end if segment.end > 0 else duration
                    self._emit(max(0.0, min(1.0, end / duration)))
        else:
            start, end = self._result_bounds(results, duration)
            segment = self._make_segment(results, start, end)
            if segment is not None:
                segments.append(segment)
        self._emit(1.0)

        if not segments:
            logger.info("GigaAM не нашёл речи в %s", audio_path)
        # GigaAM v3 — русскоязычная модель; возвращаем 'ru' независимо от
        # переданного language (он остаётся для совместимости протокола).
        return segments, "ru", duration
