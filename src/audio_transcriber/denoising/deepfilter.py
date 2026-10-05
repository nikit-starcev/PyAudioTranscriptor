"""Шумоподавление на DeepFilterNet (https://github.com/Rikorose/DeepFilterNet).

DeepFilterNet — нейросетевой подавитель шума, работающий на полнодиапазонном
аудио **48 кГц** моно. Конвейер распознавания и диаризации использует
**16 кГц**, поэтому полный цикл такой: декодировать вход (PyAV) → шумоподавление
на 48 кГц → ресемплинг в 16 кГц → временный WAV, который отдаётся как ASR,
так и диаризации (временные метки обоих движков согласованы).

Обработка **потоковая**: файл не грузится целиком в 48 кГц (это ~691 МБ на
час). Аудио декодируется кадрами, наполняет буфер, и по достижении
``chunk_seconds`` (по умолчанию 30 с) чанк прогоняется через модель. Соседние
чанки перекрываются на ``overlap_seconds`` (по умолчанию 0.5 с) и склеиваются
crossfade-overlap-add: без щелчков, пропусков и дублирования сэмплов. В памяти
живут только текущий чанк, хвост перекрытия и итоговый 16-кГц waveform.
Ресемплинг 48 → 16 кГц выполняется одним постоянным ``AudioResampler``,
которому чанки подаются последовательно, а в конце сбрасывается буфер — так
фильтр сохраняет непрерывность на стыках.

``df.enhance`` в начале каждого вызова сбрасывает состояние модели
(``model.reset_h0()``), поэтому чанки обрабатываются независимо и корректно.

Компонент спроектирован для мягкой деградации: если DeepFilterNet не
установлен, модель не загружается или аудио не декодируется, этап
пропускается с предупреждением в лог, а конвейер получает исходный файл.
Это критично: шумоподавление включено по умолчанию.

Модель DeepFilterNet при первом запуске скачивается в кэш
(``~/.cache/DeepFilterNet``). Дальше инференс полностью локальный и офлайн.
В этом окружении ``deepfilternet`` ставится вручную (сборка libDF требует
Rust), см. README.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import types
import wave
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import av
import numpy as np

from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    WaveformAccumulator,
    encode_pcm16,
    estimate_sample_count,
)
from audio_transcriber.utils.exceptions import AudioFileError

logger = logging.getLogger(__name__)

#: Частота дискретизации, с которой работает DeepFilterNet.
DF_SAMPLE_RATE = 48000

#: Длительность чанка денойза по умолчанию (секунды). Определяет пиковую память
#: стадии: 30 с при 48 кГц float32 — это ~5.5 МБ на буфер.
DEFAULT_CHUNK_SECONDS = 30.0

#: Длина перекрытия соседних чанков по умолчанию (секунды). Нужна для
#: кроссфейда на стыках; должна быть меньше половины чанка.
DEFAULT_OVERLAP_SECONDS = 0.5

# DeepFilterNet читает свои опции конфигурации из переменных окружения
# (``df.config`` использует ``os.environ[option.upper()]``). Переменные из
# ``config.env`` (например ``MODEL`` — это модель whisper) при экспорте через
# run.sh попадают в окружение и ломают загрузку DeepFilterNet ("No module named
# 'df.large-v3-turbo'"). На время инициализации убираем такие имена.
_COLLIDING_ENV_VARS = (
    "MODEL",
    "DEVICE",
    "LANGUAGE",
    "EPOCH",
    "SR",
    "LOG_LEVEL",
    "POST_FILTER",
    "MASK_ONLY",
)


@contextmanager
def _sanitized_env() -> Iterator[None]:
    """Временно убирает из окружения переменные, конфликтующие с настройками DeepFilterNet."""

    saved = {name: os.environ.pop(name) for name in _COLLIDING_ENV_VARS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)

# Загруженная модель кэшируется на уровне модуля: при обработке очереди файлов
# (TUI) она переиспользуется, а не грузится заново на каждый файл.
_MODEL_CACHE: tuple[Any, Any] | None = None
_MODEL_FAILED = False


def _install_torchaudio_backend_shim() -> None:
    """Добавляет совместимость DeepFilterNet с torchaudio >= 2.9.

    ``df.io`` импортирует ``AudioMetaData`` из ``torchaudio.backend.common``,
    который был удалён в новых torchaudio. Этот тип используется только в
    аннотациях, поэтому достаточно зарегистрировать заглушку до импорта ``df``.
    Ничего в site-packages не меняется.
    """

    try:
        import torchaudio.backend.common  # noqa: F401
    except ModuleNotFoundError:
        backend = types.ModuleType("torchaudio.backend")
        common = types.ModuleType("torchaudio.backend.common")
        setattr(common, "AudioMetaData", type("AudioMetaData", (), {}))  # noqa: B010
        setattr(backend, "common", common)  # noqa: B010
        sys.modules.setdefault("torchaudio.backend", backend)
        sys.modules.setdefault("torchaudio.backend.common", common)


def _load_deepfilter() -> tuple[Any, Any] | None:
    """Импортирует DeepFilterNet и загружает модель; ``None`` — движок недоступен."""

    try:
        _install_torchaudio_backend_shim()
        from df import init_df

        with _sanitized_env():
            model, df_state, _ = init_df(log_level="WARNING")
    except Exception as exc:  # noqa: BLE001 — любая ошибка ведёт к мягкому пропуску
        logger.warning(
            "DeepFilterNet недоступен — шумоподавление будет пропущено "
            "(проверьте установку: см. README, раздел про денойз): %s",
            exc,
        )
        return None

    logger.info("Модель DeepFilterNet загружена (частота %d Гц)", DF_SAMPLE_RATE)
    return model, df_state


def _ensure_model() -> tuple[Any, Any] | None:
    """Возвращает закэшированную модель, загружая её при первом обращении."""

    global _MODEL_CACHE, _MODEL_FAILED

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if _MODEL_FAILED:
        return None

    loaded = _load_deepfilter()
    if loaded is None:
        # Не пытаемся грузить модель снова на каждый файл очереди.
        _MODEL_FAILED = True
        return None
    _MODEL_CACHE = loaded
    return _MODEL_CACHE


class _SampleQueue:
    """FIFO-очередь моно-фрагментов с дешёвым доступом к первым N сэмплам.

    Позволяет сдвигать окно на ``hop`` сэмплов, оставляя ``overlap`` для
    кроссфейда, без повторной склейки всего буфера.
    """

    __slots__ = ("_chunks", "_size")

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, samples: np.ndarray) -> None:
        """Добавляет фрагмент в конец очереди."""
        if samples.size:
            self._chunks.append(samples)
            self._size += int(samples.shape[0])

    def peek(self, count: int) -> np.ndarray:
        """Возвращает первые ``count`` сэмплов (копию/представление), не удаляя их."""
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        if count >= self._size:
            if len(self._chunks) == 1:
                return self._chunks[0]
            return np.concatenate(list(self._chunks))
        parts: list[np.ndarray] = []
        remaining = count
        for chunk in self._chunks:
            take = min(remaining, int(chunk.shape[0]))
            parts.append(chunk if take == chunk.shape[0] else chunk[:take])
            remaining -= take
            if remaining == 0:
                break
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def drop(self, count: int) -> None:
        """Удаляет первые ``count`` сэмплов из очереди."""
        remaining = count
        while remaining > 0 and self._chunks:
            chunk = self._chunks[0]
            length = int(chunk.shape[0])
            if length <= remaining:
                self._chunks.popleft()
                self._size -= length
                remaining -= length
            else:
                self._chunks[0] = chunk[remaining:]
                self._size -= remaining
                remaining = 0


def _iter_decoded_frames(container: Any, stream: Any) -> Iterator[np.ndarray]:
    """Потоково декодирует поток в моно 48 кГц float32, отдавая кадры по мере чтения."""

    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout="mono", rate=DF_SAMPLE_RATE
    )
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            yield resampled.to_ndarray()[0]
    # Сброс буфера ресемплера — иначе теряется «хвост» потока.
    for resampled in resampler.resample(None):
        yield resampled.to_ndarray()[0]


def _require_length(enhanced: np.ndarray, expected: int) -> np.ndarray:
    """Проверяет, что модель вернула столько же сэмплов, сколько получила."""

    result = np.asarray(enhanced, dtype=np.float32).reshape(-1)
    if result.shape[0] != expected:
        raise ValueError(
            f"DeepFilterNet вернул {result.shape[0]} сэмплов вместо {expected}"
        )
    return result


def _iter_resolved_48k(
    frames: Iterator[np.ndarray],
    *,
    chunk_size: int,
    overlap: int,
    enhance: Callable[[np.ndarray], np.ndarray],
    on_chunk: Callable[[int], None] | None = None,
) -> Iterator[np.ndarray]:
    """Прогоняет поток через ``enhance`` чанками с overlap-add и отдаёт готовые куски.

    Соседние чанки перекрываются на ``overlap`` сэмплов. Кроссфейд (линейный,
    сумма весов = 1) сохраняет амплитуду и убирает щелчки. Возвращаемые куски
    идут строго последовательно и в сумме дают ровно столько же сэмплов, сколько
    пришло на вход (без пропусков и дублирования).

    ``on_chunk`` (если задан) вызывается с числом обработанных сэмплов входа
    после каждого чанка — для отображения прогресса длинной стадии денойза.
    """

    hop = chunk_size - overlap
    fade_in = (
        np.linspace(0.0, 1.0, num=overlap, endpoint=False, dtype=np.float32)
        if overlap > 0
        else np.empty(0, dtype=np.float32)
    )
    fade_out = 1.0 - fade_in
    queue = _SampleQueue()
    prev_tail: np.ndarray | None = None
    processed = 0

    for frame in frames:
        queue.push(frame)
        while len(queue) >= chunk_size:
            segment = queue.peek(chunk_size)
            queue.drop(hop)
            enhanced = _require_length(enhance(segment), chunk_size)
            if prev_tail is None:
                # Первый чанк: отдаём всё, кроме хвоста перекрытия.
                yield enhanced[:hop]
            else:
                # Стык: кроссфейд хвоста предыдущего чанка с головой текущего.
                if overlap > 0:
                    yield (prev_tail * fade_out + enhanced[:overlap] * fade_in).astype(
                        np.float32
                    )
                if hop > overlap:
                    yield enhanced[overlap:hop]
            prev_tail = enhanced[hop:]
            processed += hop
            if on_chunk is not None:
                on_chunk(processed)

    remaining = len(queue)
    if remaining > 0:
        segment = queue.peek(remaining)
        enhanced = _require_length(enhance(segment), remaining)
        if prev_tail is None:
            yield enhanced
        else:
            boundary = min(overlap, remaining)
            if boundary > 0:
                yield (
                    prev_tail[:boundary] * fade_out[:boundary]
                    + enhanced[:boundary] * fade_in[:boundary]
                ).astype(np.float32)
            if remaining > boundary:
                yield enhanced[boundary:]
        processed += remaining
        if on_chunk is not None:
            on_chunk(processed)
    elif prev_tail is not None and prev_tail.size:
        yield prev_tail


class DeepFilterDenoiser:
    """Шумоподавление через DeepFilterNet с мягкой деградацией.

    Создаёт временные WAV в собственном каталоге, который освобождается в
    :meth:`close` (также поддерживается контекстный менеджер).

    :param on_progress: приёмник событий прогресса (мягко необязателен).
        Атрибут публичный — :class:`~audio_transcriber.cache.denoiser.
        CachingDenoiser` прокидывает в него колбэк конвейера.
    """

    def __init__(
        self,
        *,
        output_sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate должен быть положительным")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds должен быть положительным")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds не может быть отрицательным")

        #: Исходные параметры конвейера (секунды/Гц) — нужны ключу кэша
        #: шумоподавления, чтобы смена чанкинга/частоты сбрасывала старый
        #: денойзенный WAV (#83). В сэмплах хранить их нельзя: зависят от SR.
        self._chunk_seconds = float(chunk_seconds)
        self._overlap_seconds = float(overlap_seconds)
        self._chunk_size = round(chunk_seconds * DF_SAMPLE_RATE)
        self._overlap = round(overlap_seconds * DF_SAMPLE_RATE)
        if self._chunk_size <= 0:
            raise ValueError("chunk_seconds слишком мал")
        if self._overlap * 2 > self._chunk_size:
            raise ValueError("overlap_seconds должен быть не больше половины chunk_seconds")

        self._output_sample_rate = output_sample_rate
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        #: Декодированный waveform последнего успешного ``denoise`` (частота
        #: конвейера). Нужен, чтобы диаризация переиспользовала уже декодированное
        #: аудио и не читала временный WAV повторно.
        self._last_waveform: np.ndarray | None = None
        #: Приёмник событий прогресса; публичный, чтобы обёртка могла прокинуть
        #: колбэк внутрь (см. ``CachingDenoiser``).
        self.on_progress = on_progress

    @property
    def chunk_seconds(self) -> float:
        """Целевая длина чанка денойза (с) — параметр ключа кэша (#83)."""
        return self._chunk_seconds

    @property
    def overlap_seconds(self) -> float:
        """Перекрытие соседних чанков денойза (с) — параметр ключа кэша (#83)."""
        return self._overlap_seconds

    @property
    def output_sample_rate(self) -> int:
        """Частота дискретизации выходного WAV (Гц) — параметр ключа кэша (#83)."""
        return self._output_sample_rate

    @property
    def last_waveform(self) -> np.ndarray | None:
        """Моно waveform 16 кГц последнего успешного :meth:`denoise` или ``None``.

        Отдаётся следующей стадии (диаризации) для переиспользования: исходный
        файл уже декодирован и отресемплен в частоту конвейера, повторное чтение
        временного WAV не требуется. ``None`` — денойз пропущен/не удался или
        еще не запускался; потребитель в этом случае декодирует файл сам.
        """
        return self._last_waveform

    def denoise(self, input_path: Path) -> Path:
        """Возвращает путь к очищенному аудио 16 кГц моно WAV.

        При любой проблеме (движок недоступен, файл не декодируется, сбой
        инференса) возвращает ``input_path`` без исключения.
        """

        self._last_waveform = None
        try:
            container = av.open(str(input_path))
        except Exception as exc:  # noqa: BLE001 — любая ошибка ведёт к мягкому пропуску
            logger.warning("Шумоподавление пропущено для %s: %s", input_path.name, exc)
            return input_path

        try:
            try:
                stream = container.streams.audio[0]
            except IndexError:
                logger.warning(
                    "Шумоподавление пропущено для %s: нет аудиодорожки", input_path.name
                )
                return input_path

            # Модель грузим только для валидного аудио: битый файл не должен
            # утаскивать за собой загрузку DeepFilterNet.
            model_and_state = _ensure_model()
            if model_and_state is None:
                return input_path

            model, df_state = model_and_state
            try:
                return self._denoise_stream(container, stream, input_path, model, df_state)
            except Exception as exc:  # noqa: BLE001 — не роняем конвейер из-за денойза
                logger.warning("Шумоподавление не удалось для %s: %s", input_path.name, exc)
                return input_path
        finally:
            container.close()

    def close(self) -> None:
        """Удаляет временные файлы (идемпотентно) и освобождает waveform."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        # Диаризация к этому моменту уже получила waveform (или её нет) —
        # удерживать декодированный массив нет смысла.
        self._last_waveform = None

    def __enter__(self) -> DeepFilterDenoiser:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _enhance(self, waveform: np.ndarray, model: Any, df_state: Any) -> np.ndarray:
        """Прогоняет 48 кГц waveform через модель; возвращает 48 кГц float32."""

        import torch
        from df import enhance

        contiguous = np.ascontiguousarray(waveform, dtype=np.float32)
        tensor = torch.from_numpy(contiguous).unsqueeze(0)
        with torch.no_grad(), _sanitized_env():
            enhanced = enhance(model, df_state, tensor)
        return enhanced.squeeze(0).detach().cpu().numpy()

    def _denoise_stream(
        self,
        container: Any,
        stream: Any,
        input_path: Path,
        model: Any,
        df_state: Any,
    ) -> Path:
        """Потоково обрабатывает открытый контейнер и пишет временный WAV."""

        expected = estimate_sample_count(container, stream, self._output_sample_rate)
        # ``expected`` — верхняя оценка числа сэмплов на частоте конвейера.
        # Переводим её в 48 кГц, в которых считает ``_iter_resolved_48k``, чтобы
        # отдавать реальную долю обработанного аудио (детерминированный прогресс
        # вместо «немого» этапа на минуты). Длительность неизвестна (0) — прогресс
        # не эмитим, только финальные 100 %.
        total_samples = (
            round(expected * DF_SAMPLE_RATE / self._output_sample_rate)
            if expected > 0 and self._output_sample_rate > 0
            else 0
        )

        def enhance(segment: np.ndarray) -> np.ndarray:
            return self._enhance(segment, model, df_state)

        resolved = _iter_resolved_48k(
            _iter_decoded_frames(container, stream),
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            enhance=enhance,
            on_chunk=lambda done: self._emit_progress(done, total_samples),
        )
        target = self._prepare_temp_path(input_path)
        self._write_temp_stream(target, resolved, expected)

        assert self._last_waveform is not None
        duration = self._last_waveform.shape[0] / self._output_sample_rate
        logger.info("Шумоподавление: %s → %s (%.1f с)", input_path.name, target.name, duration)
        self._emit(1.0)
        return target

    def _emit(self, fraction: float | None) -> None:
        """Отправляет событие прогресса денойза, если задан приёмник."""

        emit = self.on_progress
        if emit is not None:
            emit(ProgressEvent("denoise", "Шумоподавление", fraction=fraction))

    def _emit_progress(self, done: int, total: int) -> None:
        """Эмитит долю обработанного аудио (0…1); без оценки длины — молчит."""

        if total > 0:
            self._emit(min(done / total, 1.0))

    def _prepare_temp_path(self, input_path: Path) -> Path:
        """Создаёт (при необходимости) временный каталог и путь к результату."""

        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="audio-transcriber-denoise-")
        return Path(self._tmpdir.name) / f"{input_path.stem}.denoised.wav"

    def _write_temp_stream(
        self,
        target: Path,
        resolved: Iterator[np.ndarray],
        expected_samples: int,
    ) -> None:
        """Ресемплит 48 кГц поток в частоту конвейера и пишет его потоково в WAV.

        Один постоянный ``AudioResampler`` подаётся чанк за чанком, поэтому
        фильтр сохраняет непрерывность на стыках (новый ресемплер на каждый
        чанк терял бы или дублировал сэмплы). В конце буфер сбрасывается.
        """

        resampler: Any = None
        if self._output_sample_rate != DF_SAMPLE_RATE:
            resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="mono", rate=self._output_sample_rate
            )
        accumulator = WaveformAccumulator(expected_samples)

        with wave.open(str(target), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._output_sample_rate)
            for piece in resolved:
                if resampler is None:
                    self._append_and_write(wf, accumulator, piece)
                    continue
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(piece, dtype=np.float32).reshape(1, -1),
                    format="fltp",
                    layout="mono",
                )
                frame.sample_rate = DF_SAMPLE_RATE
                for resampled in resampler.resample(frame):
                    self._append_and_write(wf, accumulator, resampled.to_ndarray()[0])
            if resampler is not None:
                for resampled in resampler.resample(None):
                    self._append_and_write(wf, accumulator, resampled.to_ndarray()[0])

        if accumulator.size == 0:
            raise AudioFileError(f"Шумоподавление не вернуло сэмплов для {target.name}")
        # Запоминаем waveform, из которого записан WAV (сэмплы совпадают с
        # точностью до s16-квантования): диаризация возьмёт его напрямую, и
        # временные метки останутся согласованы с ASR.
        self._last_waveform = accumulator.finish()

    @staticmethod
    def _append_and_write(
        wf: wave.Wave_write,
        accumulator: WaveformAccumulator,
        samples: np.ndarray,
    ) -> None:
        """Добавляет моно-чанк в accumulator и сразу пишет его в WAV как s16."""

        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        accumulator.append(mono)
        wf.writeframes(encode_pcm16(mono))
