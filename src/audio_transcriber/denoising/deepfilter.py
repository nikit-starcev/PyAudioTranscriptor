"""Шумоподавление через внешний Rust-CLI ``deep-filter`` (DeepFilterNet).

DeepFilterNet распространяется как Python-пакет ``deepfilternet``/``DeepFilterLib``,
у которого нет колёс для Python > 3.11 и который тянет ``numpy<2``. Чтобы не
ломать резолв зависимостей проекта (Python 3.14, numpy 2.x), денойз выполняется
готовым бинарником ``deep-filter`` из релизов DeepFilterNet (v0.5.6+).

``deep-filter`` — обычное CLI: ``deep-filter -D <вход.wav> -o <каталог>``. Флаг
``-D`` (``--compensate-delay``) обязателен: без него выход сдвинут на задержку
STFT/модели (~30 мс). Бинарник работает на полнодиапазонном аудио **48 кГц** моно;
конвейер распознавания и диаризации использует **16 кГц**, поэтому полный цикл
такой: декодировать вход (PyAV) → шумоподавление на 48 кГц → ресемплинг в 16 кГц
→ временный WAV, который отдаётся как ASR, так и диаризации (временные метки
обоих движков согласованы).

Модель у бинарника встроена: никаких ``torch``/``numpy`` и скачивания весов не
требуется — инференс полностью локальный и офлайн.

Обработка **потоковая и чанковая**. CLI загружает переданный ему файл целиком,
поэтому на часовой записи пик памяти превысил бы 1 ГБ. Чтобы этого не
допустить, вход режется на чанки по ``chunk_seconds`` (по умолчанию 30 с),
каждый чанк отдаётся ``deep-filter`` отдельным процессом, а результат
склеивается crossfade-overlap-add. Соседние чанки перекрываются на
``overlap_seconds`` (по умолчанию 0.5 с): без щелчков, пропусков и дублирования
сэмплов. В Python-памяти живут только текущий чанк, хвост перекрытия и итоговый
16-кГц waveform. Каждый вызов CLI загружает модель встроенную (~57 МБ RSS) и
обрабатывает ровно один чанк, поэтому память не растёт с длиной записи.

Компонент спроектирован для мягкой деградации: если бинарник не найден,
аудио не декодируется или процесс упал, этап пропускается с предупреждением в
лог, а конвейер получает исходный файл. Это критично: шумоподавление включено
по умолчанию.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np

from audio_transcriber.config.defaults import DEFAULT_DEEP_FILTER_BINARY
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    WaveformAccumulator,
    encode_pcm16,
    estimate_sample_count,
    write_wav,
)
from audio_transcriber.utils.exceptions import AudioFileError
from audio_transcriber.utils.subprocess_registry import register_process, terminate_process

logger = logging.getLogger(__name__)

#: Частота дискретизации, с которой работает DeepFilterNet.
DF_SAMPLE_RATE = 48000

#: Длительность чанка денойза по умолчанию (секунды). Определяет пиковую память
#: стадии: 30 с при 48 кГц float32 — это ~5.5 МБ на буфер. Ровно один чанк
#: живёт в памяти Python и подаётся внешнему ``deep-filter``.
DEFAULT_CHUNK_SECONDS = 30.0

#: Длина перекрытия соседних чанков по умолчанию (секунды). Нужна для
#: кроссфейда на стыках; должна быть меньше половины чанка.
DEFAULT_OVERLAP_SECONDS = 0.5

#: Версия реализации денойза. Входит в ключ кэша: переход с Python-пакета
#: ``deepfilternet`` на Rust-CLI меняет результат при тех же параметрах, поэтому
#: старый кэш должен пересчитаться ровно один раз.
DENOISE_IMPL_VERSION = 2

#: Чанки короче этого числа сэмплов (0.1 с) не отправляются в CLI: модели нужен
#: минимальный контекст STFT, а «хвост» записи такой длиной на слух неразличим.
MIN_ENHANCE_SAMPLES = DF_SAMPLE_RATE // 10

#: Допустимое укорочение выхода ``deep-filter -D``. С флагом ``-D`` CLI отдаёт
#: выровненный сигнал без задержки STFT/модели (~30 мс на 48 кГц), то есть на
#: эту задержку короче входа. Недостающие сэмплы добиваются нулями: они лежат в
#: зоне перекрытия чанков (0.5 с) и полностью замещаются головой следующего
#: чанка при crossfade. 0.1 с — с запасом.
MAX_DELAY_SAMPLES = DF_SAMPLE_RATE // 10

#: Таймаут одного запуска ``deep-filter`` (секунды). Чанк ограничен по длине,
#: поэтому щедрый предел защищает от зависания, не мешая штатной работе.
DEFAULT_DEEP_FILTER_TIMEOUT = 600.0


def _resolve_binary(binary: str) -> str | None:
    """Возвращает путь к ``deep-filter`` или ``None``, если он недоступен.

    Принимает как имя в ``PATH``, так и явный путь к файлу. Пустая строка —
    «не задано» — трактуется как ``None`` (мягкий пропуск денойза).
    """

    raw = (binary or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    try:
        if candidate.is_file():
            return str(candidate)
    except OSError:
        pass
    return shutil.which(raw)


def _read_pcm16_wav(path: Path, *, expected_rate: int = DF_SAMPLE_RATE) -> np.ndarray:
    """Читает моно PCM16 WAV как float32 в диапазоне ``[-1, 1)``.

    ``deep-filter`` пишет 16-битный PCM WAV с той же частотой, что у входа
    (48 кГц). Читаем его напрямую модулем ``wave`` (без ресемплера), чтобы
    побитово восстановить сэмплы.
    """

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (OSError, wave.Error) as exc:
        raise AudioFileError(f"Не удалось прочитать выход deep-filter {path}: {exc}") from exc

    if width != 2:
        raise AudioFileError(
            f"deep-filter вернул WAV с {width * 8}-битным форматом вместо 16-битного"
        )
    if rate != expected_rate:
        raise AudioFileError(
            f"deep-filter вернул WAV {rate} Гц вместо {expected_rate} Гц"
        )

    data = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if channels > 1:
        # Моно — ожидаемый случай; на всякий случай усредняем каналы.
        data = data.reshape(-1, channels).mean(axis=1)
    return data / 32768.0


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
    """Проверяет, что движок вернул столько же сэмплов, сколько получил."""

    result = np.asarray(enhanced, dtype=np.float32).reshape(-1)
    if result.shape[0] != expected:
        raise ValueError(
            f"deep-filter вернул {result.shape[0]} сэмплов вместо {expected}"
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
    """Шумоподавление через внешний CLI ``deep-filter`` с мягкой деградацией.

    Создаёт временные WAV (чанки и итог) в собственном каталоге, который
    освобождается в :meth:`close` (также поддерживается контекстный менеджер).

    :param binary: имя бинарника в ``PATH`` или путь к нему. Значение по
        умолчанию — ``DEEP_FILTER_BINARY`` (``deep-filter``).
    :param on_progress: приёмник событий прогресса (мягко необязателен).
        Атрибут публичный — :class:`~audio_transcriber.cache.denoiser.
        CachingDenoiser` прокидывает в него колбэк конвейера.
    """

    def __init__(
        self,
        *,
        binary: str = DEFAULT_DEEP_FILTER_BINARY,
        output_sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
        timeout: float | None = DEFAULT_DEEP_FILTER_TIMEOUT,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate должен быть положительным")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds должен быть положительным")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds не может быть отрицательным")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout должен быть положительным или None")

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
        self._binary_setting = (binary or "").strip()
        self._timeout = float(timeout) if timeout is not None else None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._chunk_counter = 0
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
    def binary(self) -> str:
        """Настроенный бинарник ``deep-filter`` — параметр ключа кэша."""
        return self._binary_setting

    @property
    def impl_version(self) -> int:
        """Версия реализации денойза — параметр ключа кэша."""
        return DENOISE_IMPL_VERSION

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

        При любой проблеме (бинарник недоступен, файл не декодируется, сбой
        процесса) возвращает ``input_path`` без исключения.
        """

        self._last_waveform = None

        binary = _resolve_binary(self._binary_setting)
        if binary is None:
            logger.warning(
                "deep-filter не найден (%s) — шумоподавление будет пропущено. "
                "Задайте DEEP_FILTER_BINARY или установите бинарник deep-filter "
                "(см. README, раздел про денойз).",
                self._binary_setting or "<не задан>",
            )
            return input_path

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

            try:
                return self._denoise_stream(container, stream, input_path, binary)
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

    def _ensure_tmpdir(self) -> Path:
        """Возвращает каталог временных файлов, создавая его при необходимости."""
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="audio-transcriber-denoise-")
        return Path(self._tmpdir.name)

    def _enhance(self, waveform: np.ndarray, binary: str) -> np.ndarray:
        """Прогоняет один 48-кГц чанк через внешний ``deep-filter``.

        Чанк пишется во временный WAV, рядом создаётся каталог вывода, а
        результат читается обратно как 48-кГц float32. Возвращает ровно столько
        же сэмплов, сколько получил на вход.
        """

        samples = np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
        if samples.shape[0] < MIN_ENHANCE_SAMPLES:
            # Слишком короткий «хвост»: не гоняем модель, отдаём как есть.
            return samples.copy()

        tmpdir = self._ensure_tmpdir()
        index = self._chunk_counter
        self._chunk_counter += 1
        input_wav = tmpdir / f"in-{index:06d}.wav"
        output_dir = tmpdir / f"out-{index:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        write_wav(input_wav, samples, sample_rate=DF_SAMPLE_RATE)
        self._run_deep_filter(binary, input_wav, output_dir)

        output_wav = output_dir / input_wav.name
        if not output_wav.is_file():
            raise AudioFileError(
                f"deep-filter не создал выходной файл {output_wav.name}"
            )
        enhanced = _read_pcm16_wav(output_wav)
        expected = samples.shape[0]
        if enhanced.shape[0] != expected:
            # ``-D`` убирает задержку, поэтому выход короче входа на её величину.
            deficit = expected - enhanced.shape[0]
            if 0 < deficit <= MAX_DELAY_SAMPLES:
                enhanced = np.concatenate(
                    [enhanced, np.zeros(deficit, dtype=np.float32)]
                )
            else:
                raise ValueError(
                    f"deep-filter вернул {enhanced.shape[0]} сэмплов вместо {expected}"
                )
        return enhanced

    def _run_deep_filter(self, binary: str, input_wav: Path, output_dir: Path) -> None:
        """Запускает ``deep-filter -D <вход> -o <каталог>`` и проверяет код возврата.

        ``-D`` (``--compensate-delay``) обязателен: без него CLI сдвигает выход
        на задержку STFT/модели (~30 мс), и временные метки ASR/диаризации уехали
        бы относительно исходной записи.
        """

        command = [binary, "-D", str(input_wav), "-o", str(output_dir)]
        logger.debug("Запуск deep-filter: %s", " ".join(command))
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise AudioFileError(f"Бинарник deep-filter не найден: {binary}") from exc

        # Регистрируем процесс в общем реестре: при SIGINT/SIGTERM/отмене он не
        # останется висеть.
        register_process(proc)
        try:
            try:
                _stdout, stderr = proc.communicate(timeout=self._timeout)
            except subprocess.TimeoutExpired:
                terminate_process(proc)
                raise AudioFileError(
                    f"deep-filter не завершился за {self._timeout:g} с"
                ) from None
        finally:
            terminate_process(proc)

        if proc.returncode != 0:
            detail = (stderr or "").strip()[-2000:]
            message = f"deep-filter завершился с кодом {proc.returncode}"
            if detail:
                message = f"{message}: {detail}"
            raise AudioFileError(message)

    def _denoise_stream(
        self,
        container: Any,
        stream: Any,
        input_path: Path,
        binary: str,
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
            return self._enhance(segment, binary)

        resolved = _iter_resolved_48k(
            _iter_decoded_frames(container, stream),
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            enhance=enhance,
            on_chunk=lambda done: self._emit_progress(done, total_samples),
        )
        target = self._prepare_temp_path(input_path)
        self._write_temp_stream(target, resolved, expected)

        waveform = self._last_waveform
        if waveform is None:
            # Явная проверка вместо ``assert``: под ``-O`` он выключился бы и
            # упал бы сырой ``AttributeError``; ошибка должна быть понятной.
            raise AudioFileError(
                f"Шумоподавление не вернуло waveform для {input_path.name}"
            )
        duration = waveform.shape[0] / self._output_sample_rate
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

        return self._ensure_tmpdir() / f"{input_path.stem}.denoised.wav"

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
