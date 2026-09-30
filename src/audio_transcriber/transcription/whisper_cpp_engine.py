"""Реализация распознавания речи через whisper.cpp (CLI).

В отличие от faster-whisper, whisper.cpp ускоряется на GPU не через
CUDA/PyTorch, а через Vulkan — поэтому работает и на AMD-картах
(например, Polaris/RX 4xx–5xx), которые ROCm/PyTorch не поддерживают.

Движок запускает собранный бинарник ``whisper-cli`` как внешний процесс
и разбирает его JSON-вывод, сохраняя интерфейс :class:`SpeechRecognizer`,
чтобы остальной конвейер (диаризация, объединение, экспорт) не менялся.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

from audio_transcriber.domain.models import TranscriptionSegment
from audio_transcriber.progress import ProgressCallback, ProgressEvent
from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    AudioProbe,
    load_waveform,
    probe_audio,
    write_wav,
)
from audio_transcriber.utils.env import with_library_path
from audio_transcriber.utils.exceptions import AudioFileError, TranscriptionError
from audio_transcriber.utils.subprocess_registry import register_process, terminate_process

logger = logging.getLogger(__name__)

#: Версия реализации ASR whisper.cpp. Участвует в ключе кэша (см.
#: ``pipeline._asr_cache_params``): при изменении логики, влияющей на результат
#: при тех же параметрах (например, отказ от лишнего перекодирования входа),
#: старый кэш должен инвалидироваться.
ASR_IMPL_VERSION = 2

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+(?:\.\d+)?)%")


def _is_native_wav(probe: AudioProbe) -> bool:
    """True, если whisper-cli прочитает файл напрямую, без перекодирования.

    Подходят 16-кГц моно PCM WAV — именно такой формат отдаёт шумоподавление.
    Прочие WAV (другая частота/каналы) и не-WAV (mp4/webm/mp3/…) конвертируются
    во временный WAV, как и раньше: убирать эту конвертацию шире, чем для
    «родного» формата, рискованно для качества и совместимости.
    """

    formats = {part.strip() for part in (probe.format_name or "").split(",")}
    return (
        "wav" in formats
        and probe.sample_rate == SAMPLE_RATE
        and probe.channels == 1
        and (probe.codec or "").startswith("pcm_")
    )


# Параметры VAD по умолчанию — как в faster-whisper
# (``faster_whisper.vad.VadOptions``), чтобы оба движка отсекали тишину/не-речь
# по одинаковым условиям. whisper.cpp требует отдельную Silero-VAD-модель
# (``--vad-model``), поэтому VAD включается, только если путь задан.
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_DURATION_MS = 0
VAD_MIN_SILENCE_DURATION_MS = 2000
VAD_SPEECH_PAD_MS = 400

# Служебные токены whisper.cpp в полном JSON (``-ojf``): ``[_BEG_]``,
# ``[_TT_129]``, ``[_EOT_]`` и т.п. У них тоже есть вероятность ``p``, но она
# относится не к речи, поэтому в среднюю уверенность не входит.
_SPECIAL_TOKEN_RE = re.compile(r"^\[_.*\]$")


def _segment_avg_logprob(item: dict) -> float | None:
    """Средняя логвероятность сегмента из полного JSON whisper.cpp (``-ojf``).

    whisper.cpp отдаёт вероятность ``p`` каждого токена. Аналог
    ``avg_logprob`` faster-whisper — среднее натуральных логарифмов ``p`` по
    «речевым» токенам (служебные ``[_...]`` пропускаются). Если токенов с
    вероятностями нет (например, запуск без ``-ojf``), возвращается ``None`` —
    фича мягко деградирует и не роняет конвейер.
    """
    tokens = item.get("tokens")
    if not isinstance(tokens, list):
        return None

    logprobs: list[float] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = str(token.get("text", "")).strip()
        if _SPECIAL_TOKEN_RE.match(text):
            continue
        probability = token.get("p")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            continue
        probability = float(probability)
        if probability <= 0.0:
            continue
        logprobs.append(math.log(probability))

    if not logprobs:
        return None
    return sum(logprobs) / len(logprobs)


class WhisperCppRecognizer:
    """Распознаёт речь через whisper.cpp. Реализует протокол ``SpeechRecognizer``."""

    def __init__(
        self,
        model_path: Path,
        *,
        binary: str = "whisper-cli",
        library_path: str | None = None,
        threads: int | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        on_progress: ProgressCallback | None = None,
        vad_filter: bool = True,
        vad_model: Path | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._binary = binary
        self._library_path = library_path
        self._threads = threads
        self._initial_prompt = initial_prompt
        self._hotwords = hotwords
        self._on_progress = on_progress
        self._vad_filter = vad_filter
        self._vad_model = Path(vad_model) if vad_model is not None else None

    def _emit(self, fraction: float | None = None, detail: str = "") -> None:
        if self._on_progress is not None:
            self._on_progress(
                ProgressEvent("asr", "Распознавание речи", fraction=fraction, detail=detail)
            )

    def _build_prompt(self) -> str | None:
        """Собирает подсказку для ASR из initial_prompt и hotwords.

        whisper.cpp не имеет отдельного механизма hotwords (в отличие от
        faster-whisper), поэтому оба источника объединяются в ``--prompt``.
        """
        parts: list[str] = []
        if self._initial_prompt and self._initial_prompt.strip():
            parts.append(self._initial_prompt.strip())
        if self._hotwords and self._hotwords.strip():
            parts.append(self._hotwords.strip())
        return " ".join(parts) or None

    def _vad_args(self) -> list[str]:
        """Флаги VAD для whisper-cli, выровненные с faster-whisper.

        whisper.cpp включает VAD только вместе с моделью Silero (``--vad-model``).
        Если фильтр включён, а модель не задана/не найдена, VAD мягко
        пропускается (как и прочие необязательные возможности проекта), чтобы
        не ронять распознавание.
        """
        if not self._vad_filter:
            return []
        if self._vad_model is None:
            logger.debug(
                "VAD включён, но модель whisper.cpp VAD не задана "
                "(WHISPER_CPP_VAD_MODEL) — распознавание идёт без VAD"
            )
            return []
        if not self._vad_model.is_file():
            logger.warning(
                "Модель whisper.cpp VAD не найдена: %s — распознавание идёт без VAD",
                self._vad_model,
            )
            return []
        return [
            "--vad",
            "-vm",
            str(self._vad_model),
            # Значения — как в faster_whisper.vad.VadOptions по умолчанию.
            "-vt",
            str(VAD_THRESHOLD),
            "-vspd",
            str(VAD_MIN_SPEECH_DURATION_MS),
            "-vsd",
            str(VAD_MIN_SILENCE_DURATION_MS),
            "-vp",
            str(VAD_SPEECH_PAD_MS),
        ]

    def _prepare_input(self, audio_path: Path, tmpdir_path: Path) -> tuple[Path, float]:
        """Готовит вход для whisper-cli и возвращает (путь, длительность, с).

        «Родной» для whisper-cli вход (16-кГц моно PCM WAV, например результат
        шумоподавления) отдаётся в ``-f`` как есть: лишний round-trip
        декодирование→запись вносил разницу в 1 LSB, из-за которой whisper.cpp
        терял речь (~59 с на реальном файле). Остальные форматы, которые
        miniaudio не декодирует (mp4/webm/mp3/…), а также WAV другой частоты или
        числа каналов по-прежнему перекодируются во временный 16-кГц WAV.

        Длительность берём из заголовка при passthrough — декодировать сэмплы
        для этого не нужно; иначе она равна длине декодированного waveform.
        """

        probe: AudioProbe | None
        try:
            probe = probe_audio(audio_path)
        except AudioFileError:
            probe = None

        if probe is not None and _is_native_wav(probe):
            logger.debug("whisper.cpp: вход отдаётся напрямую (%s)", audio_path)
            duration = probe.duration_seconds or 0.0
            return audio_path, duration

        wav_path = tmpdir_path / "audio.wav"
        waveform = load_waveform(audio_path)
        write_wav(wav_path, waveform)
        # Реальная длительность аудио (включая хвостовую тишину), а не конец
        # последнего сегмента — VAD отсекает тишину, из-за чего ``end``
        # последней реплики систематически занижает длительность.
        return wav_path, len(waveform) / SAMPLE_RATE

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        if not self._model_path.is_file():
            raise TranscriptionError(f"Модель whisper.cpp не найдена: {self._model_path}")

        with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_base = tmpdir_path / "result"

            input_path, audio_duration = self._prepare_input(audio_path, tmpdir_path)

            cmd = [
                self._binary,
                "-m",
                str(self._model_path),
                "-f",
                str(input_path),
                "-l",
                language or "auto",
                # -ojf (полный JSON) дополнительно отдаёт вероятности токенов,
                # по которым считается средняя уверенность реплики
                # (аналог avg_logprob faster-whisper).
                "-ojf",
                "-of",
                str(output_base),
                # -pp печатает «progress = N%» в stderr — по нему TUI
                # показывает реальный прогресс распознавания.
                "-pp",
            ]
            cmd += self._vad_args()
            if self._threads:
                cmd += ["-t", str(self._threads)]
            prompt = self._build_prompt()
            if prompt:
                cmd += ["--prompt", prompt]

            env = with_library_path(os.environ, self._library_path)

            logger.debug("Запуск whisper.cpp: %s", " ".join(cmd))
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
            except FileNotFoundError as exc:
                raise TranscriptionError(f"Бинарник whisper-cli не найден: {self._binary}") from exc

            # Регистрируем процесс в общем реестре: при SIGINT/SIGTERM или
            # аварийном выходе whisper-cli не останется висеть.
            register_process(proc)
            stderr_tail: list[str] = []
            try:
                assert proc.stderr is not None
                for line in proc.stderr:
                    stderr_tail.append(line)
                    if len(stderr_tail) > 40:
                        stderr_tail.pop(0)
                    match = _PROGRESS_RE.search(line)
                    if match:
                        self._emit(fraction=float(match.group(1)) / 100.0)

                proc.wait()
            finally:
                terminate_process(proc)

            json_path = Path(str(output_base) + ".json")
            if proc.returncode != 0 or not json_path.exists():
                detail = "".join(stderr_tail).strip()[-2000:]
                raise TranscriptionError(
                    f"whisper.cpp завершился с ошибкой (код {proc.returncode}): {detail}"
                )

            data = json.loads(json_path.read_text(encoding="utf-8"))

        segments = [
            TranscriptionSegment(
                start=item["offsets"]["from"] / 1000.0,
                end=item["offsets"]["to"] / 1000.0,
                text=item["text"].strip(),
                avg_logprob=_segment_avg_logprob(item),
            )
            for item in data.get("transcription", [])
            if item.get("text", "").strip()
        ]

        detected = data.get("result", {}).get("language")
        # Длительность берём из декодированного аудио. Если по какой-то причине
        # она неизвестна, откатываемся к концу последней реплики.
        duration = audio_duration if audio_duration > 0 else (segments[-1].end if segments else 0.0)

        return segments, language or detected, duration
