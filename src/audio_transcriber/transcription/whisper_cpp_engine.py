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
from audio_transcriber.utils.audio import load_waveform, write_wav
from audio_transcriber.utils.env import with_library_path
from audio_transcriber.utils.exceptions import TranscriptionError
from audio_transcriber.utils.subprocess_registry import register_process, terminate_process

logger = logging.getLogger(__name__)

_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+(?:\.\d+)?)%")

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
    ) -> None:
        self._model_path = Path(model_path)
        self._binary = binary
        self._library_path = library_path
        self._threads = threads
        self._initial_prompt = initial_prompt
        self._hotwords = hotwords
        self._on_progress = on_progress

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

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[list[TranscriptionSegment], str, float]:
        if not self._model_path.is_file():
            raise TranscriptionError(f"Модель whisper.cpp не найдена: {self._model_path}")

        with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_base = tmpdir_path / "result"

            # whisper-cli (miniaudio) не декодирует все форматы (например,
            # WebM), поэтому перекодируем вход в 16-кГц WAV через PyAV.
            wav_path = tmpdir_path / "audio.wav"
            write_wav(wav_path, load_waveform(audio_path))

            cmd = [
                self._binary,
                "-m",
                str(self._model_path),
                "-f",
                str(wav_path),
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
        duration = segments[-1].end if segments else 0.0

        return segments, language or detected, duration
