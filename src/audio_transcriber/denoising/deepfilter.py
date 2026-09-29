"""Шумоподавление на DeepFilterNet (https://github.com/Rikorose/DeepFilterNet).

DeepFilterNet — нейросетевой подавитель шума, работающий на полнодиапазонном
аудио **48 кГц** моно. Конвейер распознавания и диаризации использует
**16 кГц**, поэтому полный цикл такой: декодировать вход (PyAV) → 48 кГц →
шумоподавление → ресемплинг в 16 кГц → временный WAV, который отдаётся как
ASR, так и диаризации (временные метки обоих движков согласованы).

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
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

from audio_transcriber.utils.audio import (
    SAMPLE_RATE,
    load_waveform,
    resample_waveform,
    write_wav,
)

logger = logging.getLogger(__name__)

#: Частота дискретизации, с которой работает DeepFilterNet.
DF_SAMPLE_RATE = 48000

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


class DeepFilterDenoiser:
    """Шумоподавление через DeepFilterNet с мягкой деградацией.

    Создаёт временные WAV в собственном каталоге, который освобождается в
    :meth:`close` (также поддерживается контекстный менеджер).
    """

    def __init__(self, *, output_sample_rate: int = SAMPLE_RATE) -> None:
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate должен быть положительным")
        self._output_sample_rate = output_sample_rate
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def denoise(self, input_path: Path) -> Path:
        """Возвращает путь к очищенному аудио 16 кГц моно WAV.

        При любой проблеме (движок недоступен, файл не декодируется, сбой
        инференса) возвращает ``input_path`` без исключения.
        """

        try:
            waveform = load_waveform(input_path, sample_rate=DF_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 — любая ошибка ведёт к мягкому пропуску
            logger.warning("Шумоподавление пропущено для %s: %s", input_path.name, exc)
            return input_path

        model_and_state = _ensure_model()
        if model_and_state is None:
            return input_path

        model, df_state = model_and_state
        try:
            enhanced = self._enhance(waveform, model, df_state)
            return self._write_temp(input_path, enhanced)
        except Exception as exc:  # noqa: BLE001 — не роняем конвейер из-за денойза
            logger.warning("Шумоподавление не удалось для %s: %s", input_path.name, exc)
            return input_path

    def close(self) -> None:
        """Удаляет временные файлы (идемпотентно)."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

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
        with torch.no_grad():
            enhanced = enhance(model, df_state, tensor)
        return enhanced.squeeze(0).detach().cpu().numpy()

    def _write_temp(self, input_path: Path, samples_48k: np.ndarray) -> Path:
        """Ресемплит результат в частоту конвейера и сохраняет во временный WAV."""

        resampled = resample_waveform(
            samples_48k, source_rate=DF_SAMPLE_RATE, target_rate=self._output_sample_rate
        )
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="audio-transcriber-denoise-")
        target = Path(self._tmpdir.name) / f"{input_path.stem}.denoised.wav"
        write_wav(target, resampled, sample_rate=self._output_sample_rate)
        duration = resampled.shape[0] / self._output_sample_rate
        logger.info("Шумоподавление: %s → %s (%.1f с)", input_path.name, target.name, duration)
        return target
