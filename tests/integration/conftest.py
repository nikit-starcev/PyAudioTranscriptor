"""Фикстуры интеграционных тестов (маркер ``integration``).

Тесты каталога ``tests/integration`` запускаются только по явному запросу
(``pytest -m integration``): по умолчанию маркер исключён в ``pyproject.toml``.
Каждый тест мягко пропускается (``pytest.skip``), если в окружении нет нужного
бинарника, модели или токена, — интеграционный прогон на «чистой» машине не
должен падать, а обязан честно сообщить причину пропуска.

Разрешение внешних ресурсов (единый порядок для всех фикстур):

1. переменная окружения ``INTEGRATION_*`` — явное переопределение для запуска
   (удобно в CI и при нестандартных путях);
2. значение из ``config.env`` (``WHISPER_CPP_*``, ``PYANNOTE_LOCAL_MODEL``,
   ``HF_TOKEN``, ``MODEL``) — «боевые» локальные пути пользователя;
3. для бинарника whisper.cpp — ``PATH`` (``whisper-cli``).

Короткий тестовый фрагмент генерируется тут же из уже хранящегося в репозитории
``tests/tests_jfk.flac`` (отдельный бинарный файл не коммитится).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform, write_wav
from audio_transcriber.utils.config_env import find_config_env, parse_config_env

from ._helpers import (
    REFERENCE_AUDIO,
    SHORT_FIXTURE_SECONDS,
    PyannoteTarget,
    WhisperCppPaths,
)


@pytest.fixture
def config_env_values() -> dict[str, str]:
    """Значения ``config.env`` из корня проекта (пустой словарь, если файла нет)."""
    path = find_config_env()
    if path is None:
        return {}
    try:
        return parse_config_env(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


@pytest.fixture
def short_speech_wav(tmp_path: Path) -> Path:
    """Короткий (``SHORT_FIXTURE_SECONDS`` с) WAV с реальной речью.

    Генерируется из эталонной записи ``tests/tests_jfk.flac``; если её нет,
    тест пропускается.
    """
    if not REFERENCE_AUDIO.is_file():
        pytest.skip(f"эталонная запись не найдена: {REFERENCE_AUDIO}")

    waveform = load_waveform(REFERENCE_AUDIO)
    samples = int(SHORT_FIXTURE_SECONDS * SAMPLE_RATE)
    if waveform.shape[0] < samples:
        pytest.skip(f"запись короче {SHORT_FIXTURE_SECONDS:g} с: {REFERENCE_AUDIO}")

    path = tmp_path / "short_speech.wav"
    write_wav(path, waveform[:samples])
    return path


def _env_or_config(
    os_var: str, config: dict[str, str], config_key: str, default: str = ""
) -> str:
    """Возвращает переменную ``INTEGRATION_*``, иначе значение config.env, иначе default."""
    override = os.environ.get(os_var, "").strip()
    if override:
        return override
    return config.get(config_key, "").strip() or default


@pytest.fixture
def whisper_cpp_paths(config_env_values: dict[str, str]) -> WhisperCppPaths:
    """Пути whisper.cpp из окружения/config.env; иначе ``pytest.skip``."""
    raw_binary = _env_or_config(
        "INTEGRATION_WHISPER_CPP_BINARY",
        config_env_values,
        "WHISPER_CPP_BINARY",
        default="whisper-cli",
    )
    binary = shutil.which(raw_binary) or (
        raw_binary if Path(raw_binary).is_file() else None
    )
    if binary is None:
        pytest.skip(f"бинарник whisper.cpp не найден: {raw_binary}")

    raw_model = _env_or_config(
        "INTEGRATION_WHISPER_CPP_MODEL", config_env_values, "WHISPER_CPP_MODEL"
    )
    if not raw_model:
        pytest.skip("модель whisper.cpp не задана (WHISPER_CPP_MODEL)")
    model = Path(raw_model)
    if not model.is_file():
        pytest.skip(f"модель whisper.cpp не найдена: {model}")

    library_path = (
        _env_or_config(
            "INTEGRATION_WHISPER_CPP_LIB_PATH", config_env_values, "WHISPER_CPP_LIB_PATH"
        )
        or None
    )
    return WhisperCppPaths(binary=binary, model=model, library_path=library_path)


@pytest.fixture
def faster_whisper_models(config_env_values: dict[str, str]) -> list[str]:
    """Кандидаты моделей faster-whisper в порядке предпочтения (от лёгких к тяжёлым).

    Сначала явное переопределение ``INTEGRATION_FASTER_WHISPER_MODEL``, затем
    лёгкие ``tiny.en``/``tiny`` (скачиваются при первом запуске) и лишь потом
    модель ``MODEL`` из ``config.env`` (обычно тяжёлая, но может быть в кэше).
    """
    candidates: list[str] = []
    override = os.environ.get("INTEGRATION_FASTER_WHISPER_MODEL", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(["tiny.en", "tiny"])
    configured = config_env_values.get("MODEL", "").strip()
    if configured:
        candidates.append(configured)
    # Уникальные с сохранением порядка предпочтения.
    return list(dict.fromkeys(candidates))


@pytest.fixture
def pyannote_target(config_env_values: dict[str, str]) -> PyannoteTarget:
    """Локальная модель и/или HF-токен pyannote; иначе ``pytest.skip``.

    Приоритет у локальной модели (работает офлайн). Если её нет, нужен токен
    Hugging Face — из ``INTEGRATION_HF_TOKEN``, ``config.env`` или окружения.
    """
    raw_local = _env_or_config(
        "INTEGRATION_PYANNOTE_LOCAL_MODEL", config_env_values, "PYANNOTE_LOCAL_MODEL"
    )
    local_model = Path(raw_local) if raw_local else None
    if local_model is not None and not local_model.exists():
        pytest.skip(f"локальная модель pyannote не найдена: {local_model}")

    token = (
        os.environ.get("INTEGRATION_HF_TOKEN", "").strip()
        or config_env_values.get("HF_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    if local_model is None and not token:
        pytest.skip(
            "нет локальной модели pyannote (PYANNOTE_LOCAL_MODEL) и HF_TOKEN — "
            "диаризация пропущена"
        )
    return PyannoteTarget(local_model=local_model, hf_token=token or None)
