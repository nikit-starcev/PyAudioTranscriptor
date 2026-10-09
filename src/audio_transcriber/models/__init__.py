"""Каталог моделей и скачивание с Hugging Face — без зависимости от веб-слоя.

Пакет переиспользуют и веб-API (:mod:`audio_transcriber.web.models`
переэкспортирует публичный API), и CLI-команда ``models``.
"""

from __future__ import annotations

from audio_transcriber.models.catalog import (
    KIND_GIGAAM,
    KIND_LLM,
    KIND_PYANNOTE,
    KIND_SHERPA,
    KIND_WHISPER,
    MODEL_CATALOG,
    LocalModelStatus,
    ModelEntry,
    ModelError,
    ModelFile,
    configured_path,
    delete_model_files,
    dir_size,
    find_model,
    free_space,
    local_status,
    primary_path,
    resolve_target,
)
from audio_transcriber.models.download import (
    DEFAULT_HEARTBEAT,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_DOWNLOADING,
    STATUS_ERROR,
    STATUS_IDLE,
    DownloadBus,
    DownloadCancelled,
    Downloader,
    DownloadState,
    HfDownloader,
    ModelDownloadManager,
    model_payload,
)

__all__ = [
    "DEFAULT_HEARTBEAT",
    "KIND_GIGAAM",
    "KIND_LLM",
    "KIND_PYANNOTE",
    "KIND_SHERPA",
    "KIND_WHISPER",
    "MODEL_CATALOG",
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_DOWNLOADING",
    "STATUS_ERROR",
    "STATUS_IDLE",
    "DownloadBus",
    "DownloadCancelled",
    "DownloadState",
    "Downloader",
    "HfDownloader",
    "LocalModelStatus",
    "ModelDownloadManager",
    "ModelEntry",
    "ModelError",
    "ModelFile",
    "configured_path",
    "delete_model_files",
    "dir_size",
    "find_model",
    "free_space",
    "local_status",
    "model_payload",
    "primary_path",
    "resolve_target",
]
