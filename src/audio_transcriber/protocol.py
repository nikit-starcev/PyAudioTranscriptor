"""Формирование «протокола» встречи по запросу.

Протокол — это стенограмма плюс резюме, выгруженные в форматы
:attr:`AppConfig.export_formats` (txt/docx/...). В отличие от «авто»-режима
конвейера (:attr:`AppConfig.protocol_auto`), здесь резюме **пересчитывается по
текущей стенограмме** — уже с применёнными именами говорящих, — поэтому правки
имён находят отражение в итоговом документе.

Модуль переиспользует те же компоненты, что и конвейер (LLM-резюме,
экспортёры, таймлайн), и создаёт собственный LLM-клиент, если он не передан.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from audio_transcriber.config.settings import AppConfig
from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.factory import create_exporter
from audio_transcriber.export.timeline import build_speaker_tracks, write_timeline
from audio_transcriber.llm.base import LlmClient
from audio_transcriber.llm.chunking import chunk_chars_for_context
from audio_transcriber.llm.client import create_llm_client
from audio_transcriber.llm.summary import summarize_meeting
from audio_transcriber.progress import ProgressCallback, ProgressEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProtocolArtifacts:
    """Итог формирования протокола: пути документов и текст резюме.

    ``paths`` — выгруженные файлы (в порядке :attr:`AppConfig.export_formats`
    плюс таймлайн, если он строился). ``summary`` — резюме, вошедшее в
    документ (``None``, если LLM выключена или резюме не удалось получить).
    """

    paths: tuple[Path, ...] = ()
    summary: str | None = None


def _recompute_summary(
    config: AppConfig,
    result: TranscriptionResult,
    *,
    llm_client: LlmClient | None,
    emit: ProgressCallback,
) -> str | None:
    """Пересчитывает резюме по текущей стенограмме (или ``None``).

    Резюме считается только при включённой LLM и включённом
    :attr:`AppConfig.llm_summary`. Клиент, созданный здесь, закрывается в
    ``finally`` — ``llama-server`` не должен оставаться висеть.
    """
    if not (config.llm_enabled and config.llm_summary):
        return result.summary

    owns_client = llm_client is None
    client = llm_client
    if client is None:
        client = create_llm_client(config)
    if client is None:
        logger.warning(
            "Протокол без резюме: LLM включена, но клиент не создан "
            "(проверьте LLM_MODEL/LLM_BASE_URL/LLM_MODEL_NAME)"
        )
        return result.summary

    emit(ProgressEvent("llm", "Резюме встречи", fraction=None))
    try:
        summary = summarize_meeting(
            result.entries,
            result.speakers,
            llm=client,
            max_chunk_chars=chunk_chars_for_context(config.llm_context_size),
            system_prompt=config.llm_summary_prompt,
        )
    except Exception as exc:  # noqa: BLE001 — протокол выгружаем и без резюме
        logger.warning("Резюме для протокола не удалось: %s", exc)
        summary = result.summary
    finally:
        if owns_client and client is not None:
            client.close()

    if summary:
        emit(ProgressEvent("llm", "Резюме встречи", fraction=1.0))
    return summary


def generate_protocol(
    config: AppConfig,
    result: TranscriptionResult,
    *,
    llm_client: LlmClient | None = None,
    on_progress: ProgressCallback | None = None,
) -> ProtocolArtifacts:
    """Формирует протокол: резюме по текущей стенограмме + экспорт документов.

    Возвращает :class:`ProtocolArtifacts` с путями выгруженных файлов и
    текстом резюме. Если LLM выключена или недоступна, документ выгружается
    без резюме. Имена говорящих берутся из ``result`` как есть — поэтому
    правки в редакторе говорящих попадают в итоговый документ.
    """
    emit = on_progress or (lambda _event: None)
    config.ensure_output_dir()

    summary = _recompute_summary(config, result, llm_client=llm_client, emit=emit)
    document = replace(result, summary=summary) if summary != result.summary else result

    paths: list[Path] = []
    for export_format in config.export_formats:
        output_path = config.output_dir / f"{config.input_file.stem}.{export_format.value}"
        logger.info("Протокол: экспорт в %s — %s", export_format.value, output_path)
        emit(ProgressEvent("export", f"Экспорт {export_format.value}", fraction=None))
        create_exporter(export_format).export(document, output_path)
        paths.append(output_path)

    if config.timeline and build_speaker_tracks(document):
        timeline_path = config.output_dir / f"{config.input_file.stem}.timeline.html"
        emit(ProgressEvent("export", "Экспорт таймлайна", fraction=None))
        if write_timeline(document, timeline_path):
            paths.append(timeline_path)

    emit(ProgressEvent("done", "Протокол готов", fraction=1.0))
    return ProtocolArtifacts(paths=tuple(paths), summary=summary)
