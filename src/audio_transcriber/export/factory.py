"""Выбор реализации экспортёра по формату."""

from __future__ import annotations

from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.export.base import ResultExporter
from audio_transcriber.export.docx_exporter import DocxExporter
from audio_transcriber.export.json_exporter import JsonExporter
from audio_transcriber.export.srt_exporter import SrtExporter
from audio_transcriber.export.txt_exporter import TxtExporter

_EXPORTERS: dict[ExportFormat, type[ResultExporter]] = {
    ExportFormat.TXT: TxtExporter,
    ExportFormat.DOCX: DocxExporter,
    ExportFormat.JSON: JsonExporter,
    ExportFormat.SRT: SrtExporter,
}


def create_exporter(export_format: ExportFormat) -> ResultExporter:
    return _EXPORTERS[export_format]()
