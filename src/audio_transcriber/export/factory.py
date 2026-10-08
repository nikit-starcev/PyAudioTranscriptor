"""Выбор реализации экспортёра по формату."""

from __future__ import annotations

from audio_transcriber.domain.enums import ExportFormat
from audio_transcriber.export.base import ResultExporter
from audio_transcriber.export.docx_exporter import DocxExporter
from audio_transcriber.export.json_exporter import JsonExporter
from audio_transcriber.export.markdown_exporter import MarkdownExporter
from audio_transcriber.export.pdf_exporter import PdfExporter
from audio_transcriber.export.srt_exporter import SrtExporter
from audio_transcriber.export.txt_exporter import TxtExporter
from audio_transcriber.export.vtt_exporter import VttExporter

_EXPORTERS: dict[ExportFormat, type[ResultExporter]] = {
    ExportFormat.TXT: TxtExporter,
    ExportFormat.DOCX: DocxExporter,
    ExportFormat.JSON: JsonExporter,
    ExportFormat.MARKDOWN: MarkdownExporter,
    ExportFormat.PDF: PdfExporter,
}

#: Субтитровые форматы, поддерживающие пословную подсветку (#45).
_HIGHLIGHT_EXPORTERS: dict[ExportFormat, type[SrtExporter] | type[VttExporter]] = {
    ExportFormat.SRT: SrtExporter,
    ExportFormat.VTT: VttExporter,
}


def create_exporter(
    export_format: ExportFormat, *, highlight_words: bool = False
) -> ResultExporter:
    if export_format in _HIGHLIGHT_EXPORTERS:
        return _HIGHLIGHT_EXPORTERS[export_format](highlight_words=highlight_words)
    return _EXPORTERS[export_format]()
