"""Экспорт стенограммы в документ PDF (.pdf).

Используется лёгкая библиотека ``fpdf2``. Встроенные шрифты PDF не содержат
кириллицы, поэтому подключается TrueType-шрифт с кириллическим покрытием из
системы (DejaVu Sans/Liberation Sans/Noto Sans). Если подходящий шрифт не
найден, экспорт завершается понятной ошибкой ``ExportError``.
"""

from __future__ import annotations

from pathlib import Path

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.export.annotations import entry_markers
from audio_transcriber.export.timestamps import format_timestamp
from audio_transcriber.utils.exceptions import ExportError

#: Пары (regular, bold) TrueType-шрифтов с кириллицей, которые ищутся в системе.
_FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ),
)


def find_cyrillic_font() -> tuple[Path, Path]:
    """Возвращает пути к regular/bold TrueType-шрифту с кириллицей."""

    for regular, bold in _FONT_CANDIDATES:
        regular_path = Path(regular)
        bold_path = Path(bold)
        if regular_path.is_file() and bold_path.is_file():
            return regular_path, bold_path
    for regular, _ in _FONT_CANDIDATES:
        regular_path = Path(regular)
        if regular_path.is_file():
            return regular_path, regular_path
    raise ExportError(
        "Для экспорта в PDF не найден TrueType-шрифт с кириллицей "
        "(DejaVu Sans / Liberation Sans / Noto Sans). Установите, например, "
        "пакет fonts-dejavu."
    )


class PdfExporter:
    """Реализует протокол ``ResultExporter`` для формата PDF."""

    def export(self, result: TranscriptionResult, output_path: Path) -> None:
        try:
            from fpdf import FPDF
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise ExportError(
                "Для экспорта в PDF требуется библиотека fpdf2. "
                "Установите её: uv pip install fpdf2"
            ) from exc

        regular, bold = find_cyrillic_font()
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_font("Cyrillic", "", str(regular))
        pdf.add_font("Cyrillic", "B", str(bold))
        pdf.add_page()

        pdf.set_font("Cyrillic", "B", 16)
        pdf.multi_cell(
            0, 9, text=result.source_path.name, new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(1)

        pdf.set_font("Cyrillic", "", 10)
        if result.language is not None:
            pdf.multi_cell(
                0, 6, text=f"Язык: {result.language}", new_x="LMARGIN", new_y="NEXT"
            )
        pdf.multi_cell(
            0,
            6,
            text=f"Длительность: {format_timestamp(result.duration)}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1)

        if result.participants:
            pdf.set_font("Cyrillic", "B", 12)
            pdf.multi_cell(0, 7, text="Участники", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Cyrillic", "", 10)
            for participant in result.participants:
                pdf.multi_cell(
                    0, 6, text=f"• {participant}", new_x="LMARGIN", new_y="NEXT"
                )
            pdf.ln(1)

        if result.summary:
            pdf.set_font("Cyrillic", "B", 12)
            pdf.multi_cell(0, 7, text="Резюме встречи", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Cyrillic", "", 10)
            for line in result.summary.splitlines():
                if line.strip():
                    pdf.multi_cell(0, 6, text=line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        pdf.set_font("Cyrillic", "B", 12)
        pdf.multi_cell(0, 7, text="Расшифровка", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        threshold = result.low_confidence_threshold
        for entry in result.entries:
            pdf.set_font("Cyrillic", "B", 10)
            pdf.write(6, f"[{format_timestamp(entry.start)}] {entry.speaker_label}: ")
            pdf.set_font("Cyrillic", "", 10)
            pdf.write(6, entry.text + entry_markers(entry, threshold))
            pdf.ln(6)

        try:
            pdf.output(str(output_path))
        except OSError as exc:
            raise ExportError(f"Не удалось сохранить файл {output_path}: {exc}") from exc
