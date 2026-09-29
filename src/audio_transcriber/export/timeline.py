"""Таймлайн «кто когда говорил»: текстовая сводка и самодостаточный HTML.

Строится по результату диаризации: подряд идущие реплики одного говорящего
объединяются в интервалы, по которым рисуется общая временная ось. HTML не
содержит внешних зависимостей/CDN и открывается локально в браузере.

Если данных о говорящих нет (диаризация выключена или ни одна реплика не
привязана к говорящему), таймлайн считается пустым и не создаётся — это не
ошибка, а мягкий пропуск.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.domain.models import TranscriptionResult
from audio_transcriber.utils.exceptions import ExportError

# Палитра дорожек: чередуется по кругу, чтобы соседние говорящие различались.
_TRACK_COLORS: tuple[str, ...] = (
    "#4c6ef5",
    "#12b886",
    "#f76707",
    "#ae3ec9",
    "#1098ad",
    "#e8590c",
    "#d6336c",
    "#37b24d",
)

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #f8f9fb; color: #1b1d23;
}
h1 { font-size: 20px; margin: 0 0 4px; }
.meta { margin: 0 0 16px; color: #6b7280; font-size: 13px; }
.chart {
  display: grid; gap: 8px; padding: 16px;
  background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
}
.row {
  display: grid; grid-template-columns: 200px 1fr;
  align-items: center; gap: 12px;
}
.label {
  font-size: 13px; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.track { position: relative; height: 26px; background: #eef1f5; border-radius: 6px; }
.block { position: absolute; top: 0; bottom: 0; border-radius: 6px; min-width: 2px; }
.ruler-row { height: 18px; }
.ruler { position: relative; height: 18px; color: #9ca3af; font-size: 11px; }
.tick { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.tick:first-child { transform: translateX(0); }
.tick:last-child { transform: translateX(-100%); }
"""


@dataclass(frozen=True, slots=True)
class SpeakerTrack:
    """Дорожка говорящего: идентификатор, имя и интервалы речи по порядку."""

    speaker_id: str
    display_name: str
    intervals: tuple[tuple[float, float], ...]


def format_clock(seconds: float) -> str:
    """Компактная метка времени: ``ММ:СС`` (для часов и более — ``Ч:ММ:СС``)."""

    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_speaker_tracks(result: TranscriptionResult) -> list[SpeakerTrack]:
    """Группирует подряд идущие реплики одного говорящего в интервалы.

    Дорожки возвращаются в порядке первого появления говорящего. Реплики без
    говорящего разрывают серию: в таймлайне «кто когда говорил» им места нет.
    """

    grouped: dict[str, list[tuple[float, float]]] = {}
    names: dict[str, str] = {}
    order: list[str] = []
    last_id: str | None = None

    for entry in result.entries:
        speaker = entry.speaker
        if speaker is None:
            last_id = None
            continue
        if speaker.id not in names:
            names[speaker.id] = speaker.display_name
            grouped[speaker.id] = []
            order.append(speaker.id)
        if speaker.id == last_id and grouped[speaker.id]:
            start = grouped[speaker.id][-1][0]
            grouped[speaker.id][-1] = (start, entry.end)
        else:
            grouped[speaker.id].append((entry.start, entry.end))
        last_id = speaker.id

    return [
        SpeakerTrack(speaker_id, names[speaker_id], tuple(grouped[speaker_id]))
        for speaker_id in order
        if grouped[speaker_id]
    ]


def _speaker_caption(label: str, display_name: str) -> str:
    """Подпись говорящего без дублирования имени.

    Если ``display_name`` пуст или совпадает с базовой меткой («Спикер 1»),
    показываем имя один раз — просто ``label``.
    """

    name = display_name.strip()
    if not name or name == label:
        return label
    return f"{label} ({name})"


def render_timeline_text(result: TranscriptionResult) -> str:
    """Текстовая сводка вида «Спикер N (Имя): 00:00–00:29, 00:53–00:59».

    Имя не дублируется: если ``display_name`` совпадает с меткой говорящего
    («Спикер 1») или пусто, выводится только метка. Пустая строка, если данных
    о говорящих нет.
    """

    tracks = build_speaker_tracks(result)
    if not tracks:
        return ""

    numbering: dict[str, int] = {
        speaker.id: index for index, speaker in enumerate(result.speakers, start=1)
    }
    next_index = len(result.speakers)
    lines: list[str] = []
    for track in tracks:
        if track.speaker_id not in numbering:
            next_index += 1
            numbering[track.speaker_id] = next_index
        ranges = ", ".join(
            f"{format_clock(start)}–{format_clock(end)}" for start, end in track.intervals
        )
        caption = _speaker_caption(
            f"Спикер {numbering[track.speaker_id]}", track.display_name
        )
        lines.append(f"{caption}: {ranges}")
    return "\n".join(lines)


def _timeline_duration(result: TranscriptionResult, tracks: Sequence[SpeakerTrack]) -> float:
    """Длительность общей оси: заявленная длительность или самый поздний конец."""

    candidates = [result.duration or 0.0]
    candidates.extend(end for track in tracks for _start, end in track.intervals)
    duration = max(candidates, default=0.0)
    return duration if duration > 0 else 1.0


def render_timeline_html(result: TranscriptionResult) -> str:
    """Самодостаточный HTML-таймлайн (без внешних зависимостей/CDN)."""

    tracks = build_speaker_tracks(result)
    duration = _timeline_duration(result, tracks)
    source_name = html.escape(result.source_path.name)
    total_clock = format_clock(duration)

    rows: list[str] = []
    for index, track in enumerate(tracks):
        color = _TRACK_COLORS[index % len(_TRACK_COLORS)]
        blocks: list[str] = []
        for start, end in track.intervals:
            left = max(0.0, min(100.0, start / duration * 100))
            width = min(max(0.4, (end - start) / duration * 100), 100.0 - left)
            label = f"{format_clock(start)}–{format_clock(end)}"
            blocks.append(
                f'<span class="block" style="left:{left:.3f}%;'
                f'width:{width:.3f}%;background:{color}" title="{label}"></span>'
            )
        name = html.escape(track.display_name)
        speaker_id = html.escape(track.speaker_id)
        rows.append(
            '<div class="row">'
            f'<div class="label" title="{speaker_id}">{name}</div>'
            f'<div class="track">{"".join(blocks)}</div>'
            "</div>"
        )

    ticks = "".join(
        f'<span class="tick" style="left:{percent}%">'
        f"{format_clock(duration * percent / 100)}</span>"
        for percent in (0, 25, 50, 75, 100)
    )
    ruler_row = (
        '<div class="row ruler-row"><div class="label"></div>'
        f'<div class="ruler">{ticks}</div></div>'
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="ru">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Таймлайн: {source_name}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Кто когда говорил</h1>\n"
        f'<p class="meta">{source_name} · всего {total_clock}</p>\n'
        '<div class="chart">\n'
        f"{ruler_row}\n"
        f'{"".join(rows)}\n'
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )


def write_timeline(result: TranscriptionResult, output_path: Path) -> bool:
    """Сохраняет HTML-таймлайн. Возвращает ``False``, если данных о говорящих нет."""

    if not build_speaker_tracks(result):
        return False

    try:
        output_path.write_text(render_timeline_html(result), encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"Не удалось сохранить таймлайн {output_path}: {exc}") from exc
    return True
