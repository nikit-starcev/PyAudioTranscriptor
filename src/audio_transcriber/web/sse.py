"""Общие помощники Server-Sent Events (SSE) для веб-роутов.

Кадр события и заголовки ответа раньше дублировались в
:mod:`audio_transcriber.web.app` и :mod:`audio_transcriber.web.chat_api` —
теперь это единый источник, как и ретрансляция шины прогресса.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from fastapi.responses import StreamingResponse

#: Заголовки SSE: без кэша и без буферизации прокси.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: Событие шины прогресса (``dict``) либо ``None`` — сигнал keep-alive.
BusEvent = dict[str, object]


class EventBus(Protocol):
    """Минимальный контракт шины прогресса для SSE-ретрансляции."""

    def history(self, after: int | None = None) -> list[BusEvent]: ...

    def subscribe(self) -> AsyncIterator[BusEvent | None]: ...


def sse_frame(event: Mapping[str, object]) -> str:
    """Кадр SSE с JSON-пейлоадом события.

    ``seq`` (если есть) идёт как SSE-поле ``id``: браузер сам пришлёт его в
    ``Last-Event-ID`` при переподключении. В самом JSON поле тоже остаётся.
    """
    seq = event.get("seq")
    prefix = f"id: {seq}\n" if isinstance(seq, int) else ""
    return f"{prefix}data: {json.dumps(event, ensure_ascii=False)}\n\n"


def parse_last_event_id(header: str | None) -> int | None:
    """Разбирает заголовок ``Last-Event-ID`` в ``int`` (``None`` при мусоре)."""
    if not header:
        return None
    try:
        return int(header)
    except ValueError:
        return None


async def stream_bus(bus: EventBus, after: int | None) -> AsyncIterator[str]:
    """Отдаёт историю шины (``after``), затем живой поток как SSE-кадры.

    ``None`` от ``subscribe`` — сигнал keep-alive (комментарий-пинг).
    """
    for event in bus.history(after=after):
        yield sse_frame(event)
    async for update in bus.subscribe():
        yield ": ping\n\n" if update is None else sse_frame(update)


def sse_response(source: AsyncIterator[str]) -> StreamingResponse:
    """Обёртка ``StreamingResponse`` с едиными SSE-заголовками."""
    return StreamingResponse(
        source, media_type="text/event-stream", headers=dict(SSE_HEADERS)
    )
