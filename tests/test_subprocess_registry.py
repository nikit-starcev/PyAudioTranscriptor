"""Тесты фонового чтения stderr подпроцессов (#127).

llama.cpp с ``-v`` печатает сырые байты токенов, среди которых встречаются
незавершённые UTF-8-последовательности. Чтение stderr не должно из-за этого
падать: иначе канал перестаёт опустошаться, pipe переполняется, и подпроцесс
намертво встаёт на записи в stderr.
"""

from __future__ import annotations

import io

from audio_transcriber.utils.subprocess_registry import StderrReader


def _drain(stream: io.BytesIO | io.StringIO) -> str:
    reader = StderrReader(stream)
    reader.start()
    reader._thread.join(2.0)
    assert not reader._thread.is_alive()
    return reader.full_text()


def test_stderr_reader_survives_invalid_utf8() -> None:
    stream = io.BytesIO(
        "старт\n".encode()
        + b"next token: 3648 '\xd0\xb0\xd0'\n"
        + "хвост\n".encode()
    )

    text = _drain(stream)

    assert "next token" in text
    assert "хвост" in text


def test_stderr_reader_accepts_text_stream() -> None:
    assert _drain(io.StringIO("строка 1\nстрока 2\n")) == "строка 1\nстрока 2"
