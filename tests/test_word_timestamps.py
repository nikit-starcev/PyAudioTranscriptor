"""Тесты пословных таймстемпов из токенов whisper.cpp (#45).

Проверяются группировка под-токенов в слова (пунктуация, служебные токены),
сшивка слов на стыке кусков с офсетом чанка и флаг включения стадии. Реальный
бинарник не запускается: ``subprocess.Popen`` подменяется заглушкой.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.cache.serialization import asr_from_payload, asr_payload
from audio_transcriber.domain.models import TranscriptionSegment, WordTimestamp
from audio_transcriber.transcription import whisper_cpp_engine as whisper_module
from audio_transcriber.transcription.whisper_cpp_engine import (
    WhisperCppRecognizer,
    _stitch_chunk_words,
    _word_timestamps,
)
from audio_transcriber.utils.audio import AudioProbe

SAMPLE_RATE = 16000


def _token(text: str, start_ms: int, end_ms: int, p: float = 0.9) -> dict:
    return {"text": text, "offsets": {"from": start_ms, "to": end_ms}, "p": p}


# --- Группировка токенов в слова --------------------------------------------


def test_word_timestamps_group_subtokens_and_punctuation() -> None:
    tokens = [
        {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}, "p": 0.95},
        _token(" 40", 120, 3160, 0.828),
        _token(" сек", 3160, 4740, 0.994),
        _token("унд", 5280, 6320, 0.99),
        _token(".", 6320, 6320, 0.53),
        {"text": "[_TT_316]", "offsets": {"from": 6320, "to": 6320}, "p": 0.248},
    ]

    words = _word_timestamps(tokens)

    assert [word.text for word in words] == ["40", "секунд."]
    assert words[0].start == pytest.approx(0.12)
    assert words[0].end == pytest.approx(3.16)
    assert words[0].probability == pytest.approx(0.828)
    # Конец слова — по последнему под-токену; пунктуация в вероятность не
    # входит (иначе низкая ``p`` точки занижала бы слово).
    assert words[1].start == pytest.approx(3.16)
    assert words[1].end == pytest.approx(6.32)
    assert words[1].probability == pytest.approx(0.99)
    assert all(word.continuation is False for word in words)


def test_word_timestamps_handles_missing_and_inverted_offsets() -> None:
    tokens = [
        {"text": " без", "p": 0.9},  # нет offsets — токен пропускается
        {"text": " Я", "offsets": {"from": 7480, "to": 6470}, "p": 0.98},
        {"text": " да", "offsets": {"from": 7530, "to": 7810}, "p": 1.0},
    ]

    words = _word_timestamps(tokens)

    assert [word.text for word in words] == ["Я", "да"]
    # Инвертированный ``offsets.to`` не даёт отрицательной длительности.
    assert words[0].start == pytest.approx(7.48)
    assert words[0].end == pytest.approx(7.48)
    assert words[1].start == pytest.approx(7.53)


def test_word_timestamps_applies_chunk_offset() -> None:
    tokens = [_token(" привет", 0, 500, 0.9)]

    words = _word_timestamps(tokens, chunk_offset=28.0)

    assert words[0].start == pytest.approx(28.0)
    assert words[0].end == pytest.approx(28.5)


def test_word_timestamps_marks_continuation_word() -> None:
    tokens = [
        _token(" сек", 0, 200, 0.9),
        _token("унд", 200, 500, 0.9),
    ]

    words = _word_timestamps(tokens)

    assert [word.text for word in words] == ["секунд"]
    assert words[0].continuation is False

    continuation = _word_timestamps([_token("унд", 0, 300, 0.9)])
    assert continuation[0].continuation is True


def test_word_timestamps_ignores_non_list_tokens() -> None:
    assert _word_timestamps(None) == []
    assert _word_timestamps("нет") == []


# --- Сшивка слов на стыке кусков --------------------------------------------


def test_stitch_chunk_words_glues_continuation() -> None:
    first = TranscriptionSegment(
        start=0.0,
        end=5.0,
        text="сек",
        words=[WordTimestamp("сек", 4.74, 5.0, 0.99)],
    )
    second = TranscriptionSegment(
        start=5.0,
        end=6.0,
        text="унд",
        words=[WordTimestamp("унд", 5.0, 6.0, 0.9, continuation=True)],
    )

    result = _stitch_chunk_words([first, second])

    assert [word.text for word in result[0].words] == ["секунд"]
    assert result[0].words[0].start == pytest.approx(4.74)
    assert result[0].words[0].end == pytest.approx(6.0)
    assert result[0].words[0].probability == pytest.approx(0.9)
    assert result[0].words[0].continuation is False
    assert result[1].words == []
    assert result[1].text == "унд"


def test_stitch_chunk_words_clears_flag_without_previous() -> None:
    only = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="ло",
        words=[WordTimestamp("ло", 0.0, 1.0, 0.5, continuation=True)],
    )

    result = _stitch_chunk_words([only])

    assert result[0].words[0].continuation is False
    assert result[0].words[0].text == "ло"


def test_stitch_chunk_words_keeps_unique_segments_untouched() -> None:
    segments = [
        TranscriptionSegment(0.0, 1.0, "привет", words=[WordTimestamp("привет", 0.0, 1.0)]),
        TranscriptionSegment(1.0, 2.0, "мир", words=[WordTimestamp("мир", 1.0, 2.0)]),
    ]

    result = _stitch_chunk_words(segments)

    assert [word.text for word in result[0].words] == ["привет"]
    assert [word.text for word in result[1].words] == ["мир"]


# --- Кэш ---------------------------------------------------------------------


def test_asr_payload_round_trips_words() -> None:
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=1.0,
            text="привет",
            words=[WordTimestamp("привет", 0.0, 1.0, 0.9)],
        )
    ]

    restored, language, duration = asr_from_payload(asr_payload(segments, "ru", 1.0))

    assert language == "ru"
    assert duration == pytest.approx(1.0)
    assert restored[0].words == [WordTimestamp("привет", 0.0, 1.0, 0.9)]


def test_asr_from_payload_without_words_is_backwards_compatible() -> None:
    payload = {
        "language": "ru",
        "duration": 1.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "привет", "avg_logprob": -0.1}],
    }

    segments, _language, _duration = asr_from_payload(payload)

    assert segments[0].words == []


# --- Прокидка слов в результат ----------------------------------------------


def test_merger_propagates_words_to_entries() -> None:
    from audio_transcriber.merging.aligner import OverlapSegmentMerger

    segment = TranscriptionSegment(
        start=0.0,
        end=1.0,
        text="привет",
        words=[WordTimestamp("привет", 0.0, 1.0, 0.9)],
    )

    entries, _speakers = OverlapSegmentMerger().merge([segment], [])

    assert entries[0].words == [WordTimestamp("привет", 0.0, 1.0, 0.9)]


def test_result_from_payload_preserves_words() -> None:
    from audio_transcriber.web.speakers import result_from_payload

    payload = {
        "language": "ru",
        "duration": 1.0,
        "speakers": [],
        "entries": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "привет",
                "words": [
                    {"text": "привет", "start": 0.0, "end": 1.0, "probability": 0.9}
                ],
            }
        ],
    }

    result = result_from_payload(payload, source_path=Path("x"))

    assert result.entries[0].words == [WordTimestamp("привет", 0.0, 1.0, 0.9)]


def test_hybrid_shift_clamped_words() -> None:
    from audio_transcriber.transcription.hybrid import _shift_clamped_words

    words = [
        WordTimestamp("до", 0.0, 0.05),
        WordTimestamp("в", 0.3, 0.6),
        WordTimestamp("после", 0.9, 1.2),
    ]

    result = _shift_clamped_words(words, offset=10.0, low=10.1, high=11.0)

    # Слово «до» целиком вне окна — отброшено; остальные сдвинуты и обрезаны.
    assert [word.text for word in result] == ["в", "после"]
    assert result[0].start == pytest.approx(10.3)
    assert result[1].end == pytest.approx(11.0)


def test_sentence_merger_concatenates_words() -> None:
    from audio_transcriber.domain.models import TranscriptEntry
    from audio_transcriber.merging.sentence_merger import SentenceMerger

    first = TranscriptEntry(
        start=0.0, end=1.0, text="привет", words=[WordTimestamp("привет", 0.0, 0.5)]
    )
    second = TranscriptEntry(
        start=1.0, end=2.0, text="мир", words=[WordTimestamp("мир", 1.0, 1.5)]
    )

    merged = SentenceMerger().merge([first, second])

    assert len(merged) == 1
    assert [word.text for word in merged[0].words] == ["привет", "мир"]


def test_web_result_serializes_words_when_present() -> None:
    from audio_transcriber.domain.models import TranscriptEntry
    from audio_transcriber.web.results import entry_to_dict

    entry = TranscriptEntry(
        start=0.0,
        end=1.0,
        text="привет",
        words=[WordTimestamp("привет", 0.0, 1.0, 0.9)],
    )
    assert entry_to_dict(entry, None)["words"] == [
        {"text": "привет", "start": 0.0, "end": 1.0, "probability": 0.9}
    ]

    without = TranscriptEntry(start=0.0, end=1.0, text="привет")
    # При выключенной стадии форма реплики прежняя — поля words нет.
    assert "words" not in entry_to_dict(without, None)


# --- Движок: флаг и офсет чанка ---------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.stderr = iter(["progress = 100%\n"])

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _segment_payload(text: str, start_ms: int, end_ms: int, token: dict) -> dict:
    return {
        "offsets": {"from": start_ms, "to": end_ms},
        "text": f" {text}",
        "tokens": [token],
    }


def _install_popen(
    monkeypatch: pytest.MonkeyPatch,
    payload_for: Callable[[int], list[dict]],
) -> None:
    def _fake_popen(cmd: list[str], *_args: object, **_kwargs: object) -> _FakeProc:
        base = Path(cmd[cmd.index("-of") + 1])
        suffix = base.name.rsplit("_", 1)[-1]
        index = int(suffix) if suffix.isdigit() else 0
        payload = {"result": {"language": "ru"}, "transcription": payload_for(index)}
        Path(str(base) + ".json").write_text(json.dumps(payload), encoding="utf-8")
        return _FakeProc()

    monkeypatch.setattr(whisper_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        whisper_module, "load_waveform", lambda _p: np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    )
    monkeypatch.setattr(whisper_module, "write_wav", lambda _path, _waveform: None)


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake")
    return model


def test_engine_populates_words_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        whisper_module, "probe_audio", lambda _p: AudioProbe("wav", "pcm_s16le", SAMPLE_RATE, 1, 2.0)
    )
    _install_popen(
        monkeypatch,
        lambda _i: [_segment_payload("привет мир", 0, 1000, _token(" привет", 0, 500, 0.9))],
    )

    segments, _language, _duration = WhisperCppRecognizer(
        _model(tmp_path), word_timestamps=True
    ).transcribe(Path("input.wav"))

    assert [word.text for word in segments[0].words] == ["привет"]


def test_engine_skips_words_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        whisper_module, "probe_audio", lambda _p: AudioProbe("wav", "pcm_s16le", SAMPLE_RATE, 1, 2.0)
    )
    _install_popen(
        monkeypatch,
        lambda _i: [_segment_payload("привет", 0, 1000, _token(" привет", 0, 500, 0.9))],
    )

    recognizer = WhisperCppRecognizer(_model(tmp_path), word_timestamps=False)
    segments, _language, _duration = recognizer.transcribe(Path("input.wav"))

    assert recognizer.word_timestamps is False
    assert segments[0].words == []


def test_chunked_engine_shifts_word_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Слова из чанков получают офсет куска, как и границы сегментов."""

    monkeypatch.setattr(
        whisper_module,
        "probe_audio",
        lambda _p: AudioProbe("wav", "pcm_s16le", SAMPLE_RATE, 1, 65.0),
    )

    def payload_for(index: int) -> list[dict]:
        # Куски: [0,30], [28,58], [56,65]; слово — в начале куска.
        return [_segment_payload(f"слово{index}", 0, 400, _token(" слово", 0, 400, 0.9))]

    # ``_install_popen`` переопределяет load_waveform/write_wav — применяем
    # «длинное» аудио после него, чтобы чанкинг реально сработал.
    _install_popen(monkeypatch, payload_for)
    monkeypatch.setattr(
        whisper_module,
        "load_waveform",
        lambda _p: np.zeros(65 * SAMPLE_RATE, dtype=np.float32),
    )
    written: list[Path] = []
    monkeypatch.setattr(
        whisper_module, "write_wav", lambda path, _waveform: written.append(Path(path))
    )

    recognizer = WhisperCppRecognizer(
        _model(tmp_path), chunk_seconds=30.0, chunk_overlap=2.0, word_timestamps=True
    )
    segments, _language, _duration = recognizer.transcribe(Path("input.wav"))

    starts = [round(segment.start, 3) for segment in segments]
    assert starts == [0.0, 28.0, 56.0]
    assert [round(segment.words[0].start, 3) for segment in segments] == starts
    assert [round(segment.words[0].end, 3) for segment in segments] == [0.4, 28.4, 56.4]
