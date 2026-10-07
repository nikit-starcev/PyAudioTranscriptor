"""Регресс-тесты корректности таймкодов (#14).

Проверяются инварианты, из-за нарушения которых времена реплик перестают
соответствовать реальному положению в аудио:

* декодирование аудио в шкалу конвейера (44.1/48/22.05 … кГц → 16 кГц) сохраняет
  длительность исходника и выравнивание по времени (ресемплер PyAV обязан
  сбрасывать внутренний буфер, иначе теряется «хвост» и всё аудио сдвигается на
  задержку фильтра);
* склейка соседних реплик одного говорящего не укорачивает интервал: границы
  объединяются, а не «побеждает последняя реплика» — на стыке ASR-кусков
  сегменты могут перекрываться.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.domain.models import Speaker, TranscriptEntry
from audio_transcriber.merging.sentence_merger import SentenceMerger
from audio_transcriber.utils.audio import SAMPLE_RATE, load_waveform

#: Допуск выравнивания импульса — пара сэмплов 16 кГц (задержка/звон фильтра).
_ALIGN_TOLERANCE_SAMPLES = 2


def _write_wav(path: Path, waveform: np.ndarray, *, sample_rate: int) -> None:
    pcm = np.clip(np.rint(waveform * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _synthetic_seconds(seconds: float, *, sample_rate: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(round(seconds * sample_rate)) * 0.1).astype(np.float32)


@pytest.mark.parametrize("source_rate", [8000, 16000, 22050, 32000, 44100, 48000, 96000])
def test_load_waveform_preserves_source_duration(
    tmp_path: Path, source_rate: int
) -> None:
    """Сигнал длительностью T ⇒ декодированная дорожка ровно T секунд (#14).

    Иначе таймкоды считаются в сжатой шкале и «уезжают» относительно аудио.
    """
    seconds = 3.0
    source = tmp_path / f"tone-{source_rate}.wav"
    _write_wav(source, _synthetic_seconds(seconds, sample_rate=source_rate), sample_rate=source_rate)

    decoded = load_waveform(source)

    assert decoded.dtype == np.float32
    # Число сэмплов на 16 кГц, соответствующее ровно `seconds` секундам.
    assert decoded.shape[0] == round(seconds * SAMPLE_RATE)


def test_load_waveform_preserves_impulse_alignment(tmp_path: Path) -> None:
    """Ресемплинг 44.1 → 16 кГц не сдвигает содержимое по времени (без flush — +1 мс)."""
    source_rate = 44100
    seconds = 5.0
    waveform = np.zeros(round(seconds * source_rate), dtype=np.float32)
    click_times = [0.5, 2.0, 4.5]
    impulse = 40
    for seconds_at in click_times:
        start = int(seconds_at * source_rate)
        waveform[start : start + impulse] = 1.0
    source = tmp_path / "clicks.wav"
    _write_wav(source, waveform, sample_rate=source_rate)

    decoded = load_waveform(source)
    envelope = np.abs(decoded)

    for seconds_at in click_times:
        expected = round(seconds_at * SAMPLE_RATE)
        low = max(0, expected - 400)
        high = min(envelope.shape[0], expected + 400)
        found = low + int(np.argmax(envelope[low:high]))
        assert abs(found - expected) <= _ALIGN_TOLERANCE_SAMPLES, (
            f"импульс {seconds_at} с оказался на {found / SAMPLE_RATE:.5f} с"
        )


def test_load_waveform_keeps_tail_impulse(tmp_path: Path) -> None:
    """«Хвост» потока не теряется: импульс у самого конца декодируется."""
    source_rate = 44100
    seconds = 2.0
    waveform = np.zeros(round(seconds * source_rate), dtype=np.float32)
    # Импульс за 1 мс до конца — ровно та зона, что пропадала без flush.
    tail = waveform.shape[0] - int(0.001 * source_rate) - 20
    waveform[tail : tail + 20] = 1.0
    source = tmp_path / "tail.wav"
    _write_wav(source, waveform, sample_rate=source_rate)

    decoded = load_waveform(source)

    # Последние ~2 мс должны содержать ненулевые сэмплы (хвост не срезан).
    tail_samples = decoded[-int(0.002 * SAMPLE_RATE) :]
    assert float(np.max(np.abs(tail_samples))) > 0.0


def _entry(start: float, end: float, text: str, speaker: Speaker | None = None) -> TranscriptEntry:
    return TranscriptEntry(start=start, end=end, text=text, speaker=speaker)


_SPEAKER = Speaker(id="SPEAKER_00", display_name="Спикер 1")


def test_sentence_merger_unions_overlapping_intervals() -> None:
    """Перекрывающиеся реплики одного говорящего дают объединённый интервал.

    Наивная склейка с ``end=entry.end`` укоротила бы реплику [0,10] до [0,7],
    и её границы перестали бы соответствовать сказанному (#14).
    """
    merged = SentenceMerger().merge(
        [
            _entry(0.0, 10.0, "раз два", _SPEAKER),
            _entry(5.0, 7.0, "три", _SPEAKER),
        ]
    )

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 10.0
    assert merged[0].text == "раз два три"


def test_sentence_merger_extends_to_last_end() -> None:
    """Если следующий сегмент заходит дальше — интервал расширяется, не сжимается."""
    merged = SentenceMerger().merge(
        [
            _entry(0.0, 5.0, "первая", _SPEAKER),
            _entry(4.5, 9.0, "вторая", _SPEAKER),
        ]
    )

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 9.0


def test_sentence_merger_interval_never_shrinks_across_chain() -> None:
    """Цепочка склеек не «съезжает»: интервал не укорачивается ни на шаге."""
    entries = [
        _entry(0.0, 10.0, "a", _SPEAKER),
        _entry(5.0, 6.0, "b", _SPEAKER),
        _entry(9.0, 12.0, "c", _SPEAKER),
        _entry(11.0, 11.5, "d", _SPEAKER),
    ]

    merged = SentenceMerger().merge(entries)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 12.0
