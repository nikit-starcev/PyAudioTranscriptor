"""Тесты движка диаризации NeMo-Speech.cpp (без реального ``nemo-speech``).

Проверяются: парсинг RTTM (включая граничные случаи), нормализация меток
говорящих, защита от затенения libstdc++/libgcc_s, мягкая деградация при
отсутствии бинарника и сбое subprocess, лимит Sortformer в 4 спикера и проброс
настроек окружения.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio_transcriber.diarization import nemo_speech_engine as nemo
from audio_transcriber.utils.env import effective_library_path

_RTTM_SINGLE = (
    "SPEAKER rec 1 0.091 2.388 <NA> <NA> speaker_1 <NA> <NA>\n"
    "SPEAKER rec 1 3.131 1.588 <NA> <NA> speaker_1 <NA> <NA>\n"
    "SPEAKER rec 1 5.211 5.829 <NA> <NA> speaker_1 <NA> <NA>\n"
)


def _completed(stdout: str = "", *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["nemo-speech"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _patch_binary(monkeypatch: pytest.MonkeyPatch, *, available: bool = True) -> None:
    monkeypatch.setattr(nemo, "binary_available", lambda _binary: available)


def _patch_run(monkeypatch: pytest.MonkeyPatch, result) -> list[dict[str, object]]:
    """Подменяет ``subprocess.run``, запоминая kwargs каждого вызова."""
    calls: list[dict[str, object]] = []

    def _fake(command: object, **kwargs: object) -> object:
        calls.append({"command": command, **kwargs})
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(nemo.subprocess, "run", _fake)
    return calls


# --- парсинг RTTM ------------------------------------------------------------


def test_parse_rttm_basic_segments() -> None:
    segments = nemo.parse_rttm(_RTTM_SINGLE)

    assert [(s.start, s.end, s.speaker_id) for s in segments] == [
        (0.091, pytest.approx(2.479), "SPEAKER_00"),
        (3.131, pytest.approx(4.719), "SPEAKER_00"),
        (5.211, pytest.approx(11.040), "SPEAKER_00"),
    ]


def test_parse_rttm_normalizes_speaker_indexing() -> None:
    text = (
        "SPEAKER rec 1 0.0 1.0 <NA> <NA> speaker_1 <NA> <NA>\n"
        "SPEAKER rec 1 1.0 1.0 <NA> <NA> speaker_2 <NA> <NA>\n"
        "SPEAKER rec 1 2.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    )

    assert [s.speaker_id for s in nemo.parse_rttm(text)] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]


def test_parse_rttm_skips_malformed_and_foreign_lines() -> None:
    text = "\n".join(
        [
            "# комментарий",
            "not rttm at all",
            "SPEAKER rec 1 0.0 <NA> <NA> <NA> speaker_1 <NA>",  # мало полей
            "SPEAKER rec 1 abc 1.0 <NA> <NA> speaker_1 <NA> <NA>",  # битый start
            "SPEAKER rec 1 1.0 1.0 <NA> <NA> <NA> <NA> <NA>",  # нет имени
            "SPEAKER rec 1 2.0 0.0 <NA> <NA> speaker_1 <NA> <NA>",  # нулевая длит.
            "SPEAKER rec 1 3.0 -1.0 <NA> <NA> speaker_1 <NA> <NA>",  # отриц. длит.
            "SPEAKER rec 1 4.0 2.0 <NA> <NA> speaker_1 <NA> <NA>",  # валидная
        ]
    )

    segments = nemo.parse_rttm(text)

    assert len(segments) == 1
    assert (segments[0].start, segments[0].end, segments[0].speaker_id) == (
        4.0,
        6.0,
        "SPEAKER_00",
    )


def test_parse_rttm_filters_by_recording_id() -> None:
    text = (
        "SPEAKER a 1 0.0 1.0 <NA> <NA> speaker_1 <NA> <NA>\n"
        "SPEAKER b 1 1.0 1.0 <NA> <NA> speaker_1 <NA> <NA>\n"
    )

    assert len(nemo.parse_rttm(text, recording_id="a")) == 1
    assert len(nemo.parse_rttm(text, recording_id="b")) == 1
    assert nemo.parse_rttm(text, recording_id="c") == []


def test_normalize_speaker_label_keeps_unknown() -> None:
    assert nemo.normalize_speaker_label("speaker_4") == "SPEAKER_03"
    assert nemo.normalize_speaker_label("SPEAKER_02") == "SPEAKER_02"
    assert nemo.normalize_speaker_label("custom") == "custom"


# --- каталог библиотек -------------------------------------------------------


def test_effective_library_path_returns_safe_directory(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libggml-vulkan.so").write_text("", encoding="utf-8")

    assert effective_library_path(str(lib), forbidden_libs=nemo.NEMO_SPEECH_FORBIDDEN_LIBS) == str(
        lib
    )


def test_effective_library_path_rejects_shadowing_libstdcxx(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libstdc++.so.6").write_text("", encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = effective_library_path(str(lib), forbidden_libs=nemo.NEMO_SPEECH_FORBIDDEN_LIBS)

    assert result is None
    assert any("libstdc++.so.6" in record.message for record in caplog.records)


def test_effective_library_path_missing_directory_is_none(tmp_path: Path) -> None:
    assert effective_library_path(str(tmp_path / "nope")) is None


# --- движок ------------------------------------------------------------------


def test_diarize_parses_output_and_computes_overlaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_binary(monkeypatch)
    captured = _patch_run(monkeypatch, _completed(_RTTM_SINGLE + "SPEAKER rec 1 6.0 1.0 <NA> <NA> speaker_2 <NA> <NA>\n"))
    diarizer = nemo.NemoSpeechSpeakerDiarizer("vulkan")

    segments = diarizer.diarize(
        Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32)
    )

    assert {s.speaker_id for s in segments} == {"SPEAKER_00", "SPEAKER_01"}
    # speaker_2 активен 6.0–7.0 внутри сегмента speaker_1 5.211–11.040.
    assert diarizer.overlap_regions()
    assert diarizer.overlap_regions()[0].speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    # Временный WAV удалён после запуска.
    temp_arg = Path(captured[0]["command"][2])
    assert not temp_arg.exists()


def test_diarize_passes_expected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_binary(monkeypatch)
    captured = _patch_run(monkeypatch, _completed(""))
    diarizer = nemo.NemoSpeechSpeakerDiarizer(
        "vulkan", binary="/opt/nemo/bin/nemo-speech", model="sortformer"
    )

    diarizer.diarize(Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32))

    command = captured[0]["command"]
    assert command[0] == "/opt/nemo/bin/nemo-speech"
    assert command[1] == "diarize"
    assert "--device" in command and command[command.index("--device") + 1] == "vulkan"
    assert "--model" in command and command[command.index("--model") + 1] == "sortformer"
    assert command[command.index("--format") + 1] == "rttm"


def test_diarize_reuses_waveform_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_binary(monkeypatch)
    _patch_run(monkeypatch, _completed(""))

    def _unexpected_decode(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("load_waveform не должен вызываться при переданном waveform")

    monkeypatch.setattr(nemo, "load_waveform", _unexpected_decode)
    diarizer = nemo.NemoSpeechSpeakerDiarizer()

    diarizer.diarize(Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32))


def test_diarize_missing_binary_degrades_softly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_binary(monkeypatch, available=False)
    diarizer = nemo.NemoSpeechSpeakerDiarizer()

    with caplog.at_level("WARNING"):
        segments = diarizer.diarize(
            Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32)
        )

    assert segments == []
    assert diarizer.overlap_regions() == []
    assert any("не найден" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "result",
    [
        OSError("cannot execute"),
        subprocess.TimeoutExpired(cmd="nemo-speech", timeout=1),
        _completed(returncode=2, stderr="boom"),
    ],
)
def test_diarize_subprocess_failure_degrades_softly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    result,
) -> None:
    _patch_binary(monkeypatch)
    _patch_run(monkeypatch, result)
    diarizer = nemo.NemoSpeechSpeakerDiarizer()

    with caplog.at_level("WARNING"):
        segments = diarizer.diarize(
            Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32)
        )

    assert segments == []
    assert caplog.records


def test_diarize_warns_over_speaker_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_binary(monkeypatch)
    _patch_run(monkeypatch, _completed(_RTTM_SINGLE))
    diarizer = nemo.NemoSpeechSpeakerDiarizer()

    with caplog.at_level("WARNING"):
        segments = diarizer.diarize(
            Path("audio.wav"),
            num_speakers=6,
            waveform=np.zeros(16000, dtype=np.float32),
        )

    assert segments  # не падает, возвращает то, что дал движок
    assert any("не более 4" in record.message for record in caplog.records)


def test_diarize_sets_ld_library_path_only_when_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_binary(monkeypatch)
    safe_lib = tmp_path / "safe"
    safe_lib.mkdir()
    captured = _patch_run(monkeypatch, _completed(""))

    nemo.NemoSpeechSpeakerDiarizer(lib_path=str(safe_lib)).diarize(
        Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32)
    )

    env = captured[0]["env"]
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(safe_lib)


def test_diarize_omits_shadowing_library_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_binary(monkeypatch)
    bad_lib = tmp_path / "bad"
    bad_lib.mkdir()
    (bad_lib / "libgcc_s.so.1").write_text("", encoding="utf-8")
    captured = _patch_run(monkeypatch, _completed(""))

    nemo.NemoSpeechSpeakerDiarizer(lib_path=str(bad_lib)).diarize(
        Path("audio.wav"), waveform=np.zeros(16000, dtype=np.float32)
    )

    env = captured[0]["env"]
    assert str(bad_lib) not in env.get("LD_LIBRARY_PATH", "")
