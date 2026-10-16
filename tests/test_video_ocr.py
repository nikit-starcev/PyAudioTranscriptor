"""Юнит-тесты выбора OCR-движка (``video.ocr``) с моками, без реального OCR."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from audio_transcriber.video import ocr as ocr_module
from audio_transcriber.video.ocr import (
    OcrUnavailableError,
    RapidOcrEngine,
    TesseractEngine,
    clean_ocr_text,
    find_ocr_engine,
)


def test_clean_ocr_text_removes_formfeed_and_blanks() -> None:
    assert clean_ocr_text("строка1\x0c\n\n  строка2  \n\n") == "строка1\nстрока2"


class _FakeRapid:
    name = "rapidocr"

    def __call__(self, _path: str) -> object:
        return type("Out", (), {"txts": ("hello", "мир")})()


def test_rapidocr_texts_from_object() -> None:
    assert ocr_module._rapidocr_texts(_FakeRapid()("x")) == ["hello", "мир"]


def test_rapidocr_texts_from_legacy_tuple() -> None:
    result = ([[[0, 0], [1, 1]], "hello", 0.9], [[[0, 0], [1, 1]], "world", 0.8])
    assert ocr_module._rapidocr_texts((result, 0.1)) == ["hello", "world"]


def test_rapidocr_engine_swallows_errors() -> None:
    def boom(_path: str) -> object:
        raise RuntimeError("сбой модели")

    assert RapidOcrEngine(boom).recognize(Path("x.png")) == ""


def test_find_ocr_engine_prefers_rapidocr(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = RapidOcrEngine(_FakeRapid())
    monkeypatch.setattr(ocr_module, "_create_rapidocr", lambda: sentinel)
    monkeypatch.setattr(ocr_module, "binary_available", lambda _binary: True)

    assert find_ocr_engine() is sentinel


def test_find_ocr_engine_falls_back_to_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_module, "_create_rapidocr", lambda: None)
    monkeypatch.setattr(ocr_module, "binary_available", lambda binary: binary == "tesseract")

    engine = find_ocr_engine()

    assert isinstance(engine, TesseractEngine)
    assert engine.name == "tesseract"


def test_find_ocr_engine_raises_without_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_module, "_create_rapidocr", lambda: None)
    monkeypatch.setattr(ocr_module, "binary_available", lambda _binary: False)

    with pytest.raises(OcrUnavailableError, match="Не найден OCR-движок"):
        find_ocr_engine()


def test_find_ocr_engine_unknown_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(OcrUnavailableError, match="Неизвестный OCR-движок"):
        find_ocr_engine(prefer="magic")


def test_find_ocr_engine_explicit_tesseract_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_module, "binary_available", lambda _binary: False)

    with pytest.raises(OcrUnavailableError, match="Tesseract недоступен"):
        find_ocr_engine(prefer="tesseract")


def _completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["tesseract"], returncode=code, stdout=stdout, stderr="")


def test_tesseract_recognize_parses_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        return _completed("Привет\x0c\nworld")

    monkeypatch.setattr(ocr_module.subprocess, "run", fake_run)

    text = TesseractEngine().recognize(tmp_path / "frame.png")

    assert text == "Привет\nworld"
    assert captured[0][0] == "tesseract"
    assert "-l" in captured[0]
    assert "rus+eng" in captured[0]


def test_tesseract_falls_back_to_eng(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        language = command[command.index("-l") + 1]
        if language == "rus+eng":
            return _completed("", code=1)
        return _completed("english text")

    monkeypatch.setattr(ocr_module.subprocess, "run", fake_run)

    assert TesseractEngine().recognize(tmp_path / "frame.png") == "english text"


def test_tesseract_returns_empty_when_binary_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(_command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed("", code=1)

    monkeypatch.setattr(ocr_module.subprocess, "run", fake_run)

    assert TesseractEngine().recognize(tmp_path / "frame.png") == ""
