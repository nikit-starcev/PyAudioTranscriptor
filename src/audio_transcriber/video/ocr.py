"""Выбор OCR-движка для распознавания текста слайдов (RU/EN, CPU).

Движок выбирается по доступности:

* **RapidOCR** (ONNX Runtime) — если установлен (extra ``slides``), основной:
  лёгкий, без torch, поддерживает кириллицу;
* **Tesseract** (внешний CLI ``tesseract`` с языками ``rus``+``eng``) — запасной;
* если недоступен ни один — :class:`OcrUnavailableError` с понятным текстом
  (вызывающий код отвечает 503, а не падает).

Все ошибки распознавания мягко деградируют: движок возвращает пустую строку, а
не роняет извлечение слайдов.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from audio_transcriber.utils.env import binary_available

logger = logging.getLogger(__name__)

#: Имя/путь внешнего CLI ``tesseract`` (ищется в ``PATH``, если не задан путь).
DEFAULT_TESSERACT_BINARY = "tesseract"

#: Языки OCR по умолчанию (кириллица + латиница).
DEFAULT_OCR_LANGUAGES: tuple[str, ...] = ("rus", "eng")

#: Таймаут одного запуска Tesseract (секунды).
_TESSERACT_TIMEOUT_SECONDS = 60.0

#: Кандидаты модулей/фабрик RapidOCR (новый и legacy-пакет).
_RAPIDOCR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("rapidocr", "RapidOCR"),
    ("rapidocr_onnxruntime", "RapidOCR"),
)


class OcrUnavailableError(RuntimeError):
    """Ни один OCR-движок не доступен (нет RapidOCR и Tesseract)."""


@runtime_checkable
class OcrEngine(Protocol):
    """Интерфейс OCR-движка: имя и распознавание текста с картинки-файла."""

    @property
    def name(self) -> str:
        """Короткое имя движка для API/метаданных (``rapidocr``/``tesseract``)."""
        ...

    def recognize(self, image_path: Path) -> str:
        """Распознаёт текст на изображении (пустая строка при неудаче)."""
        ...


def clean_ocr_text(text: str) -> str:
    """Чистит вывод OCR: убирает form feed, выравнивает и схлопывает пустые строки."""
    normalized = text.replace("\x0c", "\n")
    lines = [line.strip() for line in normalized.splitlines()]
    return "\n".join(line for line in lines if line).strip()


class TesseractEngine:
    """OCR через внешний CLI ``tesseract`` (языки ``rus``+``eng`` по умолчанию)."""

    def __init__(
        self,
        *,
        binary: str = DEFAULT_TESSERACT_BINARY,
        languages: Sequence[str] = DEFAULT_OCR_LANGUAGES,
    ) -> None:
        self.binary = binary
        self.languages: tuple[str, ...] = tuple(languages) or DEFAULT_OCR_LANGUAGES

    @property
    def name(self) -> str:
        return "tesseract"

    def recognize(self, image_path: Path) -> str:
        """Распознаёт текст; при отсутствии языковых данных пробует только ``eng``."""
        attempts: list[tuple[str, ...]] = [self.languages]
        if self.languages != ("eng",):
            attempts.append(("eng",))
        for languages in attempts:
            text = self._run(image_path, languages)
            if text is not None:
                return text
        return ""

    def _run(self, image_path: Path, languages: Sequence[str]) -> str | None:
        command = [
            self.binary,
            str(image_path),
            "stdout",
            "-l",
            "+".join(languages),
            "--psm",
            "6",
        ]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_TESSERACT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Tesseract не запустился: %s", exc)
            return None
        if proc.returncode != 0:
            logger.debug(
                "Tesseract вернул код %s для языков %s: %s",
                proc.returncode,
                languages,
                (proc.stderr or "").strip()[:200],
            )
            return None
        return clean_ocr_text(proc.stdout)


class RapidOcrEngine:
    """OCR через RapidOCR (ONNX Runtime): ленивая обёртка над готовым движком."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @property
    def name(self) -> str:
        return "rapidocr"

    def recognize(self, image_path: Path) -> str:
        """Распознаёт текст; любая ошибка движка — пустая строка."""
        try:
            result = self._engine(str(image_path))
        except Exception as exc:  # noqa: BLE001 — сбой OCR не должен ронять извлечение
            logger.warning("RapidOCR не распознал кадр %s: %s", image_path.name, exc)
            return ""
        return clean_ocr_text("\n".join(_rapidocr_texts(result)))


def _rapidocr_texts(result: Any) -> list[str]:
    """Извлекает строки текста из результата RapidOCR (разные версии API)."""
    texts = getattr(result, "txts", None)
    if texts is not None:
        return [str(item) for item in texts]
    if isinstance(result, tuple) and result:
        result = result[0]
    collected: list[str] = []
    if isinstance(result, (list, tuple)):
        for item in result:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], str)
            ):
                collected.append(item[1])
    return collected


def _import_rapidocr() -> Any | None:
    """Импортирует класс RapidOCR (новый или legacy-пакет) либо ``None``."""
    for module_name, attribute in _RAPIDOCR_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 — модуль не установлен/не импортируется
            continue
        candidate = getattr(module, attribute, None)
        if candidate is not None:
            return candidate
    return None


def _create_rapidocr() -> RapidOcrEngine | None:
    """Пытается создать RapidOCR-движок; ``None``, если он недоступен."""
    engine_class = _import_rapidocr()
    if engine_class is None:
        return None
    try:
        return RapidOcrEngine(engine_class())
    except Exception as exc:  # noqa: BLE001 — нет моделей/сломанная установка
        logger.debug("RapidOCR не инициализировался: %s", exc)
        return None


def find_ocr_engine(
    *,
    prefer: str | None = None,
    tesseract_binary: str = DEFAULT_TESSERACT_BINARY,
    languages: Sequence[str] = DEFAULT_OCR_LANGUAGES,
) -> OcrEngine:
    """Выбирает доступный OCR-движок или бросает :class:`OcrUnavailableError`.

    ``prefer`` — принудительный выбор движка (``rapidocr``/``tesseract``); без
    него сначала пробуется RapidOCR, затем Tesseract.
    """
    requested = prefer.strip().casefold() if isinstance(prefer, str) else None
    if requested not in (None, "rapidocr", "tesseract"):
        raise OcrUnavailableError(f"Неизвестный OCR-движок: {prefer!r}")

    if requested in (None, "rapidocr"):
        rapid = _create_rapidocr()
        if rapid is not None:
            return rapid
    if requested in (None, "tesseract") and binary_available(tesseract_binary):
        return TesseractEngine(binary=tesseract_binary, languages=languages)

    if requested == "rapidocr":
        raise OcrUnavailableError(
            "RapidOCR недоступен: установите extra 'slides' "
            "(uv pip install --python .venv/bin/python '.[slides]')"
        )
    if requested == "tesseract":
        raise OcrUnavailableError(
            "Tesseract недоступен: установите tesseract с языками rus и eng"
        )
    raise OcrUnavailableError(
        "Не найден OCR-движок: установите RapidOCR (extra 'slides') "
        "или tesseract с языками rus и eng"
    )
