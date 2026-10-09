"""Точка входа frozen-бандла: запускает Typer-приложение CLI.

PyInstaller выполняет этот файл как ``__main__``. Отдельный файл нужен,
чтобы до старта приложения вызвать ``multiprocessing.freeze_support()`` —
это защищает от повторного запуска CLI в дочерних процессах на Windows.
"""

from __future__ import annotations

import multiprocessing

from audio_transcriber.cli.app import app

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()
