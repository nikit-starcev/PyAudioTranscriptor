"""Состояние очереди файлов TUI."""

from __future__ import annotations

import queue as queue_module
from pathlib import Path


class QueueController:
    """Состояние очереди файлов TUI: строки таблицы, статусы и очередь на обработку.

    Повторное добавление уже известного файла не создаёт дублирующей строки.
    """

    def __init__(self) -> None:
        self.pending: queue_module.Queue[Path] = queue_module.Queue()
        self.rows: list[Path] = []
        self.status: dict[Path, str] = {}

    def is_waiting(self, path: Path) -> bool:
        """Файл уже в очереди или обрабатывается прямо сейчас."""
        return self.status.get(path) in ("pending", "running")

    def is_empty(self) -> bool:
        return self.pending.empty()

    def first(self) -> Path:
        """Первый файл очереди (без извлечения)."""
        return self.pending.queue[0]

    def add(self, path: Path) -> None:
        """Ставит файл в очередь, переиспользуя существующую строку."""
        if path not in self.rows:
            self.rows.append(path)
        self.status[path] = "pending"
        self.pending.put(path)

    def mark(self, path: Path, status: str) -> None:
        self.status[path] = status
