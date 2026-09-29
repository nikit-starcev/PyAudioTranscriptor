"""Библиотека голосов для веб-интерфейса: каталог, листинг и безопасный доступ.

Каталог библиотеки — тот же, что у TUI/CLI (``VOICES_DIR`` из ``config.env``
или ``./voices``). Файлы ``<Имя>.wav`` верхнего уровня — образцы голоса; имя
участника — это stem файла. Все операции доступа к файлу идут через листинг,
поэтому обойти каталог через подставное имя невозможно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.config.defaults import DEFAULT_VOICES_DIR
from audio_transcriber.utils.playback import read_duration
from audio_transcriber.web.config import env_defaults


def resolve_voices_dir(explicit: Path | None = None) -> Path:
    """Каталог библиотеки голосов: явный аргумент, ``VOICES_DIR`` или ``voices``."""
    if explicit is not None:
        return Path(explicit)
    raw = env_defaults().get("VOICES_DIR", "").strip()
    return Path(raw) if raw else Path(DEFAULT_VOICES_DIR)


@dataclass(frozen=True, slots=True)
class VoiceSample:
    """Один файл-образец в библиотеке голосов."""

    name: str
    path: Path

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API (имя, файл, длительность, размер)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {
            "name": self.name,
            "filename": self.path.name,
            "duration": round(read_duration(self.path), 2),
            "size": size,
        }


def list_voice_samples(directory: Path) -> list[VoiceSample]:
    """Образцы ``*.wav`` верхнего уровня, отсортированные по имени.

    Отсутствующий/недоступный каталог даёт пустой список (мягкая деградация).
    """
    try:
        if not directory.is_dir():
            return []
        files = sorted(
            (path for path in directory.iterdir() if path.is_file()),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        return []

    samples: list[VoiceSample] = []
    for path in files:
        if path.suffix.lower() != ".wav":
            continue
        name = path.stem.strip()
        if name:
            samples.append(VoiceSample(name=name, path=path))
    return samples


def find_voice_sample(directory: Path, name: str) -> VoiceSample | None:
    """Находит образец по точному имени (stem) внутри каталога библиотеки.

    Возвращает ``None``, если имени нет среди файлов каталога. Поиск идёт по
    листингу, поэтому ``name`` не может вывести за пределы библиотеки.
    """
    for sample in list_voice_samples(directory):
        if sample.name == name:
            return sample
    return None
