"""Библиотека голосов для веб-интерфейса: каталог, листинг и безопасный доступ.

Каталог библиотеки — тот же, что у TUI/CLI (``VOICES_DIR`` из ``config.env``
или ``./voices``). Файлы ``<Имя>.wav`` верхнего уровня — образцы голоса; у
одного человека их может быть несколько: ``Иван.wav``, ``Иван (2).wav`` и т.д.
(см. :mod:`audio_transcriber.diarization.voices`). Образцы группируются по
базовому имени человека; идентификатором конкретного файла служит его имя
(``filename``). Все операции доступа к файлу идут через листинг, поэтому обойти
каталог через подставное имя невозможно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audio_transcriber.config.defaults import DEFAULT_VOICES_DIR
from audio_transcriber.diarization.voices import collect_voice_library
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
    """Один файл-образец в библиотеке голосов.

    ``name`` — базовое (человеческое) имя участника (общее для дубликатов),
    ``path`` — конкретный файл. Идентификатор образца в API — ``filename``.
    """

    name: str
    path: Path

    @property
    def filename(self) -> str:
        """Имя файла образца (уникальный идентификатор в API)."""
        return self.path.name

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API (имя, файл, длительность, размер)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {
            "name": self.name,
            "filename": self.filename,
            "duration": round(read_duration(self.path), 2),
            "size": size,
        }


@dataclass(frozen=True, slots=True)
class VoiceGroup:
    """Все образцы одного человека: базовое имя и список образцов."""

    name: str
    samples: tuple[VoiceSample, ...]

    def as_dict(self) -> dict[str, object]:
        """Плоское представление для API (имя, число образцов, сами образцы)."""
        return {
            "name": self.name,
            "count": len(self.samples),
            "samples": [sample.as_dict() for sample in self.samples],
        }


def list_voice_samples(directory: Path) -> list[VoiceSample]:
    """Все образцы ``*.wav`` верхнего уровня (плоский список, по имени группы).

    Отсутствующий/недоступный каталог даёт пустой список (мягкая деградация).
    Внутри группы основной образец идёт первым, затем дубликаты по номеру.
    """
    samples: list[VoiceSample] = []
    for name, paths in collect_voice_library(directory).items():
        samples.extend(VoiceSample(name=name, path=path) for path in paths)
    return samples


def list_voice_groups(directory: Path) -> list[VoiceGroup]:
    """Группы образцов по человеку, отсортированные по имени (без учёта регистра)."""
    groups = [
        VoiceGroup(name=name, samples=tuple(VoiceSample(name=name, path=path) for path in paths))
        for name, paths in collect_voice_library(directory).items()
    ]
    groups.sort(key=lambda group: group.name.casefold())
    return groups


def find_voice_sample(directory: Path, name: str) -> VoiceSample | None:
    """Находит образец по точному stem файла, иначе — первый образец человека.

    Так работают и старые вызовы ``/api/voices/<имя>``: для одиночного файла
    ``Анна.wav`` имя группы совпадает со stem; для дубликатов возвращается
    основной (первый) образец. Поиск идёт по листингу, поэтому ``name`` не может
    вывести за пределы библиотеки.
    """
    samples = list_voice_samples(directory)
    for sample in samples:
        if sample.path.stem == name:
            return sample
    for sample in samples:
        if sample.name == name:
            return sample
    return None


def find_voice_sample_file(directory: Path, filename: str) -> VoiceSample | None:
    """Находит конкретный образец по имени файла (``Иван (2).wav``)."""
    if not filename or "/" in filename or "\\" in filename:
        return None
    for sample in list_voice_samples(directory):
        if sample.filename == filename:
            return sample
    return None
