"""Хелперы для сборки окружения внешних процессов.

Движки на базе C++ (llama.cpp, whisper.cpp) загружают разделяемые библиотеки
по ``LD_LIBRARY_PATH``; логика подмешивания каталога библиотек к текущему
окружению вынесена сюда, чтобы клиент LLM и движок whisper.cpp не дублировали
её.
"""

from __future__ import annotations

from collections.abc import Mapping


def with_library_path(
    env: Mapping[str, str], library_path: str | None
) -> dict[str, str]:
    """Возвращает копию ``env`` с добавленным ``LD_LIBRARY_PATH``.

    Если ``library_path`` не задан, окружение возвращается без изменений
    (копией). Иначе каталог библиотек ставится первым, сохраняя прежнее
    значение ``LD_LIBRARY_PATH`` после него.
    """
    result = dict(env)
    if library_path:
        existing = result.get("LD_LIBRARY_PATH")
        result["LD_LIBRARY_PATH"] = library_path + (":" + existing if existing else "")
    return result
