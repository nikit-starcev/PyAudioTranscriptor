"""Устойчивое извлечение JSON-объекта из ответа LLM.

Модель нередко добавляет пояснения до/после JSON или оборачивает его в
markdown-блок ```json … ```. Общий разбор используется всеми LLM-этапами
(имена, термины, семантические правки), чтобы не дублировать логику.
"""

from __future__ import annotations

import json
import re


def extract_json_object(raw: str) -> dict:
    """Извлекает первый JSON-объект из ответа LLM (устойчиво к лишнему тексту).

    :raises ValueError: если в ответе нет корректного JSON-объекта.
    """
    text = raw.strip()
    # Снимаем ```json ... ``` обёртку, если LLM её добавила.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("В ответе LLM нет JSON-объекта")

    return json.loads(text[start : end + 1])
