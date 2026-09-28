"""LLM-постобработка стенограммы.

Подпакет объединяет три части:

- :mod:`audio_transcriber.llm.client` — тонкий клиент к локальному
  ``llama-server`` (OpenAI-совместимый эндпоинт ``/v1/chat/completions``);
- :mod:`audio_transcriber.llm.glossary` — детерминированный матчер терминов
  по пользовательскому глоссарию (без LLM);
- :mod:`audio_transcriber.llm.postprocess` — извлечение имён участников и
  правка терминов, связывающие LLM и глоссарий в единый этап конвейера.
"""

from __future__ import annotations
