"""LLM-постобработка стенограммы.

Подпакет объединяет части:

- :mod:`audio_transcriber.llm.client` — тонкий клиент к локальному
  ``llama-server`` (OpenAI-совместимый эндпоинт ``/v1/chat/completions``);
- :mod:`audio_transcriber.llm.glossary` — детерминированный матчер терминов
  по пользовательскому глоссарию (без LLM);
- :mod:`audio_transcriber.llm.chunking` — нарезка длинной стенограммы на
  фрагменты под контекст модели (общая для всех этапов);
- :mod:`audio_transcriber.llm.summary` — резюме встречи (map-reduce);
- :mod:`audio_transcriber.llm.prompts` — пользовательские доп. инструкции и
  сохранение фактических промптов;
- :mod:`audio_transcriber.llm.postprocess` — оркестрация этапов (имена,
  термины, резюме), связывающая LLM и глоссарий в единый шаг конвейера.
"""

from __future__ import annotations
