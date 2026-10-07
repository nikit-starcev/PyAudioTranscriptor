# Интеграционные тесты

Тесты, которые запускают **реальные** модели и бинарники (faster-whisper,
whisper.cpp, pyannote.audio) на коротком аудио. Они помечены маркером
`integration` и **не** входят в обычный прогон `pytest` (в `pyproject.toml`
задан `addopts = "-ra -m 'not integration'"`).

## Запуск

```bash
# только интеграционные
.venv/bin/python -m pytest -m integration

# только этот каталог
.venv/bin/python -m pytest tests/integration -m integration

# с подробным выводом причин пропусков
.venv/bin/python -m pytest -m integration -rs

# uv-обёртка (как в README проекта)
uv run pytest -m integration
```

Обычный `pytest` (без `-m integration`) эти тесты не запускает и не требует
моделей — можно убедиться: `pytest --collect-only -q`.

## Принцип «skip, а не fail»

Если в окружении нет нужного ресурса, тест **пропускается** с понятной причиной
(а не падает). Разрешение ресурсов идёт в таком порядке:

1. переменная окружения `INTEGRATION_*` (явное переопределение);
2. значение из `config.env` в корне проекта;
3. для whisper-cli — поиск в `PATH`.

| Тест | Нужно | Переменная | Ключ `config.env` |
|------|-------|------------|-------------------|
| `test_asr_whisper_cpp.py` | бинарник и модель whisper.cpp | `INTEGRATION_WHISPER_CPP_BINARY`, `INTEGRATION_WHISPER_CPP_MODEL`, `INTEGRATION_WHISPER_CPP_LIB_PATH` | `WHISPER_CPP_BINARY`, `WHISPER_CPP_MODEL`, `WHISPER_CPP_LIB_PATH` |
| `test_asr_faster_whisper.py` | модель faster-whisper | `INTEGRATION_FASTER_WHISPER_MODEL` | `MODEL` |
| `test_diarization_pyannote.py` | локальная модель pyannote и/или HF-токен | `INTEGRATION_PYANNOTE_LOCAL_MODEL`, `INTEGRATION_HF_TOKEN` | `PYANNOTE_LOCAL_MODEL`, `HF_TOKEN` |

faster-whisper пробует модели от лёгких к тяжёлым (`tiny.en` → `tiny` →
`MODEL`), поэтому при отсутствии сети тест либо скачает `tiny`, либо
пропустится. Диаризация предпочитает локальную модель (работает офлайн);
если её нет — используется HF-токен, а при недоступности обоих тест
пропускается.

## Что проверяется

- **ASR (whisper.cpp и faster-whisper):** на 5-секундном фрагменте речи
  возвращаются непустые сегменты и текст, длительность совпадает с длиной
  фрагмента (допуск 0.25 с), сегменты упорядочены и не выходят за границы аудио.
- **Диаризация pyannote:** smoke — пайплайн не падает и возвращает корректные
  реплики говорящих (непустой `speaker_id`, `end > start >= 0`).

## Тестовый аудиофрагмент

Короткий WAV (5 с) генерируется в `tmp_path` из уже хранящейся в репозитории
эталонной записи `tests/tests_jfk.flac` (см. фикстуру `short_speech_wav`).
Отдельный бинарный файл не коммитится; если эталонной записи нет, тест
пропускается.
