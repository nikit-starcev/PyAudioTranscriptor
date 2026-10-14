# Карта компонентов

Навигация: [Обзор](README.md) · [Конвейер](pipeline.md) · [Движки](engines.md) ·
[Веб](web.md) · [Конфигурация](configuration.md) · [Упаковка](packaging.md) ·
[Корп-режим](corporate-mode.md) · [ADR](decisions.md)

Документ описывает карту пакетов/модулей и **протоколы-барьеры**, через которые
стадии конвейера общаются с реализациями. Факты — из `src/audio_transcriber/**`.

## 1. Протоколы (контракты) и их реализации

Все ключевые швы описаны через `typing.Protocol`. Это позволяет заменить
компонент, не меняя `pipeline.py` (см. `pipeline.py:1-8`). Реализации
подставляются фабриками или собираются в `run_pipeline` по `AppConfig`.

| Протокол | Где объявлен | Реализации |
|---|---|---|
| `SpeechRecognizer` | `transcription/base.py:17` | `WhisperSpeechRecognizer` (faster-whisper, `transcription/whisper_engine.py:18`), `WhisperCppRecognizer` (`transcription/whisper_cpp_engine.py:868`), `GigaAmRecognizer` (`transcription/gigaam_engine.py:93`), `HybridSpeechRecognizer` (`transcription/hybrid.py:196`) |
| `SpeakerDiarizer` | `diarization/base.py:19` | `PyannoteSpeakerDiarizer` (`diarization/pyannote_engine.py:85`), `NemoSpeechSpeakerDiarizer` (`diarization/nemo_speech_engine.py:243`), `HybridSpeakerDiarizer` (`diarization/hybrid_engine.py:468`), `SherpaSpeakerDiarizer` (`diarization/sherpa_engine.py:130`) |
| `SegmentMerger` | `merging/base.py:16` | `OverlapSegmentMerger` (`merging/aligner.py:224`) |
| `DenoiserProtocol` | `denoising/base.py:16` | `DeepFilterDenoiser` (`denoising/deepfilter.py:304`), `CachingDenoiser` (`cache/denoiser.py:28`, обёртка) |
| `ArtifactCleanerProtocol` / `RepetitionCleanerProtocol` / `TextNormalizerProtocol` | `cleaning/base.py:17` / `:29` / `:53` | `ArtifactCleaner` (`cleaning/artifact_filter.py:423`), `RepetitionCleaner` (`cleaning/repetition_filter.py:127`), `TextNormalizer` (`cleaning/text_normalizer.py:72`) |
| `TextCorrector` | `correction/base.py:11` | `MorphTextCorrector` (`correction/morph_corrector.py:75`) |
| `ResultExporter` | `export/base.py:12` | `TxtExporter`, `DocxExporter`, `JsonExporter`, `SrtExporter`, `VttExporter`, `MarkdownExporter`, `PdfExporter` (`export/{txt,docx,json,srt,vtt,markdown,pdf}_exporter.py`); выбор — `export/factory.py` |
| `LlmClient` / `StreamingLlmClient` | `llm/base.py:10` / `:29` | `LlamaServerClient` (`llm/client.py:272`), `OpenAIClient` (`llm/client.py:605`); выбор — `create_llm_client` (`llm/client.py:917`) |
| `SpeakerEmbeddingEngine` | `diarization/enrollment.py:92` | `PyannoteEmbeddingEngine` (`diarization/enrollment.py:105`), CAM++-эмбеддер (через `SpeakerEmbedder`, `diarization/embeddings.py:231`) |
| `SpeakerDiarizer` (как источник `overlap_regions`) | `diarization/base.py:49` | pyannote и nemo-speech отдают зоны перекрытий; `sherpa` — всегда пусто (`diarization/sherpa_engine.py:1-24`) |

Базовые интерфейсы объявляют лишь минимум:

```python
# transcription/base.py:20-29
def transcribe(self, audio_path: Path, *, language: str | None = None) -> tuple[list[TranscriptionSegment], str, float]: ...

# diarization/base.py:22-47
def diarize(self, audio_path, *, num_speakers=None, min_speakers=None, max_speakers=None, waveform=None) -> list[SpeakerSegment]: ...
def overlap_regions(self) -> list[SpeakerOverlap]: ...
```

## 2. Карта пакетов

### 2.1. Точки входа и интерфейсы

| Пакет/модуль | Ответственность |
|---|---|
| `cli/app.py` | Typer-приложение: команды `transcribe` (`:204`), `doctor` (`:1331`), `tui` (`:1353`), `web` (`:1411`), подкоманда `glossary` (`:1476`). Сборка `AppConfig` с приоритетом явных CLI-флагов над `config.env` (`_explicit_overrides`, `:155`) |
| `cli/env_config.py` | Перенос значений `config.env` в kwargs CLI-опций |
| `cli/models_cmd.py` | Подкоманда `models` (каталог/скачивание) + `download_for_web` |
| `tui/` | Textual-TUI: `app.py`, `screens.py`, `widgets.py`, `queue.py` (очередь), `results.py`, `options.py`, `config_builder.py` |
| `web/` | FastAPI-слой (см. [`web.md`](web.md)) |

### 2.2. Ядро конвейера

| Пакет/модуль | Ответственность |
|---|---|
| `pipeline.py` | `run_pipeline` (`:510`) — последовательная сборка стадий, стадийный кэш, контрольные точки отмены |
| `protocol.py` | `generate_protocol` (`:94`) — «протокол встречи» (пересчёт резюме + экспорт) по кнопке |
| `progress.py` | `ProgressEvent`, `ProgressCallback` — единый формат событий прогресса |
| `config/settings.py` | `AppConfig` (`:104`) — вся конфигурация одного запуска |
| `config/defaults.py` | Значения по умолчанию, не зависящие от подсистем (пороги, дефолты движков) |
| `domain/models.py` | Доменные модели: `WordTimestamp`, `TranscriptionSegment`, `SpeakerSegment`, `Speaker`, `SpeakerOverlap`, `TranscriptEntry`, `TranscriptionResult` |
| `domain/enums.py` | `Device`, `AsrBackend`, `ExportFormat` |
| `domain/editing.py` | Операции правки доменной модели (используются web-слоем) |

### 2.3. Стадии

| Пакет | Ответственность | Ключевые модули |
|---|---|---|
| `denoising/` | Шумоподавление | `base.py`, `deepfilter.py` (внешний Rust-CLI `deep-filter`) |
| `transcription/` | ASR | `base.py`, `whisper_engine.py`, `whisper_cpp_engine.py`, `gigaam_engine.py`, `hybrid.py` |
| `diarization/` | Диаризация, оценка числа говорящих, enrollment | `base.py`, `factory.py`, `pyannote_engine.py`, `nemo_speech_engine.py`, `hybrid_engine.py`, `sherpa_engine.py`, `embeddings.py`, `enrollment.py`, `speaker_count.py`, `reference.py`, `samples.py`, `voices.py`, `dedup.py`, `track.py`, `overlap.py`, `nemo_speech_assets.py` |
| `merging/` | Объединение ASR и диаризации в реплики | `base.py`, `aligner.py` (`OverlapSegmentMerger`), `sentence_merger.py`, `split.py` (`SpeakerChangeSplitter`), `overlap.py`, `same_name.py`, `context.py` |
| `cleaning/` | Очистка текста | `base.py`, `artifact_filter.py`, `repetition_filter.py`, `text_normalizer.py` |
| `correction/` | Автоисправление опечаток (pymorphy3) | `base.py`, `morph_corrector.py`, `editorial.py`, `defaults.py` |
| `llm/` | LLM-постобработка и чат | `base.py`, `client.py`, `postprocess.py`, `summary.py`, `glossary.py`, `semantic.py`, `chunking.py`, `prompts.py`, `chat.py`, `json_utils.py` |
| `export/` | Экспорт | `base.py`, `factory.py`, `txt/docx/json/srt/vtt/markdown/pdf_exporter.py`, `timeline.py`, `timestamps.py`, `words.py`, `annotations.py` |

### 2.4. Кэш, хранилища, модели, утилиты

| Пакет/модуль | Ответственность |
|---|---|
| `cache/store.py` | `StageCache` (`:58`), `compute_cache_key` (`:41`), `CACHE_FORMAT_VERSION` (`:29`) |
| `cache/serialization.py` | Сериализация ASR/диаризации для кэша (`asr_payload`, `diarization_payload`, `*_from_payload`) |
| `cache/denoiser.py` | `CachingDenoiser` — кэширование денойза + `cache_key`/`last_hit`/`last_waveform` |
| `storage/glossary_db.py` | SQLite-БД глоссария (`GlossaryDB`, `:147`), общая для CLI (`glossary`) и web |
| `storage/glossary_builder.py` | `build_active_glossary` — сборка активного глоссария для прогона |
| `storage/prompts_db.py` | `PromptsDB` (`:65`) — пользовательские шаблоны промпта резюме (#97) |
| `storage/chat_db.py` | `ChatDB` (`:65`) — история чата по стенограмме (#54/#96) |
| `storage/archive_db.py` | `ArchiveDB` (`:193`) — архив + полнотекстовый индекс FTS5 (#99) |
| `web/storage/jobs_db.py` | `JobsDB` (`:268`), `Job` (`:163`) — очередь/состояние задач (web-only) |
| `models/catalog.py` | `MODEL_CATALOG` (`:64`) — каталог моделей и их целевых путей |
| `models/download.py` | `HfDownloader` (`:104`), `ModelDownloadManager` (`:272`), `DownloadBus` (`:175`) |
| `utils/` | Общие утилиты: `audio.py` (декод/запись, `SAMPLE_RATE`), `device.py` (resolve_device), `env.py` (поиск бинарников, effective_library_path), `config_env.py` (поиск/разбор `config.env`), `hotwords.py`, `logging.py`, `net.py` (порты), `notifications.py`, `playback.py`, `subprocess_registry.py` (гашение дочерних процессов), `text.py`, `glossary_paths.py`, `exceptions.py` |
| `doctor.py` | Самопроверка окружения (`run_doctor`; вызывается из CLI `cli/app.py:1346`) |

## 3. Разделение хранилищ: root `storage/` vs `web/storage/`

- `storage/` (корневой) — **общие** предметные БД, доступные и CLI/TUI, и web:
  глоссарий, шаблоны промптов, чат, архив.
- `web/storage/` — только `jobs_db.py` (очередь и состояние web-задач), т.к. это
  сугубо веб-сущность.

SQLite-хранилища открываются per-запрос/операцию (например, `PromptsDB` в
`app.py:848` внутри `active_summary_prompt_body`), а не держатся глобально.

## 4. Что НЕ является частью ядра

- Веб-слой (`web/`) и SPA (`webui/`) — надстройка над `run_pipeline`; их можно
  не запускать (CLI/TUI работают без них).
- Внешние бинарники и модели — не в репозитории; ставятся отдельно
  ([`packaging.md`](packaging.md), [`configuration.md`](configuration.md)).
- Live-режим (`docs/design/live-mode.md`) — **дизайн-док**, в коде отсутствует.

## 5. Перелинковка

Дальше: как всё это исполняется на прогоне — [`pipeline.md`](pipeline.md);
как выбираются движки — [`engines.md`](engines.md); как это обёрнуто сервером —
[`web.md`](web.md).
