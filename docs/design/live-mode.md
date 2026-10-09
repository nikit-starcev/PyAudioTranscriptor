# Live-режим: потоковая транскрибация, предпросмотр и захват звука (#44)

Статус: **дизайн-док** (исследование + эмпирическая проверка выполнены). Дата: 2026-10-09.

## 1. Цель и объём

Дать возможность транскрибировать **во время записи**: потоковый ASR, живой предпросмотр последних
секунд и **одновременный захват системного звука и микрофона** (встречи/звонки «на лету»). Сейчас
live-режима нет вообще — только файловая (батчевая) обработка.

Вне объёма: онлайн-перевод, TTS, замена батчевого конвейера.

Связанные задачи: **#47** (потоковая диаризация Sortformer), **#108** (дорожка говорящего),
**#42/#55** (удалённый режим — тонкий клиент).

## 2. Проверенные факты (эмпирика на целевом железе)

Железо: **AMD RX 590 (Polaris gfx803), только Vulkan, без ROCm/CUDA**; PyTorch — CPU.

Проект **уже** использует `nemo-speech` (`diarization/nemo_speech_engine.py`) и `whisper.cpp`
(`whisper-cli`). Проверено напрямую бинарником `nemo-speech 0.1.0` (`nemo-speech-native/`):

- `transcribe --live` — транскрибация **с микрофона** до Ctrl-C; `--stream <wav>` — стриминг по
  записанному WAV; `--device vulkan[:N]`; `--diarize`/`--diar-model` (Sortformer);
  `--endpointing`, `--vad-model`, `--speech-context` (hotwords), `--format text|json|srt|vtt`.
- Дефолтная ASR-модель — **`nvidia/nemotron-3.5-asr-streaming-0.6b`** (q8_0, 707 МиБ; ru-RU —
  «transcription-ready»). Режим `streaming`, шаг **160 мс**, `backend=Vulkan0`.
- **Замеры:** русский WAV 25 с — **Vulkan 4 с**, CPU 13 с (**3.25×**); английский JFK распознан
  дословно и без ошибок. `--diarize` (streaming ASR + Sortformer) на Vulkan отработал.
- `serve` — HTTP API + **realtime WebSocket `/v1/realtime`** + OpenAI-совместимые роуты
  (`/health`, `/v1/models`); требует явных `--asr-model`/`--diar-model`/`--config` (иначе
  «no models were loaded»), порт поднимается ~28 с (загрузка модели).
- `doctor`: features `asr backend_vulkan diarization http integrated_vad model_pull realtime_websocket`;
  device[0] `AMD Radeon RX 590 (RADV POLARIS10)`.
- **`whisper.cpp` `stream` в бандле отсутствует** (только `whisper-cli`) → стриминг через whisper.cpp
  без пересборки недоступен.
- `--live` захватывает **только дефолтный микрофон** (не системный звук).

Ограничения, снятия которых нет: качество на шумном русском аудио ниже, чем на чистом (нужны
VAD/денойз/hotwords); `--live` не умеет «система+микрофон» сам.

## 3. Требования

**Функциональные:** старт/стоп сессии; живой текст с латентностью единиц секунд; предпросмотр
последних ~10–20 с; одновременный захват system audio + mic; сохранение итога как обычного
результата (редактор/экспорт/голоса/протокол).

**Нефункциональные:** изоляция от батч-очереди; не блокировать и не срывать длинные прогоны;
один GPU — упорядоченный доступ; кроссплатформенность UI; работа офлайн.

## 4. Ключевое архитектурное решение

**Тезис «live нужно делать отдельным приложением» — не подтверждается.** Практика делится:
- стриминговые web-сервисы (WhisperLiveKit, WhisperLive, whisper_streaming, SimulStreaming,
  RealtimeSTT-сервер, NeMo-Speech.cpp `serve`) выносят live в **отдельный сервис/процесс + WS + UI**;
- десктоп-приложения (Buzz, Meetily) **встраивают** live в одно приложение (и тоже используют
  whisper.cpp/Vulkan).

Критерий — **отдельный процесс и свой жизненный цикл (always-on)**, а не «новое приложение».
Для нас отдельное приложение означало бы дублирование движков/UI/упаковки без выгоды.

**Решение: отдельный ПРОЦЕСС (live-воркер) в том же репозитории/пакете + новый live-экран в текущем
SPA.** Ядро стриминга — уже существующий `nemo-speech` (CLI `--live`/`--stream` или `serve`).

## 5. Архитектура

```
[источник звука]                         (system audio + mic)
  ├─ браузер: getDisplayMedia+getUserMedia (основной путь, без нативных хелперов)
  └─ нативно (Linux/Windows): PulseAudio monitor/loopback + микс → PCM 16k mono
        │  PCM-чанки
        ▼
[live-воркер — отдельный процесс]        ← переиспользует пакет audio_transcriber
  ├─ nemo-speech serve  (HTTP + WS /v1/realtime)   ← ядро: streaming ASR + Sortformer
  └─ либо nemo-speech transcribe --live (дефолтный микрофон)
        │  частичный текст + таймкоды (+ говорящие)
        ▼
[FastAPI: LiveSessionManager + WS/SSE]     (транспорт в UI; НЕ через JobsDB/JobRunner)
        ▼
[React SPA: новый экран «Live»]            (уровень, живой текст, предпросмотр)
        ▼
[финализация] → обычный result/job → существующие редактор, экспорт, голоса, протокол
```

### 5.1 Слой захвата
- **Основной путь — браузер:** `getDisplayMedia({audio:true, systemAudio:"include"})` +
  `getUserMedia({audio:true})`, микс в WebAudio → PCM-чанки в WS. Не требует нативных хелперов,
  работает на Linux/Windows/macOS (Chrome, localhost — secure context).
- **Нативная опция:** Linux — PulseAudio/PipeWire monitor-источник + `parec`/`pw-record`,
  микрофон вторым потоком, микс (`ffmpeg amix`/`pw-loopback`) — без root. Windows — WASAPI loopback.
  macOS — только через виртуальное устройство (BlackHole).
- Интерфейс: `AudioSource → np.ndarray` (16 кГц, моно) — ровно формат конвейера (`SAMPLE_RATE`).

### 5.2 Ядро стриминга
- **`nemo-speech serve`** (HTTP/WS) — предпочтителен: реальный стриминг, диаризация, устройство
  `vulkan`, OpenAI-совместимый контракт. Живёт **отдельным процессом** (свой порт, always-on).
- Альтернатива — `nemo-speech transcribe --live` (проще, но только микрофон и без WS-контракта).
- whisper.cpp `stream` — **недоступен** без пересборки; fast-whisper/SimulStreaming — не подходят
  (потоковость/VRAM).

### 5.3 Менеджер сессий и транспорт
Новый `LiveSessionManager` (id сессии → буфер сегментов и превью) + канал в UI
(`WS` или `SSE`). Существующий `JobEventBus` несёт только прогресс и закрывается на терминальных
статусах — для бессрочной live-сессии нужен **отдельный** канал.

### 5.4 UI
Новый hash-роут `#/live` + экран: индикатор уровня (есть `utils/playback.amplitude_envelope`),
живой текст (переиспользовать компоненты `TranscriptTable`/`EditorPanel`), кнопки старт/стоп,
предпросмотр последних секунд (аналог плеера `app/playback.tsx`).

### 5.5 Финализация
По стоп-сессии — прогнать накопленные сегменты через уже имеющиеся чистые преобразования
(`merging/`, `cleaning/`, `correction/`, `export/`, `protocol.py`) и сохранить как обычный результат,
чтобы работали редактор, экспорт (TXT/DOCX/JSON/SRT/VTT/MD/PDF), протокол, голоса.

## 6. Точки интеграции в коде

- **Новый стриминговый протокол** рядом с `transcription/base.py` (сейчас принимает только `Path`) —
  напр. `feed(chunk) -> Iterator[Partial]` / `finalize()`.
- **Слой захвата** — вне текущего `utils/audio.py` (только файлы/PyAV).
- **Менеджер live-сессий + роут** — в `web/` (не через `JobsDB`/`JobRunner`).
- **Фронт**: `webui/src/app/routes.ts`, `AppShell.tsx`, `Sidebar.tsx` + новая страница.
- **Заново не пишем**: `AppConfig`/`config.env`, каталог/скачивание моделей
  (`models/`), doctor, глоссарий/hotwords, голоса/enrollment, всю постобработку/экспорт.

## 7. Сосуществование с батчем (один GPU)

`JobRunner` намеренно последовательный (одна задача за раз). Политика:
**live на время сессии приоритетнее — батч ставится на паузу**, live-воркер — отдельный процесс,
чтобы конкуренция за RX 590 была управляемой. Явного параллельного GPU-доступа не требуется.

## 8. Рассмотренные альтернативы

| Вариант | Вердикт |
| --- | --- |
| Отдельное **приложение** | ❌ дублирование движков/UI/упаковки; практика Buzz/Meetily против |
| Стриминг через **whisper.cpp `stream`** | ❌ бинарника нет; нужна пересборка с SDL2+Vulkan (не проверено на gfx803) |
| **SimulStreaming** (AlignAtt) | ❌ torch, ≥10 ГБ VRAM — не наш класс железа |
| **faster-whisper** streaming | ❌ сам не стримит; GPU — только CUDA |
| **Vosk** (RU, CPU) | △ запасной CPU-путь, если GPU занят/недоступен |
| **nemo-speech** (`serve`/`--live`) | ✅ проверено: стриминг ASR + Sortformer на Vulkan, ru-RU |
| Захват только нативно | △ работает, но требует хелперов на каждую ОС → браузер проще |

## 9. Открытые вопросы и риски

- **Система+микрофон одновременно**: `nemo-speech --live` умеет только микрофон — нужен наш слой
  захвата/микса (Linux/Windows) либо браузерный микс.
- **`getDisplayMedia`** system-audio — не Baseline; проверить на целевой связке Linux+PipeWire+Chrome.
- **Качество RU** на шумном аудио — включить `--vad-model` (Silero), hotwords/`--speech-context`,
  при необходимости — денойз до стриминга.
- **Потоковая диаризация (#47):** Sortformer ≤4 спикера, EN-центричность; в live — только как
  подсказка имён (enrollment по образцам требует отдельного эмбеддера).
- **Упаковка:** захват аудио нельзя делать внутри Docker (нет устройств) — live только на хосте.
- **Латентность/надёжность** WS-сессии, реконнект.

## 10. План реализации

- **P0 — прототип:** поднять `nemo-speech serve` как стриминговое ядро; минимальный live-экран
  (старт/стоп, уровень, живой текст) с **браузерным микрофоном**; проверить сквозную латентность.
- **P1 — захват:** «система+микрофон» (браузерный микс как основной путь; нативный Linux-хелпер
  `parec`/loopback+amix — опцией).
- **P2 — сессии и финализация:** `LiveSessionManager` + WS/SSE; сохранение итога обычным результатом
  с постобработкой/экспортом.
- **P3 — качество и диаризация:** VAD/hotwords/денойз; потоковая диаризация (#47) как подсказка имён.
- **P4 — упаковка и кроссплатформа:** Windows (WASAPI loopback), portable; политика «live vs batch».

## 11. Источники

- NeMo-Speech.cpp — <https://github.com/NVIDIA/NeMo-Speech.cpp> (cli.md, build.md); модель
  <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>; Sortformer streaming —
  <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2>, <https://arxiv.org/abs/2507.18446>.
- Аналоги: WhisperLiveKit, WhisperLive, ufal/whisper_streaming (LocalAgreement),
  ufal/SimulStreaming (AlignAtt), KoljaB/RealtimeSTT; Buzz, Meetily.
- Захват: PipeWire/PulseAudio (monitor/loopback), WASAPI loopback (MS Learn),
  BlackHole, `getDisplayMedia` (MDN).
- Внутренние наблюдения: `#289` (nemo-speech Vulkan на RX 590), проверка streaming ASR (см. историю).
