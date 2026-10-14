# Анализ потребления оперативной памяти в PyAudioTranscriptor

> Режим: **только чтение**. Код не менялся, сервер 8790 не трогался.
> Репозиторий: `PyAudioTranscriptor`, ветка `main`, HEAD `93d5028`.
> Все факты — со ссылками `path:line`. Числа, которых нет в коде/доках,
> помечены **[ДОПУЩЕНИЕ]** и не выдаются за измеренные.

## 0. Опорные величины (для расчётов)

- Моно `float32` **16 кГц**: `16000 × 4 = 64 000 байт/с` ≈ **230 МБ/час**
  (в README это же значение: `README.md:1084-1085` — 30 мин ≈ 115 МБ).
- Моно `float32` **48 кГц**: ≈ **691 МБ/час** (считается только на денойзе и
  только по одному чанку — см. §1.1).
- Модели (файлы, каталог `models/catalog.py`):

| Модель | Файл | Ссылка |
|---|---|---|
| whisper.cpp base | 148 МБ | `catalog.py:110-119` |
| whisper.cpp small | 488 МБ | `catalog.py:99-108` |
| whisper.cpp medium | 1.53 ГБ | `catalog.py:88-97` |
| whisper.cpp large-v3-turbo | 1.62 ГБ | `catalog.py:66-75` |
| whisper.cpp large-v3 | 3.10 ГБ | `catalog.py:77-86` |
| Qwen2.5-7B Q4_K_M (LLM) | 4.68 ГБ (2 шарда) | `catalog.py:121-133` |
| pyannote community-1 (snapshot) | ~32 МБ (каталог) | `catalog.py:135-146` |
| GigaAM v3 ONNX (snapshot) | 4.46 ГБ (fp32+int8 варианты) | `catalog.py:148-163` |
| sherpa CAM++ (эмбеддинги) | 28 МБ | `catalog.py:165-183` |

**Важно:** размер *файла* ≠ RSS при инференсе. Реальный RSS модели **не измерялся**
в этом анализе; где он нужен — помечен **[ДОПУЩЕНИЕ]**.

## 1. Карта памяти по стадиям

| Стадия | Главный потребитель | Что ещё живёт одновременно | Освобождается? |
|---|---|---|---|
| Декод/буферы | waveform `float32` 16кГц | сам контейнер PyAV (закрыт) | да, но массив держится потребителем |
| Денойз (#42) | внешний `deep-filter` (процесс) | 1 чанк 48кГц + итоговый 16кГц waveform | да, `close()` |
| ASR whisper.cpp | процесс `whisper-cli` (RSS) | waveform **всего файла** (Python) | проц. — да; waveform — до возврата |
| ASR faster-whisper | модель **в процессе** | — | с процессом-раннером |
| ASR GigaAM | модель ONNX **в процессе** | waveform всего файла | с раннером |
| Диаризация pyannote | torch + pipeline **в процессе** | waveform (+ torch-вид) | с раннером |
| Диаризация nemo-speech | процесс `nemo-speech` | temp WAV (диск) | проц. — да |
| Диаризация sherpa | CAM++ ONNX **в процессе** | waveform + N×N матрица | с раннером |
| Диаризация hybrid | `nemo-speech` (окна) + CAM++ | waveform + observations | с раннером |
| Постобработка | списки `entries`/`words` (текст) | — | по GC |
| LLM | процесс `llama-server` (RSS+VRAM) | — | `close()` в конце стадии |
| Кэш | JSON-строки стадий | модель/буферы | transient |
| Web результата | `TranscriptionResult` + JSON | waveform, если ещё жив | по GC |

### 1.1. Декодирование и буферы (`utils/audio.py`, `utils/playback.py`)

- `WaveformAccumulator` (`audio.py:25-61`) — предвыделенный буфер, растёт
  геометрически (`:53`), `finish()` отдаёт contiguous-срез **без копии**
  (`:59-61`). Это уже оптимизировано: пик равен длине результата, а не её
  удвоению (что было бы при `np.concatenate` списка чанков — см. docstring
  `:27-32`).
- `load_waveform` (`audio.py:136-172`) — потоково декодирует **весь файл** в один
  массив `float32` 16 кГц. Именно этот массив потом многократно переиспользуется
  (диаризация/enrollment/образцы), но всё равно упирается в 230 МБ/час.
- `estimate_sample_count` (`audio.py:64-80`) — дешёвая оценка по заголовку для
  предвыделения (без декода).
- `resample_waveform` (`audio.py:175-213`) — **держит список чанков и результат**
  одновременно и склеивает через `np.concatenate` (`:213`), т.е. пик ≈ двойная
  длина. Используется для смены частоты вне основного пути денойза.
- `encode_pcm16` (`audio.py:216-228`) — на **весь** waveform создаёт временные
  `float32` (`samples = clip * 32768`) и `int16` (ещё ~½ длины): при записи
  целого файла это транзиентный пик ≈ 1.5× длины массива.
- `playback.py`: `_read_wav_samples` (`:72-96`) читает **весь** WAV в память ради
  амплитудной огибающей (`amplitude_envelope`, `:118`); `read_duration` (`:53`)
  и сам плеер (`:186`) — дешёвые (заголовок / внешний процесс).

### 1.2. Денойз (`denoising/deepfilter.py`) — #42

- Обработка **чанковая** (30 с, перекрытие 0.5 с): `DEFAULT_CHUNK_SECONDS`
  (`:70`), `_iter_resolved_48k` (`:230-301`), один внешний процесс на чанк
  (`_enhance` `:460-501`, `_run_deep_filter` `:503`).
- В Python живут: текущий 48кГц-чанк (~5.5 МБ), хвост перекрытия и **итоговый
  16кГц waveform** — аккумулятор на весь файл (`_write_temp_stream`, `:609-654`;
  `accumulator = WaveformAccumulator(expected_samples)` `:627`).
- `close()` (`:439-446`) удаляет temp-каталог и **обнуляет** `_last_waveform`.
- Пик стадии для часовой записи задокументирован как **0.2–0.3 ГБ**
  (`README.md:1136-1140`), процесс `deep-filter` ~57 МБ RSS независимо от длины.
- Важный факт: итоговый 16кГц-массив (230 МБ/час) **не освобождается после
  денойза**, а уходит дальше как `last_waveform` (см. §1.4).

### 1.3. ASR

- **whisper.cpp** (`transcription/whisper_cpp_engine.py`):
  - Длинное аудио режется на 30-с чанки (`DEFAULT_CHUNK_SECONDS` `:72`,
    `DEFAULT_CHUNK_OVERLAP` `:76`), каждый — отдельный процесс `whisper-cli`
    (`_run_whisper` `:1020-1126`) → RSS уходит с процессом.
  - **НО** `_transcribe_chunked` сначала делает `load_waveform(input_path)`
    (`:1132`) и держит **waveform всего файла** (230 МБ/час) на всё время
    чанкинга; каждый чанк пишется на диск `write_wav` (`:1154`) — это view +
    транзиентный encode.
  - Если вход не «родной» WAV, `_prepare_input` (`:987-1018`) дополнительно
    декодирует файл и пишет целый WAV (`:1013-1014`) → **двойной декод + пик
    encode** (см. §0).
  - `stderr_tail` ограничен 40 строками (`:1084`); поток-читатель джойнится
    (`:1103`, `:1115`).
- **faster-whisper** (`transcription/whisper_engine.py`): модель грузится лениво
  и **живёт в процессе** (`_load_model` `:46-69`, `self._model` `:59`), выгрузки
  нет. Аудио обрабатывает сам движок (не наш waveform).
- **GigaAM** (`transcription/gigaam_engine.py`): модель ONNX **в процессе**
  (`_load_adapter` `:158-197`) **и** `load_waveform` всего файла (`:279`).
- **Гибрид** (`transcription/hybrid.py`): держит **весь waveform** (`:321`) и
  создаёт вырезки-сегменты во временный WAV (`_refine_segment` `:252`), плюс
  второй движок (тоже модель в процессе).

### 1.4. Единый waveform и его «хвост» (`pipeline.py`)

- `_SharedWaveform` (`pipeline.py:109-136`) декодирует аудио **один раз** и
  переиспользует.
- Создаётся при `diarization_enabled` (`:614-618`), беря `denoiser.last_waveform`
  при промахе кэша.
- Используется на `:716` (диаризация), `:786` (enrollment), `:965` (образцы).
- **Ключевая находка:** `shared_waveform` — локальная переменная `run_pipeline`,
  она **не освобождается** и держится до самого конца функции, т.е. массив
  230 МБ/час живёт **сквозь** merge/clean/correct/LLM/export — задолго после
  того, как диаризация и enrollment закончились. Явного `del`/сброса нет;
  `denoiser.close()` (`:798-800`) снимает только ссылку денойзера, но не
  `shared_waveform._waveform`.

### 1.5. Диаризация

- **pyannote** (`diarization/pyannote_engine.py`): pipeline грузится в процесс и
  **кэшируется** (`self._pipeline` `:226`), `pipeline.to(device)` (`:225`).
  `torch.from_numpy(waveform).unsqueeze(0)` (`:248`) — **без копии** (torch
  разделяет буфер numpy). Модель/рантайм torch — отдельный крупный потребитель.
- **nemo-speech** (`diarization/nemo_speech_engine.py`): внешний процесс
  (`subprocess.run` `:210`); Python пишет **целый** temp 16кГц WAV
  (`_prepare_wav` `:154-167`) — это диск, не RAM.
- **sherpa** (`diarization/sherpa_engine.py`, ONNX CPU): CAM++ **в процессе**
  (`_resolve_embedder` `:182-202`), держит waveform (`:246-247`) и список
  векторов по окнам 1.5 с (`:267-291`).
- **hybrid** (`diarization/hybrid_engine.py`): окна 90 с, внутри — `nemo-speech`;
  держит весь waveform (`:788`), CAM++ и `observations`; эмбеддинги собираются
  `np.stack` (`:397`), клипы — `np.concatenate` (`:360`).
- **Оценщик числа говорящих** (`diarization/speaker_count.py`) при `auto` грузит
  silero VAD + CAM++ (`detect_speech_spans` `:210`, `estimate_speaker_count`
  `:357`); модель эмбеддингов ленивая и переиспользуемая
  (`embeddings.py:257-272`).
- **Кластеризация** (`diarization/embeddings.py:298-359`): для cosine строится
  **N×N**-матрица `precomputed`, но уже с **одной** аллокацией (in-place
  `:344-347`). Для часовой записи при окнах 1.5 с N≈2400 → ~23 МБ
  [ДОПУЩЕНИЕ по числу окон]; растёт квадратично.

### 1.6. Постобработка, LLM, экспорт

- `merging/*`, `cleaning/*`, `correction/*` работают со списками `entries`/`words`
  (текст), не с аудио. Память пропорциональна объёму стенограммы (единицы–десятки
  МБ), не длине аудио.
- **LLM** (`llm/client.py`): `llama-server` — **подпроцесс** (`_spawn` `:405-447`),
  RSS уходит с процессом. Пайплайн создаёт и **сам закрывает** клиент:
  `llm/postprocess.py:594,722`; `protocol.py:61-87`. Текст режется на фрагменты
  (`llm/chunking.py:140,170`), в память целиком не тащится.
- **Экспорт** (`export/*`, `protocol.py`): DOCX/PDF/MD/HTML строятся в памяти
  (timeline HTML — конкатенация строк, `export/timeline.py:185-226`) — обычно
  небольшие, но для очень длинных стенограмм дают всплеск.

### 1.7. Кэш, хранилище, веб

- `cache/store.py`: `save` делает `json.dumps` всего payload (`:117`), `load` —
  `read_text` всего файла (`:90`); аудио копируется (`save_audio` `:139-148`).
  **Автоэвикции/лимита нет** — кэш только `clear()` вручную (`:182`). ASR-JSON
  с пословными метками на длинном файле — самый крупный JSON.
- `cache/denoiser.py`: на попадании `last_waveform=None` (`:121`), на промахе
  отдаёт waveform (`:127`).
- `web/runner.py`: единственный воркер (`:152`), результат держится в памяти
  целиком: `serialize_result` → dict → `json.dumps(..., indent=2)` (`:591-603`).
  Для длинной стенограммы это **2-3× размер** JSON одновременно.
- `web/events.py`: история SSE на задачу **не ограничена** — `_history[job_id]`
  копит все события (`:88`, `:105`). События мелкие, но число за долгий прогон
  растёт.
- `web/app.py`: раздача файлов **потоковая** (`_file_chunks` `:4516`), загрузка
  **потоковая** 1 МБ (`:4234`), кэш длительностей — LRU 512 (`:4479-4504`).
  Однако `_decode_upload` декодирует загруженные байты целиком (`:3997-4004`).
- `web/results.py`: `load_result_file` читает весь JSON в память (`:158`).

## 2. Пиковые драйверы

### 2.1. «Тяжёлый» сценарий (по умолчанию / CLI)
Денойз + диаризация (`auto`) + `faster-whisper large-v3-turbo` + LLM (`Qwen-7B`),
экспорт + образцы, часовой файл. Оценки (не измерялись, **[ДОПУЩЕНИЕ]**):

- waveform 16кГц — ~0.23 ГБ (живёт сквозь весь прогон, §1.4);
- `faster-whisper` large-v3-turbo в процессе — ~1.5-2 ГБ;
- pyannote + torch runtime — сотни МБ (снапшот 32 МБ + рантайм);
- `llama-server` (процесс) — ~4.7 ГБ веса + KV-кэш при `LLM_CONTEXT=4096`;
- транзиенты JSON/экспорта — десятки МБ.

**Итог [ДОПУЩЕНИЕ]: >7 ГБ** на пике → уверенно не влезает в 4 ГБ, впритык в 8 ГБ.

### 2.2. «Лёгкий» сценарий
`whisper-cpp base/small` + `sherpa` диаризация + без LLM, часовой файл:

- `whisper-cli` (base ~148 МБ файл) процесс — ~0.3-0.5 ГБ [ДОПУЩЕНИЕ];
- CAM++ sherpa — 28 МБ + рантайм ONNX;
- waveform 16кГц — 0.23 ГБ + чанковые транзиенты;
- SSE/кэш-транзиенты — единицы МБ.

**Итог [ДОПУЩЕНИЕ]: ~1-1.5 ГБ** → влезает в 4 ГБ.

### 2.3. Что реально одновременно
- waveform жив **сквозь** merge→LLM→export (§1.4);
- модель ASR/диаризации жива от загрузки до конца объекта раннера; в web объекты
  создаются на прогон (`pipeline.py:544,703`) → между задачами освобождаются, но
  **внутри** задачи модель ASR и модель диаризации живут параллельно;
- LLM-процесс поднимается на стадии и закрывается после (`postprocess.py:722`),
  т.е. **не** пересекается с ASR/диаризацией — это плюс;
- `nemo-speech`/`whisper-cli`/`llama-server` изолированы в процессах (RSS уходит
  с процессом) — тоже плюс;
- **минус:** pyannote/faster-whisper/GigaAM/sherpa живут **в** процессе.

## 3. Рычаги оптимизации (сложность / эффект)

Сложность — по T-shirt (XS…XL, в духе bare-minimum). Эффект — по памяти.

| # | Рычаг | Сложность | Эффект | Где |
|---|---|---|---|---|
| O1 | Меньшие/квантованные модели: `whisper-cpp small/base`, `GIGAAM_QUANTIZATION=int8`, LLM 1.5–3B Q4 | XS (конфиг) | Очень высокий | `catalog.py:99-119,148-163`, `config/defaults.py:252-255` |
| O2 | Отключить стадии: `--no-denoise`, `DIARIZATION_ENABLED=false`, `LLM_ENABLED=false`, `protocol_auto=false`, `EXPORT_SPEAKER_SAMPLES=false` | XS (конфиг) | Очень высокий | `pipeline.py:602,672,892,926,953` |
| O3 | Выбор движка под слабое железо: `whisper-cpp` (процесс) вместо `faster-whisper`; `sherpa` (ONNX, без torch) вместо `pyannote` | XS–S | Высокий | `factory.py:50-53`, `engines.md` |
| O4 | Освобождать `shared_waveform` сразу после enrollment (перед merge), а для образцов декодировать заново | S | ~0.23 ГБ/час на длинном «хвосте» | `pipeline.py:595,615,786,965` |
| O5 | Не декодировать файл повторно в `_transcribe_chunked`: переиспользовать `shared_waveform`/читать чанки потоково | S–M | ~0.23 ГБ/час + время | `whisper_cpp_engine.py:1132` |
| O6 | Ленивая выгрузка моделей между стадиями (`del`+`gc.collect`, отказ от кэша `_pipeline`) | S–M | Высокий при последовательных стадиях | `whisper_engine.py:59`, `pyannote_engine.py:226`, `gigaam_engine.py:164` |
| O7 | Лимит/эвикция `StageCache` (размер/TTL), стриминговая сериализация ASR-JSON | S–M | Диск + пик JSON | `cache/store.py:90,117,182` |
| O8 | Ограничить историю SSE (maxlen по числу/байтам) | XS | Малый, но ограничивает рост | `web/events.py:88,105` |
| O9 | Изоляция pyannote/faster-whisper в сабпроцесс (по образцу nemo-speech) | M–L | Гарантированное освобождение RSS | `pyannote_engine.py`, `whisper_engine.py` |
| O10 | Обработка по сегментам вместо всего файла на ASR (расширить чанкинг #42/whisper.cpp на декод) | M | Снимает waveform всего файла | `whisper_cpp_engine.py:1128-1215` |
| O11 | Стриминговая запись результата (без полного dict+`json.dumps`) | M | Ниже пик на длинных стенограммах | `web/runner.py:591-603` |
| O12 | Субсэмплинг окон перед N×N-кластеризацией | S | Квадратичный член под контролем | `embeddings.py:344-347` |

Bare-minimum фильтр (для каждого рычага: MVP-критичен? нельзя отложить? нужен
архвыбор сейчас?):
- **O1–O3** — да/да/нет (чистая конфигурация, wow-эффект сразу).
- **O4, O5** — да/нет/нет (локальные правки, можно отложить, но дёшевы).
- **O6–O8** — нет/да/нет (гигиена памяти, не блокирует MVP).
- **O9–O11** — нет/нет/**да** (требуют архитектурного решения, если закладываться
  на «слабое железо» всерьёз).

## 4. Гид «слабое железо» (CPU без GPU, мало RAM)

> Скорости — из README (реальные замеры): денойз RTF ≈ 0.16
> (`README.md:1158-1159`), whisper.cpp на 300-с сигнале ≈ 48 с
> (`README.md:1159`). Оценки RAM — **[ДОПУЩЕНИЕ]**.

### 4.1. 4 ГБ RAM
- ASR: `whisper-cpp` + **base** (148 МБ) или **small** (488 МБ);
- диаризация: **выключена** или `sherpa` (CAM++ 28 МБ, ONNX CPU);
- денойз: выключить (экономит temp-WAV и постоянный waveform, если диаризация off);
- LLM: **выключить**; `protocol_auto=false`; образцы — выключить;
- `WORD_TIMESTAMPS` можно оставить (текстовый объём мал).
- Ожидание: часовой файл влезает (~1-1.5 ГБ пик, §2.2); whisper.cpp base — быстро,
  качество ниже.

### 4.2. 8 ГБ RAM
- ASR: `whisper-cpp` + **small/medium**, либо `faster-whisper` **int8**;
- денойз: можно включить (0.2-0.3 ГБ/час);
- диаризация: `sherpa` (без torch) или `nemo-speech` (процесс); `pyannote` —
  на грани из-за torch;
- LLM: только **лёгкая** (1.5–3B Q4, ~1-2 ГБ); 7B Q4 (4.7 ГБ) — не одновременно с
  тяжёлым ASR/диаризацией.

### 4.3. 16 ГБ RAM
- Комфортно: `whisper-cpp large-v3-turbo/medium` + `pyannote` + 7B Q4;
- избегать одновременного `faster-whisper large` + `pyannote` + 7B;
- можно держать `protocol_auto=true` и образцы.

### 4.4. Общие рекомендации
- **Движок ASR на CPU:** `whisper-cpp` (процесс, RSS изолирован, Vulkan/CPU) —
  предпочтительнее `faster-whisper` (модель в процессе) на слабой RAM.
- **Движок диаризации:** `sherpa` (ONNX, без torch) или `nemo-speech` (процесс)
  вместо `pyannote` (torch в процессе).
- **GigaAM**: снапшот 4.46 ГБ, но при `GIGAAM_QUANTIZATION=int8` берётся int8-часть
  (`catalog.py:148-163`, `config/defaults.py:252-255`); всё равно грузит весь
  waveform — на 4 ГБ не рекомендуется [ДОПУЩЕНИЕ по runtime RSS].
- **Отключить в первую очередь:** LLM → денойз (если диаризация выключена) →
  диаризацию → `protocol_auto`/образцы.
- Ограничение `WHISPER_CPP_CHUNK_SECONDS` не влияет на Python-декод в
  `_transcribe_chunked` (он всё равно читает весь файл) — экономии RAM не даст, пока
  не сделан O5.

## 5. Что уже сделано (не дублировать)

- **#42 — чанковый денойз** с crossfade: `deepfilter.py:19-27,230-301`; пик
  0.2–0.3 ГБ/час вместо ~1.4 ГБ (`README.md:1131-1141`).
- **Предвыделенные буферы** `WaveformAccumulator` без двойной аллокации:
  `audio.py:25-61`.
- **Единый waveform на прогон** (декод один раз): `pipeline.py:109-136`.
- **#91 — постадийный кэш-store**: `cache/store.py` (атомарные записи, мягкая
  деградация).
- **#50 — переход денойза на CLI** `deep-filter`: вместо Python-пакета
  (прежний ~1.15 ГБ RSS на минуту → ~146-174 МБ): `README.md:1155-1163`.
- **#83 — подпись фактического денойза** в ключах ASR/диаризации:
  `pipeline.py:295-312`, `cache/denoiser.py:51-55`.
- **#94 — сшивка стыков ASR-чанков**: `whisper_cpp_engine.py:417-468,1195-1209`.
- **Кластеризация N×N с одной аллокацией** (in-place): `embeddings.py:339-347`.
- **Потоковая загрузка/отдача**: `web/app.py:4234` (1 МБ), `_file_chunks` `:4516`;
  LRU-кэш длительностей `:4479-4504`.
- **Изоляция в сабпроцессах** для `whisper-cli`, `llama-server`, `nemo-speech`,
  `deep-filter` + реестр процессов/`atexit`/сигналы:
  `utils/subprocess_registry.py`.
- **Закрытие LLM-процесса** после стадии: `llm/postprocess.py:722`,
  `protocol.py:61-87`.
- **Ограниченные буферы stderr**: `subprocess_registry.py:29-32,121`,
  `whisper_cpp_engine.py:1084`.

## 6. Phased delivery summary (bare minimum, JIRA)

> Только предложение (анализ). Стратегия деплоя — **Feature Flags**: все рычаги —
> это уже существующие env/флаги (`config.env`), откат мгновенный.

## PHASE 1: LIGHTWEIGHT_MODE — CONFIG PROFILES
PAT-1 Профили конфигурации «слабое железо» (4/8/16 ГБ)
**Description:**
• Набор документированных пресетов `config.env`: выбор движка+модели+стадий (O1–O3).
• Ноль изменений кода — только конфигурация и README-раздел.
**Blocks:**
└─ — (стартовая задача)
Delivery Commitment (PAT-1):
- Priority: High · Story Points: 2 · Sprint: Phase 1 (Sprint 1)
- Production Ready: «Пользователь выбирает пресет — сразу влезает в целевую RAM»
- Rollback Strategy: «Смена пресета = правка config.env»
- Zero Downtime: «Совместимо с текущим прогоном (применяется к новой задаче)»
- Monitoring: «Лог выбранных движка/модели; ручная проверка RSS»

## PHASE 2: FREE_WAVEFORM — RELEASE TOP CONSUMER
PAT-2 Освобождать `shared_waveform` после enrollment; переиспользовать чанковый декод
**Description:**
• O4: сброс `shared_waveform` после enrollment, повторный декод только для образцов.
• O5: `_transcribe_chunked` читает чанки потоково вместо `load_waveform` всего файла.
• Feature-flag по умолчанию выкл. на первый прогон, затем вкл.
**Blocks:**
└─ PAT-1 (профили уже могут уменьшить нагрузку)
Delivery Commitment (PAT-2):
- Priority: Medium · Story Points: 5 · Sprint: Phase 2 (Sprint 3)
- Production Ready: «−0.23 ГБ/час на длинных файлах»
- Rollback Strategy: «Instant rollback via feature flag»
- Zero Downtime: «Совместимо; стадийный кэш переиспользуется»
- Monitoring: «Сравнение пика RSS и таймингов стадий до/после»

## PHASE 3: MODEL_LIFECYCLE — LAZY UNLOAD / ISOLATION
PAT-3 Выгрузка моделей между стадиями и (опц.) изоляция pyannote/faster-whisper
**Description:**
• O6: явный сброс моделей + `gc.collect` между стадиями.
• O9: (опц.) сабпроцесс для pyannote/faster-whisper по образцу `nemo-speech`.
• Требует архитектурного решения — только после 1–2.
**Blocks:**
└─ PAT-2 (сначала убрать дешёвые пики)
└─ Измерения RSS baseline (внешний блокер: нужен воспроизводимый замер)
Delivery Commitment (PAT-3):
- Priority: Low · Story Points: 13 · Sprint: Phase 3 (Sprint 5)
- Production Ready: «Модель гарантированно освобождается до следующей задачи»
- Rollback Strategy: «Feature flag: in-process vs subprocess»
- Zero Downtime: «Изоляция не меняет контракт движка»
- Monitoring: «RSS процесса-раннера между задачами; alert при росте»

## PHASE 4: HYGIENE — CACHE & STREAMING
PAT-4 Лимит кэша, стриминговая сериализация, лимит истории SSE, субсэмплинг окон
**Description:**
• O7, O8, O11, O12 — гигиена, не блокирует MVP.
• Feature-flag для стриминговой записи результата.
**Blocks:**
└─ PAT-1 (профили как база)
Delivery Commitment (PAT-4):
- Priority: Low · Story Points: 8 · Sprint: Phase 4 (Sprint 7)
- Production Ready: «Диск и пик JSON под контролем на очень длинных записях»
- Rollback Strategy: «Instant rollback via feature flag»
- Zero Downtime: «Совместимо; формат результата не меняется»
- Monitoring: «Размер каталога кэша; рост истории SSE»

## 7. Открытые вопросы / непроверенное [ДОПУЩЕНИЕ]

1. Реальный RSS моделей (whisper.cpp, faster-whisper, pyannote, GigaAM, LLM) в
   процессе **не измерялся** — все оценки RAM помечены.
2. Точное число окон кластеризации для реального часа (влияет на N×N) —
   оценено по окнам 1.5 с (`sherpa_engine.py:73`), не измерено.
3. Возвращает ли GC освобождённую память ОС (torch/CT2 аллокаторы могут
   удерживать арены) — зависит от версии рантайма; требует замера.
4. `pyannote community-1` снапшот в каталоге — 32 МБ, но полный рантайм-потреб
   (torch + подмодели) в каталоге не отражён.
