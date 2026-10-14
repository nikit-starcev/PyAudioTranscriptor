# Профиль оперативной памяти конвейера PyAudioTranscriptor

**Дата:** 2026-10-14 · **Файл-вход:** `вход.webm` (18 МБ, ~121 с речи, скопирован в `/tmp/opencode/input.webm`)
**Репозиторий:** `/home/user/LLM/установка транскрибера и ui/PyAudioTranscriptor` (venv `.venv`), **файлы репозитория не изменялись**.
**Артефакты профилирования:** `/tmp/opencode/mem-profile/` (samples/stages/procpeak/summary на каждый прогон), логи CLI — `/tmp/opencode/runs/<label>/logs/`.

---

## 1. Железо и окружение

| Параметр | Значение |
|---|---|
| CPU | AMD Ryzen 5 3600, 6 ядер / 12 потоков |
| RAM | 31 GiB (+ swap 23 GiB) |
| GPU | AMD Radeon RX 590, 8 ГБ VRAM, драйвер RADV (Vulkan 1.4) |
| ASR-бэкенд | `whisper-cpp`, модель `ggml-large-v3-turbo.bin` (1.62 ГБ), чанкинг 30 с / перекрытие 2 с → **5 вызовов whisper-cli** |
| Диаризация | `auto` → сработал **hybrid** (оценщик увидел ≈7 говорящих > cap 4): nemo-speech (Vulkan) + sherpa-onnx эмбеддинги |
| LLM | llama-server, `qwen2.5-7b-instruct-q4_k_m`, `-ngl 99` → 29/29 слоёв на GPU |
| Денойз | **не выполнен:** бинарник `deep-filter` (DeepFilterNet) в системе отсутствует → стадия мягко пропущена (в логе: «deep-filter не найден … шумоподавление будет пропущено») |

> Важно: «тяжёлый» прогон формально с `DENOISE=true`, но реального денойза не было (нет бинарника). Стоимость денойза по памяти **не измерена**; фактически стадия = no-op.

### Инструмент измерения
`/tmp/opencode/run_profile.py` — мониторит **дерево процессов** (`/proc/<pid>/status` `VmRSS`, суммарно Python + дети) и метки стадий из вывода CLI; для прогонов под лимитом дополнительно читает cgroup systemd-скоупа (`memory.current`/`memory.peak`/`memory.events`). `psutil` нет — всё через `/proc`. Логи CLI в этом окружении `rich` печатает в **stdout**, поэтому harness объединяет stdout+stderr.

---

## 2. Команды прогонов

Все прогоны — из-под CLI (`audio-transcriber transcribe`), cwd `/tmp/opencode` (там свой `config.env`, чтобы не трогать репозиторий), `--no-cache` (считать всё заново), `--no-notify`.

```bash
# (H) ТЯЖЁЛЫЙ: denoise + diarization(auto→hybrid) + word-timestamps + LLM 7B
audio-transcriber transcribe /tmp/opencode/input.webm \
  --output-dir /tmp/opencode/runs/heavy --no-cache --no-notify

# (L) ЛЁГКИЙ: только ASR (без денойза/диаризации/LLM), word-timestamps по умолчанию
#     cwd=/tmp/opencode/light, где config.env с LLM_ENABLED=false (флага --no-llm нет)
audio-transcriber transcribe /tmp/opencode/input.webm \
  --output-dir /tmp/opencode/runs/light \
  --no-cache --no-notify --no-denoise --no-diarization --no-speaker-samples

# (C) CPU-only ASR: whisper-cli с -ng (обёртка) + --device cpu
audio-transcriber transcribe /tmp/opencode/input.webm \
  --output-dir /tmp/opencode/runs/cpu \
  --no-cache --no-notify --no-denoise --no-diarization --no-speaker-samples \
  --device cpu --whisper-cpp-binary /tmp/opencode/bin/whisper-cli-cpu

# (ограничение RAM) пример — CPU-ASR под 1.8 ГБ:
systemd-run --user --scope --unit=memprof-cpu18g -p MemoryMax=1800M -p MemorySwapMax=0 \
  audio-transcriber transcribe /tmp/opencode/input.webm ... --device cpu ...
```

Обёртка `/tmp/opencode/bin/whisper-cli-cpu` = `exec whisper-cli -ng "$@"` (штатный pipeline флаг `-ng` не пробрасывает).

---

## 3. Пиковая RSS по стадиям

### 3.1 Тяжёлый прогон (H) — пик 5369 МБ, стена ~80 с

| стадия | длит., с | RSS Python, МБ | RSS детей, МБ | суммарный пик, МБ | кто даёт пик |
|---|---:|---:|---:|---:|---|
| denoise | 0.0 | ~527 | 0 | ~527 | нет бинарника deep-filter → no-op |
| **ASR** | 29.1 | 527 | 190 | **718** | whisper-cli |
| **diarization** | 32.3 | 721 | 106 | **828** | nemo-speech (+ sherpa эмбеддинги в Python) |
| clean/merge | ~0.1 | 777 | 0 | 777 | Python |
| **LLM** | 15.4 | 780 | **4589** | **5369** | llama-server |
| export | 0.1 | 772 | 0 | 772 | Python |
| **ОБЩИЙ ПИК** | | 801 | 4589 | **5369** | llama-server |

### 3.2 Лёгкий прогон (L) — только ASR, пик 718 МБ, стена ~31 с

| стадия | длит., с | RSS Python, МБ | RSS детей, МБ | суммарный пик, МБ |
|---|---:|---:|---:|---:|
| **ASR** | 28.4 | 527 | 190 (whisper-cli) | **718** |
| export/done | ~1 | 533 | 0 | 533 |
| **ОБЩИЙ ПИК** | | 533 | 190 | **718** |

### 3.3 CPU-only ASR (C) — пик 1961 МБ, стена ~264 с

| стадия | длит., с | RSS Python, МБ | RSS детей, МБ | суммарный пик, МБ |
|---|---:|---:|---:|---:|
| **ASR** | 263.5 | 92 | **1869** (whisper-cli, CPU) | **1961** |
| export/done | ~0.2 | 85 | 0 | 85 |
| **ОБЩИЙ ПИК** | | 92 | 1869 | **1961** |

> `--device cpu` убирает импорт `torch` → Python-процесс падает с ~527 до ~92 МБ. При этом whisper.cpp **всё равно** идёт на GPU (Vulkan) — флаг `--device` влияет только на torch-части (faster-whisper/pyannote), но не на whisper-cli/nemo-speech.
> Скорость: CPU-ASR ~264 с против ~29 с на Vulkan для одного и того же файла — **≈9× медленнее**.

---

## 4. Разбор пиков (кто и сколько берёт)

Прямые микрозамеры (30-с кусок, `RssAnon` / `RssFile` / VRAM):

| процесс / режим | peak RSS | RssAnon | RssFile | VRAM (Δ) | время |
|---|---:|---:|---:|---:|---:|
| whisper-cli **GPU** (Vulkan) | 164 МБ | 115 МБ | 50 МБ | **~1554 МБ** | 5.2 с |
| whisper-cli **CPU** (`-ng`) | **1863 МБ** | **1821 МБ** | 42 МБ | ~0 | 51.5 с |
| llama-server 7B Q4 **GPU** (`-ngl 99`) | 4590 МБ | **60 МБ** | **4530 МБ** | **~4066 МБ** | загрузка+idle |
| llama-server 0.5B Q4 **CPU** (`-ngl 0`) | 636 МБ | 108 МБ | 528 МБ | 0 | — |
| Python: `resolve_device(auto)` (импорт torch) | **~527 МБ** | — | — | — | 1.15 с |
| Python: `--device cpu` (без torch) | ~92 МБ | — | — | — | — |

**Выводы по пикам:**
- **ASR.** Модель whisper.cpp на GPU живёт в **VRAM** (~1.55 ГБ), в RAM остаётся ~115 МБ. На CPU вся модель (1.62 ГБ) — в анонимной памяти (~1.82 ГБ RSS). Именно поэтому CPU-прогон и упирается в ~1.9 ГБ.
- **Диаризация.** nemo-speech — маленький процесс (~100 МБ, основное — VRAM). Дороже обходится **hybrid-маршрутизация**: она поднимает в самом Python sherpa-onnx эмбеддинги (+~200 МБ к Python, 527→721 МБ).
- **LLM.** llama-server показывает RSS ~4.5 ГБ, но это почти целиком **`RssFile`** — memory-mapped файл весов (страницы page cache, разделяемые/вытесняемые), приватной анонимной памяти всего ~60 МБ. Реально модель занимает **~4.1 ГБ VRAM** (при `-ngl 99`). То есть «пик 5.37 ГБ» из таблицы 3.1 сильно переоценивает давление на RAM — но эти страницы обязаны быть резидентными (иначе бесконечные перечитывания с диска).
- **Базовая стоимость Python.** `resolve_device(Device.AUTO)` лениво импортирует `torch` → **+~435 МБ и ~1.1 с** на каждый запуск, даже когда ASR идёт через whisper.cpp и torch не нужен.
- **Освобождение памяти.** Дочерние процессы освобождают RSS **полностью** при выходе: после последнего whisper-cli суммарная RSS падает 630→520 МБ; после остановки llama-server — 1224→771 МБ. RSS **самого Python не уменьшается**: он монотонно растёт 527→721→800 МБ (torch + модели диаризации) и держится до конца прогона; освобождается только при завершении процесса.

---

## 5. Прогоны под ограничением RAM (cgroup, `MemorySwapMax=0`)

| прогон | лимит | результат | стадия/время сбоя | пик RSS / cgroup | детали |
|---|---:|---|---|---:|---|
| CPU-ASR | **2.0 ГБ** | ✅ прошёл | — | 1961 / ~2048 МБ | впритык |
| CPU-ASR | **1.8 ГБ** | ❌ **OOM** | ASR, 26.7 с | 1869 / 1798 МБ | `oom_kill=1`, rc=-15, убит whisper-cli |
| GPU + `--device auto` | 600 МБ | ✅ прошёл | — | 718 / 438 МБ | страницы весов вытесняемы |
| GPU + `--device auto` | **400 МБ** | ❌ **SIGKILL** | старт ASR, 2.7 с | 604 / 321 МБ | Python уже 527 МБ (torch) > 400 |
| GPU + `--device cpu` | 512 МБ | ✅ прошёл | — | 283 / 206 МБ | |
| GPU + `--device cpu` | **256 МБ** | ✅ прошёл | — | 283 / **205 МБ** | минимальный рабочий профиль |

**Где падает:** критическая точка — **стадия ASR**. На CPU потолок ~1.9–2.0 ГБ (модель в RAM). С `--device auto` ломается уже на старте из-за импорта torch (~527 МБ): лимит 400 МБ → SIGKILL за 2.7 с.

---

## 6. Что реально нужно слабому железу

**Минимальные профили (файл 2 мин, whisper large-v3-turbo):**

| Сценарий | RAM (host) | VRAM | Комментарий |
|---|---:|---:|---|
| **GPU, только ASR, `--device cpu`** | **~0.21–0.3 ГБ** | ~1.6 ГБ | самый экономный; влезает в cgroup 256 МБ |
| GPU, только ASR, `--device auto` | ~0.44–0.72 ГБ | ~1.6 ГБ | лишние ~0.4 ГБ на torch |
| GPU + диаризация (hybrid) | ~0.8–1.0 ГБ | +немного | +~200 МБ Python (sherpa) + nemo |
| **CPU, только ASR** | **~1.9–2.0 ГБ** | 0 | без GPU-варианта |
| CPU + LLM 7B Q4 (`-ngl 0`) | +~4.5 ГБ | 0 | 7B — тяжело для слабого ПК |
| GPU + LLM 7B Q4 (`-ngl 99`) | +~0.06 ГБ aнон | **+~4.1 ГБ** | в 8 ГБ VRAM ещё влезает вместе с ASR |

**Практические выводы:**
1. **Память ASR определяется бэкендом, а не размером файла.** whisper.cpp на GPU = ~0.12 ГБ RAM + 1.6 ГБ VRAM; на CPU = ~1.85 ГБ RAM. Для 1–2-минутного файла чанкинг (5×30 с) не увеличивает пик — whisper-cli-процессы последовательны.
2. **`--device cpu` — важная ручка для слабого ПК:** убирает импорт torch (−435 МБ) и всё равно оставляет whisper.cpp/LLM на Vulkan. Но на машине без рабочего GPU это не спасёт — там ASR упрётся в ~2 ГБ RAM.
3. **Осторожно с оценкой по RSS:** у llama-server 4.5 ГБ — это mmap весов (`RssFile`), приватной памяти ~60 МБ; основной расход — **VRAM (~4 ГБ)**. При нехватке RAM эти страницы вытесняются (ценой скорости), но при нехватке VRAM GPU-инференс деградирует/падает.
4. **Денойз (DeepFilterNet) не измерен** — бинарник отсутствует. На слабом ПК это ещё один внешний процесс; в текущем виде стадия просто пропускается.
5. **Рекомендуемые требования:** GPU-ASR-only — от 1 ГБ RAM + 2 ГБ VRAM; с диаризацией — 4 ГБ RAM; CPU-only ASR — **не менее 4 ГБ RAM** (пик ~2 ГБ + запас на ОС/браузер); с LLM 7B — 16 ГБ RAM (CPU) или 8 ГБ VRAM (GPU); для совсем слабых — LLM 0.5B (~0.6 ГБ) вместо 7B.
6. **Возможная оптимизация (за рамками задачи):** не импортировать torch в `resolve_device`, когда бэкенд — whisper-cpp (экономия ~435 МБ и ~1 с на каждый прогон).

---

## 7. Приложения

- Harness: `/tmp/opencode/run_profile.py` (сбор RSS по дереву `/proc`, метки стадий, опц. cgroup)
- Микрозамеры: `/tmp/opencode/memstat_run.py`
- Сводки прогонов: `/tmp/opencode/mem-profile/<label>.summary.json` + `.samples.csv`, `.stages.csv`, `.procpeak.csv`, `<label>.stderr.log`
- Логи CLI: `/tmp/opencode/runs/<label>/logs/audio-transcriber_*.log`
- Метки: `heavy`, `light`, `cpu_asr`, `cpu_2g`, `cpu_18g`, `gpu_dcpu_512`, `gpu_auto_600`, `gpu_auto_400`, `gpu_dcpu_256`

**Ограничения:** один короткий файл (~2 мин), по одному прогону на конфиг (без статистики разброса); денойз не измерен (нет бинарника); RV590/драйвер один.
