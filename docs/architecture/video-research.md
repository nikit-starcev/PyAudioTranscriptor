# #121 — Извлечение контента презентации из видео: OCR слайдов + активный говорящий по видеоряду

**Контекст:** локальный офлайн-инструмент (`PyAudioTranscriptor`), видео уже декодируется через **PyAV**.
Железо: **AMD RX 590 (Polaris/gfx803, только Vulkan, без CUDA/ROCm)**, CPU **Ryzen 5 3600 (6c/12t)**.
Слабый профиль проекта ориентирован на **~4 ГБ RAM** (см. `config.example.env` §«Слабое железо» и `docs/architecture/memory*.md`).

> **Главный вывод (кратко):** Технически осуществимо и хорошо ложится на текущий стек. Рекомендуемая схема:
> PyAV/ffmpeg → детект смены сцен → дедуп кадров (pHash/ImageHash) → OCR **RapidOCR (ONNX Runtime, CPU)** как основной (RU+EN), **Tesseract** как «сверхлёгкий» fallback. ASD (активный говорящий) — **опционально и по кнопке** (MediaPipe Face Landmarker как лёгкая эвристика на CPU; Light-ASD/TalkNet — тяжёлый CPU-путь). GPU ускорения для OCR/ASD на этой карте **нет** (ROCm не поддерживает RX 590, у ONNX Runtime нет Vulkan EP) — всё пойдёт на CPU. Коммерческие и OSS-аналоги почти всегда **не** OCR-ят слайды; OCR слайдов делает лишь горстка небольших OSS-проектов.

---

## Карта связей

```
Контент презентации из видео
├── Кадры/слайды
│   ├── Детект смены сцен → ffmpeg scdet / select[1] → PySceneDetect ContentDetector[4]
│   │   └── PyAV уже есть → av.filter.Graph (select/scdet в процессе)[3]
│   └── Дедуп похожих кадров → perceptual hash (imagehash)[6] → hamming
│       ├── PySceneDetect HashDetector (pHash, порог 0.35)[5]
│       └── «стабильный слайд = кадр перед следующей сценой» → эвристика + ручная ревизия[12]
├── OCR офлайн CPU (RU+EN)
│   ├── RapidOCR (ONNX, cyrillic, Apache-2.0)[9] → рекомендуется (лёгкий, RU)
│   ├── Tesseract (rus+eng, Apache-2.0)[7] → самый лёгкий, слабее на разметке
│   ├── PaddleOCR (paddlepaddle — тяжёлый)[8] / EasyOCR (torch)[10] / Surya (650M, torch)[11]
│   └── docTR (torch)[13], tesserocr (wrapper, MIT)[14]
├── Текст на слайдах/схемах
│   ├── Детекция боксов (DB/DBNet) vs полный OCR[8][9] → текст в блоках диаграмм
│   ├── Склейка по слайду → lecture-to-notes ground_slides.py[15]
│   └── Повторы слайдов → OCR token Jaccard[12] + semantic dedup[15]
└── Активный говорящий по видео
    ├── Подсвеченная «плитка» → раскладко-зависимо, типового OSS нет[19]; Recall.ai берёт из API платформы[20]
    ├── Лип-движение → MediaPipe Face Landmarker (blendshape jawOpen и др., CPU)[17]
    └── Модели ASD → TalkNet[21] / Light-ASD[22] / fast-asd[23] (torch, CPU-only у нас)
```

---

## 1. Извлечение кадров/слайдов

### Детект смены сцен
- **ffmpeg `scdet`**: `scdet=t=10` (порог по умолчанию 10); выводит `lavfi.scd.score`/`lavfi.scd.time`. Для фильмов/доков рекомендуют 8–14. Порог `select` — **другая шкала**, старт `select='gt(scene,0.35)'` (0.25–0.3 если режет мало, 0.45–0.5 если много ложных). Замер: полный декод 1080p H.264 → **16.92× realtime на Core i9-14900KF (32 потока)** — то есть скорость лимитируется декодированием каждого кадра. [FFmpeg Scene Detection](https://ffmpeg-cookbook.com/en/articles/scene-detect/) · [FFmpeg `scdet`](https://ffmpeg.org/ffmpeg-filters.html#scdet) · [FFmpeg `select`](https://ffmpeg.org/ffmpeg-filters.html#select_002c-aselect)
- **Caveat:** фейды/диссолвы `scdet` часто пропускает (малое межкадровое различие) — комбинировать с `blackdetect`. [ibid.](https://ffmpeg-cookbook.com/en/articles/scene-detect/)
- **PyAV в процессе:** PyAV умеет запускать libavfilter-графы (`av.filter.Graph`, `add_buffer`), т.е. `select`/`scdet` можно применять без внешнего ffmpeg — важно, т.к. видео у нас уже декодируется PyAV. [PyAV Filters](https://pyav.org/docs/develop/api/filter.html)
- **PySceneDetect (BSD-3-Clause)** — удобнее как библиотека: детекторы `ContentDetector` (HSV, `threshold=27.0`, `min_scene_len=15` кадров по умолчанию), `AdaptiveDetector` (rolling average, `adaptive_threshold=3.0`, `window_width=2`, `min_content_val=15.0`), `ThresholdDetector` (фейды), `HistogramDetector` (Y в YUV), `HashDetector` (pHash между соседними кадрами, `threshold=0.35`, `lowpass=2`). Бэкенды: **opencv**, **pyav (`VideoStreamAv`)**, moviepy, concat — то есть можно работать через наш PyAV. [Detectors](https://www.scenedetect.com/docs/latest/api/detectors.html) · [Backends](https://www.scenedetect.com/docs/latest/api/backends.html) · [Лицензия BSD-3](https://www.scenedetect.com/copyright/)

### Дедуп похожих кадров
- **imagehash (BSD-2)**: aHash / pHash / dHash / wHash / colorhash / crop-resistant; сравнение по **hamming distance**. Зависимости: Pillow + numpy + scipy. [ImageHash · PyPI](https://pypi.org/project/ImageHash/)
- **PySceneDetect HashDetector** уже содержит pHash-логику (DCT, бинаризация по медиане) — не обязательно тянуть imagehash отдельно. [Detectors](https://www.scenedetect.com/docs/latest/api/detectors.html)
- **SSIM** (scikit-image/`structural_similarity` или OpenCV) — как дополнительный порог для «почти совпадающих» кадров.

### Выбор «слайда целиком» и подводные камни (важно!)
Практика показывает, что **ни один простой признак не работает на шаблонных слайдах**:
- PySceneDetect по умолчанию (HSV-гистограмма) — **шаблон слайдов почти не сдвигает гистограмму**;
- **pHash в одиночку** — «тот же шаблон обманывает хеш»;
- **SSIM в одиночку** — та же проблема;
- **OCR Jaccard в одиночку** — соседние слайды по теме делят лексику.

Рабочая двухуровневая схема из `slide-extractor`: (1) **онлайн-кластеризация контента** — каждый сэмплированный кадр OCR-ится, кадры кластеризуются по **token Jaccard**, новый слайд открывается, когда пересечение с предыдущими ниже порога; (2) **ручная ревизия** (contact sheet) для остаточных ~5% (анимации/переходы). Заявленная точность auto ≈ **95%**. [liaw-boy/slide-extractor](https://github.com/liaw-boy/slide-extractor)
- Выбор «полного слайда»: брать кадр **перед** следующей сменой сцены (после доигрывания анимаций) — у `slide-extractor` это «animation-aware transition detection». [ibid.](https://github.com/liaw-boy/slide-extractor)
- **Демонстрация экрана**: эти подходы нацелены именно на **screen-recording слайдов** (PowerPoint/Keynote/Google Slides, конференции со скрин-кастом). Явно **не** работают: whiteboard/рукописное, software-demo скринкасты, чистый talking-head, Prezi-зумы, видео <30 с. [ibid.](https://github.com/liaw-boy/slide-extractor)
- **Периодическая выборка** (напр. 1 кадр / 2–5 с) — самый дешёвый стартовый вариант для слабого CPU; нарезка по сценам точнее, но требует полного декода.

---

## 2. OCR офлайн на CPU, RU+EN

### Сравнительная таблица движков

| Движок | Лицензия | Backend/зависимости | RU | Размер/вес | CPU-скорость (из открытых сравнений) | Вердикт для нас |
|---|---|---|---|---|---|---|
| **Tesseract 5** | Apache-2.0 | C++ `libtesseract` + CLI; `tessdata` (`rus.traineddata`, `eng.traineddata`) | ✅ `rus` | core ~10 МБ; `tessdata_fast`: eng 4.1 МБ, rus 3.86 МБ, osd 10.5 МБ; runtime 50–200 МБ | **чистый документ ~1.5 с, сложная вёрстка ~2.5 с, scene-текст ~3.5 с** (точность на scene-text плохая) | **Самый лёгкий fallback**; отлично на чистых слайдах, слабее на разметке |
| **RapidOCR** | Apache-2.0 | `pip install rapidocr onnxruntime`; ONNX/OpenVINO/MNN/TensorRT/Paddle/PyTorch; модели PaddleOCR→ONNX | ✅ `cyrillic` (PP-OCRv5/v3 rec mobile) | ONNX Runtime wheel + модели ~10–20 МБ | Mobile-модели в ONNX, быстрее полного Paddle-пайплайна на CPU | **Рекомендуемый основной** (лёгкий, RU, совпадает с уже используемым ONNX Runtime) |
| **PaddleOCR** | Apache-2.0 | `paddleocr` + **`paddlepaddle`** (тяжёлый); PP-OCRv4/v5/v6; CPU/GPU/XPU/NPU | ✅ (PP-OCRv5 multiling, 109 языков, cyrillic) | модели 10–20 МБ/язык; CPU runtime 200–500 МБ + сам paddle | **CPU ~4–5 с/изображение**; GPU (RTX 3080) 0.8–1.2 с | Избыточен по зависимостям; RapidOCR даёт те же модели в ONNX |
| **EasyOCR** | Apache-2.0 | **PyTorch** (CRAFT + CRNN); `gpu=False` для CPU | ✅ (Cyrillic) | модели ~**5 ГБ** (у `slide-extractor`) | torch на CPU медленнее ONNX | Тяжёл по диску/памяти; не под слабый CPU |
| **docTR** | Apache-2.0 (код) | **PyTorch**, two-stage (детекция+распознавание), раньше Mindee, теперь t2k | ✅ (латиница/кириллица через модели) | тянет torch + веса | CPU медленно | Дублирует уже имеющийся у нас torch; избыточно |
| **Surya (datalab)** | код Apache-2.0, **модель OpenRAIL-M** | **torch≥2.7 + transformers≥5.12 + torchvision** | ✅ (91 язык) | **650M параметров** | 5 стр/с на RTX 5090 (**GPU**); на CPU очень тяжело | Только как «третий, точный» путь по кнопке, не для слабого CPU |
| **tesserocr** | MIT | Python-обёртка над Tesseract (C API), без CLI-спавна | ✅ (через tessdata) | как Tesseract | как Tesseract, чуть меньше оверхеда | Если не хочется спавнить CLI |

Источники: [Tesseract](https://github.com/tesseract-ocr/tesseract) · [tessdata](https://github.com/tesseract-ocr/tessdata) · [tessdata_best (rus/eng/ukr)](https://github.com/tesseract-ocr/tessdata_best) · [tesserocr · PyPI](https://pypi.org/project/tesserocr/) · [RapidOCR](https://github.com/RapidAI/RapidOCR) · [RapidOCR model list (cyrillic)](https://rapidai.github.io/RapidOCRDocs/main/model_list/) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [EasyOCR](https://github.com/JaidedAI/EasyOCR) · [docTR](https://github.com/mindee/doctr) · [Surya](https://github.com/VikParuchuri/surya) · [surya-ocr deps · PyPI](https://pypi.org/pypi/surya-ocr/json) · [PaddleOCR vs Tesseract (тайминги/точность/memory)](https://mljourney.com/paddleocr-vs-tesseract-comprehensive-comparison-for-ocr-implementation/)

**Ключевые цифры точности** (открытое сравнение): на чистых документах Tesseract и PaddleOCR дают **95–99%**; на scene-text (ICDAR2015) PaddleOCR **85–90%** против Tesseract **60–70%**. [mljourney](https://mljourney.com/paddleocr-vs-tesseract-comprehensive-comparison-for-ocr-implementation/)

**Что реально стоит брать:** **RapidOCR** (основной, ONNX Runtime CPU, cyrillic-модель) + **Tesseract `rus+eng`** (fallback). Оба — Apache-2.0, без torch, десятки МБ. Готовый референс именно такого выбора — `lecture-to-notes`, где **RapidOCR — baseline Stage B**, **Surya — опциональный Stage B2**, а PaddleOCR-VL — резервный (в отдельных venv, чтобы torch/paddle не конфликтовали). [drpwchen/lecture-to-notes](https://github.com/drpwchen/lecture-to-notes) · [ocr_bench](https://github.com/drpwchen/lecture-to-notes/tree/main/ocr_bench)

> **Важно про бенчмарки:** `lecture-to-notes` специально держит `ocr_bench`, потому что «движки, выигрывающие на документных бенчмарках, регулярно разваливаются на слайдах» (сонограммы, MRI, плотный CJK, декоративные титулы). Выбирать движок на **своих** слайдах. [ocr_bench README](https://github.com/drpwchen/lecture-to-notes/tree/main/ocr_bench)

---

## 3. Текст на слайдах/схемах

- **Детекция текста vs полный OCR:** RapidOCR/PaddleOCR/docTR = **детекция боксов (DB/DBNet) + распознавание** → находят текст в произвольных местах, в т.ч. **внутри блоков диаграмм**; Tesseract делает rule-based layout analysis, слабее на нестандартной разметке («текст на любой угол», «curved text» — сильная сторона детекции на нейросетях). [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [mljourney](https://mljourney.com/paddleocr-vs-tesseract-comprehensive-comparison-for-ocr-implementation/)
- **Колонки/вёрстка:** у Tesseract PSM-режимы: `3` (auto), `6` (единый блок), `11` (sparse — «найти как можно больше текста»), `13` (одна строка). Для слайдов часто полезен `--psm 11`/`6`. [mljourney](https://mljourney.com/paddleocr-vs-tesseract-comprehensive-comparison-for-ocr-implementation/)
- **Склейка текста по слайду и привязка к речи:** `lecture-to-notes` хранит для каждого канонического слайда его OCR-текст **и** сегменты транскрипта, произнесённые пока слайд был на экране (`ground_slides.py`) → получается `slides_grounded.json`. [lecture-to-notes](https://github.com/drpwchen/lecture-to-notes)
- **Повторы слайдов:** сценарии — токен-Jaccard кластеризация (см. §1) + семантический дедуп (`dedup_semantic.py`). [slide-extractor](https://github.com/liaw-boy/slide-extractor) · [lecture-to-notes](https://github.com/drpwchen/lecture-to-notes)
- **Шум (веб-камеры/люди):** один из OSS-проектов явно выкидывает кадры с лицами (face recognition) и оставляет только слайды + дедуп + извлечение текста. [SamrajSingh1/Slides-Extraction-from-Video](https://github.com/SamrajSingh1/Slides-Extraction-from-Video)
- **Разбиение/порядок чтения:** Surya даёт layout analysis (table/image/header) и reading order — полезно, если нужны «схемы/таблицы» как структура, но это тяжёлый путь. [Surya](https://github.com/VikParuchuri/surya)

---

## 4. Активный говорящий по видео (ASD)

### 4a. Подсвеченная «плитка» в раскладках ВКС
- Типового **готового OSS-детектора подсветки активного участника** (рамка/бордер в Zoom/Teams/Meet) найти **не удалось** — это раскладочно-зависимая эвристика (анализ бордера/яркости фиксированных регионов). **Помечаю как неподтверждённое.**
- Индустриальный путь — **не** визуальный: попасть в API платформы и получить события активного говорящего. Recall.ai отдаёт **speaker timeline** (`speech_on`/`speech_off`): для **Zoom** события идут по underlying audio (чаще), для **остальных платформ** — только на **смену** говорящего; из screen-share говорящий не эмитится. [Recall.ai Speaker Timelines](https://docs.recall.ai/docs/speaker-timeline)
- Для офлайн-инструмента (без API платформы) остаётся визуальная эвристика по плиткам, либо отказ от неё в пользу аудио-диаризации.

### 4b. Face detection + движение губ (лёгкий путь, CPU)
- **MediaPipe Face Landmarker** (Apache-2.0): 3D-landmarks лица + **52 blendshape-коэффициента** (подтверждены имена `jawOpen`, `mouthClose`, `mouthFunnel`, `mouthPucker`, `mouthSmileLeft`) + матрица трансформации; режимы IMAGE/VIDEO/LIVE_STREAM, `num_faces`, пороги confidence; модели TFLite, идут на **CPU**. Это дешёвый сигнал «губы шевелятся» для плитки-кандидата. [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) · [имена blendshape в коде](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc)

### 4c. Готовые аудио-визуальные модели ASD

| Модель | Венue/год | Метрика | Backend | Доп. требования | Применимость на AMD без CUDA |
|---|---|---|---|---|---|
| **TalkNet** | ACM MM 2021 | mAP **92.3** val / **90.8** test (AVA) | PyTorch (15.01M params) | torch, torchaudio, scenedetect, opencv, `python_speech_features`; для «в дикой природе» — **face detection S3FD** | **Только CPU** (torch CPU). Медленно, но работает |
| **Light-ASD** | CVPR 2023 | mAP **94.06** val; F1 **95.5%** (Columbia, TalkSet-веса) | PyTorch | torch, cv2, `python_speech_features`, scenedetect, **S3FD** | Только CPU |
| **LR-ASD** | IJCV 2025 | «lightweight & robust», расширение Light-ASD | PyTorch | как выше | Только CPU |
| **fast-asd** (Sieve) | production | оптимизированный TalkNet; variable FPS | PyTorch | отдельная standalone face detection (YOLO) | Только CPU; часть функций требует Sieve API |

Источники: [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) · [TalkNet requirements](https://github.com/TaoRuijie/TalkNet-ASD/blob/main/requirement.txt) · [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) · [LR-ASD](https://github.com/Junhua-Liao/LR-ASD) · [fast-asd](https://github.com/sieve-community/fast-asd) · другие (ASDNet, lookwhostalking, SPELL, TS-TalkNet): [GitHub search «active speaker detection»](https://github.com/search?q=active+speaker+detection&type=repositories) и [awesomeASD](https://github.com/TaoRuijie/TalkNet-ASD/blob/main/awesomeASD.md)

### 4d. Ограничения железа (критично)
- **ROCm на RX 590 не работает.** ROCm 7.2.1 поддерживает Radeon 9000 (RDNA4) и часть 7000 (RDNA3) + Ryzen APU; **Polaris (RX 590/gfx803) в список не входит**. [ROCm Radeon/Ryzen compatibility](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html)
- **У ONNX Runtime нет Vulkan Execution Provider** (открытый feature request), т.е. OCR (RapidOCR/Paddle) и любые ONNX-модели на этой карте ускорить нельзя. [onnxruntime issue #21917](https://github.com/microsoft/onnxruntime/issues/21917)
- Вывод: **OCR и ASD считаем строго на CPU** (torch CPU уже есть в проекте; для OCR torch не нужен вовсе).
- **Как подсказка к аудио-диаризации:** ASD даёт «кто говорит» во времени; при перекрытиях/ошибках pyannote его можно использовать как мягкий prior (не единственный источник истины). Полностью опираться на ASD в ВКС-раскладках рискованно (см. 4a).

---

## 5. Аналоги/практика

**Коммерческие (meeting notes):**
- **Recall.ai** — отдаёт **раздельные** аудио/видео потоки по участникам, «100% diarized с именами», данные screen-share, метаданные; speaker timeline из API платформы. [Recall.ai Transcription API](https://www.recall.ai/product/meeting-transcription-api) · [Speaker Timelines](https://docs.recall.ai/docs/speaker-timeline)
- **Otter / Fireflies / tl;dv** — наличие **OCR слайдов** подтвердить **не удалось** (поиск блокировался), помечаю как **непроверенное**.

**OSS-транскрибация (без OCR слайдов):**
- **WhisperX** — word-level timestamps (wav2vec2 alignment), VAD, диаризация pyannote; faster-whisper <8 ГБ GPU для large-v2; CUDA-ориентирован, CPU возможен. **OCR слайдов нет.** [m-bain/whisperX](https://github.com/m-bain/whisperX)
- **Buzz** (MIT) — офлайн Whisper, **Vulkan-ускорение whisper.cpp** (дружелюбно к AMD), speaker ID. **OCR слайдов нет.** [chidiwilliams/buzz](https://github.com/chidiwilliams/buzz)
- **Meetily** — privacy-first локальный ассистент, сборка Whisper на **Vulkan** (AVX2 CPU; CUDA только через сборку под NVIDIA). **OCR слайдов нет.** [Zackriya-Solutions/meeting-minutes](https://github.com/Zackriya-Solutions/meeting-minutes)

**OSS, которые реально OCR-ят слайды (наши прямые аналоги):**
- **drpwchen/lecture-to-notes** (108★) — Whisper ASR + `extract_slides` (кадры+perceptual dedup) + `quick_ocr` (RapidOCR) + `dedup_semantic` + `ocr_surya` (опц.) + `vlm_signals` (локальная VLM) + `ground_slides`; офлайн, stage-основанный; отдельный `ocr_bench` для подбора движка. [repo](https://github.com/drpwchen/lecture-to-notes)
- **liaw-boy/slide-extractor** (MIT*) — EasyOCR + pHash, animation-aware, web-review UI, auto ≈95%. *Только для личного изучения по лицензии/notice. [repo](https://github.com/liaw-boy/slide-extractor)
- **Adityakeshav/youtube-slide-extractor**, **SamrajSingh1/Slides-Extraction-from-Video**, **lybhb8/video-to-pptx**, **r0mar1n/LectureLens** — небольшие (1–11★) пайплайны «видео → уникальные слайды → OCR → PPTX/заметки». Найдено через [GitHub search](https://github.com/search?q=slide+extraction+video&type=repositories).

**Что пишут про сложность/ценность:** мотивация `lecture-to-notes` — «никто на самом деле не пересматривает часы видео»; цель — **traceability** (каждое утверждение привязывается к моменту в транскрипте и слайду на экране), а дешёвые стадии (A–E: кадры/OCR/дедуп/привязка) стоят **0 токенов LLM**. Это ровно та ценность, которую можно закрыть локально. [lecture-to-notes](https://github.com/drpwchen/lecture-to-notes)

---

## 6. Итог по осуществимости

**Вердикт: ОСУЩЕСТВИМО, приоритет — OCR слайдов (высокая ценность/низкая цена), ASD — опционально по кнопке.** Всё на CPU; GPU ускорение на RX 590 для OCR/ASD нереализуемо.

### Рекомендуемый набор
| Слой | Что брать | Лицензия | Зависимости | Почему |
|---|---|---|---|---|
| Кадры | **PyAV** (уже есть) + `av.filter.Graph` со `select`/`scdet`, **или** `PySceneDetect` (backend `pyav`) | BSD-3 (PySceneDetect) | — / scenedetect | Не добавляет тяжёлое; PyAV уже декодирует |
| Дедуп | **ImageHash** (pHash/dHash, hamming) + опц. SSIM | BSD-2 | Pillow, numpy, scipy | Лёгкий, детерминированный |
| OCR | **RapidOCR** (ONNX RT, `cyrillic`+`eng`) — основной; **Tesseract `rus+eng`** (`tessdata_fast`) — fallback | Apache-2.0 / Apache-2.0 | onnxruntime / CLI+данные | Без torch; RU+EN; совпадает с уже используемым ONNX Runtime (`sherpa`, `onnx-asr`) |
| Слайды | Кластеризация по OCR token-Jaccard + «кадр перед сменой сцены» | — | — | Практика: простая гипотеза «pHash/SSIM/HSV» на шаблонных слайдах не работает |
| ASD (опция) | **MediaPipe Face Landmarker** (blendshapes, CPU) как эвристика; **Light-ASD/TalkNet** — тяжёлый путь | Apache-2.0 / MIT(Apache?) | mediapipe / torch+S3FD | Лёгкая эвристика на CPU; тяжёлое — по кнопке |

### Примерная стоимость на CPU (условно, порядок величины)
- **Декод/детект сцен**: полный проход декодирует все кадры. Ориентир 16.92× realtime — на 32-потоках i9; на Ryzen 5 3600 (12 потоков) при 1080p ожидаемо **~4–8× realtime** (точную цифру **на нашем железе не измерял**). При периодической выборке декод дешевле.
- **OCR/кадр**: Tesseract ≈ **1.5–3.5 с** (по типу кадра); RapidOCR (mobile-модели в ONNX) обычно **быстрее**, порядок ~**0.3–1 с/кадр** — точное число для RapidOCR **подтвердить не удалось** (публичного свежего ms-бенчмарка не нашёл).
- **Рычаги экономии:** (1) OCR только по **каноническим/стабильным** кадрам (после дедупа — обычно единицы-десятки слайдов на час, а не тысячи кадров); (2) **downscale** кадра перед OCR; (3) периодическая выборка вместо покадрового прохода; (4) crop к области слайда (отсечь веб-камеры/панели).

### Размер зависимостей
- **Лёгкий путь:** RapidOCR (onnxruntime ~ десятки МБ + модели ~10–20 МБ) + Tesseract (~10 МБ + ~4 МБ/язык) + ImageHash/Pillow/numpy/scipy. **Без torch.**
- **Тяжёлый путь (по кнопке):** Surya (torch≥2.7, transformers, 650M), EasyOCR (~5 ГБ моделей), PaddleOCR (paddlepaddle), либо ASD (torch+S3FD). Всё это резко поднимает RAM — только опционально.

### Что опционально / по кнопке
- **По умолчанию:** только кадры+дедуп+OCR (RapidOCR) → текст слайдов, привязанный к таймлайну транскрипта (аналогично `lecture-to-notes`).
- **По кнопке:** точный OCR (Surya/PaddleOCR-VL) в отдельном venv; ASD (MediaPipe-эвристика; при желании Light-ASD/TalkNet на CPU); экспорт в PPTX/заметки.

### Что НЕ удалось подтвердить
1. Точная **CPU-скорость RapidOCR** (ms/кадр) — свежего бенчмарка не нашёл (repo описывает только «extreme speed»).
2. **OCR слайдов у Otter/Fireflies/tl;dv** — не подтверждено (поиск блокировался Cloudflare/капчей DDG/Bing).
3. Единый **OSS-детектор подсвеченной плитки** ВКС — не существует типового; это раскладочная эвристика.
4. Точные требования RAM/VRAM у TalkNet/Light-ASD **на CPU** (README даёт метрики качества, не латентность CPU).
5. Наличие **LoCoNet/новейших SOTA** ASD — не проверял по первоисточникам (только через awesome-списки/поиск).

---

## Confidence
- **High:** движки OCR, лицензии/зависимости, детекторы и пороги PySceneDetect/ffmpeg, метрики TalkNet/Light-ASD, blendshape-имена MediaPipe, ограничения AMD (ROCm/ONNX Vulkan EP), факты Recall.ai, наличие OSS-аналогов.
- **Medium:** сквозная оценка CPU-стоимости на нашем железе (экстраполяция, не замер); практическая точность «склейки» слайдов на конкретных записях проекта.
- **Low / не подтверждено:** CPU-скорость RapidOCR в ms; OCR слайдов у коммерческих meeting-нот; наличие типового OSS-детектора активной плитки.

> Нужен полный список источников — напиши **«sources»**.
