# Сборка portable-бандла

Каталог содержит всё необходимое для сборки автономного one-dir бандла
`audio-transcriber` (CLI + веб-интерфейс) через PyInstaller.

## Быстрый старт

```bash
python scripts/build_portable.py --target linux --out ./dist-portable
```

Результат — каталог `./dist-portable/audio-transcriber/` с исполняемым файлом
`audio-transcriber` (на Windows — `audio-transcriber.exe`). Внутри лежат
Python-код проекта, зависимости, собранный SPA (`web/static`) и данные
словарей/VAD.

Флаги:

- `--target {linux,windows,macos}` — целевая ОС. **Обязана совпадать с
  host-ОС**: PyInstaller не кросс-компилирует; в CI runner совпадает с
  matrix-ОС. При несовпадении скрипт падает с понятной ошибкой.
- `--out <DIR>` — куда положить бандл.
- `--clean` — удалить прежний бандл и build-окружение перед сборкой.

## Как готовится окружение

Скрипт не трогает основное окружение репозитория. Он создаёт изолированный
venv (`<OUT>/.build/venv`) через `uv` и ставит туда:

```bash
uv pip install --python <venv>/bin/python --no-sources --torch-backend cpu "<repo>[web]" pyinstaller
```

- `--no-sources` — игнорирует `[tool.uv.sources]` из `pyproject.toml`, который
  на Linux/Windows тянет CUDA-сборку PyTorch (`cu126`). Для portable-бандла
  CUDA не нужна, поэтому `--torch-backend cpu` даёт лёгкую CPU-сборку и не
  тянет пакеты `nvidia-*`.
- extra `web` — FastAPI/uvicorn/python-multipart для веб-интерфейса.
- `pyinstaller` — build-инструмент, а не рантайм-зависимость; в
  `[project.dependencies]` его нет.

## Блокеры заморозки и их решения

- **`pyannote.audio` требует `torchcodec`** (жёсткая зависимость) и при импорте
  пытается загрузить его нативные библиотеки через FFmpeg/CUDA. В проекте
  `torchcodec` не используется: диаризация получает waveform in-memory
  (`utils/audio.py`, PyAV), а `pyannote.audio` `try/except`-ом деградирует при
  отсутствии модуля. Поэтому в спеке `torchcodec` (и `nvidia`) добавлен в
  `excludes` — модуль в бандл не попадает, ложное предупреждение подавляется
  штатным фильтром (`utils/logging.py`). Это проверено: `import pyannote.audio`
  без рабочего `torchcodec` завершается успешно, диаризация идёт по waveform.
- **Динамические импорты** (uvicorn-протоколы, textual-драйверы, ленивые
  импорты pyannote/lightning/наших движков) не видны статическому анализу —
  собраны через `collect_submodules`.
- **Данные** (`web/static` SPA, словари `pymorphy3_dicts_ru`, VAD faster-whisper
  `silero_vad_v6.onnx`) не являются Python-кодом — собраны через
  `collect_data_files`.
- **Нативные бинарники не зашиваются**: `whisper-cli`, `llama-server`,
  `nemo-speech`, `deep-filter` и модели устанавливаются отдельно (#98/#115).
  Бандл — это сам `audio-transcriber`; `doctor` покажет их отсутствие как
  некритичные предупреждения.

## Ручная сборка (для отладки)

```bash
uv venv --python 3.14 /tmp/portable-venv
uv pip install --python /tmp/portable-venv/bin/python \
  --no-sources --torch-backend cpu ".[web]" pyinstaller
/tmp/portable-venv/bin/python -m PyInstaller \
  --noconfirm --clean --log-level WARN \
  --distpath /tmp/portable --workpath /tmp/portable/.build \
  packaging/pyinstaller/audio-transcriber.spec
```

Спека использует `SPECPATH`, поэтому запускать её можно из любого каталога.

## Smoke-проверка бандла

```bash
./dist-portable/audio-transcriber/audio-transcriber --version
./dist-portable/audio-transcriber/audio-transcriber --help
./dist-portable/audio-transcriber/audio-transcriber doctor
```

`doctor` может ругаться на внешние бинарники/модели — это ожидаемо, они
ставятся отдельно. Важно, что процесс стартует и импорты движков резолвятся.

## Готовые сборки из CI

Workflow `build-portable.yml` собирает бандл по тегам `v*` и вручную
(`workflow_dispatch`). При запуске по тегу каталог бандла упаковывается в один
архив (`audio-transcriber-<версия>-<os>-<arch>.tar.gz` для Linux/macOS, `.zip`
для Windows) и **прикладывается к GitHub Release** тега; при ручном запуске —
только workflow-артефакт `portable-<os>-<arch>`.
