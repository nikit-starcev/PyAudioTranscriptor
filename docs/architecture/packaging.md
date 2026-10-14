# Точки входа, упаковка, Docker и релизы

Навигация: [Обзор](README.md) · [Компоненты](components.md) · [Конвейер](pipeline.md) ·
[Движки](engines.md) · [Веб](web.md) · [Конфигурация](configuration.md) ·
[Корп-режим](corporate-mode.md) · [ADR](decisions.md)

## 1. Точки входа

- Консольный скрипт: `audio-transcriber = "audio_transcriber.cli.app:app"`
  (`pyproject.toml:26`).
- Модульный запуск: `python -m audio_transcriber` (`__main__.py` вызывает
  `cli.app.app`).
- Команды Typer (`cli/app.py`): `transcribe` (`:204`), `doctor` (`:1331`),
  `tui` (`:1353`), `web` (`:1411`), подкоманда `glossary` (`:1476`),
  подкоманда `models` (`cli/models_cmd.py`).
- Оболочки-лаунчеры в репозитории: `run.sh`/`run.bat`/`run.ps1` (CLI/TUI),
  `web.sh`/`web.bat`/`web.ps1` (веб), `launch-tui.sh`.

Питон-требование: `>=3.12,<3.15` (`pyproject.toml:9`).

## 2. Зависимости и extras

Базовые зависимости — `pyproject.toml:10-23` (`av`, `faster-whisper`,
`fpdf2`, `numpy`, `pyannote-audio`, `python-docx`, `rich`, `textual`, `typer`,
`torch`/`torchaudio`, `pymorphy3`). Необязательные extras (`:32-52`):

| Extra | Что даёт |
|---|---|
| `gigaam` | GigaAM v3 через `onnx-asr` (без torch) |
| `sherpa` | `sherpa-onnx` (оценщик числа говорящих + гибрид) |
| `web` | `fastapi`, `uvicorn[standard]`, `python-multipart` |

Денойз **не** Python-extra: внешний Rust-CLI `deep-filter` (DeepFilterNet)
ставится отдельно (`pyproject.toml:28-31`). PyTorch на не-macOS тянется из
индекса CUDA `cu126` (`[tool.uv.sources]`, `:75-83`), версии `torch<2.13`/
`torchaudio<2.12` закреплены намеренно (`:70-74`).

## 3. Docker

`Dockerfile` — многостадийный:

| Стадия | Назначение |
|---|---|
| `whisper-cpp` (`:30`) | официальный Vulkan-образ ggml-org; копируются бинарник и `.so` |
| `llama-cpp` (`:39`) | сборка `llama-server` (Vulkan) из исходников на том же базовом образе (совместимость glibc, RPATH `$ORIGIN`) |
| `builder` (`:76`) | установка проекта в `.venv` через `uv` |
| `runtime` (`:127`) | финальный образ: непривилегированный пользователь `app`, копии движков, `.venv`, `config.example.env → /app/config.env` |

- GPU — только через **Vulkan** (проброс `/dev/dri`), без CUDA/ROCm
  (`Dockerfile:3-7`, `docker-compose.yml`). ROCm — экспериментально через
  `TORCH_INDEX`.
- `EXPOSE 8790` (`:220`), `HEALTHCHECK` — `GET /api/health` (`:228`),
  `ENTRYPOINT entrypoint.sh`, `CMD audio-transcriber web --no-browser ...`
  (`:233-234`).
- **Rootless/userns-remap** различается в `entrypoint.sh`: rootful выравнивает
  владельца тома `/data` (`chown -R app:app`) и сбрасывает привилегии через
  `setpriv`; при userns-remap chown **не** делается (иначе ломается владелец
  хостовых файлов) и приложение стартует от контейнерного root
  (`entrypoint.sh:1-31`).
- `docker-compose.yml` публикует порт только на localhost
  (`127.0.0.1:8790`), монтирует `./web-data:/data` и `./voices:/data/voices`,
  задаёт `AUDIO_TRANSCRIBER_WEB_DATA=/data`, healthcheck и `restart:
  unless-stopped`; секции GPU (`devices`, `group_add`) закомментируются для
  CPU-only.

## 4. Portable-бандл (one-dir, PyInstaller)

- Скрипт сборки — `scripts/build_portable.py` (`--target {linux,windows,macos}`,
  `--out`, `--clean`). PyInstaller **не кросс-компилирует**: `--target` обязан
  совпадать с host-ОС (`packaging/README.md:18-23`, `:5-7`).
- Окружение: изолированный venv через `uv`, установка `"<repo>[web]"` с
  `--no-sources --torch-backend cpu` и PyInstaller (`packaging/README.md:25-40`).
- Спека — `packaging/pyinstaller/audio-transcriber.spec`; вход —
  `packaging/pyinstaller/entry.py`.
- Блокеры заморозки и решения (`packaging/README.md:42-61`): `torchcodec`
  исключён (диаризация идёт по waveform); динамические импорты собраны
  `collect_submodules`; данные (SPA `web/static`, словари pymorphy3, VAD
  faster-whisper) собраны `collect_data_files`; **нативные бинарники не
  зашиваются** — ставятся отдельно (#98/#115).
- Результат — `dist-portable/audio-transcriber/` с исполняемым файлом;
  smoke-проверка — `--version`, `--help`, `doctor` (`packaging/README.md:77-86`).

## 5. Инсталляторы

- Скрипт — `scripts/build_installer.py`: берёт готовый one-dir бандл, генерирует
  иконки, вызывает платформенный сборщик и кладёт `.<sha256>`; подпись — при
  наличии переменных окружения (`scripts/build_installer.py:1-7`).
- Платформенные сборщики (`packaging/installers/`):

| ОС | Каталог | Инструмент |
|---|---|---|
| Linux | `linux/` | AppImage (`build-appimage.sh`), fallback (`build-fallback.sh`), `AppRun`, `.desktop` |
| Windows | `windows/` | Inno Setup (`audio-transcriber.iss`, `build-inno.ps1`) |
| macOS | `macos/` | `build-app.sh`, `build-dmg.sh`, `Info.plist`, launcher |

- Общие лаунчеры — `packaging/launchers/run-web.{sh,command,bat,ps1}`.
- Подпись артефактов — `packaging/SIGNING.md`.
- Иконки генерируются `packaging/installers/generate_icons.py`.

## 6. Версионирование

Версия **не** хранится в `pyproject.toml` — её выводит `hatch-vcs` из git-тега
(`pyproject.toml:1-5`, `[tool.hatch.version] source = "vcs"`, `:60-61`).
Единственный источник истины — тег `vX.Y.Z` (`RELEASING.md:1-16`):

| Состояние | Версия |
|---|---|
| ровно на теге `v0.3.1` | `0.3.1` |
| 17 коммитов после | `0.3.2.dev17+g3f5f115` |
| грязное дерево | `…dev17+g3f5f115.d20261002` |

## 7. CI и релизы

Файлы в `.github/workflows/`:

| Workflow | Триггеры | Что делает |
|---|---|---|
| `ci.yml` | push, pull_request | проверка `uv.lock`, линтер (ruff), типы (mypy), тесты (pytest); CPU-only PyTorch |
| `release.yml` | теги `v*` | сборка sdist/wheel, проверка, что версия сборки совпадает с тегом, извлечение заметок из `CHANGELOG.md`, создание GitHub Release |
| `build-portable.yml` | теги `v*`, `workflow_dispatch` | сборка one-dir бандла по matrix ОС; на теге упаковка в `audio-transcriber-<версия>-<os>-<arch>.tar.gz`/`.zip` и приложение к Release |
| `build-installers.yml` | теги `v*`, `workflow_dispatch` | сборка инсталляторов (AppImage/.exe/.dmg), подпись и выгрузка в Release |

Порядок релиза (обновить `CHANGELOG.md` → коммит → аннотированный тег → push)
описан в `RELEASING.md:30-60`. Заголовок в `CHANGELOG.md` ищется ровно как
`## [<версия>]` для сборки заметок релиза (`RELEASING.md:36-46`).

## 8. Что НЕ входит в артефакты

- Внешние бинарники (`whisper-cli`, `llama-server`, `nemo-speech`,
  `deep-filter`) и модели — устанавливаются пользователем/из UI
  (`packaging/README.md:58-61`).
- `config.env` и `web-data/` исключены из git (`.gitignore`); Docker даёт
  встроенный `config.example.env`.
- Runtime-данные (`web-data/`) монтируются томом в Docker.
