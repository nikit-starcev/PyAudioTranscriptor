#!/usr/bin/env bash
# Универсальный запуск AudioTranscriptor для Linux и macOS.
#
# Что делает:
#   1. Устанавливает uv, если он ещё не установлен.
#   2. Подтягивает Python нужной версии и все зависимости (это делает uv run
#      автоматически при первом запуске — отдельно ничего ставить не нужно).
#   3. Запускает транскрибацию с параметрами из config.env.
#
# Использование:
#   ./run.sh путь/к/записи.mp3
#
# Перед первым запуском:
#   cp config.example.env config.env
#   chmod 600 config.env   # в файле токены — доступ только владельцу
#   # затем откройте config.env и укажите свой токен Hugging Face и параметры

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "${1:-}" ]; then
    echo "Использование: $0 путь/к/аудиозаписи.mp3"
    exit 1
fi
AUDIO_FILE="$1"

if [ ! -f "config.env" ]; then
    echo "Не найден config.env."
    echo "Скопируйте config.example.env в config.env и укажите там параметры (см. README.md)."
    exit 1
fi

set -a
# shellcheck disable=SC1091
source config.env
set +a

if ! command -v uv >/dev/null 2>&1; then
    echo "uv не найден — устанавливаю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Внимание: HF_TOKEN не задан в config.env — определение говорящих завершится ошибкой (см. README)."
fi

# Маппинг config.env → флаги CLI живёт в одном месте — в коде
# (audio_transcriber.cli.env_config): CLI сам читает config.env (как веб и TUI),
# а флаги лишь переопределяют настройки. Раньше обёртки дублировали этот
# маппинг, и он расходился (CLEAR_CACHE и env-only WHISPER_CPP_CHUNK_*,
# issue #90). Поэтому здесь не перечисляем переменные, а передаём только
# обязательный путь к файлу: остальное подхватит CLI.
#
# config.env уже загружен выше (`set -a; source config.env`), поэтому
# переменные, которые код читает напрямую из окружения
# (WHISPER_CPP_VAD_MODEL, WHISPER_CPP_CHUNK_* и т.п.), тоже доступны.
#
# Секреты (HF_TOKEN, LLM_API_KEY) намеренно НЕ пробрасываются флагами argv:
# они видны в `ps`/`/proc/<pid>/cmdline`. CLI берёт их из окружения/файла сам.

# Для бэкенда whisper-cpp (гибрид на AMD) используется CPU-сборка torch,
# установленная вручную в .venv. `uv run` сверяется с uv.lock и может
# переустановить CUDA-сборку torch, поэтому вызываем бинарник напрямую.
if [ -x ".venv/bin/audio-transcriber" ]; then
    .venv/bin/audio-transcriber transcribe "$AUDIO_FILE"
else
    uv run audio-transcriber transcribe "$AUDIO_FILE"
fi
