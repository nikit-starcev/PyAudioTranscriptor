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

ARGS=(transcribe "$AUDIO_FILE")
[ -n "${OUTPUT_DIR:-}" ] && ARGS+=(--output-dir "$OUTPUT_DIR")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
[ -n "${LANGUAGE:-}" ] && ARGS+=(--language "$LANGUAGE")
[ -n "${DEVICE:-}" ] && ARGS+=(--device "$DEVICE")

if [ -n "${FORMATS:-}" ]; then
    IFS=',' read -ra FORMAT_LIST <<< "$FORMATS"
    for fmt in "${FORMAT_LIST[@]}"; do
        ARGS+=(--format "$(echo "$fmt" | xargs)")
    done
fi

[ -n "${NUM_SPEAKERS:-}" ] && ARGS+=(--num-speakers "$NUM_SPEAKERS")

if [ -n "${SPEAKER_NAMES:-}" ]; then
    IFS=',' read -ra NAME_LIST <<< "$SPEAKER_NAMES"
    for name in "${NAME_LIST[@]}"; do
        ARGS+=(--speaker-name "$(echo "$name" | xargs)")
    done
fi

[ -n "${HF_TOKEN:-}" ] && ARGS+=(--hf-token "$HF_TOKEN")
[ "${ENABLE_CORRECTION:-false}" = "true" ] && ARGS+=(--enable-correction)
[ -n "${CORRECTION_MIN_WORD_LENGTH:-}" ] && ARGS+=(--correction-min-word-length "$CORRECTION_MIN_WORD_LENGTH")
[ -n "${CORRECTION_MIN_SIMILARITY:-}" ] && ARGS+=(--correction-min-similarity "$CORRECTION_MIN_SIMILARITY")
[ -n "${CORRECTION_MAX_CANDIDATES:-}" ] && ARGS+=(--correction-max-candidates "$CORRECTION_MAX_CANDIDATES")
[ -n "${HOTWORDS:-}" ] && ARGS+=(--hotwords "$HOTWORDS")
[ "${VERBOSE:-false}" = "true" ] && ARGS+=(--verbose)

uv run audio-transcriber "${ARGS[@]}"
