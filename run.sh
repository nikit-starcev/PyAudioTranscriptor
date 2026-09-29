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
[ -n "${ASR_BACKEND:-}" ] && ARGS+=(--asr-backend "$ASR_BACKEND")
[ -n "${WHISPER_CPP_MODEL:-}" ] && ARGS+=(--whisper-cpp-model "$WHISPER_CPP_MODEL")
[ -n "${WHISPER_CPP_BINARY:-}" ] && ARGS+=(--whisper-cpp-binary "$WHISPER_CPP_BINARY")
[ -n "${WHISPER_CPP_LIB_PATH:-}" ] && ARGS+=(--whisper-cpp-lib-path "$WHISPER_CPP_LIB_PATH")

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

if [ -n "${SPEAKER_REFERENCES:-}" ]; then
    IFS=',' read -ra REF_LIST <<< "$SPEAKER_REFERENCES"
    for ref in "${REF_LIST[@]}"; do
        ARGS+=(--speaker-reference "$(echo "$ref" | xargs)")
    done
fi
[ -n "${ENROLLMENT_MIN_SIMILARITY:-}" ] && ARGS+=(--enrollment-min-similarity "$ENROLLMENT_MIN_SIMILARITY")

[ -n "${HF_TOKEN:-}" ] && ARGS+=(--hf-token "$HF_TOKEN")
[ -n "${PYANNOTE_LOCAL_MODEL:-}" ] && ARGS+=(--pyannote-local-model "$PYANNOTE_LOCAL_MODEL")
[ "${ENABLE_CORRECTION:-false}" = "true" ] && ARGS+=(--enable-correction)
[ "${CLEAN_ARTIFACTS:-true}" = "false" ] && ARGS+=(--no-clean)
[ "${COLLAPSE_REPEATS:-true}" = "false" ] && ARGS+=(--no-collapse-repeats)
[ -n "${REPEAT_MIN_WORDS:-}" ] && ARGS+=(--repeat-min-words "$REPEAT_MIN_WORDS")
[ -n "${REPEAT_SIMILARITY:-}" ] && ARGS+=(--repeat-similarity "$REPEAT_SIMILARITY")
[ "${NORMALIZE_TEXT:-true}" = "false" ] && ARGS+=(--no-normalize)
[ "${MARK_OVERLAP:-true}" = "false" ] && ARGS+=(--no-overlap)
[ -n "${LOW_CONFIDENCE_THRESHOLD:-}" ] && ARGS+=(--low-confidence-threshold "$LOW_CONFIDENCE_THRESHOLD")
[ "${DENOISE:-true}" = "false" ] && ARGS+=(--no-denoise)
[ "${USE_CACHE:-true}" = "false" ] && ARGS+=(--no-cache)
[ "${CLEAR_CACHE:-false}" = "true" ] && ARGS+=(--clear-cache)
[ -n "${CACHE_DIR:-}" ] && ARGS+=(--cache-dir "$CACHE_DIR")
[ "${NOTIFICATIONS:-true}" = "false" ] && ARGS+=(--no-notify)
[ -n "${CORRECTION_MIN_WORD_LENGTH:-}" ] && ARGS+=(--correction-min-word-length "$CORRECTION_MIN_WORD_LENGTH")
[ -n "${CORRECTION_MIN_SIMILARITY:-}" ] && ARGS+=(--correction-min-similarity "$CORRECTION_MIN_SIMILARITY")
[ -n "${CORRECTION_MAX_CANDIDATES:-}" ] && ARGS+=(--correction-max-candidates "$CORRECTION_MAX_CANDIDATES")
[ -n "${HOTWORDS:-}" ] && ARGS+=(--hotwords "$HOTWORDS")
[ "${VERBOSE:-false}" = "true" ] && ARGS+=(--verbose)

# --- LLM-постобработка (llama.cpp) ---
[ "${LLM_ENABLED:-false}" = "true" ] && ARGS+=(--llm)
[ -n "${LLM_MODEL:-}" ] && ARGS+=(--llm-model "$LLM_MODEL")
[ -n "${LLM_BINARY:-}" ] && ARGS+=(--llm-binary "$LLM_BINARY")
[ -n "${LLM_LIB_PATH:-}" ] && ARGS+=(--llm-lib-path "$LLM_LIB_PATH")
[ "${LLM_GPU:-true}" = "false" ] && ARGS+=(--llm-cpu)
[ -n "${LLM_CONTEXT:-}" ] && ARGS+=(--llm-context "$LLM_CONTEXT")
[ "${LLM_EXTRACT_NAMES:-true}" = "false" ] && ARGS+=(--llm-no-names)
[ "${LLM_SUMMARY:-true}" = "false" ] && ARGS+=(--no-llm-summary)
[ "${LLM_SUGGEST_TERMS:-false}" = "true" ] && ARGS+=(--llm-suggest-terms)
[ -n "${LLM_PROMPT_EXTRA:-}" ] && ARGS+=(--llm-prompt-extra "$LLM_PROMPT_EXTRA")
[ -n "${LLM_PROMPT_FILE:-}" ] && ARGS+=(--llm-prompt-file "$LLM_PROMPT_FILE")
[ -n "${GLOSSARY_PATH:-}" ] && ARGS+=(--glossary "$GLOSSARY_PATH")

# Для бэкенда whisper-cpp (гибрид на AMD) используется CPU-сборка torch,
# установленная вручную в .venv. `uv run` сверяется с uv.lock и может
# переустановить CUDA-сборку torch, поэтому вызываем бинарник напрямую.
if [ -x ".venv/bin/audio-transcriber" ]; then
    .venv/bin/audio-transcriber "${ARGS[@]}"
else
    uv run audio-transcriber "${ARGS[@]}"
fi
