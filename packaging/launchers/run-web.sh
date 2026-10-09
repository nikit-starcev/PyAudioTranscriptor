#!/usr/bin/env bash
# Запуск портативного бандла AudioTranscriptor (веб-интерфейс).
#
# Рассчитан на one-dir-бандл, собранный `python scripts/build_portable.py`:
# каталог `audio-transcriber/` с исполняемым файлом внутри. Исполняемый файл
# ищется РЯДОМ С САМИМ ЛАУНЧЕРОМ (все пути считаются относительно него),
# поэтому лаунчер можно положить как внутрь каталога бандла, так и рядом с ним.
#
# Использование:
#   ./run-web.sh                              # http://127.0.0.1:8790 + браузер
#   AUDIO_TRANSCRIBER_WEB_PORT=9000 ./run-web.sh
#   ./run-web.sh --no-browser                 # флаги CLI пробрасываются дальше
#
# Остановить сервер — Ctrl+C.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Порядок поиска: лаунчер внутри бандла → рядом с бандлом → на уровень выше
# (например, лаунчер лежит в packaging/launchers, а бандл — в dist/).
EXE=""
for candidate in \
    "$SCRIPT_DIR/audio-transcriber" \
    "$SCRIPT_DIR/audio-transcriber/audio-transcriber" \
    "$SCRIPT_DIR/../audio-transcriber/audio-transcriber"
do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        EXE="$candidate"
        break
    fi
done

if [ -z "$EXE" ]; then
    {
        echo "Ошибка: рядом с лаунчером не найден исполняемый файл бандла."
        echo "Положите этот скрипт внутрь каталога бандла или рядом с ним."
        echo "Ожидался один из путей:"
        echo "  $SCRIPT_DIR/audio-transcriber"
        echo "  $SCRIPT_DIR/audio-transcriber/audio-transcriber"
        echo "  $SCRIPT_DIR/../audio-transcriber/audio-transcriber"
        echo "Соберите бандл: python scripts/build_portable.py --target linux --out dist"
    } >&2
    # При двойном запуске терминал закрылся бы сразу — даём прочитать ошибку.
    if [ -t 0 ]; then
        read -r -p "Нажмите Enter, чтобы закрыть..." _ || true
    fi
    exit 1
fi

# Порт по умолчанию; переопределяется переменной окружения. Если пользователь
# уже передал --port в аргументах, дефолт не добавляем.
PORT="${AUDIO_TRANSCRIBER_WEB_PORT:-8790}"
PORT_ARG=(--port "$PORT")
for arg in "$@"; do
    case "$arg" in
        --port | --port=*)
            PORT_ARG=()
            break
            ;;
    esac
done

echo "Запуск AudioTranscriptor (веб-интерфейс). Ctrl+C — остановить."
exec "$EXE" web "${PORT_ARG[@]}" "$@"
