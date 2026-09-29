#!/usr/bin/env bash
# Запуск веб-интерфейса AudioTranscriptor для Linux и macOS.
#
# Поднимает локальный сервер (`audio-transcriber web`) на свободном порту
# 127.0.0.1 и открывает браузер. При первом запуске `uv` сам установит Python
# нужной версии и базовые зависимости; для веб-интерфейса нужен extra `web`.
#
# Использование:
#   ./web.sh                 # http://127.0.0.1:8765/ + браузер
#   ./web.sh --port 9000     # свой порт
#   ./web.sh --no-browser    # не открывать браузер автоматически
#   ./web.sh --host 0.0.0.0  # слушать на всех интерфейсах (осторожно!)
#
# Остановить сервер — Ctrl+C.

set -euo pipefail
# Переходим в каталог скрипта: пути с пробелами/кириллицей должны работать.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Значения по умолчанию совпадают с CLI `audio-transcriber web`.
HOST="127.0.0.1"
PORT="8765"

# Читаем host/port из аргументов только для того, чтобы напечатать верный URL;
# сами аргументы передаём в CLI без изменений (он парсит их сам).
args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
    case "${args[$i]}" in
        --host) HOST="${args[$((i + 1))]:-$HOST}" ;;
        --host=*) HOST="${args[$i]#--host=}" ;;
        --port) PORT="${args[$((i + 1))]:-$PORT}" ;;
        --port=*) PORT="${args[$i]#--port=}" ;;
    esac
    i=$((i + 1))
done

if [ -x ".venv/bin/python" ]; then
    if ! .venv/bin/python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
        echo "Веб-интерфейс недоступен: не установлены веб-зависимости."
        echo 'Установите их: uv pip install --python .venv/bin/python ".[web]"'
        read -rp "Нажмите Enter, чтобы закрыть..."
        exit 1
    fi
    if [ -x ".venv/bin/audio-transcriber" ]; then
        RUNNER=(.venv/bin/audio-transcriber)
    else
        RUNNER=(.venv/bin/python -m audio_transcriber)
    fi
else
    # .venv ещё нет — `uv run` создаст окружение (extra `web` нужно доустановить).
    RUNNER=(uv run audio-transcriber)
fi

echo "Веб-интерфейс: http://${HOST}:${PORT}/"
echo "Ctrl+C — остановить."

exec "${RUNNER[@]}" web "$@"
