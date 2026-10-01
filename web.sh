#!/usr/bin/env bash
# Запуск веб-интерфейса AudioTranscriptor для Linux и macOS.
#
# Поднимает локальный сервер (`audio-transcriber web`) на свободном порту
# 127.0.0.1 и открывает браузер. При первом запуске `uv` сам установит Python
# нужной версии и базовые зависимости; для веб-интерфейса нужен extra `web`.
#
# Использование:
#   ./web.sh                 # 8765 либо ближайший свободный порт + браузер
#   ./web.sh --port 9000     # строго заданный порт (занят — понятная ошибка)
#   ./web.sh --no-browser    # не открывать браузер автоматически
#   ./web.sh --host 0.0.0.0  # слушать на всех интерфейсах (осторожно!)
#
# Фактический адрес и порт печатает сама команда (`audio-transcriber web`),
# поэтому оболочка не дублирует URL: при автоподборе порта он может отличаться
# от значения по умолчанию.
#
# Остановить сервер — Ctrl+C.

set -euo pipefail
# Переходим в каталог скрипта: пути с пробелами/кириллицей должны работать.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

echo "Ctrl+C — остановить."

exec "${RUNNER[@]}" web "$@"
