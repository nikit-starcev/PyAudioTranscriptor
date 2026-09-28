#!/usr/bin/env bash
# Запуск интерактивного интерфейса AudioTranscriptor (TUI).
# Можно запускать двойным щелчком по ярлыку — откроется терминал с интерфейсом.
set -u

cd "$(dirname "$(readlink -f "$0")")" || exit 1

if [ ! -x ".venv/bin/audio-transcriber" ]; then
    echo "Не найдено окружение .venv — установите зависимости (uv sync)."
    read -rp "Нажмите Enter, чтобы закрыть..."
    exit 1
fi

.venv/bin/audio-transcriber tui
status=$?

if [ "$status" -ne 0 ]; then
    echo
    echo "Программа завершилась с кодом $status."
    read -rp "Нажмите Enter, чтобы закрыть окно..."
fi
