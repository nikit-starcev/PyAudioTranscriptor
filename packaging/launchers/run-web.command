#!/usr/bin/env bash
# macOS: запуск веб-интерфейса двойным щелчком из Finder.
# Расширение .command открывает Терминал и выполняет этот файл.
# Вся логика — в run-web.sh из того же каталога.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$DIR/run-web.sh" ]; then
    echo "Ошибка: рядом не найден run-web.sh." >&2
    echo "Положите run-web.command и run-web.sh в один каталог." >&2
    read -r -p "Нажмите Enter, чтобы закрыть..." _ || true
    exit 1
fi

exec "$DIR/run-web.sh" "$@"
