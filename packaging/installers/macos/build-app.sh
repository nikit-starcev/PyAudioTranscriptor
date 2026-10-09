#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUNDLE_DIR=""
OUT_DIR=""
ICON=""
VERSION="0.0.0"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --icon) ICON="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        *) echo "Неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BUNDLE_DIR" ] || [ -z "$OUT_DIR" ]; then
    echo "Использование: build-app.sh --bundle-dir DIR --out-dir DIR [--icon ICNS] [--version VER]" >&2
    exit 2
fi
if [ ! -d "$BUNDLE_DIR" ]; then
    echo "Ошибка: каталог бандла не найден: $BUNDLE_DIR" >&2
    exit 1
fi

APP_ID="AudioTranscriptor"
APP_DIR="$OUT_DIR/$APP_ID.app"
CONTENTS="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES"

cp -a "$BUNDLE_DIR/." "$MACOS_DIR/"
install -m 0755 "$SCRIPT_DIR/launcher" "$MACOS_DIR/$APP_ID"

sed "s/__VERSION__/$VERSION/g" "$SCRIPT_DIR/Info.plist" > "$CONTENTS/Info.plist"

if [ -n "$ICON" ] && [ -f "$ICON" ]; then
    cp "$ICON" "$RESOURCES/AppIcon.icns"
fi

echo "Готово: $APP_DIR"
