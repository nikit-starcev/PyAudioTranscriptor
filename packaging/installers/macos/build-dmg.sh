#!/usr/bin/env bash
set -euo pipefail

APP_DIR=""
OUT_DIR=""
VERSION="0.0.0"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --app) APP_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        *) echo "Неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$APP_DIR" ] || [ -z "$OUT_DIR" ]; then
    echo "Использование: build-dmg.sh --app PATH --out-dir DIR [--version VER]" >&2
    exit 2
fi
if [ ! -d "$APP_DIR" ]; then
    echo "Ошибка: .app не найден: $APP_DIR" >&2
    exit 1
fi

if ! command -v hdiutil >/dev/null 2>&1; then
    echo "Ошибка: hdiutil доступен только на macOS." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUTPUT="$OUT_DIR/audio-transcriber-$VERSION.dmg"
rm -f "$OUTPUT"

if command -v create-dmg >/dev/null 2>&1; then
    create-dmg \
        --volname "AudioTranscriptor" \
        --icon-size 100 \
        --icon "$(basename "$APP_DIR")" 150 185 \
        --app-drop-link 600 185 \
        --no-internet-enable \
        "$OUTPUT" "$APP_DIR"
else
    STAGING="$(mktemp -d)"
    trap 'rm -rf "$STAGING"' EXIT
    cp -R "$APP_DIR" "$STAGING/"
    ln -s /Applications "$STAGING/Applications"
    hdiutil create -volname "AudioTranscriptor" -srcfolder "$STAGING" -ov -format UDZO "$OUTPUT"
fi

if [ ! -f "$OUTPUT" ]; then
    echo "Ошибка: не удалось создать .dmg" >&2
    exit 1
fi

echo "Готово: $OUTPUT"
