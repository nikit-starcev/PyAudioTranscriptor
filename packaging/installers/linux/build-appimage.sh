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
    echo "Использование: build-appimage.sh --bundle-dir DIR --out-dir DIR [--icon PNG] [--version VER]" >&2
    exit 2
fi
if [ ! -d "$BUNDLE_DIR" ]; then
    echo "Ошибка: каталог бандла не найден: $BUNDLE_DIR" >&2
    exit 1
fi

ARCH="${ARCH:-$(uname -m)}"
case "$ARCH" in
    x86_64 | amd64) ARCH="x86_64" ;;
    aarch64 | arm64) ARCH="aarch64" ;;
    *) echo "Ошибка: неподдерживаемая архитектура: $ARCH" >&2; exit 1 ;;
esac

APP_NAME="audio-transcriber"
APPDIR="$OUT_DIR/AppDir"
OUTPUT="$OUT_DIR/$APP_NAME-$VERSION-$ARCH.AppImage"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -a "$BUNDLE_DIR/." "$APPDIR/usr/bin/"
install -m 0755 "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
install -m 0644 "$SCRIPT_DIR/$APP_NAME.desktop" "$APPDIR/$APP_NAME.desktop"

if [ -n "$ICON" ] && [ -f "$ICON" ]; then
    cp "$ICON" "$APPDIR/$APP_NAME.png"
    cp "$ICON" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_NAME.png"
else
    echo "Предупреждение: иконка не задана, appimagetool использует заглушку." >&2
fi

CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/audio-transcriber/tools"
TOOL=""
if [ -n "${APPIMAGETOOL:-}" ] && [ -x "$APPIMAGETOOL" ]; then
    TOOL="$APPIMAGETOOL"
elif command -v appimagetool >/dev/null 2>&1; then
    TOOL="$(command -v appimagetool)"
else
    CACHED="$CACHE/appimagetool-$ARCH.AppImage"
    if [ ! -x "$CACHED" ]; then
        mkdir -p "$CACHE"
        URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
        echo "Скачиваю appimagetool: $URL"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -o "$CACHED" "$URL" || true
        fi
        if [ ! -s "$CACHED" ] && command -v wget >/dev/null 2>&1; then
            wget -q -O "$CACHED" "$URL" || true
        fi
        chmod +x "$CACHED" 2>/dev/null || true
    fi
    if [ -x "$CACHED" ]; then
        TOOL="$CACHED"
    fi
fi

if [ -z "$TOOL" ]; then
    echo "appimagetool недоступен (нет в PATH и не скачался)." >&2
    exit 3
fi

RUN_ARGS=()
case "$TOOL" in
    *.AppImage) RUN_ARGS+=(--appimage-extract-and-run) ;;
esac

echo "Сборка AppImage: $OUTPUT"
ARCH="$ARCH" "$TOOL" "${RUN_ARGS[@]}" --no-appstream "$APPDIR" "$OUTPUT"
chmod +x "$OUTPUT"
echo "Готово: $OUTPUT"
