#!/usr/bin/env bash
set -euo pipefail

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
    echo "Использование: build-fallback.sh --bundle-dir DIR --out-dir DIR [--icon PNG] [--version VER]" >&2
    exit 2
fi
if [ ! -d "$BUNDLE_DIR" ]; then
    echo "Ошибка: каталог бандла не найден: $BUNDLE_DIR" >&2
    exit 1
fi

BUNDLE_DIR="$(cd "$BUNDLE_DIR" && pwd)"
BUNDLE_NAME="$(basename "$BUNDLE_DIR")"
BUNDLE_PARENT="$(dirname "$BUNDLE_DIR")"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
    x86_64 | amd64) ARCH="amd64"; BARCH="x86_64" ;;
    aarch64 | arm64) ARCH="arm64"; BARCH="aarch64" ;;
    *) echo "Ошибка: неподдерживаемая архитектура: $ARCH_RAW" >&2; exit 1 ;;
esac

TARBALL="$OUT_DIR/audio-transcriber-$VERSION-linux-$BARCH.tar.gz"
tar -czf "$TARBALL" -C "$BUNDLE_PARENT" "$BUNDLE_NAME"
echo "Готово: $TARBALL"

if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb недоступен — .deb пропущен." >&2
    exit 0
fi

ROOT="$OUT_DIR/.deb-root"
DEB="$OUT_DIR/audio-transcriber_${VERSION}_${ARCH}.deb"
rm -rf "$ROOT"
mkdir -p "$ROOT/DEBIAN" \
    "$ROOT/opt/audio-transcriber" \
    "$ROOT/usr/bin" \
    "$ROOT/usr/share/applications" \
    "$ROOT/usr/share/icons/hicolor/256x256/apps"

cp -a "$BUNDLE_DIR/." "$ROOT/opt/audio-transcriber/"

cat > "$ROOT/usr/bin/audio-transcriber" <<'WRAPPER'
#!/bin/sh
if [ "$#" -eq 0 ]; then
    set -- web
fi
exec /opt/audio-transcriber/audio-transcriber "$@"
WRAPPER
chmod 0755 "$ROOT/usr/bin/audio-transcriber"

cat > "$ROOT/usr/share/applications/audio-transcriber.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=AudioTranscriptor
Comment=Локальная транскрибация аудио с разделением говорящих
Exec=audio-transcriber
Icon=audio-transcriber
Terminal=false
Categories=AudioVideo;
DESKTOP

if [ -n "$ICON" ] && [ -f "$ICON" ]; then
    cp "$ICON" "$ROOT/usr/share/icons/hicolor/256x256/apps/audio-transcriber.png"
fi

INSTALLED_KB="$(du -sk "$ROOT/opt/audio-transcriber" | cut -f1)"
cat > "$ROOT/DEBIAN/control" <<CONTROL
Package: audio-transcriber
Version: $VERSION
Architecture: $ARCH
Maintainer: PyAudioTranscriptor <noreply@example.com>
Installed-Size: $INSTALLED_KB
Section: sound
Priority: optional
Description: Локальная транскрибация аудио с разделением говорящих
CONTROL

dpkg-deb --build --root-owner-group "$ROOT" "$DEB"
rm -rf "$ROOT"
echo "Готово: $DEB"
