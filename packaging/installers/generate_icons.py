"""Генерация иконок инсталлятора из ``favicon.svg`` без внешних Python-зависимостей.

SVG растрируется штатным конвертером (ImageMagick / librsvg / Inkscape /
cairosvg), иначе иконка рисуется программно. Форматы PNG/ICO/ICNS собираются
вручную (PNG-кадры внутри ICO и ICNS), поэтому Pillow не требуется.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

BRAND_COLOR = (134, 59, 255, 255)
ICON_SIZES = (512, 256, 128, 64, 48, 32, 16)
ICO_SIZES = (16, 32, 48, 64, 128, 256)
ICNS_FRAMES = ((b"ic08", 256), (b"ic09", 512))

_RASTERIZER_ORDER = ("magick", "convert", "rsvg-convert", "inkscape")


def build_parser() -> argparse.ArgumentParser:
    """CLI-парсер генератора иконок."""
    parser = argparse.ArgumentParser(
        prog="generate_icons.py",
        description="Сгенерировать PNG/ICO/ICNS для инсталлятора audio-transcriber.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Исходный SVG.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Каталог для иконок.")
    parser.add_argument("--name", default="audio-transcriber", help="Базовое имя файлов иконок.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    return build_parser().parse_args(argv)


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Возвращает (width, height) PNG из IHDR."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("не PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def encode_png(rgba: bytes, size: int) -> bytes:
    """Кодирует RGBA-буфер (size×size) в PNG без Pillow."""
    stride = size * 4
    raw = bytearray()
    for row in range(size):
        raw.append(0)
        raw += rgba[row * stride : (row + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _inside_round_rect(
    x: int, y: int, left: int, top: int, right: int, bottom: int, radius: int
) -> bool:
    """Проверяет попадание точки в прямоугольник со скруглёнными углами."""
    cx = min(max(x, left + radius), right - radius)
    cy = min(max(y, top + radius), bottom - radius)
    dx = x - cx
    dy = y - cy
    return dx * dx + dy * dy <= radius * radius


def render_fallback_png(size: int) -> bytes:
    """Рисует минимальную иконку (скруглённый квадрат + эквалайзер) в PNG."""
    pad = max(2, size // 10)
    radius = max(2, size // 6)
    bar_width = max(1, size // 14)
    gap = max(1, size // 9)
    ratios = (0.30, 0.52, 0.74, 0.52, 0.30)
    total = len(ratios) * bar_width + (len(ratios) - 1) * gap
    start_x = (size - total) // 2
    center_y = size // 2
    bars = []
    for index, ratio in enumerate(ratios):
        left = start_x + index * (bar_width + gap)
        height = int(size * ratio)
        bars.append((left, center_y - height // 2, left + bar_width, center_y + height // 2))

    buffer = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            if not _inside_round_rect(x, y, pad, pad, size - pad, size - pad, radius):
                continue
            offset = (y * size + x) * 4
            color = BRAND_COLOR
            for left, top, right, bottom in bars:
                if left <= x < right and top <= y < bottom:
                    color = (255, 255, 255, 255)
                    break
            buffer[offset : offset + 4] = bytes(color)
    return encode_png(bytes(buffer), size)


def _rasterize_tool(tool: str, source: Path, dest: Path, size: int) -> bool:
    """Растрирует SVG одним инструментом в квадрат ``size×size``."""
    name = Path(tool).name
    if name in ("magick", "convert"):
        cmd = [
            tool,
            "-background",
            "none",
            "-density",
            "384",
            str(source),
            "-resize",
            f"{size}x{size}",
            "-gravity",
            "center",
            "-background",
            "none",
            "-extent",
            f"{size}x{size}",
            "-depth",
            "8",
            "-strip",
            str(dest),
        ]
    elif name == "rsvg-convert":
        cmd = [tool, "-w", str(size), "-h", str(size), "-o", str(dest), str(source)]
    elif name == "inkscape":
        cmd = [
            tool,
            "--export-type=png",
            "--export-filename",
            str(dest),
            "-w",
            str(size),
            "-h",
            str(size),
            str(source),
        ]
    else:
        return False
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return dest.is_file()


def rasterize_size(source: Path, size: int) -> bytes | None:
    """Возвращает PNG-байты квадратной иконки или None, если не удалось."""
    with tempfile.TemporaryDirectory(prefix="audio-transcriber-icon-") as tmp:
        for name in _RASTERIZER_ORDER:
            tool = shutil.which(name)
            if tool is None:
                continue
            dest = Path(tmp) / f"{size}.png"
            if not _rasterize_tool(tool, source, dest, size):
                continue
            data = dest.read_bytes()
            try:
                width, height = png_dimensions(data)
            except ValueError:
                continue
            if width == size and height == size:
                return data
    try:
        import cairosvg
    except ImportError:
        return None
    dest = Path(tempfile.mkdtemp(prefix="audio-transcriber-icon-")) / f"{size}.png"
    try:
        cairosvg.svg2png(url=str(source), write_to=str(dest), output_width=size, output_height=size)
    except (OSError, ValueError):
        return None
    return dest.read_bytes()


def collect_frames(source: Path) -> dict[int, bytes]:
    """Готовит PNG-кадры всех размеров (растр или программный фолбэк)."""
    frames: dict[int, bytes] = {}
    used_fallback = False
    for size in ICON_SIZES:
        data = rasterize_size(source, size) if source.is_file() else None
        if data is None:
            if not used_fallback:
                print(f"Конвертер SVG недоступен; рисую иконку программно ({source}).")
                used_fallback = True
            data = render_fallback_png(size)
        frames[size] = data
    return frames


def pack_ico(frames: dict[int, bytes]) -> bytes:
    """Собирает ICO из PNG-кадров (PNG внутри ICO, Windows Vista+)."""
    entries = b""
    payload = b""
    offset = 6 + 16 * len(ICO_SIZES)
    for size in ICO_SIZES:
        data = frames[size]
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(data), offset)
        payload += data
        offset += len(data)
    return struct.pack("<HHH", 0, 1, len(ICO_SIZES)) + entries + payload


def pack_icns(frames: dict[int, bytes]) -> bytes:
    """Собирает ICNS из PNG-кадров 256/512 (macOS 10.7+)."""
    chunks = b""
    for code, size in ICNS_FRAMES:
        data = frames[size]
        chunks += code + struct.pack(">I", len(data) + 8) + data
    return b"icns" + struct.pack(">I", len(chunks) + 8) + chunks


def generate(source: Path, out_dir: Path, name: str) -> dict[str, Path]:
    """Генерирует PNG (512/256), ICO и ICNS, возвращает пути."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(Path(source))

    png_main = out_dir / f"{name}.png"
    png_256 = out_dir / f"{name}-256.png"
    ico_path = out_dir / f"{name}.ico"
    icns_path = out_dir / f"{name}.icns"

    png_main.write_bytes(frames[512])
    png_256.write_bytes(frames[256])
    ico_path.write_bytes(pack_ico(frames))
    icns_path.write_bytes(pack_icns(frames))

    return {"png": png_main, "png256": png_256, "ico": ico_path, "icns": icns_path}


def main(argv: list[str] | None = None) -> int:
    """Точка входа генератора иконок."""
    args = parse_args(argv)
    paths = generate(args.source, args.out_dir, args.name)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
