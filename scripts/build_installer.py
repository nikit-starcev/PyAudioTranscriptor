"""Сборка инсталляторов ``audio-transcriber`` (Inno Setup / .dmg / AppImage).

Скрипт берёт готовый one-dir бандл (результат ``scripts/build_portable.py``),
генерирует иконки, вызывает платформенный сборщик из
``packaging/installers/<os>/``, при наличии переменных окружения подписывает
артефакт и рядом кладёт ``.sha256``.

PyInstaller не кросс-компилирует, поэтому сборка возможна только под host-ОС:
``--target`` обязан совпадать с ней.

Пример::

    python scripts/build_installer.py --target linux \
        --bundle-dir dist/audio-transcriber --out dist-installers
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BUNDLE_DIRNAME = "audio-transcriber"

SUPPORTED_TARGETS = ("linux", "windows", "macos")

EXECUTABLE_NAMES = {
    "linux": "audio-transcriber",
    "windows": "audio-transcriber.exe",
    "macos": "audio-transcriber",
}

_OS_TO_TARGET = {
    "linux": "linux",
    "windows": "windows",
    "darwin": "macos",
}

DEFAULT_BUNDLE_DIR = Path("dist") / BUNDLE_DIRNAME

INSTALLERS_SUBDIR = Path("packaging") / "installers"

ICON_SOURCE = Path("webui") / "public" / "favicon.svg"

ARTIFACT_GLOBS: dict[str, tuple[str, ...]] = {
    "linux": ("*.AppImage", "*.deb", "*.tar.gz"),
    "windows": ("*-setup*.exe",),
    "macos": ("*.dmg",),
}

WINDOWS_SIGN_ENV = ("WINDOWS_CERT_FILE", "WINDOWS_CERT_PASSWORD")
MACOS_SIGN_ENV = (
    "APPLE_CERT_P12",
    "APPLE_CERT_PASSWORD",
    "APPLE_ID",
    "APPLE_TEAM_ID",
    "APPLE_APP_PASSWORD",
)

WINDOWS_TIMESTAMP_URL = "http://timestamp.digicert.com"

APPIMAGE_TOOL_UNAVAILABLE = 3


@dataclass(frozen=True)
class SigningPlan:
    """Решение о подписи для целевой ОС и значения из окружения."""

    target: str
    enabled: bool
    missing: tuple[str, ...] = ()
    values: Mapping[str, str] = field(default_factory=dict)


def host_target() -> str:
    """Целевая ОС текущей машины в терминах ``--target``."""
    os_name = platform.system().lower()
    target = _OS_TO_TARGET.get(os_name)
    if target is None:
        raise SystemExit(f"Неподдерживаемая host-ОС: {platform.system()!r}")
    return target


def repo_root() -> Path:
    """Корень репозитория (на два уровня выше ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


def executable_name(target: str) -> str:
    """Имя исполняемого файла бандла для целевой ОС."""
    return EXECUTABLE_NAMES[target]


def bundle_executable(bundle_dir: Path, target: str) -> Path:
    """Путь к исполняемому файлу внутри one-dir бандла."""
    return Path(bundle_dir) / executable_name(target)


def installer_script(target: str, name: str) -> Path:
    """Путь к платформенному сборщику внутри ``packaging/installers/``."""
    return repo_root() / INSTALLERS_SUBDIR / target / name


def validate_target(target: str, host: str) -> None:
    """Проверяет, что целевая ОС совпадает с host (иначе — понятная ошибка)."""
    if target not in SUPPORTED_TARGETS:
        raise SystemExit(
            f"Неизвестный --target {target!r}. Допустимо: {', '.join(SUPPORTED_TARGETS)}."
        )
    if target != host:
        raise SystemExit(
            f"Инсталляторы не кросс-собираются: запрошен --target {target!r}, "
            f"а эта машина — {host!r}. Запускайте сборку на целевой ОС "
            "(в CI runner совпадает с matrix-ОС)."
        )


def build_parser() -> argparse.ArgumentParser:
    """CLI-парсер аргументов сборки инсталлятора."""
    parser = argparse.ArgumentParser(
        prog="build_installer.py",
        description="Собрать инсталлятор audio-transcriber (Inno Setup / .dmg / AppImage).",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=SUPPORTED_TARGETS,
        help="Целевая ОС (обязана совпадать с host-ОС).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Каталог one-dir бандла (результат build_portable.py).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Каталог, куда положить инсталлятор и .sha256.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    return build_parser().parse_args(argv)


def parse_version(text: str, default: str = "0.0.0") -> str:
    """Извлекает ``X.Y.Z`` из вывода ``--version`` (отбрасывает dev-суффикс)."""
    match = re.search(r"(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else default


def detect_version(exe: Path) -> str:
    """Определяет версию по ``<exe> --version`` с безопасным фолбэком."""
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "0.0.0"
    return parse_version(result.stdout or result.stderr)


def sha256_file(path: Path) -> str:
    """SHA-256 файла в hex."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: Path) -> Path:
    """Пишет рядом с файлом ``<name>.sha256`` в формате ``sha256sum``."""
    sidecar = Path(path).with_name(Path(path).name + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {Path(path).name}\n", encoding="utf-8")
    return sidecar


def find_artifacts(out_dir: Path, target: str) -> list[Path]:
    """Возвращает собранные артефакты целевой ОС из ``out_dir``."""
    found: list[Path] = []
    for pattern in ARTIFACT_GLOBS[target]:
        found.extend(sorted(Path(out_dir).glob(pattern)))
    return found


def signing_plan(target: str, env: Mapping[str, str] | None = None) -> SigningPlan:
    """Определяет необходимость и параметры подписи по переменным окружения.

    Подпись опциональна: если обязательных переменных нет, возвращается
    выключенный план с перечнем отсутствующих имён, а сборка не падает.
    """
    values = dict(os.environ if env is None else env)
    if target == "linux":
        return SigningPlan(target=target, enabled=False)
    required = WINDOWS_SIGN_ENV if target == "windows" else MACOS_SIGN_ENV
    missing = tuple(name for name in required if not values.get(name))
    if missing:
        return SigningPlan(target=target, enabled=False, missing=missing)
    selected = {name: values[name] for name in required}
    extra = ("APPLE_SIGN_IDENTITY",)
    for name in extra:
        if values.get(name):
            selected[name] = values[name]
    return SigningPlan(target=target, enabled=True, values=selected)


def materialize_certificate(raw: str, suffix: str, workdir: Path) -> Path:
    """Готовит файл сертификата: путь как есть либо base64 → временный файл."""
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    try:
        data = base64.b64decode("".join(raw.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SystemExit(
            f"Сертификат задан не путём и не корректным base64 ({suffix}): {exc}"
        ) from exc
    dest = Path(workdir) / f"certificate{suffix}"
    dest.write_bytes(data)
    return dest


def signtool_command(cert: Path, password: str, artifact: Path) -> list[str]:
    """Команда ``signtool`` для подписи артефакта Windows."""
    return [
        "signtool",
        "sign",
        "/f",
        str(cert),
        "/p",
        password,
        "/fd",
        "sha256",
        "/tr",
        WINDOWS_TIMESTAMP_URL,
        "/td",
        "sha256",
        str(artifact),
    ]


def codesign_command(identity: str, target: Path) -> list[str]:
    """Команда ``codesign`` для подписи .app с hardened runtime."""
    return [
        "codesign",
        "--force",
        "--deep",
        "--options",
        "runtime",
        "--timestamp",
        "--sign",
        identity,
        str(target),
    ]


def notarytool_command(dmg: Path, apple_id: str, team_id: str, password: str) -> list[str]:
    """Команда ``notarytool submit --wait`` для нотаризации .dmg."""
    return [
        "xcrun",
        "notarytool",
        "submit",
        str(dmg),
        "--apple-id",
        apple_id,
        "--team-id",
        team_id,
        "--password",
        password,
        "--wait",
    ]


def stapler_command(dmg: Path) -> list[str]:
    """Команда ``stapler staple`` для прикрепления нотариального тикета."""
    return ["xcrun", "stapler", "staple", str(dmg)]


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    """Запускает команду, печатает её и возвращает код завершения."""
    print("+ " + " ".join(cmd), flush=True)
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"Не найдена команда {cmd[0]!r}: {exc}") from exc
    if check and result.returncode != 0:
        raise SystemExit(f"Команда завершилась с кодом {result.returncode}: {cmd[0]}")
    return result.returncode


def _format_size(num_bytes: int) -> str:
    """Человекочитаемый размер файла."""
    size = float(num_bytes)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ"):
        if size < 1024.0 or unit == "ГиБ":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} ГиБ"


def generate_icons(
    out_dir: Path, source: Path = ICON_SOURCE, name: str = BUNDLE_DIRNAME
) -> dict[str, Path]:
    """Генерирует PNG/ICO/ICNS и возвращает пути к ним."""
    script = repo_root() / INSTALLERS_SUBDIR / "generate_icons.py"
    if not script.is_file():
        raise SystemExit(f"Не найден генератор иконок: {script}")
    icon_source = Path(source)
    if not icon_source.is_absolute():
        icon_source = repo_root() / icon_source
    icons_dir = Path(out_dir) / "icons"
    _run(
        [
            sys.executable,
            str(script),
            "--source",
            str(icon_source),
            "--out-dir",
            str(icons_dir),
            "--name",
            name,
        ]
    )
    return {
        "png": icons_dir / f"{name}.png",
        "ico": icons_dir / f"{name}.ico",
        "icns": icons_dir / f"{name}.icns",
    }


def _report_signing(target: str, plan: SigningPlan) -> None:
    """Печатает понятное сообщение о пропуске/неприменимости подписи."""
    if target == "linux":
        print("Подпись не требуется для linux.")
    elif not plan.enabled:
        print(f"Подпись пропущена ({target}): не заданы {', '.join(plan.missing)}.")


def _sign_windows(artifact: Path, plan: SigningPlan) -> None:
    """Подписывает артефакт Windows через ``signtool``."""
    with tempfile.TemporaryDirectory(prefix="audio-transcriber-sign-") as tmp:
        cert = materialize_certificate(plan.values["WINDOWS_CERT_FILE"], ".pfx", Path(tmp))
        cmd = signtool_command(cert, plan.values["WINDOWS_CERT_PASSWORD"], artifact)
        cmd[0] = shutil.which("signtool") or cmd[0]
        _run(cmd)


def _import_p12(cert: Path, password: str) -> str:
    """Импортирует P12 во временную связку ключей и возвращает её имя."""
    keychain = "audio-transcriber-signing.keychain-db"
    _run(["security", "create-keychain", "-p", "", keychain], check=False)
    _run(["security", "unlock-keychain", "-p", "", keychain])
    _run(
        ["security", "import", str(cert), "-k", keychain, "-P", password, "-T", "/usr/bin/codesign"]
    )
    _run(
        [
            "security",
            "set-key-partition-list",
            "-S",
            "apple-tool:,apple:",
            "-s",
            "-k",
            "",
            keychain,
        ]
    )
    return keychain


def _detect_codesign_identity() -> str:
    """Находит identity ``Developer ID Application`` в связке ключей."""
    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        match = re.search(r'"(Developer ID Application[^"]+)"', line)
        if match:
            return match.group(1)
    raise SystemExit("Не найден identity 'Developer ID Application' в связке ключей.")


def _sign_macos_app(app_dir: Path, plan: SigningPlan) -> None:
    """Импортирует сертификат и подписывает .app."""
    with tempfile.TemporaryDirectory(prefix="audio-transcriber-sign-") as tmp:
        cert = materialize_certificate(plan.values["APPLE_CERT_P12"], ".p12", Path(tmp))
        _import_p12(cert, plan.values["APPLE_CERT_PASSWORD"])
        identity = plan.values.get("APPLE_SIGN_IDENTITY") or _detect_codesign_identity()
        _run(codesign_command(identity, app_dir))


def _notarize_macos(dmg: Path, plan: SigningPlan) -> None:
    """Нотаризует .dmg и прикрепляет тикет через ``stapler``."""
    _run(
        notarytool_command(
            dmg,
            plan.values["APPLE_ID"],
            plan.values["APPLE_TEAM_ID"],
            plan.values["APPLE_APP_PASSWORD"],
        )
    )
    _run(stapler_command(dmg))


def _build_linux(
    bundle_dir: Path, out_dir: Path, icons: Mapping[str, Path], version: str
) -> list[Path]:
    """Собирает AppImage, при недоступности тулчейна — фолбэк .deb/.tar.gz."""
    script = installer_script("linux", "build-appimage.sh")
    rc = _run(
        [
            "bash",
            str(script),
            "--bundle-dir",
            str(bundle_dir),
            "--out-dir",
            str(out_dir),
            "--icon",
            str(icons["png"]),
            "--version",
            version,
        ],
        check=False,
    )
    if rc == APPIMAGE_TOOL_UNAVAILABLE:
        print("AppImage-тулчейн недоступен; собираю фолбэк .deb и .tar.gz.")
        _run(
            [
                "bash",
                str(installer_script("linux", "build-fallback.sh")),
                "--bundle-dir",
                str(bundle_dir),
                "--out-dir",
                str(out_dir),
                "--icon",
                str(icons["png"]),
                "--version",
                version,
            ]
        )
    elif rc != 0:
        raise SystemExit(f"Сборка AppImage завершилась с кодом {rc}.")
    return find_artifacts(out_dir, "linux")


def _build_windows(
    bundle_dir: Path, out_dir: Path, icons: Mapping[str, Path], version: str
) -> list[Path]:
    """Собирает установщик Inno Setup и при наличии env подписывает его."""
    _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(installer_script("windows", "build-inno.ps1")),
            "-BundleDir",
            str(bundle_dir),
            "-OutDir",
            str(out_dir),
            "-Icon",
            str(icons["ico"]),
            "-Version",
            version,
        ]
    )
    return find_artifacts(out_dir, "windows")


def _build_macos(
    bundle_dir: Path,
    out_dir: Path,
    icons: Mapping[str, Path],
    version: str,
    plan: SigningPlan,
) -> list[Path]:
    """Собирает .app + .dmg; подпись и нотаризация — только при наличии env."""
    _run(
        [
            "bash",
            str(installer_script("macos", "build-app.sh")),
            "--bundle-dir",
            str(bundle_dir),
            "--out-dir",
            str(out_dir),
            "--icon",
            str(icons["icns"]),
            "--version",
            version,
        ]
    )
    app_dir = out_dir / "AudioTranscriber.app"
    if plan.enabled:
        _sign_macos_app(app_dir, plan)
    _run(
        [
            "bash",
            str(installer_script("macos", "build-dmg.sh")),
            "--app",
            str(app_dir),
            "--out-dir",
            str(out_dir),
            "--version",
            version,
        ]
    )
    artifacts = find_artifacts(out_dir, "macos")
    if plan.enabled:
        for artifact in artifacts:
            _notarize_macos(artifact, plan)
    return artifacts


def build(target: str, bundle_dir: Path, out_dir: Path) -> list[Path]:
    """Собирает инсталлятор(ы), подписывает при наличии env и пишет .sha256."""
    host = host_target()
    validate_target(target, host)

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    exe = bundle_executable(bundle_dir, target)
    if not exe.is_file():
        raise SystemExit(
            f"Не найден исполняемый файл бандла: {exe}. "
            "Сначала соберите бандл: python scripts/build_portable.py "
            f"--target {target} --out {bundle_dir.parent}"
        )

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    version = detect_version(exe)
    print(f"Версия бандла: {version}")
    icons = generate_icons(out_dir)

    plan = signing_plan(target, os.environ)
    _report_signing(target, plan)

    if target == "linux":
        artifacts = _build_linux(bundle_dir, out_dir, icons, version)
    elif target == "windows":
        artifacts = _build_windows(bundle_dir, out_dir, icons, version)
        if plan.enabled:
            for artifact in artifacts:
                _sign_windows(artifact, plan)
    else:
        artifacts = _build_macos(bundle_dir, out_dir, icons, version, plan)

    if not artifacts:
        raise SystemExit(f"Сборка завершилась, но артефакты не найдены в {out_dir}.")

    for artifact in artifacts:
        write_sha256_sidecar(artifact)
        print(f"\nГотово: {artifact}")
        print(f"Размер: {_format_size(artifact.stat().st_size)}")
        print(f"SHA-256: {sha256_file(artifact)}")
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа скрипта."""
    args = parse_args(argv)
    build(args.target, args.bundle_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
