"""Тесты аргументов и утилит скрипта сборки инсталляторов.

Реальная сборка (AppImage/Inno Setup/.dmg) не запускается: проверяются чистые
функции — валидация цели, поиск артефактов, план подписи и команды подписи.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "build_installer.py"


def _load_build_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_installer", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_installer = _load_build_module()


def test_host_target_supported() -> None:
    assert build_installer.host_target() in build_installer.SUPPORTED_TARGETS


@pytest.mark.parametrize("target", ["linux", "windows", "macos"])
def test_validate_target_accepts_host(target: str) -> None:
    build_installer.validate_target(target, target)


def test_validate_target_rejects_cross_compile() -> None:
    with pytest.raises(SystemExit):
        build_installer.validate_target("windows", "linux")


def test_validate_target_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        build_installer.validate_target("plan9", "plan9")


def test_executable_names() -> None:
    assert build_installer.executable_name("windows") == "audio-transcriber.exe"
    assert build_installer.executable_name("linux") == "audio-transcriber"
    assert build_installer.executable_name("macos") == "audio-transcriber"


def test_bundle_executable() -> None:
    assert build_installer.bundle_executable(Path("/tmp/b"), "windows") == Path(
        "/tmp/b/audio-transcriber.exe"
    )


def test_macos_app_dir() -> None:
    assert build_installer.macos_app_dir(Path("/tmp/out")) == Path("/tmp/out/AudioTranscriptor.app")


def test_macos_app_dir_matches_builder() -> None:
    script = _REPO_ROOT / "packaging" / "installers" / "macos" / "build-app.sh"
    content = script.read_text(encoding="utf-8")
    assert f'APP_ID="{build_installer.MACOS_APP_NAME}"' in content


def test_parse_args_with_bundle_dir() -> None:
    args = build_installer.parse_args(
        ["--target", "linux", "--bundle-dir", "/tmp/bundle", "--out", "/tmp/out"]
    )
    assert args.target == "linux"
    assert args.bundle_dir == Path("/tmp/bundle")
    assert args.out == Path("/tmp/out")


def test_parse_args_default_bundle_dir() -> None:
    args = build_installer.parse_args(["--target", "linux", "--out", "/tmp/out"])
    assert args.bundle_dir == build_installer.DEFAULT_BUNDLE_DIR


def test_parse_args_requires_target_and_out() -> None:
    with pytest.raises(SystemExit):
        build_installer.parse_args([])


def test_parse_version_strips_dev_suffix() -> None:
    assert build_installer.parse_version("audio-transcriber 0.5.9.dev12+g397383ec1") == "0.5.9"


def test_parse_version_fallback() -> None:
    assert build_installer.parse_version("no digits here", default="1.2.3") == "1.2.3"


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    assert build_installer.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_write_sha256_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "audio.AppImage"
    path.write_bytes(b"data")
    sidecar = build_installer.write_sha256_sidecar(path)
    assert sidecar.name == "audio.AppImage.sha256"
    assert sidecar.read_text(encoding="utf-8") == (
        f"{hashlib.sha256(b'data').hexdigest()}  audio.AppImage\n"
    )


def test_find_artifacts_linux(tmp_path: Path) -> None:
    (tmp_path / "audio-transcriber-1.0-x86_64.AppImage").write_text("x")
    (tmp_path / "audio-transcriber_1.0_amd64.deb").write_text("x")
    (tmp_path / "audio-transcriber-1.0-linux-x86_64.tar.gz").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    found = build_installer.find_artifacts(tmp_path, "linux")
    assert [path.name for path in found] == [
        "audio-transcriber-1.0-x86_64.AppImage",
        "audio-transcriber_1.0_amd64.deb",
        "audio-transcriber-1.0-linux-x86_64.tar.gz",
    ]


def test_find_artifacts_windows(tmp_path: Path) -> None:
    (tmp_path / "audio-transcriber-1.0-setup.exe").write_text("x")
    (tmp_path / "other.exe").write_text("x")
    found = build_installer.find_artifacts(tmp_path, "windows")
    assert [path.name for path in found] == ["audio-transcriber-1.0-setup.exe"]


def test_signing_plan_linux_disabled() -> None:
    plan = build_installer.signing_plan("linux", {})
    assert plan.enabled is False
    assert plan.missing == ()


def test_signing_plan_windows_missing_env() -> None:
    plan = build_installer.signing_plan("windows", {"WINDOWS_CERT_FILE": "x"})
    assert plan.enabled is False
    assert plan.missing == ("WINDOWS_CERT_PASSWORD",)


def test_signing_plan_windows_enabled() -> None:
    env = {"WINDOWS_CERT_FILE": "cert.pfx", "WINDOWS_CERT_PASSWORD": "secret"}
    plan = build_installer.signing_plan("windows", env)
    assert plan.enabled is True
    assert plan.values["WINDOWS_CERT_PASSWORD"] == "secret"


def test_signing_plan_macos_enabled() -> None:
    env = {
        "APPLE_CERT_P12": "cert.p12",
        "APPLE_CERT_PASSWORD": "certpass",
        "APPLE_ID": "dev@example.com",
        "APPLE_TEAM_ID": "TEAM123456",
        "APPLE_APP_PASSWORD": "app-pass",
    }
    plan = build_installer.signing_plan("macos", env)
    assert plan.enabled is True
    assert plan.missing == ()
    assert plan.values["APPLE_ID"] == "dev@example.com"


def test_signtool_command() -> None:
    cmd = build_installer.signtool_command(Path("/tmp/c.pfx"), "pw", Path("/tmp/setup.exe"))
    assert cmd[:2] == ["signtool", "sign"]
    assert "/f" in cmd and "/tmp/c.pfx" in cmd
    assert cmd[-1] == "/tmp/setup.exe"


def test_codesign_command() -> None:
    cmd = build_installer.codesign_command("Developer ID Application: X", Path("/tmp/App.app"))
    assert cmd[:2] == ["codesign", "--force"]
    assert cmd[-1] == "/tmp/App.app"
    assert cmd[-2] == "Developer ID Application: X"


def test_notarytool_command() -> None:
    cmd = build_installer.notarytool_command(Path("/tmp/a.dmg"), "id", "team", "pw")
    assert cmd[:3] == ["xcrun", "notarytool", "submit"]
    assert "--wait" in cmd
    assert "/tmp/a.dmg" in cmd


def test_stapler_command() -> None:
    assert build_installer.stapler_command(Path("/tmp/a.dmg")) == [
        "xcrun",
        "stapler",
        "staple",
        "/tmp/a.dmg",
    ]


def test_materialize_certificate_from_base64(tmp_path: Path) -> None:
    raw = base64.b64encode(b"cert-bytes").decode()
    path = build_installer.materialize_certificate(raw, ".pfx", tmp_path)
    assert path.read_bytes() == b"cert-bytes"


def test_materialize_certificate_from_path(tmp_path: Path) -> None:
    cert = tmp_path / "cert.p12"
    cert.write_bytes(b"x")
    assert build_installer.materialize_certificate(str(cert), ".p12", tmp_path) == cert
