"""Тесты единого реестра внешних ресурсов и автоустановки бинарников (#98).

Сеть не трогается: загрузка подменяется заглушкой ``fetch``, которая пишет
подготовленные байты. Проверяются allowlist, подбор артефакта по ОС/архитектуре,
проверка ``sha256`` (ok/fail), распаковка zip/tar, защита от path traversal,
авто-прописывание пути в настройках, статусы/SSE и сводка мастера.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tarfile
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio_transcriber.web import assets as web_assets
from audio_transcriber.web import setup as web_setup
from audio_transcriber.web.app import create_app
from audio_transcriber.web.models import DownloadBus
from audio_transcriber.web.paths import WebPaths
from audio_transcriber.web.settings import SettingsStore, WebSettings


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _artifact(data: bytes, **overrides: object) -> web_assets.AssetArtifact:
    fields: dict[str, object] = {
        "os": "linux",
        "arch": "x86_64",
        "url": "https://example.invalid/asset.tar.gz",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "archive": "tar.gz",
        "member": "whisper-cli",
    }
    fields.update(overrides)
    return web_assets.AssetArtifact(**fields)  # type: ignore[arg-type]


def _fake_fetch(data: bytes) -> Callable[[str, Path, Callable[[int], None]], None]:
    def fetch(url: str, destination: Path, on_progress: Callable[[int], None]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        on_progress(len(data))

    return fetch


# --- реестр и подбор артефакта --------------------------------------------


def test_registry_unique_and_kinds() -> None:
    keys = [asset.key for asset in web_assets.ASSETS]
    assert len(keys) == len(set(keys))
    assert {"gigaam", "sherpa", "deep-filter", "whisper-cli", "llama-server", "nemo-speech"} <= set(keys)
    for asset in web_assets.ASSETS:
        assert web_assets.find_asset(asset.key) is asset
        assert asset.check_id
        if asset.kind == web_assets.KIND_BINARY:
            assert asset.settings_field
            assert asset.bin_name
    assert web_assets.find_asset("nope") is None
    # URL/хеши фиксированы только у артефактов, без пользовательского ввода.
    for asset in web_assets.BINARY_ASSETS:
        for artifact in asset.artifacts:
            assert artifact.url.startswith("https://github.com/")
            assert len(artifact.sha256) == 64


def test_resolve_artifact_by_platform_and_variant() -> None:
    llama = web_assets.find_asset("llama-server")
    assert llama is not None
    cpu = web_assets.resolve_artifact(llama, os_name="linux", arch="x86_64")
    vulkan = web_assets.resolve_artifact(
        llama, os_name="linux", arch="x86_64", prefer_vulkan=True
    )
    assert cpu is not None and cpu.variant == "cpu"
    assert vulkan is not None and vulkan.variant == "vulkan"
    assert web_assets.resolve_artifact(llama, os_name="plan9", arch="x86_64") is None

    nemo = web_assets.find_asset("nemo-speech")
    assert nemo is not None
    assert web_assets.resolve_artifact(nemo) is None  # скачать нельзя — инструкция


def test_current_platform_is_known() -> None:
    assert web_assets.current_os() in {"linux", "macos", "windows"}
    assert web_assets.current_arch() in {"x86_64", "arm64"} or web_assets.current_arch()


# --- скачивание и распаковка ----------------------------------------------


def test_install_none_archive_sets_executable(tmp_path: Path) -> None:
    payload = b"#!/bin/sh\necho hi\n"
    asset = web_assets.find_asset("deep-filter")
    assert asset is not None
    artifact = _artifact(payload, archive="none", filename="deep-filter", member="")
    binary = web_assets.install_binary_asset(
        asset,
        artifact,
        bin_root=tmp_path / "bin",
        fetch=_fake_fetch(payload),
        on_progress=lambda _n: None,
        on_message=lambda _m: None,
    )
    assert binary.is_file()
    assert binary.read_bytes() == payload


def test_install_checksum_mismatch_removes_file(tmp_path: Path) -> None:
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    artifact = _artifact(b"good")
    with pytest.raises(web_assets.ChecksumError):
        web_assets.install_binary_asset(
            asset,
            artifact,
            bin_root=tmp_path / "bin",
            fetch=_fake_fetch(b"evil"),
            on_progress=lambda _n: None,
            on_message=lambda _m: None,
        )
    # Скачанный файл удаляется (каталог с ключом может остаться пустым).
    assert not (tmp_path / "bin" / "whisper-cli" / "asset.tar.gz").exists()


def test_install_tar_gz_finds_member(tmp_path: Path) -> None:
    data = _tar_gz({"whisper-bin-ubuntu-x64/whisper-cli": b"bin", "readme": b"x"})
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    binary = web_assets.install_binary_asset(
        asset,
        _artifact(data),
        bin_root=tmp_path / "bin",
        fetch=_fake_fetch(data),
        on_progress=lambda _n: None,
        on_message=lambda _m: None,
    )
    assert binary.name == "whisper-cli"
    assert binary.read_bytes() == b"bin"


def test_install_zip_finds_member(tmp_path: Path) -> None:
    data = _zip({"Release/whisper-cli.exe": b"pe"})
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    artifact = _artifact(data, archive="zip", member="whisper-cli.exe")
    binary = web_assets.install_binary_asset(
        asset,
        artifact,
        bin_root=tmp_path / "bin",
        fetch=_fake_fetch(data),
        on_progress=lambda _n: None,
        on_message=lambda _m: None,
    )
    assert binary.name == "whisper-cli.exe"


def test_zip_slip_rejected(tmp_path: Path) -> None:
    data = _zip({"../evil": b"x"})
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    with pytest.raises(web_assets.AssetError):
        web_assets.install_binary_asset(
            asset,
            _artifact(data, archive="zip"),
            bin_root=tmp_path / "bin",
            fetch=_fake_fetch(data),
            on_progress=lambda _n: None,
            on_message=lambda _m: None,
        )


def test_tar_traversal_rejected(tmp_path: Path) -> None:
    data = _tar_gz({"../evil": b"x"})
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    with pytest.raises(web_assets.AssetError):
        web_assets.install_binary_asset(
            asset,
            _artifact(data),
            bin_root=tmp_path / "bin",
            fetch=_fake_fetch(data),
            on_progress=lambda _n: None,
            on_message=lambda _m: None,
        )


def test_missing_member_in_archive(tmp_path: Path) -> None:
    data = _tar_gz({"other": b"x"})
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    with pytest.raises(web_assets.AssetError):
        web_assets.install_binary_asset(
            asset,
            _artifact(data),
            bin_root=tmp_path / "bin",
            fetch=_fake_fetch(data),
            on_progress=lambda _n: None,
            on_message=lambda _m: None,
        )


# --- BinaryInstaller (фоновый поток + настройки) ---------------------------


def _installer(tmp_path: Path, data: bytes) -> web_assets.BinaryInstaller:
    store = SettingsStore(tmp_path / "settings.json")
    return web_assets.BinaryInstaller(
        bus=DownloadBus(),
        bin_root=tmp_path / "bin",
        settings_store=store,
        fetch=_fake_fetch(data),
    )


def test_binary_installer_success_writes_settings(tmp_path: Path) -> None:
    data = _tar_gz({"whisper-bin-ubuntu-x64/whisper-cli": b"bin"})
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    installer = _installer(tmp_path, data)
    artifact = _artifact(data)
    assert installer.start(asset, artifact) is True
    installer.wait(5)

    state = installer.state("whisper-cli")
    assert state.status == web_assets.STATUS_DONE
    assert state.path.endswith("whisper-cli")

    saved = SettingsStore(tmp_path / "settings.json").load()
    assert saved.whisper_cpp_binary == state.path
    assert saved.whisper_cpp_lib_path == str(Path(state.path).parent)

    events = installer.bus.history()
    assert events[-1]["status"] == "done"
    assert events[-1]["key"] == "whisper-cli"


def test_binary_installer_checksum_error_keeps_settings(tmp_path: Path) -> None:
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    installer = _installer(tmp_path, b"evil")
    assert installer.start(asset, _artifact(b"good")) is True
    installer.wait(5)

    state = installer.state("whisper-cli")
    assert state.status == web_assets.STATUS_ERROR
    assert state.error is not None
    # Путь в настройках не тронут (остаётся значение по умолчанию).
    assert SettingsStore(tmp_path / "settings.json").load().whisper_cpp_binary == "whisper-cli"


def test_binary_installer_one_at_a_time(tmp_path: Path) -> None:
    block = threading.Event()
    data = _tar_gz({"whisper-cli": b"bin"})
    asset = web_assets.find_asset("whisper-cli")

    def blocking_fetch(url: str, destination: Path, on_progress: Callable[[int], None]) -> None:
        block.wait(5)
        destination.write_bytes(data)
        on_progress(len(data))

    installer = web_assets.BinaryInstaller(
        bus=DownloadBus(), bin_root=tmp_path / "bin", fetch=blocking_fetch
    )
    assert asset is not None
    assert installer.start(asset, _artifact(data)) is True
    assert installer.is_running() is True
    assert installer.start(asset, _artifact(data)) is False
    block.set()
    installer.wait(5)


# --- payload / endpoints ---------------------------------------------------


def test_asset_payload_binary_downloadable(tmp_path: Path) -> None:
    asset = web_assets.find_asset("whisper-cli")
    assert asset is not None
    payload = web_assets.asset_payload(
        asset,
        bin_root=tmp_path / "bin",
        settings=WebSettings(),
        os_name="linux",
        arch="x86_64",
    )
    assert payload["kind"] == "binary"
    assert payload["downloadable"] is True
    assert payload["installed"] is False
    artifact = payload["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["sha256"]


def test_asset_payload_pip_installed_flag(tmp_path: Path) -> None:
    asset = web_assets.find_asset("gigaam")
    assert asset is not None
    payload = web_assets.asset_payload(
        asset, bin_root=tmp_path / "bin", settings=WebSettings()
    )
    assert payload["kind"] == "pip"
    assert payload["downloadable"] is False
    assert isinstance(payload["installed"], bool)


@pytest.fixture
def web_paths(tmp_path: Path) -> WebPaths:
    return WebPaths(tmp_path / "web-data")


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("audio_transcriber.web.app.env_defaults", lambda: {})
    monkeypatch.setattr("audio_transcriber.web.settings.env_defaults", lambda: {})
    monkeypatch.setattr(
        "audio_transcriber.web.doctor_api.load_config_env", lambda: (None, {})
    )


def test_assets_endpoint_shape(web_paths: WebPaths) -> None:
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as client:
        payload = client.get("/api/assets").json()
    assert "bin_dir" in payload
    by_key = {item["key"]: item for item in payload["assets"]}
    assert {"gigaam", "whisper-cli", "llama-server", "nemo-speech"} <= set(by_key)
    assert by_key["nemo-speech"]["downloadable"] is False
    assert by_key["gigaam"]["kind"] == "pip"


def test_install_binary_via_endpoint(web_paths: WebPaths, tmp_path: Path) -> None:
    data = _tar_gz({"whisper-bin-ubuntu-x64/whisper-cli": b"bin"})
    artifact = _artifact(data)
    app = create_app(paths=web_paths, asset_downloader=_fake_fetch(data), heartbeat=0.05)
    # Подменяем подбор артефакта: реальный URL/хеш в реестре не совпал бы с тестовыми байтами.
    with TestClient(app) as client:
        import audio_transcriber.web.app as app_module

        original = app_module.resolve_artifact
        app_module.resolve_artifact = lambda _asset, **_kw: artifact  # type: ignore[assignment]
        try:
            response = client.post("/api/assets/whisper-cli/install")
            assert response.status_code == 202
            assert response.json()["kind"] == "binary"
            client.app.state.binary_installer.wait(5)  # type: ignore[attr-defined]
        finally:
            app_module.resolve_artifact = original  # type: ignore[assignment]

        state = client.app.state.binary_installer.state("whisper-cli")  # type: ignore[attr-defined]
        assert state.status == web_assets.STATUS_DONE
        saved = client.app.state.settings_store.load()  # type: ignore[attr-defined]
        assert saved.whisper_cpp_binary == state.path


def test_install_pip_via_assets_endpoint(web_paths: WebPaths) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], on_line: Callable[[str], None]) -> int:
        commands.append(command)
        on_line("Successfully installed")
        return 0

    app = create_app(paths=web_paths, dep_install_runner=runner, heartbeat=0.05)
    with TestClient(app) as client:
        response = client.post("/api/assets/gigaam/install")
        assert response.status_code == 202
        assert response.json()["kind"] == "pip"
        client.app.state.dependency_installer.wait(5)  # type: ignore[attr-defined]
        state = client.app.state.dependency_installer.state("gigaam")  # type: ignore[attr-defined]
        assert state.status == "done"
    assert commands and commands[0][-1] == "onnx-asr[cpu,hub]"


def test_install_unknown_asset_404(web_paths: WebPaths) -> None:
    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        assert client.post("/api/assets/nope/install").status_code == 404


def test_install_nemo_has_no_artifact_400(web_paths: WebPaths) -> None:
    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        response = client.post("/api/assets/nemo-speech/install")
    assert response.status_code == 400
    assert "соберите" in response.json()["detail"].lower()


def _stage_binary(web_paths: WebPaths, key: str) -> tuple[Path, Path]:
    """Кладёт бинарник в ``bin/<key>/<key>`` и возвращает (каталог, файл)."""
    directory = web_paths.bin_dir / key
    binary = directory / key
    directory.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"bin")
    return directory, binary


def test_delete_binary_asset_removes_dir_and_resets_settings(web_paths: WebPaths) -> None:
    directory, binary = _stage_binary(web_paths, "whisper-cli")
    app = create_app(paths=web_paths, heartbeat=0.05)
    with TestClient(app) as client:
        store = client.app.state.settings_store  # type: ignore[attr-defined]
        settings = store.load()
        settings.whisper_cpp_binary = str(binary)
        settings.whisper_cpp_lib_path = str(directory)
        store.save(settings)

        response = client.delete("/api/assets/whisper-cli")

        assert response.status_code == 200
        assert response.json()["deleted"] == "whisper-cli"
        assert not directory.exists()
        saved = store.load()
        assert saved.whisper_cpp_binary == "whisper-cli"
        assert saved.whisper_cpp_lib_path == ""

        payload = client.get("/api/assets").json()
        whisper = next(item for item in payload["assets"] if item["key"] == "whisper-cli")
        assert whisper["installed"] is False
        assert whisper["managed"] is False


def test_delete_binary_asset_not_installed_404(web_paths: WebPaths) -> None:
    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        assert client.delete("/api/assets/whisper-cli").status_code == 404


def test_delete_binary_asset_unknown_404(web_paths: WebPaths) -> None:
    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        assert client.delete("/api/assets/nope").status_code == 404


def test_delete_binary_asset_pip_400(web_paths: WebPaths) -> None:
    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        assert client.delete("/api/assets/gigaam").status_code == 400


def test_delete_binary_asset_refuses_outside_bin_dir(
    web_paths: WebPaths, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep")
    web_paths.bin_dir.mkdir(parents=True, exist_ok=True)
    link = web_paths.bin_dir / "whisper-cli"
    link.symlink_to(outside, target_is_directory=True)

    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        response = client.delete("/api/assets/whisper-cli")

    assert response.status_code == 400
    assert marker.exists()
    assert link.exists()


def _assets_events_route(app: object) -> object:
    stack = list(app.routes)  # type: ignore[attr-defined]
    while stack:
        route = stack.pop()
        if getattr(route, "path", None) == "/api/assets/events":
            return route
        nested = getattr(route, "routes", None)
        if nested:
            stack.extend(nested)
        original = getattr(route, "original_router", None)
        if original is not None:
            stack.extend(getattr(original, "routes", None) or [])
    raise AssertionError("маршрут /api/assets/events не найден")


def _read_assets_events(
    client: TestClient, *, last_event_id: str | None = None
) -> tuple[list[str], list[dict[str, object]]]:
    route = _assets_events_route(client.app)  # type: ignore[arg-type]
    endpoint = route.endpoint  # type: ignore[attr-defined]

    async def scenario() -> tuple[list[str], list[dict[str, object]]]:
        response = await endpoint(last_event_id=last_event_id)
        lines: list[str] = []
        events: list[dict[str, object]] = []
        agen = response.body_iterator
        try:
            async for chunk in agen:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("data:"):
                        events.append(json.loads(line[len("data:") :].strip()))
                        return lines, events
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await aclose()
        return lines, events

    return asyncio.run(scenario())


def test_assets_events_share_bus(web_paths: WebPaths) -> None:
    with TestClient(create_app(paths=web_paths, heartbeat=0.05)) as client:
        bus = client.app.state.deps_bus  # type: ignore[attr-defined]
        bus.clear()
        bus.publish({"key": "deep-filter", "status": "running", "message": "Скачивание…"})
        bus.publish({"key": "deep-filter", "status": "done", "message": "Установлено"})
        first_seq = int(bus.history()[0]["seq"])  # type: ignore[arg-type]

        lines, events = _read_assets_events(client, last_event_id=str(first_seq))

    assert [event["status"] for event in events] == ["done"]
    assert f"id: {events[0]['seq']}" in lines


# --- сводка мастера --------------------------------------------------------


def _report(checks: list[dict[str, object]], critical_failures: int = 0) -> dict[str, object]:
    return {"checks": checks, "summary": {"critical_failures": critical_failures}}


def _check(check_id: str, status: str) -> dict[str, object]:
    return {
        "id": check_id,
        "label": check_id,
        "status": status,
        "critical": status == "fail",
        "detail": "",
        "hint": "",
        "links": [],
    }


def test_setup_binary_requirement_has_download_metadata() -> None:
    requirements = web_setup.binary_requirements(WebSettings(asr_backend="whisper-cpp"))
    whisper = next(item for item in requirements if item["key"] == "whisper-cli")
    assert whisper["kind"] == "binary"
    assert whisper["asset_key"] == "whisper-cli"
    assert isinstance(whisper["downloadable"], bool)
    assert whisper["setting_key"] == "WHISPER_CPP_BINARY"
    assert whisper["lib_setting_key"] == "WHISPER_CPP_LIB_PATH"


def test_setup_readiness_summary() -> None:
    settings = WebSettings(asr_backend="whisper-cpp", llm_enabled=True)
    required = tuple(web_setup.required_model_ids(settings))
    report = _report([_check("bin:whisper-cli", "ok")])
    models = [
        {"id": model_id, "status": {"present": model_id == "whisper-large-v3-turbo"}}
        for model_id in required
    ]

    plan = web_setup.build_setup_steps(
        settings=settings,
        report=report,
        models=models,
        hf_token_set=True,
    )

    readiness = plan["readiness"]
    assert readiness["models"]
    assert readiness["dependencies"]
    assert readiness["binaries"]
    assert "pyannote-community-1" in readiness["missing_models"]
    assert "whisper-large-v3-turbo" not in readiness["missing_models"]
    # Каждый бинарный пункт несёт размер/платформу для отображения.
    whisper = next(item for item in readiness["binaries"] if item["key"] == "whisper-cli")
    assert "platform" in whisper
    assert "size" in whisper
