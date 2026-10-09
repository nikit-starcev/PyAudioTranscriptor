# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_DIR = os.path.abspath(SPECPATH)

# Пакеты с динамическими/ленивыми импортами, которые статический анализ
# PyInstaller не проследует и которые нужны CLI и веб-серверу.
_PACKAGES = (
    "audio_transcriber",
    "uvicorn",
    "textual",
    "faster_whisper",
    "pymorphy3",
    "pymorphy3_dicts_ru",
    "pyannote.audio",
    "pyannoteai",
    "lightning",
    "lightning_fabric",
    "pytorch_lightning",
    "torchmetrics",
    "pytorch_metric_learning",
    "torch_audiomentations",
    "huggingface_hub",
)

hiddenimports = []
for _package in _PACKAGES:
    hiddenimports += collect_submodules(_package)

# Собранный SPA и иные ресурсы пакета; данные VAD faster-whisper; словари pymorphy3.
datas = collect_data_files("audio_transcriber")
datas += collect_data_files("faster_whisper")
datas += collect_data_files("pymorphy3_dicts_ru")
# pyannote.audio читает свои ресурсы из пакета (telemetry/config.yaml, sample/*).
datas += collect_data_files("pyannote.audio")
# Метаданные дистрибутива нужны importlib.metadata.version("audio-transcriber"):
# без них __version__ падает в fallback и бандл печатает неверную версию.
datas += copy_metadata("audio-transcriber")

# torchcodec не нужен: диаризация получает waveform in-memory (utils/audio.py).
# Его нативные библиотеки зависят от системного FFmpeg и CUDA, а pyannote.audio
# корректно деградирует при отсутствии модуля. CUDA/nvidia и лишние тяжёлые
# зависимости в бандл не попадают.
excludes = [
    "torchcodec",
    "nvidia",
    "torchvision",
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "mypy",
    "ruff",
    "torch.test",
    "torch.utils.tensorboard",
    "sherpa_onnx",
    "onnx_asr",
]

a = Analysis(
    [os.path.join(SPEC_DIR, "entry.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="audio-transcriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="audio-transcriber",
)
