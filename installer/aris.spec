# PyInstaller spec for building the single-file ``aris`` executable.
#
# Build:
#   pyinstaller installer/aris.spec --clean --noconfirm
#
# On Windows the output is ``dist/aris.exe``. On Linux it's ``dist/aris``
# (we still build it on Linux as a smoke test of the spec).
#
# Notes
# -----
# * PyTorch ships a lot of large CUDA / cuDNN DLLs. PyInstaller's torch hook
#   already covers most of them, but we explicitly ``collect_all`` to be safe.
# * Tiktoken loads regex via cffi and bundles its mergeable BPE table at runtime;
#   we collect_all to make sure those resources land in the bundle.
# * FastAPI / Pydantic / uvicorn have a number of dynamic imports — we use
#   ``collect_submodules`` so PyInstaller picks them up.
# * The configs YAML files and any small static assets live alongside the
#   ``llm`` package and are pulled in via ``collect_data_files``.

from __future__ import annotations

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)

block_cipher = None

hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []


def _collect(pkg: str) -> None:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)


# Heavy ML / serving deps that PyInstaller cannot fully discover on its own.
# NOTE: torch has its own builtin PyInstaller hook — calling ``collect_all('torch')``
# on top of it causes ``ImportError: cannot load module more than once per process``
# at runtime. We rely on the builtin hook and only pull torch submodules that the
# spec needs (autocast, distributed, etc.) via ``collect_submodules`` below.
for _pkg in ("tiktoken", "tiktoken_ext", "sentencepiece", "fastapi", "starlette", "pydantic", "uvicorn", "prometheus_client"):
    try:
        _collect(_pkg)
    except Exception as exc:
        # Optional deps (e.g. sentencepiece) may be missing — that's fine.
        print(f"[aris.spec] skipped {_pkg}: {exc}")

hiddenimports += collect_submodules("torch.distributed")
hiddenimports += collect_submodules("torch.optim")
hiddenimports += collect_submodules("torch.nn")

# Make sure every Aris module is present even if only imported via string lookup
# (e.g. job dispatch in /finetune).
hiddenimports += collect_submodules("llm")

# Ship the YAML configs and docs alongside the binary so ``aris.exe info`` and
# ``aris.exe serve --config llm/configs/7b.yaml`` work out of the box.
datas += collect_data_files("llm", includes=["configs/*.yaml"])


a = Analysis(
    ["aris_cli.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Heavy and unused: shave a few hundred MB off the bundle.
        "matplotlib",
        "tkinter",
        "test",
        "tests",
        "PIL",
        "scipy",
        "pandas",
    ],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="aris",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
