"""Build Script"""
import os
import sys
from pathlib import Path


def build_exe():
    """Build executable with PyInstaller"""
    import PyInstaller.__main__

    project_root = Path(__file__).parent.parent
    spec_file = project_root / "ecommerce_agent.spec"

    # Create spec file if not exists
    if not spec_file.exists():
        create_spec_file(spec_file, project_root)

    # Run PyInstaller
    PyInstaller.__main__.run([
        str(spec_file),
        "--clean"
    ])


def create_spec_file(spec_file: Path, project_root: Path):
    """Create PyInstaller spec file"""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{project_root}/backend/main.py'],
    pathex=['{project_root}'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'deepagents',
        'playwright',
        'uvicorn',
        'fastapi',
        'apscheduler',
        'chromadb',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    name='ecommerce_agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''

    with open(spec_file, "w", encoding="utf-8") as f:
        f.write(spec_content)


if __name__ == "__main__":
    build_exe()
