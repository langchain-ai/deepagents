"""Tests for the Switchyard runtime artifact assembler."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "evals" / "assemble_switchyard_runtime.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assemble_switchyard_runtime", SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assemble = _load_script()


def _adapter(root: Path) -> Path:
    adapter = root / "langchain_nvidia_switchyard"
    adapter.mkdir()
    (adapter / "__init__.py").write_text("adapter = True\n")
    (adapter / "middleware.py").write_text("middleware = True\n")
    return adapter


def _wheel(path: Path, *, native: bool = True, unsafe: bool = False) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("switchyard/libsy/algorithms.py", "noop = True\n")
        if native:
            archive.writestr(
                "switchyard_rust/_switchyard_rust.cpython-312-x86_64-linux-gnu.so",
                b"native",
            )
        if unsafe:
            archive.writestr("../outside.py", "bad = True\n")
    return path


def test_assembles_wheel_and_adapter_without_installing(tmp_path: Path) -> None:
    output = tmp_path / "runtime"

    assemble.assemble_runtime(
        _wheel(tmp_path / "switchyard.whl"),
        _adapter(tmp_path),
        output,
    )

    assert (output / "switchyard" / "libsy" / "algorithms.py").is_file()
    assert (output / "switchyard_rust" / "_switchyard_rust.cpython-312-x86_64-linux-gnu.so").is_file()
    assert (output / "langchain_nvidia_switchyard" / "middleware.py").is_file()


def test_rejects_wheel_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe wheel member"):
        assemble.assemble_runtime(
            _wheel(tmp_path / "switchyard.whl", unsafe=True),
            _adapter(tmp_path),
            tmp_path / "runtime",
        )

    assert not (tmp_path / "outside.py").exists()


def test_rejects_wheel_without_native_extension(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="native Switchyard extension"):
        assemble.assemble_runtime(
            _wheel(tmp_path / "switchyard.whl", native=False),
            _adapter(tmp_path),
            tmp_path / "runtime",
        )
