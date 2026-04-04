"""End-to-end tests verifying cross-language workflows in the wasmsh sandbox.

These tests exercise the core use case: bash and Python sharing /workspace.
They require Deno 2+ and built Pyodide assets.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from langchain_wasmsh import WasmshSandbox

try:
    from wasmsh_pyodide_runtime import get_dist_dir

    _assets_available = get_dist_dir().joinpath("pyodide.asm.wasm").exists()
except (ImportError, FileNotFoundError):
    _assets_available = False


pytestmark = [
    pytest.mark.skipif(
        shutil.which("deno") is None and shutil.which("node") is None,
        reason="deno or node is required",
    ),
    pytest.mark.skipif(
        not _assets_available,
        reason="Pyodide assets not built",
    ),
]


def test_bash_write_python_validate(sandbox: WasmshSandbox) -> None:
    """Bash writes a JSON file, Python validates its schema."""
    r1 = sandbox.execute("""echo '{"name":"test","value":42}' > /workspace/data.json""")
    assert r1.exit_code == 0

    r2 = sandbox.execute(
        'python3 -c "'
        "import json; "
        "d = json.load(open('/workspace/data.json')); "
        "assert d['name'] == 'test', f'unexpected name: {d[\"name\"]}'; "
        "assert d['value'] == 42, f'unexpected value: {d[\"value\"]}'; "
        "print('valid')\""
    )
    assert r2.exit_code == 0
    assert "valid" in r2.output


def test_python_compute_bash_verify(sandbox: WasmshSandbox) -> None:
    """Python computes a value and writes it; bash reads and verifies."""
    r1 = sandbox.execute(
        'python3 -c "'
        "result = sum(range(1, 11)); "
        "open('/workspace/result.txt', 'w').write(str(result))\""
    )
    assert r1.exit_code == 0

    r2 = sandbox.execute("cat /workspace/result.txt")
    assert r2.exit_code == 0
    assert r2.output.strip() == "55"


def test_shared_workspace_multi_step(sandbox: WasmshSandbox) -> None:
    """Multi-step workflow: bash creates structure, python writes, bash reads."""
    r1 = sandbox.execute("mkdir -p /workspace/output")
    assert r1.exit_code == 0

    r2 = sandbox.execute(
        'python3 -c "'
        "import os\n"
        "for i in range(3):\n"
        "    open(f'/workspace/output/file_{i}.txt', 'w').write(f'content-{i}')\n"
        "print('wrote', len(os.listdir('/workspace/output')), 'files')\""
    )
    assert r2.exit_code == 0
    assert "wrote 3 files" in r2.output

    r3 = sandbox.execute("cat /workspace/output/file_1.txt")
    assert r3.exit_code == 0
    assert r3.output.strip() == "content-1"

    r4 = sandbox.execute("ls /workspace/output | wc -l")
    assert r4.exit_code == 0
    assert r4.output.strip() == "3"
