from __future__ import annotations

import shutil

import pytest

from langchain_wasmsh import WasmshSandbox

pytestmark = pytest.mark.skipif(
    shutil.which("deno") is None and shutil.which("node") is None,
    reason="deno or node is required for langchain-wasmsh tests",
)


def test_execute_supports_bash_and_python() -> None:
    sandbox = WasmshSandbox(initial_files={"/workspace/seed.txt": b"seed\n"})
    try:
        result = sandbox.execute(
            "cat seed.txt && python3 -c "
            "\"print(open('/workspace/seed.txt')"
            '.read().strip())"'
        )

        assert result.exit_code == 0
        assert "seed" in result.output
    finally:
        sandbox.close()


def test_upload_and_download_roundtrip() -> None:
    sandbox = WasmshSandbox()
    try:
        upload = sandbox.upload_files([("/workspace/demo.txt", b"demo")])
        download = sandbox.download_files(["/workspace/demo.txt"])

        assert upload[0].error is None
        assert download[0].error is None
        assert download[0].content == b"demo"
    finally:
        sandbox.close()
