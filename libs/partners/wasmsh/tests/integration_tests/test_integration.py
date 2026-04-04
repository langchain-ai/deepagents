"""Integration tests using the langchain-tests standard sandbox suite."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_wasmsh import WasmshSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

try:
    from wasmsh_pyodide_runtime import get_dist_dir

    _assets_available = get_dist_dir().joinpath("pyodide.asm.wasm").exists()
except (ImportError, FileNotFoundError):
    _assets_available = False


@pytest.mark.skipif(
    shutil.which("deno") is None and shutil.which("node") is None,
    reason="deno or node is required for langchain-wasmsh integration tests",
)
@pytest.mark.skipif(
    not _assets_available,
    reason=(
        "Pyodide assets not built"
        " (run just build-pyodide"
        " && just package-pyodide-runtime)"
    ),
)
class TestWasmshSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        backend = WasmshSandbox()
        try:
            yield backend
        finally:
            backend.close()

    @pytest.mark.xfail(
        reason=(
            "Emscripten VFS does not enforce chmod — permissions are silently ignored"
        ),
    )
    def test_download_error_permission_denied(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        super().test_download_error_permission_denied(sandbox_backend)
