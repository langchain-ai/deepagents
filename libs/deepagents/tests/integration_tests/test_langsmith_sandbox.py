from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from langsmith.sandbox import SandboxClient

from deepagents.backends.langsmith import LangSmithSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestLangSmithSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if not api_key:
            msg = "Missing secrets for LangSmith integration test: set LANGSMITH_API_KEY"
            raise RuntimeError(msg)

        client = SandboxClient(api_key=api_key)
        ls_sandbox = client.create_sandbox(template_name="deepagents-cli")
        backend = LangSmithSandbox(sandbox=ls_sandbox)
        try:
            yield backend
        finally:
            client.delete_sandbox(ls_sandbox.name)

    @pytest.mark.xfail(reason="LangSmith runs as root and ignores file permissions")
    def test_download_error_permission_denied(self, sandbox_backend: SandboxBackendProtocol) -> None:
        super().test_download_error_permission_denied(sandbox_backend)

    @pytest.mark.xfail(strict=True, reason="Upstream langchain_tests uses `in` on ReadResult dataclass")
    def test_read_basic_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        super().test_read_basic_file(sandbox_backend)

    @pytest.mark.xfail(strict=True, reason="Upstream langchain_tests uses `in` on ReadResult dataclass")
    def test_edit_single_occurrence(self, sandbox_backend: SandboxBackendProtocol) -> None:
        super().test_edit_single_occurrence(sandbox_backend)

    @pytest.mark.xfail(
        strict=True,
        reason="LangSmithSandbox.write() bypasses existence check; fix stashed",
    )
    def test_write_existing_file_fails(self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str) -> None:
        super().test_write_existing_file_fails(sandbox_backend, sandbox_test_root)

    @pytest.mark.xfail(
        strict=True,
        reason="BaseSandbox.read() via execute() hangs on large content over websocket; fix stashed",
    )
    async def test_awrite_aread_adownload_large_text_with_escaped_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        await super().test_awrite_aread_adownload_large_text_with_escaped_content(sandbox_backend, sandbox_test_root)
