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
    def test_download_error_permission_denied(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        super().test_download_error_permission_denied(sandbox_backend)

    @pytest.mark.xfail(reason="LangSmith returns file_not_found for relative paths")
    def test_download_error_invalid_path_relative(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        super().test_download_error_invalid_path_relative(sandbox_backend)

    @pytest.mark.xfail(reason="LangSmith accepts relative paths on upload")
    def test_upload_relative_path_returns_invalid_path(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        super().test_upload_relative_path_returns_invalid_path(sandbox_backend)
