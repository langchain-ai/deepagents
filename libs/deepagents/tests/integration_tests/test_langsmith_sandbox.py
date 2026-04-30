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


SNAPSHOT_NAME = "deepagents-cli"
DEFAULT_IMAGE = "python:3"
DEFAULT_FS_CAPACITY = 16 * 1024**3  # 16 GiB -- mirrors CLI _LangSmithProvider default.


class TestLangSmithSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if not api_key:
            msg = "Missing secrets for LangSmith integration test: set LANGSMITH_API_KEY"
            raise RuntimeError(msg)

        client = SandboxClient(api_key=api_key)

        # Server-side filter keeps this quick even with many snapshots in the
        # workspace. name_contains is a case-insensitive substring match, so
        # match the exact name client-side.
        existing = client.list_snapshots(name_contains=SNAPSHOT_NAME)
        ready = any(snap.name == SNAPSHOT_NAME and snap.status == "ready" for snap in existing)
        if not ready:
            client.create_snapshot(
                name=SNAPSHOT_NAME,
                docker_image=DEFAULT_IMAGE,
                fs_capacity_bytes=DEFAULT_FS_CAPACITY,
            )

        ls_sandbox = client.create_sandbox(snapshot_name=SNAPSHOT_NAME)
        backend = LangSmithSandbox(sandbox=ls_sandbox)
        try:
            yield backend
        finally:
            # Never delete the snapshot -- it is shared across test runs.
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
