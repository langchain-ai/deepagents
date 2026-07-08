from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_vercel_sandbox import VercelSandbox

vercel_sandbox = pytest.importorskip("vercel.sandbox")

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestVercelSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        sandbox = vercel_sandbox.Sandbox.create(
            runtime="python3.13",
            timeout=timedelta(minutes=30),
        )
        backend = VercelSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            sandbox.stop()
