from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from e2b import Sandbox
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_e2b import E2BSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestE2BSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            pytest.skip("Missing secrets for E2B integration test: set E2B_API_KEY")

        sandbox = Sandbox.create(timeout=3600, api_key=api_key)
        backend = E2BSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            sandbox.kill()
