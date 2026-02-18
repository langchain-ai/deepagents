from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

import daytona
from langchain_daytona import DaytonaSandbox


class TestDaytonaSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        sdk = daytona.Daytona()
        sandbox = sdk.create()
        backend = DaytonaSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            sandbox.delete()
