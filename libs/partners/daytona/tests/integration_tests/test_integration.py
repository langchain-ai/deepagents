from __future__ import annotations

from typing import TYPE_CHECKING

import daytona
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_daytona import DaytonaSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


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
