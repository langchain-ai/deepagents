from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_novita import NovitaSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestNovitaSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        from novita_sandbox.code_interpreter import Sandbox

        sdk_sandbox = Sandbox.create()
        backend = NovitaSandbox(sandbox=sdk_sandbox)
        try:
            yield backend
        finally:
            sdk_sandbox.kill()
