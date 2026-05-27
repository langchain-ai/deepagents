"""Standard sandbox integration tests against a real CubeSandbox.

These tests require a reachable CubeSandbox deployment. Set the following
environment variables before running:

* `CUBE_API_URL` — base URL of the CubeAPI control plane.
* `CUBE_API_KEY` — optional, depending on the deployment.
* `CUBE_TEMPLATE_ID` — template ID to launch the sandbox from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cubesandbox
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_cubesandbox import CubeSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestCubeSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        sb = cubesandbox.Sandbox.create()
        backend = CubeSandbox(sandbox=sb)
        try:
            yield backend
        finally:
            sb.kill()
