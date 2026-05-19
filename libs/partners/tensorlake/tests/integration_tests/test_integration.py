from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from tensorlake.sandbox import Sandbox

from langchain_tensorlake import TensorlakeSandbox

if TYPE_CHECKING:
    from collections.abc import Generator

_MISSING_API_KEY_MSG = "Missing TENSORLAKE_API_KEY for Tensorlake integration test"


class TestTensorlakeSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Generator[TensorlakeSandbox, None, None]:
        api_key = os.environ.get("TENSORLAKE_API_KEY")
        org_id = os.environ.get("TENSORLAKE_ORGANIZATION_ID")
        project_id = os.environ.get("TENSORLAKE_PROJECT_ID")

        if not api_key:
            raise RuntimeError(_MISSING_API_KEY_MSG)

        sdk_sandbox = Sandbox.create(
            api_key=api_key,
            organization_id=org_id,
            project_id=project_id,
        )
        try:
            yield TensorlakeSandbox(sandbox=sdk_sandbox)
        finally:
            sdk_sandbox.terminate()
