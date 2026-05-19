from __future__ import annotations

import os
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from tensorlake.sandbox import Sandbox

from langchain_tensorlake import TensorlakeSandbox


class TestTensorlakeSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self):
        api_key = os.environ.get("TENSORLAKE_API_KEY")
        org_id = os.environ.get("TENSORLAKE_ORGANIZATION_ID")
        project_id = os.environ.get("TENSORLAKE_PROJECT_ID")

        if not api_key:
            raise RuntimeError(
                "Missing TENSORLAKE_API_KEY for Tensorlake integration test"
            )

        sdk_sandbox = Sandbox.create(
            api_key=api_key,
            organization_id=org_id,
            project_id=project_id,
        )
        try:
            yield TensorlakeSandbox(sandbox=sdk_sandbox)
        finally:
            sdk_sandbox.terminate()
