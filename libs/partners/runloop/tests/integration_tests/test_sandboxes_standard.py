from __future__ import annotations

import os

import pytest
from deepagents.backends.sandbox import SandboxClient  # noqa: TC002

from langchain_runloop import RunloopSandboxClient
from tests.integration_tests.sandboxes import SandboxClientIntegrationTests


class TestRunloopSandboxClientStandard(SandboxClientIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        api_key = os.environ.get("RUNLOOP_API_KEY")
        if not api_key:
            msg = "RUNLOOP_API_KEY is required for Runloop integration tests"
            raise RuntimeError(msg)
        return RunloopSandboxClient(api_key=api_key)

    @property
    def has_async(self) -> bool:
        return True
