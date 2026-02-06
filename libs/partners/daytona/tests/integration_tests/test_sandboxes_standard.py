from __future__ import annotations

import os

import pytest
from deepagents.backends.sandbox import SandboxClient

from langchain_daytona.sandbox import DaytonaSandboxClient
from tests.integration_tests.sandboxes import SandboxClientIntegrationTests


class TestDaytonaSandboxClientStandard(SandboxClientIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DAYTONA_API_KEY is required for Daytona integration tests"
            )
        return DaytonaSandboxClient(api_key=api_key)

    @property
    def has_async(self) -> bool:
        return False
