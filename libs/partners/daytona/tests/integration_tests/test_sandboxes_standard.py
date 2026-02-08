from __future__ import annotations

import os

import pytest
from deepagents.backends.sandbox import SandboxClient  # noqa: TC002

from langchain_daytona.sandbox import DaytonaSandboxClient
from tests.integration_tests.sandboxes import (
    SandboxClientIntegrationTests,
    SandboxIntegrationTests,
)


class TestDaytonaSandboxClientStandard(SandboxClientIntegrationTests):
    @property
    def supports_distinct_download_errors(self) -> bool:
        return False

    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            msg = "DAYTONA_API_KEY is required for Daytona integration tests"
            raise RuntimeError(msg)
        return DaytonaSandboxClient(api_key=api_key)

    @property
    def has_async(self) -> bool:
        return True


class TestDaytonaSandboxStandard(SandboxIntegrationTests):
    @property
    def supports_distinct_download_errors(self) -> bool:
        return False

    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            msg = "DAYTONA_API_KEY is required for Daytona integration tests"
            raise RuntimeError(msg)
        return DaytonaSandboxClient(api_key=api_key)
