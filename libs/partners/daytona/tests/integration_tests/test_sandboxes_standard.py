from __future__ import annotations

import os
from typing import Any

import pytest
from deepagents.backends.sandbox import SandboxProvider

from langchain_daytona.sandbox import DaytonaProvider
from tests.integration_tests.sandboxes import SandboxProviderIntegrationTests


class TestDaytonaSandboxProviderStandard(SandboxProviderIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxProvider[Any]:
        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise RuntimeError("DAYTONA_API_KEY is required for Daytona integration tests")
        return DaytonaProvider(api_key=api_key)

    @property
    def has_async(self) -> bool:
        return False
