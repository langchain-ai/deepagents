from __future__ import annotations

import pytest
from deepagents.backends.sandbox import SandboxClient  # noqa: TC002

from langchain_modal import ModalSandboxClient
from tests.integration_tests.sandboxes import SandboxClientIntegrationTests


class TestModalSandboxClientStandard(SandboxClientIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        return ModalSandboxClient()

    @property
    def has_async(self) -> bool:
        return True
