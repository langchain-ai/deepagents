from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from deepagents.backends.langsmith import LangSmithBackend, LangSmithProvider

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestLangSmithSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        provider = LangSmithProvider()
        backend = provider.get_or_create()
        try:
            yield backend
        finally:
            provider.delete(sandbox_id=backend.id)
