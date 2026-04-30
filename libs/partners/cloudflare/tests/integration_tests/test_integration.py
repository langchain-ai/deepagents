from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

from langchain_cloudflare import CloudflareSandbox


class TestCloudflareSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        base_url = os.environ.get("CLOUDFLARE_SANDBOX_BRIDGE_URL")
        api_key = os.environ.get("CLOUDFLARE_SANDBOX_API_KEY")
        sandbox_id = os.environ.get("CLOUDFLARE_SANDBOX_ID", "integration-test")
        if not base_url:
            msg = (
                "Missing CLOUDFLARE_SANDBOX_BRIDGE_URL for integration test: "
                "set the URL of your deployed Cloudflare Sandbox Bridge Worker"
            )
            raise RuntimeError(msg)

        return CloudflareSandbox(
            base_url=base_url,
            sandbox_id=sandbox_id,
            api_key=api_key,
        )
