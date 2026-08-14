from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import httpx
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_createos import CreateOSSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestCreateOSSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ["CREATEOS_API_KEY"]
        base_url = os.environ.get(
            "CREATEOS_BASE_URL", "https://api.sb.createos.sh"
        )
        client = httpx.Client(
            base_url=base_url,
            headers={"X-Api-Key": api_key},
            timeout=60.0,
        )
        resp = client.post(
            "/v1/sandboxes",
            json={"shape": "s-1vcpu-256mb"},
        )
        resp.raise_for_status()
        sandbox_id = resp.json()["data"]["id"]
        # Give the VM a moment to finish booting.
        time.sleep(2)

        backend = CreateOSSandbox(
            sandbox_id=sandbox_id,
            api_key=api_key,
            base_url=base_url,
        )
        try:
            yield backend
        finally:
            client.delete(f"/v1/sandboxes/{sandbox_id}")
            client.close()
