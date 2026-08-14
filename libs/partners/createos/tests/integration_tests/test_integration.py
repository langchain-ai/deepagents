from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import httpx
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_createos import CreateOSSandbox

_READY_TIMEOUT = 60.0
_READY_POLL_INTERVAL = 1.0
_READY_COMMAND_TIMEOUT = 5

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestCreateOSSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ["CREATEOS_API_KEY"]
        base_url = os.environ.get("CREATEOS_BASE_URL", "https://api.sb.createos.sh")
        with httpx.Client(
            base_url=base_url,
            headers={"X-Api-Key": api_key},
            timeout=60.0,
        ) as client:
            resp = client.post(
                "/v1/sandboxes",
                json={"shape": "s-1vcpu-256mb"},
            )
            resp.raise_for_status()
            sandbox_id = resp.json()["data"]["id"]

            backend = CreateOSSandbox(
                sandbox_id=sandbox_id,
                api_key=api_key,
                base_url=base_url,
            )
            try:
                _wait_until_ready(backend)
                yield backend
            finally:
                try:
                    backend.close()
                finally:
                    delete = client.delete(f"/v1/sandboxes/{sandbox_id}")
                    delete.raise_for_status()


def _wait_until_ready(backend: CreateOSSandbox) -> None:
    """Wait until the sandbox accepts commands or the boot deadline expires."""
    deadline = time.monotonic() + _READY_TIMEOUT
    while time.monotonic() < deadline:
        try:
            result = backend.execute("true", timeout=_READY_COMMAND_TIMEOUT)
            if result.exit_code == 0:
                return
        except httpx.HTTPError:
            pass
        time.sleep(_READY_POLL_INTERVAL)

    msg = f"CreateOS sandbox {backend.id} did not become ready"
    raise TimeoutError(msg)
