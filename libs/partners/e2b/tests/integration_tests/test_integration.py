from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import httpx
import pytest
from e2b import Sandbox
from e2b.exceptions import SandboxException
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_e2b import E2BSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

KILL_ATTEMPTS = 3
KILL_RETRY_DELAY_SECONDS = 1


class TestE2BSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            pytest.skip("Missing secrets for E2B integration test: set E2B_API_KEY")

        template = os.environ.get("E2B_TEMPLATE")
        if template:
            sandbox = Sandbox.create(
                template=template,
                timeout=60 * 60,
                api_key=api_key,
            )
        else:
            sandbox = Sandbox.create(
                timeout=60 * 60,
                api_key=api_key,
            )
        backend = E2BSandbox(sandbox=sandbox)
        try:
            yield backend
        finally:
            _kill_sandbox(sandbox)


def _kill_sandbox(sandbox: Sandbox) -> None:
    last_error: BaseException | None = None
    for attempt in range(KILL_ATTEMPTS):
        try:
            sandbox.kill()
        except (httpx.HTTPError, SandboxException) as exc:
            last_error = exc
            if attempt + 1 < KILL_ATTEMPTS:
                time.sleep(KILL_RETRY_DELAY_SECONDS)
        else:
            return

    msg = f"Failed to kill E2B sandbox {sandbox.sandbox_id!r}"
    raise RuntimeError(msg) from last_error
