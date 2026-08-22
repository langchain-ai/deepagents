from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from microsandbox import Sandbox

from langchain_microsandbox import MicrosandboxSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


async def _create_sandbox(name: str) -> Sandbox:
    return await Sandbox.create(
        name,
        image="python:3.13-slim",
        ephemeral=True,
    )


async def _stop_sandbox(sandbox: Sandbox) -> None:
    await sandbox.stop()


class TestMicrosandboxSandboxStandard(SandboxIntegrationTests):
    @classmethod
    @pytest.fixture(scope="class")
    def sandbox_backend(
        cls,
        sandbox: SandboxBackendProtocol,
    ) -> SandboxBackendProtocol:
        return sandbox

    @classmethod
    @pytest.fixture(scope="class")
    def sandbox(cls) -> Iterator[SandboxBackendProtocol]:
        sandbox = asyncio.run(
            _create_sandbox(f"deepagents-integration-{uuid4().hex[:8]}")
        )
        try:
            backend = asyncio.run(MicrosandboxSandbox.create(sandbox))
            yield backend
        finally:
            asyncio.run(_stop_sandbox(sandbox))
