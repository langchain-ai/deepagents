from __future__ import annotations

import asyncio
import os
import time

import pytest

try:
    from deepagents.backends.sandbox import SandboxClient
except ImportError:  # pragma: no cover
    from typing import Any as SandboxClient
from runloop_api_client import Runloop

from langchain_runloop import RunloopSandbox
from tests.integration_tests.sandboxes import (
    SandboxClientIntegrationTests,
    SandboxIntegrationTests,
    SandboxNotFoundError,
)


class _TestRunloopSandboxClient(SandboxClient):
    async def aget(self, *, sandbox_id: str, **kwargs: object) -> RunloopSandbox:
        return await asyncio.to_thread(self.get, sandbox_id=sandbox_id, **kwargs)

    async def acreate(self, **kwargs: object) -> RunloopSandbox:
        return await asyncio.to_thread(self.create, **kwargs)

    async def adelete(self, *, sandbox_id: str, **kwargs: object) -> None:
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)

    def __init__(self, *, api_key: str) -> None:
        self._client = Runloop(bearer_token=api_key)

    def get(self, *, sandbox_id: str, **kwargs: object) -> RunloopSandbox:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestRunloopSandboxClient.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            devbox = self._client.devboxes.retrieve(id=sandbox_id)
        except Exception as e:
            raise SandboxNotFoundError(sandbox_id) from e
        return RunloopSandbox(devbox)

    def create(self, *, timeout: int = 180, **kwargs: object) -> RunloopSandbox:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestRunloopSandboxClient.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        devbox = self._client.devboxes.create()

        for _ in range(timeout // 2):
            status = self._client.devboxes.retrieve(id=devbox.id)
            if status.status == "running":
                break
            time.sleep(2)
        else:
            self._client.devboxes.shutdown(id=devbox.id)
            msg = f"Devbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return RunloopSandbox(devbox)

    def delete(self, *, sandbox_id: str, **kwargs: object) -> None:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestRunloopSandboxClient.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            self._client.devboxes.shutdown(id=sandbox_id)
        except Exception:  # noqa: BLE001
            return


class TestRunloopSandboxClientStandard(SandboxClientIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        api_key = os.environ.get("RUNLOOP_API_KEY")
        if not api_key:
            msg = "RUNLOOP_API_KEY is required for Runloop integration tests"
            raise RuntimeError(msg)
        return _TestRunloopSandboxClient(api_key=api_key)

    @property
    def has_async(self) -> bool:
        return True


class TestRunloopSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        api_key = os.environ.get("RUNLOOP_API_KEY")
        if not api_key:
            msg = "RUNLOOP_API_KEY is required for Runloop integration tests"
            raise RuntimeError(msg)
        return _TestRunloopSandboxClient(api_key=api_key)
