from __future__ import annotations

import asyncio
import os
import time

import pytest
from daytona import Daytona, DaytonaConfig

try:
    from deepagents.backends.sandbox import SandboxClient
except ImportError:  # pragma: no cover
    from typing import Any as SandboxClient

from langchain_daytona.sandbox import DaytonaSandbox
from tests.integration_tests.sandboxes import (
    SandboxClientIntegrationTests,
    SandboxIntegrationTests,
    SandboxNotFoundError,
)


class _TestDaytonaSandboxClient(SandboxClient):
    async def aget(self, *, sandbox_id: str, **kwargs: object) -> DaytonaSandbox:
        return await asyncio.to_thread(self.get, sandbox_id=sandbox_id, **kwargs)

    async def acreate(self, **kwargs: object) -> DaytonaSandbox:
        return await asyncio.to_thread(self.create, **kwargs)

    async def adelete(self, *, sandbox_id: str, **kwargs: object) -> None:
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)

    def __init__(self, *, api_key: str) -> None:
        self._client = Daytona(DaytonaConfig(api_key=api_key))

    def get(self, *, sandbox_id: str, **kwargs: object) -> DaytonaSandbox:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestDaytonaSandboxClient.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception as e:
            raise SandboxNotFoundError(sandbox_id) from e
        return DaytonaSandbox(sandbox)

    def create(self, *, timeout: int = 180, **kwargs: object) -> DaytonaSandbox:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestDaytonaSandboxClient.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        sandbox = self._client.create()

        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            try:
                sandbox.delete()
            finally:
                msg = f"Daytona sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return DaytonaSandbox(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: object) -> None:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestDaytonaSandboxClient.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception:  # noqa: BLE001
            return
        try:
            self._client.delete(sandbox)
        except Exception:  # noqa: BLE001
            return


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
        return _TestDaytonaSandboxClient(api_key=api_key)

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
        return _TestDaytonaSandboxClient(api_key=api_key)


class DaytonaSandboxClient:
    """Daytona sandbox provider implementation."""

    def __init__(
        self, *, api_key: str | None = None, client: Daytona | None = None
    ) -> None:
        """Create a provider backed by the Daytona SDK."""
        if api_key and client:
            msg = "Provide either api_key or client, not both."
            raise ValueError(msg)

        if client is not None:
            self._client = client
            return

        api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            msg = "Provide either client or api_key (or set DAYTONA_API_KEY)."
            raise ValueError(msg)

        self._client = Daytona(DaytonaConfig(api_key=api_key))

    @property
    def client(self) -> Daytona:
        """Expose the underlying Daytona client instance."""
        return self._client

    def get(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing Daytona sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaProvider.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception as e:
            raise ValueError(sandbox_id) from e
        return DaytonaSandbox(sandbox)

    def create(
        self,
        *,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Create a new Daytona sandbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaProvider.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        sandbox = self._client.create()

        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: BLE001
                # Ok: startup errors vary; we retry then timeout.
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            try:
                sandbox.delete()
            finally:
                msg = f"Daytona sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return DaytonaSandbox(sandbox)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Deprecated: use get() or create()."""
        if sandbox_id is None:
            return self.create(timeout=timeout, **kwargs)
        return self.get(sandbox_id=sandbox_id, **kwargs)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a sandbox by id."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"DaytonaProvider.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = self._client.get(sandbox_id)
        except Exception:  # noqa: BLE001
            return
        try:
            self._client.delete(sandbox)
        except Exception:  # noqa: BLE001
            return
