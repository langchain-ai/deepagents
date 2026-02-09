from __future__ import annotations

import asyncio
import time

import modal
import pytest

try:
    from deepagents.backends.sandbox import SandboxClient
except ImportError:  # pragma: no cover
    from typing import Any as SandboxClient

from langchain_modal import ModalSandbox
from tests.integration_tests.sandboxes import (
    SandboxClientIntegrationTests,
    SandboxIntegrationTests,
    SandboxNotFoundError,
)


class _TestModalSandboxClient(SandboxClient):
    async def aget(self, *, sandbox_id: str, **kwargs: object) -> ModalSandbox:
        return await asyncio.to_thread(self.get, sandbox_id=sandbox_id, **kwargs)

    async def acreate(self, **kwargs: object) -> ModalSandbox:
        return await asyncio.to_thread(self.create, **kwargs)

    async def adelete(self, *, sandbox_id: str, **kwargs: object) -> None:
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)

    def __init__(self, *, app_name: str = "deepagents-sandbox") -> None:
        self._app = modal.App.lookup(name=app_name, create_if_missing=True)

    def get(self, *, sandbox_id: str, **kwargs: object) -> ModalSandbox:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestModalSandboxClient.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception as e:
            raise SandboxNotFoundError(sandbox_id) from e
        return ModalSandbox(sandbox)

    def create(
        self, *, workdir: str = "/workspace", timeout: int = 180, **kwargs: object
    ) -> ModalSandbox:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestModalSandboxClient.create() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        sandbox = modal.Sandbox.create(
            app=self._app,
            workdir=workdir,
            image=modal.Image.debian_slim(python_version="3.13"),
        )

        for _ in range(timeout // 2):
            if sandbox.poll() is not None:
                msg = "Modal sandbox terminated unexpectedly during startup"
                raise RuntimeError(msg)
            try:
                process = sandbox.exec("echo", "ready", timeout=5)
                process.wait()
                if process.returncode == 0:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            sandbox.terminate()
            msg = f"Modal sandbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return ModalSandbox(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: object) -> None:
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"_TestModalSandboxClient.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception:  # noqa: BLE001
            return
        sandbox.terminate()


class TestModalSandboxClientStandard(SandboxClientIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        return _TestModalSandboxClient()

    @property
    def has_async(self) -> bool:
        return True


class TestModalSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox_provider(self) -> SandboxClient:
        return _TestModalSandboxClient()

    @property
    def supports_distinct_download_errors(self) -> bool:
        return False

    @pytest.mark.usefixtures("sandbox_backend")
    def test_download_error_permission_denied(self) -> None:
        pytest.skip("Modal sandboxes do not reliably enforce chmod-based permissions.")
