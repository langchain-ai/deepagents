"""Runloop sandbox implementation."""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any

if importlib.util.find_spec("runloop_api_client") is None:
    msg = (
        "runloop_api_client package is required for RunloopSandboxClient. "
        "Install with `pip install runloop-api-client`."
    )
    raise ImportError(msg)

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox
from runloop_api_client import Runloop


class RunloopSandbox(BaseSandbox):
    """Sandbox backend that operates on a Runloop devbox."""

    def __init__(
        self,
        *,
        devbox_id: str,
        client: Runloop | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create a sandbox backend connected to an existing Runloop devbox."""
        if client is not None and api_key is not None:
            msg = "Provide either client or api_key, not both."
            raise ValueError(msg)

        if client is None:
            api_key = api_key or os.environ.get("RUNLOOP_API_KEY")
            if api_key is None:
                msg = "Provide either client or api_key (or set RUNLOOP_API_KEY)."
                raise ValueError(msg)
            client = Runloop(bearer_token=api_key)

        self._client = client
        self._devbox_id = devbox_id
        self._timeout = 30 * 60

    @property
    def id(self) -> str:
        """Return the devbox id."""
        return self._devbox_id

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command inside the devbox."""
        result = self._client.devboxes.execute_and_await_completion(
            devbox_id=self._devbox_id,
            command=command,
            timeout=self._timeout,
        )

        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output, exit_code=result.exit_status, truncated=False
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the devbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            resp = self._client.devboxes.download_file(self._devbox_id, path=path)
            content = resp.read()
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the devbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            self._client.devboxes.upload_file(self._devbox_id, path=path, file=content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses


class RunloopSandboxClient:
    """Runloop sandbox client implementation."""

    def __init__(
        self, *, api_key: str | None = None, client: Runloop | None = None
    ) -> None:
        """Create a provider backed by the Runloop SDK."""
        if api_key is not None and client is not None:
            msg = "Provide either api_key or client, not both."
            raise ValueError(msg)

        if client is not None:
            self._client = client
            return

        api_key = api_key or os.environ.get("RUNLOOP_API_KEY")
        if api_key is None:
            msg = "Provide either client or api_key (or set RUNLOOP_API_KEY)."
            raise ValueError(msg)

        self._client = Runloop(bearer_token=api_key)

    @property
    def client(self) -> Runloop:
        """Expose the underlying Runloop client instance."""
        return self._client

    def get(self, *, sandbox_id: str, **kwargs: Any) -> SandboxBackendProtocol:
        """Get an existing Runloop devbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"RunloopSandboxClient.get() got unsupported kwargs: {keys}"
            raise ValueError(msg)

        try:
            devbox = self._client.devboxes.retrieve(id=sandbox_id)
        except Exception as e:
            raise ValueError(sandbox_id) from e

        return RunloopSandbox(devbox_id=devbox.id, client=self._client)

    def create(self, *, timeout: int = 180, **kwargs: Any) -> SandboxBackendProtocol:
        """Create a new Runloop devbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"RunloopSandboxClient.create() got unsupported kwargs: {keys}"
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

        return RunloopSandbox(devbox_id=devbox.id, client=self._client)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a Runloop devbox."""
        if kwargs:
            keys = sorted(kwargs.keys())
            msg = f"RunloopSandboxClient.delete() got unsupported kwargs: {keys}"
            raise ValueError(msg)
        try:
            self._client.devboxes.shutdown(id=sandbox_id)
        except Exception:  # noqa: BLE001
            return
