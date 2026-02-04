"""LangSmith sandbox backend implementation."""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxListResponse,
    SandboxProvider,
)

if TYPE_CHECKING:
    from langsmith.sandbox import Sandbox, SandboxClient, SandboxTemplate

# Default template configuration
DEFAULT_TEMPLATE_NAME = os.getenv("LANGSMITH_SANDBOX_TEMPLATE_NAME", "deepagents-cli")
DEFAULT_TEMPLATE_IMAGE = os.getenv("LANGSMITH_SANDBOX_TEMPLATE_IMAGE", "python:3")


class LangSmithBackend(BaseSandbox):
    """LangSmith backend implementation."""

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize with sandbox instance."""
        self._sandbox = sandbox
        self._timeout: int = 30 * 60

    @property
    def id(self) -> str:
        """Return sandbox name as ID."""
        return self._sandbox.name

    def execute(self, command: str) -> ExecuteResponse:
        """Execute command in sandbox.

        Returns:
            ExecuteResponse with output and exit code.
        """
        result = self._sandbox.run(command, timeout=self._timeout)
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr
        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from sandbox.

        Returns:
            List of FileDownloadResponse objects.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            content = self._sandbox.read(path)
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to sandbox.

        Returns:
            List of FileUploadResponse objects.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            self._sandbox.write(path, content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses


class LangSmithProvider(SandboxProvider[dict[str, Any]]):
    """LangSmith sandbox provider implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize provider with API key.

        Raises:
            ValueError: If LANGSMITH_API_KEY is not set.
        """
        from langsmith import sandbox

        self._api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        if not self._api_key:
            msg = "LANGSMITH_API_KEY environment variable not set"
            raise ValueError(msg)
        self._client: SandboxClient = sandbox.SandboxClient()

    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[dict[str, Any]]:
        """List sandboxes (not implemented)."""
        msg = "Listing with LangSmith SDK not yet implemented"
        raise NotImplementedError(msg)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        template: SandboxTemplate | str | None = None,
        template_image: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Get existing or create new LangSmith sandbox.

        Args:
            sandbox_id: Optional existing sandbox name to reuse
            timeout: Timeout in seconds for sandbox startup
            template: Template name/ID or SandboxTemplate object
            template_image: Docker image for template creation

        Returns:
            LangSmithBackend instance

        Raises:
            RuntimeError: If sandbox connection or startup fails
        """
        if sandbox_id:
            try:
                sandbox = self._client.get_sandbox(name=sandbox_id)
            except Exception as e:
                msg = f"Failed to connect to existing sandbox '{sandbox_id}': {e}"
                raise RuntimeError(msg) from e
            return LangSmithBackend(sandbox)

        template_name = self._resolve_template_name(template)
        self._ensure_template(template_name, template_image)

        try:
            sandbox = self._client.create_sandbox(
                template_name=template_name, timeout=timeout
            )
        except Exception as e:
            msg = f"Failed to create sandbox from template '{template_name}': {e}"
            raise RuntimeError(msg) from e

        for _ in range(timeout // 2):
            try:
                result = sandbox.run("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: S110, BLE001
                pass
            time.sleep(2)
        else:
            with contextlib.suppress(Exception):
                self._client.delete_sandbox(sandbox.name)
            msg = f"LangSmith sandbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return LangSmithBackend(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Delete a sandbox."""
        self._client.delete_sandbox(sandbox_id)

    @staticmethod
    def _resolve_template_name(template: SandboxTemplate | str | None) -> str:
        if template is None:
            return DEFAULT_TEMPLATE_NAME
        if isinstance(template, str):
            return template
        return template.name

    def _ensure_template(
        self,
        template_name: str | None = None,
        template_image: str | None = None,
    ) -> None:
        from langsmith.sandbox import ResourceNotFoundError

        name = template_name or DEFAULT_TEMPLATE_NAME
        image = template_image or DEFAULT_TEMPLATE_IMAGE

        try:
            self._client.get_template(name)
        except ResourceNotFoundError as e:
            if e.resource_type != "template":
                msg = f"Unexpected resource not found: {e}"
                raise RuntimeError(msg) from e
            try:
                self._client.create_template(name=name, image=image)
            except Exception as create_err:
                msg = f"Failed to create template '{name}': {create_err}"
                raise RuntimeError(msg) from create_err
        except Exception as e:
            msg = f"Failed to check template '{name}': {e}"
            raise RuntimeError(msg) from e
