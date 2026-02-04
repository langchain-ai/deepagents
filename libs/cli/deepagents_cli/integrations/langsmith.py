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
DEFAULT_TEMPLATE_NAME = "deepagents-cli"
DEFAULT_TEMPLATE_IMAGE = "python:3"


class LangSmithBackend(BaseSandbox):
    """LangSmith backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using LangSmith's API.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the LangSmithBackend with a sandbox instance.

        Args:
            sandbox: LangSmith Sandbox instance
        """
        self._sandbox = sandbox
        self._timeout: int = 30 * 60  # 30 mins default

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.name

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        result = self._sandbox.run(command, timeout=self._timeout)

        # Combine stdout and stderr (matching other backends' approach)
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the LangSmith sandbox.

        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.

        Note:
            Error handling with standardized FileOperationError codes is not yet
            implemented in existing backends. Currently only the happy path is
            implemented. See existing backends for reference.
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            # Use LangSmith's native file read API (returns bytes)
            content = self._sandbox.read(path)
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the LangSmith sandbox.

        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.

        Note:
            Error handling with standardized FileOperationError codes is not yet
            implemented in existing backends. Currently only the happy path is
            implemented. See existing backends for reference.
        """
        responses: list[FileUploadResponse] = []

        for path, content in files:
            # Use LangSmith's native file write API
            self._sandbox.write(path, content)
            responses.append(FileUploadResponse(path=path, error=None))

        return responses


class LangSmithProvider(SandboxProvider[dict[str, Any]]):
    """LangSmith sandbox provider implementation.

    This provider manages the lifecycle of LangSmith sandboxes, including
    creation, deletion, and template management.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the LangSmith provider with API credentials.

        Args:
            api_key: Optional LangSmith API key. If not provided, will attempt
                    to read from LANGSMITH_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided and LANGSMITH_API_KEY
                       environment variable is not set.
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
        """List available sandboxes.

        Note:
            This method is not yet implemented for LangSmith SDK.

        Raises:
            NotImplementedError: Always raised as listing is not yet supported.
        """
        msg = "Listing with LangSmith SDK not yet implemented"
        raise NotImplementedError(msg)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox or create a new one.

        If sandbox_id is provided, attempts to connect to an existing sandbox.
        Otherwise, creates a new sandbox using the specified template.

        Args:
            sandbox_id: Optional existing sandbox name to reuse. If provided,
                       will attempt to connect to this existing sandbox.
            timeout: Timeout in seconds for sandbox startup (default: 180).
            **kwargs: Additional LangSmith-specific parameters:
                     - template: Template name/ID or SandboxTemplate object
                     - template_image: Docker image for template creation

        Returns:
            LangSmithBackend instance wrapping the sandbox.

        Raises:
            RuntimeError: If sandbox connection fails (when sandbox_id provided),
                         if sandbox creation fails, or if sandbox doesn't become
                         ready within the timeout period.
        """
        # Extract template parameters from kwargs
        template = kwargs.get("template")
        template_image = kwargs.get("template_image")
        # If sandbox_id is provided, try to connect to existing sandbox
        if sandbox_id:
            try:
                sandbox = self._client.get_sandbox(name=sandbox_id)
            except Exception as e:
                msg = f"Failed to connect to existing sandbox '{sandbox_id}': {e}"
                raise RuntimeError(msg) from e
            return LangSmithBackend(sandbox)

        # Resolve template name and image from various input types
        template_name, resolved_image = self._resolve_template(template, template_image)

        # Ensure template exists (create if needed)
        self._ensure_template(template_name, resolved_image)

        # Create new sandbox from template
        try:
            sandbox = self._client.create_sandbox(
                template_name=template_name, timeout=timeout
            )
        except Exception as e:
            msg = f"Failed to create sandbox from template '{template_name}': {e}"
            raise RuntimeError(msg) from e

        # Wait for sandbox to be ready (poll with exponential backoff)
        for _ in range(timeout // 2):
            try:
                result = sandbox.run("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: S110, BLE001
                pass
            time.sleep(2)
        else:
            # Cleanup failed sandbox
            with contextlib.suppress(Exception):
                self._client.delete_sandbox(sandbox.name)
            msg = f"LangSmith sandbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return LangSmithBackend(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Delete a sandbox by ID.

        Args:
            sandbox_id: Name/ID of the sandbox to delete.
        """
        self._client.delete_sandbox(sandbox_id)

    @staticmethod
    def _resolve_template(
        template: SandboxTemplate | str | None,
        template_image: str | None = None,
    ) -> tuple[str, str | None]:
        """Resolve template name and image from various input types.

        Args:
            template: Template specification - can be None, string name, or
                     SandboxTemplate object.
            template_image: Explicit image override. If None and template is a
                          SandboxTemplate, will try to use template.image.

        Returns:
            Tuple of (template_name, template_image).
        """
        if template is None:
            return DEFAULT_TEMPLATE_NAME, template_image
        if isinstance(template, str):
            return template, template_image
        # template is a SandboxTemplate object
        # Use explicit template_image if provided, otherwise try to get from template
        image = template_image
        if image is None and hasattr(template, "image"):
            image = template.image
        return template.name, image

    def _ensure_template(
        self,
        template_name: str | None = None,
        template_image: str | None = None,
    ) -> None:
        """Ensure the specified template exists, creating it if necessary.

        Args:
            template_name: Name of the template to ensure exists.
                          Defaults to DEFAULT_TEMPLATE_NAME if None.
            template_image: Docker image to use when creating the template.
                           Defaults to DEFAULT_TEMPLATE_IMAGE if None.

        Raises:
            RuntimeError: If template check fails or template creation fails.
        """
        from langsmith.sandbox import ResourceNotFoundError

        name = template_name or DEFAULT_TEMPLATE_NAME
        image = template_image or DEFAULT_TEMPLATE_IMAGE

        try:
            self._client.get_template(name)
        except ResourceNotFoundError as e:
            # Template doesn't exist - create it
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
