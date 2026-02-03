"""LangSmith sandbox backend implementation."""

from __future__ import annotations

import contextlib
import os
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
    from langsmith.sandbox import Sandbox, SandboxClient

from deepagents_cli.config import console


def _print_sandbox_failure_logs(sb: Sandbox) -> None:
    """Best-effort dump of sandbox logs to help debug startup/readiness failures.

    Args:
        sb: Sandbox instance to retrieve logs from.
    """

    def _try_print(text: object) -> bool:
        if text is None:
            return False
        rendered = str(text).strip()
        if not rendered:
            return False
        console.print("[dim]\n--- sandbox logs ---\n[/dim]" + rendered)
        return True

    with contextlib.suppress(Exception):
        for attr in ("logs", "get_logs", "read_logs", "tail_logs"):
            fn = getattr(sb, attr, None)
            if callable(fn) and _try_print(fn()):
                return

    with contextlib.suppress(Exception):
        result = sb.run(
            "set -euo pipefail; "
            "echo 'uname:'; uname -a; "
            "echo; echo 'processes:'; ps aux | head -n 50; "
            "echo; echo 'disk:'; df -h; "
            "echo; echo 'recent logs:'; "
            "(ls -la /var/log 2>/dev/null || true); "
            "(tail -n 200 /var/log/syslog 2>/dev/null || true); "
            "(tail -n 200 /var/log/messages 2>/dev/null || true); "
            "(tail -n 200 /var/log/cloud-init-output.log 2>/dev/null || true);",
            timeout=20,
        )
        combined = result.stdout or ""
        if result.stderr:
            combined = combined + ("\n" if combined else "") + result.stderr
        _try_print(combined)


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


# Default template configuration
DEFAULT_TEMPLATE_NAME = os.getenv("DEFAULT_SANDBOX_TEMPLATE_NAME", "python-slim")
DEFAULT_TEMPLATE_IMAGE = os.getenv(
    "DEFAULT_SANDBOX_TEMPLATE_IMAGE", "bracelangchain/deepagents-sandbox:v1"
)


def _ensure_template(
    client: SandboxClient, template_name: str = DEFAULT_TEMPLATE_NAME
) -> None:
    """Ensure template exists, creating it if needed.

    Args:
        client: LangSmith SandboxClient instance
        template_name: Name of the template to ensure exists

    Raises:
        RuntimeError: If template check or creation fails
    """
    from langsmith.sandbox import ResourceNotFoundError

    try:
        templates = client.list_templates()
        for template in templates:
            if template.name == template_name:
                console.print(
                    f"[green]✓ Template '{template_name}' already exists[/green]"
                )
                return
        client.get_template(template_name)
    except ResourceNotFoundError:
        console.print(f"[dim]Creating template '{template_name}'...[/dim]")
        try:
            client.create_template(name=template_name, image=DEFAULT_TEMPLATE_IMAGE)
        except Exception as e:
            msg = f"Failed to create template '{template_name}': {e}"
            raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Failed to check template '{template_name}': {e}"
        raise RuntimeError(msg) from e


def _verify_sandbox_ready(sb: Sandbox, client: SandboxClient) -> None:  # noqa: ARG001
    """Verify sandbox is ready to accept commands.

    Args:
        sb: Sandbox instance to verify
        client: SandboxClient instance (for cleanup if needed)

    Raises:
        RuntimeError: If sandbox readiness check fails
    """
    readiness_error: Exception | None = None
    try:
        result = sb.run("echo ready", timeout=5)
        if result.exit_code != 0:
            readiness_error = RuntimeError("Sandbox readiness check failed")
    except Exception as e:  # noqa: BLE001
        readiness_error = e

    if readiness_error:
        msg = f"LangSmith sandbox failed readiness check: {readiness_error}"
        raise RuntimeError(msg) from readiness_error


def _create_sandbox_instance(
    client: SandboxClient, template_name: str = DEFAULT_TEMPLATE_NAME
) -> Sandbox:
    """Create a new sandbox.

    Args:
        client: LangSmith SandboxClient instance
        template_name: Name of the template to use

    Returns:
        Ready Sandbox instance

    Raises:
        RuntimeError: If sandbox creation fails
    """
    console.print("[dim]Creating sandbox...[/dim]")
    try:
        sb = client.create_sandbox(template_name=template_name, timeout=180)
    except Exception as e:
        msg = f"Failed to create sandbox from template '{template_name}': {e}"
        raise RuntimeError(msg) from e

    # No need to verify - sandbox is ready when create_sandbox returns
    console.print(f"[green]✓ LangSmith sandbox ready: {sb.name}[/green]")
    return sb


class LangSmithProvider(SandboxProvider[dict[str, Any]]):
    """LangSmith sandbox provider implementation.

    Manages LangSmith sandbox lifecycle using the LangSmith SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        template_name: str = DEFAULT_TEMPLATE_NAME,
    ) -> None:
        """Initialize LangSmith provider.

        Args:
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
            api_endpoint: LangSmith API endpoint (defaults to standard endpoint)
            template_name: Default template name for new sandboxes

        Raises:
            ValueError: If no API key is found in parameters or environment.
        """
        from langsmith import sandbox

        self._api_key = (
            api_key
            or os.environ.get("LANGSMITH_API_KEY")
            or os.environ.get("LANGSMITH_API_KEY_PROD")
        )
        if not self._api_key:
            msg = (
                "LANGSMITH_API_KEY or LANGSMITH_API_KEY_PROD "
                "environment variable not set"
            )
            raise ValueError(msg)

        langsmith_endpoint = api_endpoint or os.environ.get(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
        sandbox_endpoint = f"{langsmith_endpoint.rstrip('/')}/v2/sandboxes"

        self._client = sandbox.SandboxClient(
            api_endpoint=sandbox_endpoint,
            api_key=self._api_key,
        )
        self._template_name = template_name

    def list(
        self,
        *,
        cursor: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxListResponse[dict[str, Any]]:
        """List available LangSmith sandboxes.

        Args:
            cursor: Optional continuation token (not currently supported)
            **kwargs: Additional parameters

        Returns:
            SandboxListResponse with sandbox info

        Note:
            Pagination is not currently implemented for LangSmith sandboxes.
        """
        sandboxes = self._client.list_sandboxes()
        items = [{"sandbox_id": sb.name, "metadata": {}} for sb in sandboxes]
        return SandboxListResponse(items=items, cursor=None)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        template_name: str | None = None,
        timeout: int = 180,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Get existing or create new LangSmith sandbox.

        Args:
            sandbox_id: Existing sandbox name to connect to (if None, creates new)
            template_name: Template name for new sandboxes (default: python-slim)
            timeout: Timeout in seconds for sandbox startup (default: 180)
            **kwargs: Additional LangSmith-specific parameters

        Returns:
            LangSmithBackend instance

        Raises:
            RuntimeError: Sandbox creation or connection failed
        """
        template = template_name or self._template_name

        if sandbox_id:
            # Connect to existing sandbox by name
            try:
                sb = self._client.get_sandbox(name=sandbox_id)
            except Exception as e:
                msg = f"Failed to connect to existing sandbox '{sandbox_id}': {e}"
                raise RuntimeError(msg) from e

            # Verify the existing sandbox is ready
            _verify_sandbox_ready(sb, self._client)
            console.print(
                f"[green]✓ Connected to existing LangSmith sandbox: {sb.name}[/green]"
            )
        else:
            # Ensure template exists and create new sandbox
            _ensure_template(self._client, template)
            sb = _create_sandbox_instance(self._client, template)

        return LangSmithBackend(sb)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Delete a LangSmith sandbox.

        Args:
            sandbox_id: Sandbox name to delete
            **kwargs: Additional parameters
        """
        # Idempotent - ignore errors if sandbox doesn't exist
        with contextlib.suppress(Exception):
            self._client.delete_sandbox(sandbox_id)
