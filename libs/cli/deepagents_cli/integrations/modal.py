"""Modal sandbox backend implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox

from deepagents_cli.integrations.sandbox_provider import (
    SandboxProvider,
)

if TYPE_CHECKING:
    import modal


class ModalBackend(BaseSandbox):
    """Modal backend implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Modal's API.
    """

    def __init__(self, sandbox: modal.Sandbox) -> None:
        """Initialize the ModalBackend with a Modal sandbox instance.

        Args:
            sandbox: Active Modal Sandbox instance
        """
        self._sandbox = sandbox
        self._default_timeout = 30 * 60

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox.object_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Note that in Modal's implementation, a timeout of 0 means
                "wait indefinitely".

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        # Execute command using Modal's exec API
        effective_timeout = timeout if timeout is not None else self._default_timeout
        process = self._sandbox.exec("bash", "-c", command, timeout=effective_timeout)

        # Wait for process to complete
        process.wait()

        # Read stdout and stderr
        stdout = process.stdout.read()
        stderr = process.stderr.read()

        # Combine stdout and stderr (matching Runloop's approach)
        output = stdout or ""
        if stderr:
            output += "\n" + stderr if output else stderr

        return ExecuteResponse(
            output=output,
            exit_code=process.returncode,
            truncated=False,  # Modal doesn't provide truncation info
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Modal sandbox.

        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.

        TODO: Implement proper error handling with standardized
        FileOperationError codes. Need to determine what exceptions
        Modal's sandbox.open() actually raises. Currently only implements
        happy path.
        """
        # This implementation relies on the Modal sandbox file API.
        # https://modal.com/doc/guide/sandbox-files
        # The API is currently in alpha and is not recommended for production
        # use. We're OK using it here as it's targeting the CLI application.
        responses = []
        for path in paths:
            with self._sandbox.open(path, "rb") as f:
                content = f.read()
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Modal sandbox.

        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.

        TODO: Implement proper error handling with standardized
        FileOperationError codes. Need to determine what exceptions
        Modal's sandbox.open() actually raises. Currently only implements
        happy path.
        """
        # This implementation relies on the Modal sandbox file API.
        # https://modal.com/doc/guide/sandbox-files
        # The API is currently in alpha and is not recommended for production
        # use. We're OK using it here as it's targeting the CLI application.
        responses = []
        for path, content in files:
            with self._sandbox.open(path, "wb") as f:
                f.write(content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses


class ModalProvider(SandboxProvider):
    """Modal sandbox provider implementation.

    Manages Modal sandbox lifecycle using the Modal SDK.
    """

    def __init__(self, app_name: str = "deepagents-sandbox") -> None:
        """Initialize Modal provider.

        Args:
            app_name: Name for the Modal app (default: "deepagents-sandbox")
        """
        import modal

        self._app_name = app_name
        self.app = modal.App.lookup(name=app_name, create_if_missing=True)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        workdir: str = "/workspace",
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002  # Required by SandboxFactory interface
    ) -> SandboxBackendProtocol:
        """Get existing or create new Modal sandbox.

        Args:
            sandbox_id: Existing sandbox ID to connect to (if None, creates new)
            workdir: Working directory for new sandboxes (default: /workspace)
            timeout: Timeout in seconds for sandbox startup (default: 180)
            **kwargs: Additional Modal-specific parameters

        Returns:
            ModalBackend instance

        Raises:
            RuntimeError: Sandbox startup failed
        """
        import modal

        if sandbox_id:
            sandbox = modal.Sandbox.from_id(sandbox_id=sandbox_id, app=self.app)  # type: ignore[call-arg]  # Modal SDK typing incomplete
        else:
            sandbox = modal.Sandbox.create(app=self.app, workdir=workdir)

            # Poll until running
            for _ in range(timeout // 2):
                if sandbox.poll() is not None:
                    msg = "Modal sandbox terminated unexpectedly during startup"
                    raise RuntimeError(msg)
                try:
                    process = sandbox.exec("echo", "ready", timeout=5)
                    process.wait()
                    if process.returncode == 0:
                        break
                except Exception:  # noqa: S110, BLE001  # Sandbox not ready yet, continue polling
                    pass
                time.sleep(2)
            else:
                sandbox.terminate()
                msg = f"Modal sandbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        return ModalBackend(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # Required by SandboxFactory interface
        """Delete a Modal sandbox.

        Args:
            sandbox_id: Sandbox ID to delete
            **kwargs: Additional parameters
        """
        import modal

        sandbox = modal.Sandbox.from_id(sandbox_id=sandbox_id, app=self.app)  # type: ignore[call-arg]  # Modal SDK typing incomplete
        sandbox.terminate()
