"""AgentCore Code Interpreter sandbox backend implementation.

This module provides a sandbox backend that uses AWS Bedrock AgentCore
Code Interpreter for remote code execution.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from deepagents_cli.integrations.sandbox_provider import SandboxProvider

if TYPE_CHECKING:
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
    from deepagents.backends.protocol import SandboxBackendProtocol

logger = logging.getLogger(__name__)

# Default session timeout from AgentCore SDK
DEFAULT_TIMEOUT = 900  # 15 minutes (can be extended up to 8 hours)


def _extract_text_from_stream(response: dict[str, Any]) -> tuple[str, int | None]:
    """Extract text output and exit code from code interpreter response stream.

    Args:
        response: Response dict from code interpreter invocation

    Returns:
        Tuple of (output_text, exit_code)
    """
    output_parts: list[str] = []
    exit_code: int | None = None

    for event in response.get("stream", []):
        if "result" in event:
            result = event["result"]

            # Check for exit code in result metadata
            if "exitCode" in result:
                exit_code = result["exitCode"]

            for content_item in result.get("content", []):
                content_type = content_item.get("type")

                if content_type == "text":
                    text = content_item.get("text", "")
                    output_parts.append(text)

                elif content_type == "error":
                    error_msg = content_item.get("text", "Unknown error")
                    output_parts.append(f"Error: {error_msg}")
                    if exit_code is None:
                        exit_code = 1

    return "\n".join(output_parts), exit_code


def _extract_files_from_stream(response: dict[str, Any]) -> dict[str, bytes]:
    """Extract file contents from code interpreter response stream.

    Parses the structured JSON response to extract file paths and contents.

    Args:
        response: Response dict from code interpreter readFiles invocation

    Returns:
        Dict mapping file paths to their contents as bytes
    """
    files: dict[str, bytes] = {}

    for event in response.get("stream", []):
        if "result" in event:
            for content_item in event["result"].get("content", []):
                if content_item.get("type") == "resource":
                    resource = content_item.get("resource", {})
                    uri = resource.get("uri", "")
                    file_path = uri.replace("file://", "")

                    if "text" in resource:
                        files[file_path] = resource["text"].encode("utf-8")
                    elif "blob" in resource:
                        files[file_path] = base64.b64decode(resource["blob"])

    return files


class AgentCoreBackend(BaseSandbox):
    """AgentCore Code Interpreter backend implementing SandboxBackendProtocol.

    Uses AWS Bedrock AgentCore Code Interpreter to execute shell commands
    and manage files in a secure, isolated MicroVM environment.

    Session Behavior:
        - Files created during a session persist for the session lifetime
        - Session timeout defaults to 15 minutes (configurable up to 8 hours)
        - Each session runs in an isolated MicroVM with dedicated resources

    Note:
        Reconnecting to a previous session (e.g., after CLI restart) is not
        currently supported. The --sandbox-id option cannot be used with
        AgentCore. However, within a single CLI session, all file operations
        work normally - files written can be read back throughout the session.

    Example:
        ```python
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        interpreter = CodeInterpreter(
            region=self._region,
            integration_source="deepagents-cli",
        )
        interpreter.start()

        backend = AgentCoreBackend(interpreter)

        # Write a file
        backend.upload_files([("hello.py", b"print('hello')")])

        # Read it back (works within same session)
        files = backend.download_files(["hello.py"])

        # Execute it
        result = backend.execute("python hello.py")

        interpreter.stop()
        ```
    """

    # Integration source identifier for telemetry tracking
    INTEGRATION_SOURCE = "deepagents-cli"

    def __init__(self, interpreter: CodeInterpreter) -> None:
        """Initialize the AgentCoreBackend with a CodeInterpreter instance.

        Args:
            interpreter: Active CodeInterpreter instance (must be started)
        """
        self._interpreter = interpreter
        self._timeout: int = DEFAULT_TIMEOUT  # 15 minutes

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend (session ID)."""
        return self._interpreter.session_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ARG002
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command string to execute
            timeout: Maximum execution time in seconds (unused —
                AgentCore does not support per-command timeouts).

        Returns:
            ExecuteResponse with output, exit code, and truncation flag
        """
        try:
            response = self._interpreter.invoke(
                method="executeCommand", params={"command": command}
            )

            output, exit_code = _extract_text_from_stream(response)

            return ExecuteResponse(
                output=output,
                exit_code=exit_code if exit_code is not None else 0,
                truncated=False,
            )
        except Exception as e:
            logger.exception("Error executing command: %s", command[:50])
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=1,
                truncated=False,
            )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the AgentCore sandbox.

        Uses AgentCore's readFiles API. Supports partial success - individual
        file downloads may fail without affecting others.

        Args:
            paths: List of file paths to download

        Returns:
            List of FileDownloadResponse objects in same order as input paths
        """
        try:
            response = self._interpreter.invoke(
                method="readFiles", params={"paths": paths}
            )

            # Parse structured JSON response
            file_contents = _extract_files_from_stream(response)

            # Build responses in order of input paths
            return [
                FileDownloadResponse(
                    path=path,
                    content=file_contents.get(path),
                    error=None if path in file_contents else "file_not_found",
                )
                for path in paths
            ]

        except Exception:
            logger.exception("Error downloading files: %s", paths)
            return [
                FileDownloadResponse(path=path, content=None, error="file_not_found")
                for path in paths
            ]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the AgentCore sandbox.

        Uses AgentCore's writeFiles API. Text files are supported directly.
        Binary files are base64 encoded.

        Args:
            files: List of (path, content) tuples to upload

        Returns:
            List of FileUploadResponse objects in same order as input files
        """
        file_list: list[dict[str, str]] = []

        for path, content in files:
            try:
                # Try to decode as text first
                text_content = content.decode("utf-8")
                file_list.append({"path": path, "text": text_content})
            except UnicodeDecodeError:
                # Binary content - base64 encode
                encoded = base64.b64encode(content).decode("ascii")
                file_list.append({"path": path, "blob": encoded})

        try:
            if file_list:
                self._interpreter.invoke(
                    method="writeFiles", params={"content": file_list}
                )
            # All files uploaded successfully
            return [FileUploadResponse(path=path, error=None) for path, _ in files]

        except Exception:
            logger.exception("Error uploading files")
            return [
                FileUploadResponse(path=path, error="permission_denied")
                for path, _ in files
            ]


class AgentCoreProvider(SandboxProvider):
    """AgentCore Code Interpreter sandbox provider.

    Manages AgentCore session lifecycle. Sessions cannot be reconnected
    after the CLI exits — the ``sandbox_id`` parameter is not supported.
    """

    def __init__(self, region: str | None = None) -> None:
        """Initialize AgentCore provider.

        Args:
            region: AWS region (defaults to AWS_REGION / AWS_DEFAULT_REGION / us-west-2)
        """
        import os

        self._region = region or os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Create a new AgentCore Code Interpreter session.

        Args:
            sandbox_id: Not supported — raises NotImplementedError if provided.
            **kwargs: Additional parameters (unused).

        Returns:
            AgentCoreBackend instance

        Raises:
            NotImplementedError: If sandbox_id is provided.
        """
        if sandbox_id:
            msg = (
                "AgentCore does not support reconnecting to existing sessions. "
                "Remove the --sandbox-id option."
            )
            raise NotImplementedError(msg)

        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        interpreter = CodeInterpreter(
            region=self._region,
            integration_source=AgentCoreBackend.INTEGRATION_SOURCE,
        )
        interpreter.start()
        return AgentCoreBackend(interpreter)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002, PLR6301
        """Stop an AgentCore session.

        Note: AgentCore sessions are identified by internal session IDs
        and cannot be reconnected. This is a best-effort cleanup.

        Args:
            sandbox_id: Session ID (used for logging only).
            **kwargs: Additional parameters (unused).
        """
        logger.info(
            "AgentCore session %s cleanup requested (sessions auto-expire)",
            sandbox_id,
        )
