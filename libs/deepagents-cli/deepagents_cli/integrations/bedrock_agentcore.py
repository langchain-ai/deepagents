"""Bedrock AgentCore Code Interpreter sandbox backend implementation.

This module provides a sandbox backend that uses AWS Bedrock AgentCore
Code Interpreter for remote code execution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter


def _extract_output_from_stream(response: Any) -> tuple[str, int | None]:
    """Extract output and exit code from code interpreter response stream.

    Args:
        response: Response from code interpreter execution

    Returns:
        Tuple of (output_text, exit_code)
    """
    output = []
    exit_code = None

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
                    output.append(text)

                    # Try to extract exit code from text if not found elsewhere
                    # Some commands include "Exit code: X" in output
                    if exit_code is None:
                        match = re.search(r"Exit code:\s*(\d+)", text)
                        if match:
                            exit_code = int(match.group(1))

                elif content_type == "resource":
                    resource = content_item.get("resource", {})
                    file_path = resource.get("uri", "").replace("file://", "")

                    if "text" in resource:
                        file_content = resource["text"]
                        output.append(f"==== File: {file_path} ====\n{file_content}\n")
                    elif "blob" in resource:
                        # Binary file (images, etc.) - just note it was created
                        output.append(f"==== Binary File: {file_path} ====\n")
                    else:
                        output.append(f"==== File: {file_path} ====\n")

                elif content_type == "error":
                    error_msg = content_item.get("text", "Unknown error")
                    output.append(f"Error: {error_msg}")
                    if exit_code is None:
                        exit_code = 1  # Default to failure for errors

    return "\n".join(output), exit_code


class AgentCoreBackend(BaseSandbox):
    """AgentCore Code Interpreter backend implementation conforming to SandboxBackendProtocol.

    This implementation uses AWS Bedrock AgentCore Code Interpreter to execute
    shell commands and manage files in a secure, isolated MicroVM environment.

    The backend inherits file operation methods from BaseSandbox (which use
    shell commands via execute()) and implements the execute() method using
    AgentCore's executeCommand API.

    Example:
        ```python
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        interpreter = CodeInterpreter(region="us-west-2")
        interpreter.start()

        backend = AgentCoreBackend(interpreter)
        result = backend.execute("ls -la")
        print(result.output)

        interpreter.stop()
        ```
    """

    def __init__(self, interpreter: CodeInterpreter) -> None:
        """Initialize the AgentCoreBackend with a CodeInterpreter instance.

        Args:
            interpreter: Active CodeInterpreter instance (must be started)
        """
        self._interpreter = interpreter
        self._timeout: int = 30 * 60  # 30 mins

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend (session ID)."""
        return self._interpreter.session_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox and return ExecuteResponse.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        try:
            response = self._interpreter.invoke(
                method="executeCommand", params={"command": command}
            )

            output, exit_code = _extract_output_from_stream(response)

            return ExecuteResponse(
                output=output,
                exit_code=exit_code if exit_code is not None else 0,
                truncated=False,
            )
        except Exception as e:
            # Return error as output with non-zero exit code
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=1,
                truncated=False,
            )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the AgentCore sandbox.

        Uses AgentCore's readFiles API to download file contents.
        Supports partial success - individual downloads may fail without
        affecting others.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order.
        """
        responses: list[FileDownloadResponse] = []

        try:
            # Read files using AgentCore's API
            response = self._interpreter.invoke(method="readFiles", params={"paths": paths})
            output, _ = _extract_output_from_stream(response)

            # Parse the output to extract file contents
            # AgentCore returns files in format: ==== File: path ====\ncontent\n
            file_contents: dict[str, bytes] = {}

            current_file = None
            current_content: list[str] = []

            for line in output.split("\n"):
                if line.startswith("==== File: ") and line.endswith(" ===="):
                    # Save previous file if exists
                    if current_file is not None:
                        content_str = "\n".join(current_content)
                        # Remove trailing empty line if present
                        content_str = content_str.removesuffix("\n")
                        file_contents[current_file] = content_str.encode("utf-8")
                    # Start new file
                    current_file = line[11:-5]  # Extract path between markers
                    current_content = []
                elif line.startswith("==== Binary File: "):
                    # Save previous file
                    if current_file is not None:
                        content_str = "\n".join(current_content)
                        file_contents[current_file] = content_str.encode("utf-8")
                    # Binary files - mark as present but can't download via text API
                    current_file = line[18:-5]
                    current_content = []
                elif current_file is not None:
                    current_content.append(line)

            # Save last file
            if current_file is not None:
                content_str = "\n".join(current_content)
                file_contents[current_file] = content_str.encode("utf-8")

            # Build responses in order of input paths
            for path in paths:
                if path in file_contents:
                    responses.append(
                        FileDownloadResponse(path=path, content=file_contents[path], error=None)
                    )
                else:
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error="file_not_found")
                    )

        except Exception:
            # Return error for all paths
            for path in paths:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="file_not_found",  # Use standard error code
                    )
                )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the AgentCore sandbox.

        Uses AgentCore's writeFiles API to upload file contents.
        Supports partial success - individual uploads may fail without
        affecting others.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        responses: list[FileUploadResponse] = []

        try:
            # Prepare files for AgentCore's writeFiles API
            # AgentCore expects: [{"path": "...", "text": "..."}]
            file_list = []
            for path, content in files:
                try:
                    text_content = content.decode("utf-8")
                    file_list.append({"path": path, "text": text_content})
                except UnicodeDecodeError:
                    # Binary content - skip for now and mark as error
                    responses.append(
                        FileUploadResponse(
                            path=path,
                            error="invalid_path",  # Using closest standard error
                        )
                    )
                    continue

            if file_list:
                # Write files using AgentCore's API
                self._interpreter.invoke(method="writeFiles", params={"content": file_list})

            # Build success responses for text files
            for path, _ in files:
                # Only add if not already in responses (from binary error)
                if not any(r.path == path for r in responses):
                    responses.append(FileUploadResponse(path=path, error=None))

        except Exception:
            # Return error for all paths not already processed
            for path, _ in files:
                if not any(r.path == path for r in responses):
                    responses.append(
                        FileUploadResponse(
                            path=path,
                            error="permission_denied",  # Use standard error code
                        )
                    )

        return responses
