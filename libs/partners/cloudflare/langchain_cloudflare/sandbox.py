"""Cloudflare sandbox backend implementation.

Communicates with the Cloudflare Sandbox Bridge HTTP API to execute commands
and manage files in isolated container sandboxes running on Cloudflare Workers.
"""

from __future__ import annotations

import base64
import json
import logging

import httpx
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT_EXIT_CODE = 124
_HTTP_NOT_FOUND = 404


class CloudflareSandbox(BaseSandbox):
    """Cloudflare sandbox implementation conforming to SandboxBackendProtocol.

    Talks to the Cloudflare Sandbox Bridge HTTP API exposed by a Cloudflare
    Worker. The bridge manages sandbox lifecycle, command execution (via SSE),
    and file I/O.

    Example:
        ```python
        from langchain_cloudflare import CloudflareSandbox

        sandbox = CloudflareSandbox(
            base_url="https://your-worker.workers.dev",
            sandbox_id="my-session",
            api_key="your-api-key",
        )
        result = sandbox.execute("echo hello")
        print(result.output)
        ```
    """

    def __init__(
        self,
        *,
        base_url: str,
        sandbox_id: str,
        api_key: str | None = None,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend connected to a Cloudflare Sandbox Bridge.

        Args:
            base_url: URL of the deployed Cloudflare Worker bridge
                (e.g. ``https://sandbox-bridge.your-subdomain.workers.dev``).
            sandbox_id: Identifier for the sandbox container. The bridge uses
                this to create or resume a sandbox via Durable Objects.
            api_key: Bearer token for authenticating with the bridge. Optional
                when the bridge is running without ``SANDBOX_API_KEY`` set.
            timeout: Default command timeout in seconds used when ``execute()``
                is called without an explicit ``timeout``.
        """
        self._base_url = base_url.rstrip("/")
        self._sandbox_id = sandbox_id
        self._default_timeout = timeout
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout + 30, connect=30),
        )

    @property
    def id(self) -> str:
        """Return the Cloudflare sandbox id."""
        return self._sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox via the bridge.

        Sends the command to the ``/v1/sandbox/:id/exec`` endpoint and
        consumes the SSE stream to collect stdout, stderr, and exit code.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to
                complete. If ``None``, uses the backend's default timeout.

        Returns:
            ``ExecuteResponse`` containing output, exit code, and truncation
            flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        payload: dict[str, object] = {
            "argv": ["sh", "-lc", command],
            "timeout_ms": effective_timeout * 1000,
        }

        try:
            response = self._client.post(
                f"/v1/sandbox/{self._sandbox_id}/exec",
                json=payload,
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(effective_timeout + 30, connect=30),
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            msg = f"Command timed out after {effective_timeout} seconds"
            return ExecuteResponse(
                output=msg,
                exit_code=COMMAND_TIMEOUT_EXIT_CODE,
                truncated=False,
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            code = exc.response.status_code
            return ExecuteResponse(
                output=f"Bridge HTTP error: {code} {detail}",
                exit_code=1,
                truncated=False,
            )

        return self._parse_exec_sse(response.text, effective_timeout)

    def _parse_exec_sse(self, body: str, timeout: int) -> ExecuteResponse:
        """Parse the SSE response from the bridge exec endpoint.

        Args:
            body: Raw SSE text body from the bridge.
            timeout: The timeout that was used for the command.

        Returns:
            Parsed ``ExecuteResponse``.
        """
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int | None = None

        for line in body.splitlines():
            if not line.startswith("data: "):
                continue

            data = line[len("data: ") :]
            event_type = _extract_preceding_event(body, line)
            parsed_exit = _dispatch_sse_event(
                event_type,
                data,
                stdout_parts,
                stderr_parts,
            )
            if parsed_exit is not None:
                exit_code = parsed_exit

        output = _combine_output(stdout_parts, stderr_parts)

        if exit_code is None:
            msg = f"Command timed out after {timeout} seconds"
            return ExecuteResponse(
                output=msg,
                exit_code=COMMAND_TIMEOUT_EXIT_CODE,
                truncated=False,
            )

        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox via the bridge file API.

        Args:
            paths: List of absolute file paths to download.

        Returns:
            List of ``FileDownloadResponse`` objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            try:
                url_path = path.lstrip("/")
                resp = self._client.get(
                    f"/v1/sandbox/{self._sandbox_id}/file/{url_path}",
                )
                resp.raise_for_status()
                responses.append(
                    FileDownloadResponse(path=path, content=resp.content, error=None)
                )
            except httpx.HTTPStatusError as exc:
                error = (
                    "file_not_found"
                    if exc.response.status_code == _HTTP_NOT_FOUND
                    else "permission_denied"
                )
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=error)
                )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox via the bridge file API.

        Args:
            files: List of ``(path, content)`` tuples to upload.

        Returns:
            List of ``FileUploadResponse`` objects, one per input file.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                url_path = path.lstrip("/")
                resp = self._client.put(
                    f"/v1/sandbox/{self._sandbox_id}/file/{url_path}",
                    content=content,
                    headers={"Content-Type": "application/octet-stream"},
                )
                resp.raise_for_status()
                responses.append(FileUploadResponse(path=path, error=None))
            except httpx.HTTPStatusError as exc:
                logger.debug("Failed to upload %s: %s", path, exc)
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
        return responses


_ERROR_EXIT_CODE = 1


def _dispatch_sse_event(
    event_type: str | None,
    data: str,
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> int | None:
    """Route a single SSE event to the appropriate output buffer.

    Args:
        event_type: The SSE event type (stdout, stderr, exit, error).
        data: The raw data payload for this event.
        stdout_parts: Accumulator for stdout chunks (mutated in place).
        stderr_parts: Accumulator for stderr chunks (mutated in place).

    Returns:
        The exit code if this was a terminal event (``exit`` or ``error``),
        otherwise ``None``.
    """
    if event_type == "stdout":
        stdout_parts.append(_decode_sse_data(data))
    elif event_type == "stderr":
        stderr_parts.append(_decode_sse_data(data))
    elif event_type == "exit":
        try:
            parsed = json.loads(data)
            return parsed.get("exit_code")
        except (json.JSONDecodeError, ValueError):
            pass
    elif event_type == "error":
        try:
            parsed = json.loads(data)
            stderr_parts.append(parsed.get("error", data))
        except (json.JSONDecodeError, ValueError):
            stderr_parts.append(data)
        return _ERROR_EXIT_CODE
    return None


def _combine_output(
    stdout_parts: list[str],
    stderr_parts: list[str],
) -> str:
    """Combine stdout and stderr parts into a single output string.

    Args:
        stdout_parts: Collected stdout chunks.
        stderr_parts: Collected stderr chunks.

    Returns:
        Combined output string.
    """
    output = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if stderr:
        output += f"\n<stderr>{stderr.strip()}</stderr>" if output else stderr.strip()
    return output


def _extract_preceding_event(body: str, data_line: str) -> str | None:
    """Extract the event type from the SSE event preceding a data line.

    Args:
        body: Full SSE body text.
        data_line: The ``data: ...`` line to find the event type for.

    Returns:
        The event type string, or ``None`` if not found.
    """
    lines = body.splitlines()
    try:
        idx = lines.index(data_line)
    except ValueError:
        return None
    for i in range(idx - 1, -1, -1):
        candidate = lines[i]
        if candidate.startswith("event: "):
            return candidate[len("event: ") :]
        if candidate.strip() == "":
            break
    return None


def _decode_sse_data(data: str) -> str:
    """Decode SSE data which may be base64-encoded.

    Args:
        data: Raw data string from an SSE ``data:`` field.

    Returns:
        Decoded string content.
    """
    try:
        return base64.b64decode(data).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001  # base64 decode may fail for non-encoded data
        return data
