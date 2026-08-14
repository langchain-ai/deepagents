"""CreateOS sandbox backend implementation."""

from __future__ import annotations

import shlex

import httpx
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

DEFAULT_BASE_URL = "https://api.sb.createos.sh"


class CreateOSSandbox(BaseSandbox):
    """CreateOS sandbox implementation conforming to SandboxBackendProtocol.

    Talks directly to the CreateOS HTTP API. Requires a sandbox that has
    already been created (status ``running``).
    """

    def __init__(
        self,
        *,
        sandbox_id: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30 * 60,
    ) -> None:
        """Create a backend wrapping an existing CreateOS sandbox.

        Args:
            sandbox_id: The ``sb_…`` id of a running sandbox.
            api_key: CreateOS API key sent as ``X-Api-Key``.
            base_url: Control-plane URL. Defaults to production.
            timeout: Default command timeout in seconds.
        """
        self._sandbox_id = sandbox_id
        self._base_url = base_url.rstrip("/")
        self._default_timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"X-Api-Key": api_key},
            timeout=httpx.Timeout(timeout=float(timeout), connect=30.0),
        )

    @property
    def id(self) -> str:
        """Return the CreateOS sandbox id."""
        return self._sandbox_id

    def close(self) -> None:
        """Close the HTTP client owned by this backend."""
        self._client.close()

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to
                complete. If None, uses the backend's default timeout.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        resp = self._client.post(
            f"/v1/sandboxes/{self._sandbox_id}/exec",
            json={"cmd": "/bin/sh", "args": ["-c", command]},
            timeout=httpx.Timeout(
                timeout=float(effective_timeout) + 30.0,
                connect=30.0,
            ),
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        result = data["result"]

        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""

        output = stdout
        if stderr.strip():
            output += f"\n<stderr>{stderr.strip()}</stderr>"

        return ExecuteResponse(
            output=output,
            exit_code=result.get("exit_code"),
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to a new file. Errors if the file already exists.

        Args:
            file_path: Absolute path for the file.
            content: UTF-8 text content to write.

        Returns:
            `WriteResult` with the written path on success or an error message.
        """
        check = self.execute(f"test -e {shlex.quote(file_path)}")
        if check.exit_code == 0:
            return WriteResult(error=f"Error: file '{file_path}' already exists")

        preflight_error = self._write_preflight(file_path)
        if preflight_error is not None:
            return preflight_error

        responses = self.upload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            msg = f"upload_files returned {len(responses)} results"
            raise AssertionError(msg)
        response = responses[0]
        if response.error:
            return WriteResult(
                error=f"Failed to write file '{file_path}': {response.error}"
            )
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Write content to a new file asynchronously.

        Args:
            file_path: Absolute path for the file.
            content: UTF-8 text content to write.

        Returns:
            `WriteResult` with the written path on success or an error message.
        """
        check = await self.aexecute(f"test -e {shlex.quote(file_path)}")
        if check.exit_code == 0:
            return WriteResult(error=f"Error: file '{file_path}' already exists")

        preflight_error = await self._awrite_preflight(file_path)
        if preflight_error is not None:
            return preflight_error

        responses = await self.aupload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            msg = f"aupload_files returned {len(responses)} results"
            raise AssertionError(msg)
        response = responses[0]
        if response.error:
            return WriteResult(
                error=f"Failed to write file '{file_path}': {response.error}"
            )
        return WriteResult(path=file_path)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            try:
                resp = self._client.get(
                    f"/v1/sandboxes/{self._sandbox_id}/files",
                    params={"path": path},
                )
                resp.raise_for_status()
                responses.append(
                    FileDownloadResponse(path=path, content=resp.content, error=None)
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 404:  # noqa: PLR2004
                    error = "file_not_found"
                elif status == 400:  # noqa: PLR2004
                    error = "is_directory"
                else:
                    error = str(exc)
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=error)
                )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                resp = self._client.put(
                    f"/v1/sandboxes/{self._sandbox_id}/files",
                    params={"path": path},
                    content=content,
                )
                resp.raise_for_status()
                responses.append(FileUploadResponse(path=path, error=None))
            except httpx.HTTPStatusError as exc:
                responses.append(FileUploadResponse(path=path, error=str(exc)))
        return responses
