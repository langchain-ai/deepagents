"""LangSmith sandbox backend.

Adapted from open-swe (agent/integrations/langsmith.py).
"""

from __future__ import annotations

import os

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox
from langsmith.sandbox import AsyncSandbox, AsyncSandboxClient, ResourceNotFoundError

DEFAULT_TEMPLATE_NAME = "deep-agent"
DEFAULT_TEMPLATE_IMAGE = "python:3"

_backends: dict[str, "LangSmithBackend"] = {}


class LangSmithBackend(BaseSandbox):
    """LangSmith sandbox backend using the async SDK.

    Overrides aexecute/awrite/adownload_files/aupload_files to use the async
    SDK natively. The sync variants raise NotImplementedError since the
    framework always calls the async versions.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._default_timeout: int = 300

    @property
    def id(self) -> str:
        return self._sandbox.name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError("Use aexecute()")

    async def aexecute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = await self._sandbox.run(command, timeout=effective_timeout)
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr
        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        raise NotImplementedError("Use awrite()")

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        try:
            await self._sandbox.write(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path, files_update=None)
        except Exception as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError("Use adownload_files()")

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            content = await self._sandbox.read(path)
            responses.append(
                FileDownloadResponse(path=path, content=content, error=None)
            )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("Use aupload_files()")

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            await self._sandbox.write(path, content)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses


async def get_or_create_sandbox(thread_id: str) -> LangSmithBackend:
    """Get a cached sandbox for this thread, or create a new one."""
    if backend := _backends.get(thread_id):
        return backend

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
        "LANGSMITH_API_KEY_PROD"
    )
    template_name = os.environ.get("SANDBOX_TEMPLATE_NAME", DEFAULT_TEMPLATE_NAME)
    template_image = os.environ.get("SANDBOX_TEMPLATE_IMAGE", DEFAULT_TEMPLATE_IMAGE)

    client = AsyncSandboxClient(api_key=api_key)
    await _ensure_template(client, template_name, template_image)
    sandbox = await client.create_sandbox(template_name=template_name, timeout=180)

    backend = LangSmithBackend(sandbox)
    _backends[thread_id] = backend
    return backend


async def _ensure_template(
    client: AsyncSandboxClient,
    template_name: str,
    template_image: str,
) -> None:
    """Ensure the sandbox template exists, creating it if needed."""
    try:
        await client.get_template(template_name)
    except ResourceNotFoundError as e:
        if e.resource_type != "template":
            raise
        await client.create_template(name=template_name, image=template_image)
