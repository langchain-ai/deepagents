"""Scope the SDK's tool-result recovery to one fresh task."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware, _routes_to_state_backend
from langchain.agents.middleware import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command


class _TaskArchives(CompositeBackend):
    """Record the durable files created by this invocation."""

    def __init__(self, backend: BackendProtocol) -> None:
        root = backend.artifacts_root if isinstance(backend, CompositeBackend) else "/"
        super().__init__(default=backend, routes={}, artifacts_root=root)
        self.paths: set[str] = set()

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        # Model tool-call IDs can repeat across parallel workers sharing one backend.
        path = f"{file_path.rsplit('/', 1)[0]}/{uuid4().hex}"
        if _routes_to_state_backend(self.default, path):
            return WriteResult(error="Task-local files expire at handoff.")
        try:
            result = await self.default.awrite(path, content)
        except Exception:  # noqa: BLE001 -- a storage failure must keep the completed tool result inline
            return WriteResult(error="Archive write failed.")
        if result is None or result.error:
            return result or WriteResult(error="Archive write failed.")
        self.paths.add(path)
        return WriteResult(path=path)


class ToolResultPaging(AgentMiddleware):
    """Delegate eviction and reads to the SDK, exposing only this task's archives."""

    def __init__(self, backend: BackendProtocol, *, has_reader: bool) -> None:
        """Keep any explicitly attached reader available for its usual paths."""
        self._archives = _TaskArchives(backend)
        self._filesystem = FilesystemMiddleware(backend=self._archives, tools=["read_file"])
        self._has_reader = has_reader

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Offer the SDK reader once there is a result to recover."""
        if self._archives.paths and not self._has_reader:
            request = request.override(tools=[*request.tools, *self._filesystem.tools])
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Use the SDK reader for exact archive paths issued to this task."""
        if request.tool_call["name"] == "read_file":
            path = request.tool_call["args"].get("file_path")
            if isinstance(path, str) and path in self._archives.paths:
                request = request.override(tool=self._filesystem.tools[0])
        return await self._filesystem.awrap_tool_call(request, handler)
