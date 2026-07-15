"""Per-session `read_file` deduplication middleware.

On session resume the agent replays its prior trajectory and re-issues
identical `read_file` calls. Without deduplication, each replayed read
re-emits the full file body into the message history, so unchanged files are
read dozens of times in a single trace, exploding context and token cost.

This middleware caches, per conversation thread, the body previously returned
for each `(file_path, offset, limit)` window. When the same window is read
again and the freshly read body is byte-for-byte identical, it returns a
compact "unchanged since it was read earlier" marker instead of the full body.
The full body is always returned on the first read, when the file changed, or
for a window not previously served.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools import ToolRuntime
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

logger = logging.getLogger(__name__)

_READ_FILE_TOOL = "read_file"


def _thread_id(runtime: ToolRuntime) -> str:
    """Return the conversation thread ID for the active run, or a default."""
    context = getattr(runtime, "context", None)
    thread_id = (
        getattr(context, "thread_id", None)
        if not isinstance(context, dict)
        else context.get("thread_id")
    )
    return thread_id if isinstance(thread_id, str) and thread_id else "_default"


def _read_window(request: ToolCallRequest) -> tuple[str, int, int] | None:
    """Return the `(file_path, offset, limit)` window for a `read_file` call."""
    if request.tool_call.get("name") != _READ_FILE_TOOL:
        return None
    args = request.tool_call.get("args") or {}
    file_path = args.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None
    offset = args.get("offset", 0)
    limit = args.get("limit", 100)
    if not isinstance(offset, int) or not isinstance(limit, int):
        return None
    return file_path, offset, limit


def _cacheable_body(result: ToolMessage | Command[Any]) -> str | None:
    """Return a `read_file` result's cacheable text body, else `None`.

    Media results (`Command` updates or `ToolMessage` carrying artifacts /
    inline media) are never cached; only plain successful text bodies are.
    """
    if not isinstance(result, ToolMessage):
        return None
    if result.status == "error" or result.artifact is not None:
        return None
    content = result.content
    return content if isinstance(content, str) else None


class ReadFileCacheMiddleware(AgentMiddleware):
    """Deduplicate repeated `read_file` reads of unchanged files per session."""

    def __init__(self) -> None:
        """Initialize the per-thread read cache."""
        super().__init__()
        self._cache: dict[tuple[str, str, int, int], str] = {}

    def _unchanged_marker(
        self, request: ToolCallRequest, body: str
    ) -> ToolMessage | None:
        """Return a compact marker when this window's body is unchanged."""
        window = _read_window(request)
        if window is None:
            return None
        key = (_thread_id(request.runtime), *window)
        if self._cache.get(key) != body:
            return None
        file_path, _offset, _limit = window
        lines = len(body.splitlines())
        return ToolMessage(
            content=(
                f"File {file_path} unchanged since it was read earlier in this "
                f"session ({lines} lines); prior content still applies."
            ),
            name=_READ_FILE_TOOL,
            tool_call_id=request.tool_call["id"],
        )

    def _record(
        self, request: ToolCallRequest, result: ToolMessage | Command[Any]
    ) -> None:
        """Cache a successful `read_file` text body for later deduplication."""
        window = _read_window(request)
        if window is None:
            return
        body = _cacheable_body(result)
        if body is None:
            return
        self._cache[_thread_id(request.runtime), *window] = body

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Return the unchanged marker for a repeated read, else the full body."""
        result = handler(request)
        body = _cacheable_body(result)
        if body is not None and (marker := self._unchanged_marker(request, body)):
            return marker
        self._record(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Return the unchanged marker for a repeated read, else the full body."""
        result = await handler(request)
        body = _cacheable_body(result)
        if body is not None and (marker := self._unchanged_marker(request, body)):
            return marker
        self._record(request, result)
        return result
