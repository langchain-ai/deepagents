"""Backend recovery at the proxy boundary, before errors reach the wire."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING

import anyio
from fastmcp.server.middleware import Middleware
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastmcp.client.transports import ClientTransport
    from fastmcp.server.middleware import CallNext, MiddlewareContext
    from fastmcp.server.providers.proxy import StatefulProxyClient
    from fastmcp.tools import ToolResult
    from mcp_types import CallToolRequestParams


def _exception_tree(exc: BaseException) -> Iterator[BaseException]:
    """Visit wrapped transport failures without looping on cyclic chains.

    Yields:
        Each exception in the chain or group once.
    """
    pending = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        pending.extend(
            cause
            for cause in (current.__cause__, current.__context__)
            if cause is not None
        )


def _is_disconnected(exc: BaseException) -> bool:
    """Recognize transport failures, never a tool's returned error content.

    Returns:
        Whether the exception tree contains a closed connection.
    """
    return any(
        isinstance(
            error,
            (
                anyio.ClosedResourceError,
                anyio.BrokenResourceError,
                anyio.EndOfStream,
                ConnectionError,
                EOFError,
                asyncio.IncompleteReadError,
            ),
        )
        or (isinstance(error, MCPError) and error.code == CONNECTION_CLOSED)
        for error in _exception_tree(exc)
    )


class MCPBackendMiddleware(Middleware):
    """Reconnect a failed backend and retry its tool call once."""

    def __init__(self, backend: StatefulProxyClient[ClientTransport]) -> None:
        """Bind recovery to the backend owned by this proxy.

        Args:
            backend: Persistent client whose transport this proxy owns.
        """
        self.backend = backend
        # A forced disconnect must not tear down another in-flight call. Other
        # servers have separate locks and continue running independently.
        self._lock = asyncio.Lock()

    async def _invalidate(self) -> None:
        from deepagents_code.mcp_tools import _close_mcp_resource, _finish_cleanup

        async def close() -> None:
            await _close_mcp_resource(
                "backend client",
                functools.partial(self.backend._disconnect, force=True),
            )
            await _close_mcp_resource("transport", self.backend.transport.close)

        await _finish_cleanup(asyncio.create_task(close()))

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Retry a broken connection once, leaving tool errors untouched.

        Args:
            context: Incoming tool request.
            call_next: Remaining proxy middleware and tool execution.

        Returns:
            The backend's tool result.
        """
        async with self._lock:
            retry = True
            while True:
                try:
                    return await call_next(context)
                except Exception as exc:
                    if not _is_disconnected(exc):
                        raise
                    await self._invalidate()
                    if not retry:
                        raise
                    retry = False
