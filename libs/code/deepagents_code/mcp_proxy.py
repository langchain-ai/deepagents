"""Backend recovery at the proxy boundary, before errors reach the wire."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import anyio
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED

from deepagents_code.mcp_auth import MCPReauthRequiredError, find_reauth_required

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastmcp.client.transports import ClientTransport
    from fastmcp.server.middleware import CallNext, MiddlewareContext
    from fastmcp.server.providers.proxy import StatefulProxyClient
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
    """Recover failed connections and preserve actionable login errors."""

    def __init__(self, backend: StatefulProxyClient[ClientTransport]) -> None:
        """Bind recovery to the backend owned by this proxy.

        Args:
            backend: Persistent client whose transport this proxy owns.
        """
        self.backend = backend
        # A forced disconnect must not tear down another in-flight call. Other
        # servers have separate locks and continue running independently.
        self._lock = asyncio.Lock()

    async def _invalidate(self) -> MCPReauthRequiredError | None:
        """Close the failed session and collect its background auth failure.

        Returns:
            A login error revealed while the background session unwinds.
        """
        from deepagents_code.mcp_tools import _close_mcp_resource, _finish_cleanup

        reauth: MCPReauthRequiredError | None = None

        async def disconnect() -> None:
            nonlocal reauth
            try:
                await self.backend._disconnect(force=True)
            except Exception as exc:
                reauth = find_reauth_required(exc)
                if reauth is None:
                    raise

        async def close() -> None:
            await _close_mcp_resource("backend client", disconnect)
            await _close_mcp_resource("transport", self.backend.transport.close)

        await _finish_cleanup(asyncio.create_task(close()))
        return reauth

    async def _recover(self, exc: Exception) -> MCPReauthRequiredError | None:
        """Inspect both the call failure and the session's eventual failure.

        Returns:
            The login error, if present, after clearing the broken session.
        """
        reauth = find_reauth_required(exc)
        session = self.backend._session_state.session_task
        if (
            reauth is None
            and not _is_disconnected(exc)
            and not (session is not None and session.done())
        ):
            raise exc
        # The dispatcher may report Connection closed before the HTTP task
        # finishes. Disconnect awaits that task, preserving its auth exception
        # and resetting FastMCP's nesting counter before the next tool call.
        background_reauth = await self._invalidate()
        return reauth or background_reauth

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
                    reauth = await self._recover(exc)
                    if reauth is not None:
                        return ToolResult(content=str(reauth), is_error=True)
                    if not retry or not _is_disconnected(exc):
                        raise
                    retry = False
