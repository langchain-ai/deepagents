"""Public metadata types for Model Context Protocol servers and tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class MCPToolInfo:
    """Metadata for a single MCP tool."""

    name: str
    """Tool name, which may include a server name prefix."""

    description: str
    """Human-readable description of what the tool does."""

    input_schema: dict[str, Any] | None = None
    """Raw MCP input schema, or `None` when unavailable."""


MCPServerStatus = Literal[
    "ok",
    "unauthenticated",
    "awaiting_reconnect",
    "error",
    "disabled",
]
"""Load state for a configured MCP server."""


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """Metadata for a configured MCP server and its tools."""

    name: str
    """Server name from the MCP configuration."""

    transport: str
    """Transport identifier for the server."""

    tools: tuple[MCPToolInfo, ...] = ()
    """Tools exposed by this server."""

    status: MCPServerStatus = "ok"
    """Current server load state."""

    error: str | None = None
    """Human-readable reason when the server is not available."""

    pending_reconnect: bool = False
    """Whether a re-enabled server is waiting to reconnect."""

    uses_oauth: bool = False
    """Whether the server connection uses OAuth."""

    def __post_init__(self) -> None:
        """Enforce status, error, tool, and reconnect consistency."""
        if self.status == "ok":
            if self.error is not None:
                msg = f"MCPServerInfo {self.name!r}: status='ok' cannot carry an error (got {self.error!r})"
                raise ValueError(msg)
        else:
            if self.error is None:
                msg = f"MCPServerInfo {self.name!r}: status={self.status!r} requires an error message"
                raise ValueError(msg)
            if self.tools:
                msg = f"MCPServerInfo {self.name!r}: status={self.status!r} cannot carry tools"
                raise ValueError(msg)
        if self.pending_reconnect and self.status != "disabled":
            msg = f"MCPServerInfo {self.name!r}: pending_reconnect requires status='disabled' (got {self.status!r})"
            raise ValueError(msg)

    def needs_attention(self) -> bool:
        """Return whether this server is blocked on user login."""
        return self.status == "unauthenticated"
