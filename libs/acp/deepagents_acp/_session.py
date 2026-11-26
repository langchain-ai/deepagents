"""Per-session state: the factory's context, the session record, and one turn's bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from acp.schema import AcpMcpServer, HttpMcpServer, McpServerStdio, SseMcpServer

from deepagents_acp._permissions import _Permissions

if TYPE_CHECKING:
    import asyncio

    from acp import PromptResponse
    from acp.schema import SessionModeState, StopReason
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Interrupt

    from deepagents_acp._tool_calls import _Chunks, _ToolCall

McpServer: TypeAlias = HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio
"""Type alias for ACP MCP server configuration variants."""


@dataclass(frozen=True, slots=True)
class AgentSessionContext:
    """Context for an agent session, including working directory, mode, and model."""

    cwd: str
    mode: str
    model: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    """Values the user picked for `extra_config_options`, as sorted `(id, value)` pairs."""
    session_id: str = ""
    mcp_servers: tuple[McpServer, ...] = ()
    """MCP servers the client supplied for this session. They belong to it alone."""

    def option(self, option_id: str) -> str | None:
        """Return the value the user picked for an extra config option, if any."""
        return dict(self.options).get(option_id)


@dataclass
class _Session:
    """Everything the server holds for one open session.

    A session is never reopened while its record exists, and closing waits for
    its turn and background tasks, so work started for a session ends with it.
    """

    cwd: str
    opening: asyncio.Task[Any] | None = None
    """The load or resume still restoring the session; it takes no prompts yet."""
    closing: bool = False
    mcp_servers: list[McpServer] = field(default_factory=list)
    mode: str | None = None
    mode_state: SessionModeState | None = None
    model: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    """Values the user picked for `extra_config_options`."""
    unsaved_options: bool = False
    title: str | None = None
    """`""` while one is being generated."""
    plan: list[dict[str, Any]] | None = None
    agent: CompiledStateGraph | None = None
    last_agent: CompiledStateGraph | None = None
    """The latest agent built, kept across option changes for metadata writes."""
    turn: asyncio.Task[PromptResponse] | None = None
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    """Background work, such as titles and command announcements; closing cancels it."""
    permissions: _Permissions = field(default_factory=_Permissions)


@dataclass
class _Turn:
    """Bookkeeping for one prompt turn: streamed tool calls and the last resumed interrupts."""

    cwd: str
    context: AgentSessionContext
    """The session's options when the turn started; a change mid-turn applies next turn."""
    calls: dict[str, _ToolCall] = field(default_factory=dict)
    chunks: _Chunks = field(default_factory=dict)
    """Streamed tool-call chunks by `(namespace, index)`, so parallel subagents don't merge."""
    interrupt_scopes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """The agent each streamed interrupt came from, keyed by interrupt id."""
    context_tokens: int | None = None
    """Input plus output tokens of the main agent's latest model call."""
    resumed: list[Interrupt] = field(default_factory=list)
    """Interrupts the previous resume answered. A task's consecutive interrupts can
    share an id, so only an identical id and value counts as already answered."""
    stop_reason: StopReason = "end_turn"
