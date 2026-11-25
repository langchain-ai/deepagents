"""Per-session state: the factory's context, the session record, and one turn's bookkeeping."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from acp.schema import AcpMcpServer, HttpMcpServer, McpServerStdio, SseMcpServer

from deepagents_acp._permissions import _Permissions

if TYPE_CHECKING:
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
    generation: int = 0
    """Changes each time the session is opened; see `AgentServerACP.is_current`."""
    mcp_servers: tuple[McpServer, ...] = ()
    """MCP servers the client supplied for this session. They belong to it alone."""

    def option(self, option_id: str) -> str | None:
        """Return the value the user picked for an extra config option, if any."""
        return dict(self.options).get(option_id)


@dataclass
class _Session:
    """Everything the server holds for one open session.

    Do not keep a reference across an `await`: the session can close, or close
    and reopen as a new record, meanwhile. Look it up by id again.
    """

    cwd: str
    generation: int = 0
    """Which opening of the session this is; work from an earlier opening is retired."""
    opening: bool = False
    """A load or resume is still restoring the session; it takes no prompts yet."""
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
    option_revision: int = 0
    """Bumped by every option change, so a build that started earlier is discarded."""
    build_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn: asyncio.Task[PromptResponse] | None = None
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
