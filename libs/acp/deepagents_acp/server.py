"""ACP server implementation for Deep Agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, TypeGuard, TypeVar, cast
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Agent as ACPAgent,
    AuthenticateResponse,
    DeleteSessionResponse,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    run_agent as run_acp_agent,
)
from acp.exceptions import RequestError
from acp.schema import (
    AcpMcpServer,
    AgentCapabilities,
    AgentMessageChunk,
    AudioContentBlock,
    AuthMethodAgent,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CloseSessionResponse,
    ConfigOptionUpdate,
    Cost,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    McpCapabilities,
    McpServerStdio,
    PromptCapabilities,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionDeleteCapabilities,
    SessionInfo,
    SessionInfoUpdate,
    SessionListCapabilities,
    SessionModeState,
    SessionResumeCapabilities,
    SseMcpServer,
    TerminalAuthMethod,
    TextContentBlock,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
)
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain_core.exceptions import ModelRefusalError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from deepagents_acp._messages import _content_updates, _prompt_content, _stop_reason
from deepagents_acp._permissions import (
    _PLAN_REJECTED,
    _all_tasks_completed,
    _execute_arguments,
    _permission_options,
    _permission_request,
    _permission_tool_call_id,
    _plan_text,
)
from deepagents_acp._replay import _history_messages, _replay_updates
from deepagents_acp._session import AgentSessionContext, McpServer, _Session, _Turn
from deepagents_acp._tool_calls import (
    ToolCallDisplay,
    _fail_open_calls,
    _finish_tool_call,
    _plan_update,
    _start_message_tool_calls,
    _start_streamed_tool_calls,
)
from deepagents_acp._version import __version__

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from acp.interfaces import Client
    from deepagents.graph import Checkpointer
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.types import Interrupt

    from deepagents_acp._permissions import _ExecuteKey

__all__ = [
    "AgentFactory",
    "AgentServerACP",
    "AgentSessionContext",
    "AuthMethod",
    "McpServer",
    "ToolCallDisplay",
]

_ShownToolCall = TypeVar("_ShownToolCall", ToolCallStart, ToolCallUpdate)

AuthMethod: TypeAlias = TerminalAuthMethod | AuthMethodAgent
"""Type alias for the sign-in methods an agent advertises in `initialize`."""

_DEFAULT_AGENT_INFO = Implementation(
    name="deepagents-acp", title="Deep Agents", version=__version__
)

_MCP_SERVER_TYPES = (HttpMcpServer, SseMcpServer, AcpMcpServer, McpServerStdio)
"""Runtime MCP server classes used to detect legacy positional `new_session` calls."""

logger = logging.getLogger(__name__)

_ACP_MODE_METADATA_KEY = "acp_mode"
_ACP_MODEL_METADATA_KEY = "acp_model"
_ACP_SESSION_METADATA_KEY = "acp_session"
_ACP_OPTIONS_METADATA_KEY = "acp_options"
_ACP_TITLE_METADATA_KEY = "acp_title"
_SESSION_PAGE_SIZE = 50
_CLOSE_DEADLINE_SECONDS = 5
_ANNOUNCE_DELAY_SECONDS = 0.05
"""Lets the response that creates a session reach the client before its commands do."""


def _session_mcp_servers(
    additional_directories: list[str] | list[McpServer] | None,
    mcp_servers: list[McpServer] | None,
) -> list[McpServer]:
    """Return `new_session`'s MCP servers, also accepting them in the old second positional slot."""
    if mcp_servers is not None:
        return mcp_servers
    if additional_directories and _is_mcp_servers(additional_directories):
        return additional_directories
    return []


def _is_mcp_servers(
    mcp_servers: list[str] | list[McpServer],
) -> TypeGuard[list[McpServer]]:
    """Return whether a value is the ACP `mcp_servers` argument."""
    return all(isinstance(server, _MCP_SERVER_TYPES) for server in mcp_servers)


_UNSUPPORTED_INTERRUPT = (
    "ACP limitation: this agent raised a free-form LangGraph interrupt(), which ACP "
    "cannot display.\n\n"
    "ACP only supports human-in-the-loop permission prompts with a fixed set of "
    "decisions (approve/reject/edit).\n"
    "Spec: https://agentclientprotocol.com/protocol/overview\n\n"
    "Fix: use LangChain HumanInTheLoopMiddleware-style interrupts "
    "(action_requests/review_configs), or override `handle_interrupt`.\n"
    "Docs: https://docs.langchain.com/oss/python/langchain/human-in-the-loop\n\n"
    "This is a protocol limitation, not a bug in the agent."
)


def _check_saved_session(session_id: str, cwd: str, metadata: Mapping[str, Any]) -> None:
    """Refuse to load a thread that isn't an ACP session or belongs to another cwd.

    Raises:
        RequestError: `resource_not_found` or `invalid_params`.
    """
    if metadata.get(_ACP_SESSION_METADATA_KEY) is not True:
        raise RequestError.resource_not_found(session_id)
    if metadata.get("cwd") != cwd:
        raise RequestError.invalid_params(
            {"cwd": "must match the working directory used to create the session"}
        )


AgentFactory: TypeAlias = Callable[
    [AgentSessionContext], CompiledStateGraph | Awaitable[CompiledStateGraph]
]
"""Builds a session's agent; may be async, for example to connect client MCP servers."""


class AgentServerACP(ACPAgent):
    """ACP agent server that bridges Deep Agents with the Agent Client Protocol."""

    _conn: Client

    internal_model_sources: ClassVar[frozenset[str]] = frozenset({"summarization"})
    """`lc_source` values that mark model calls the harness makes for itself.

    Their streamed output is not the agent speaking, so it is not shown.
    """

    def __init__(  # noqa: PLR0913  # optional settings are keyword-only
        self,
        agent: CompiledStateGraph | AgentFactory,
        *,
        modes: SessionModeState | None = None,
        models: list[dict[str, str]] | None = None,
        load_sessions: bool = False,
        agent_info: Implementation | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """Initialize the ACP agent server with the given agent factory or compiled graph.

        Args:
            agent: Either a compiled state graph or a factory function that creates one
            modes: Optional approval modes, offered as the `mode` config option and the
              legacy `modes` field
            models: Optional list of available models with 'value', 'name', and optionally
              'description'
            load_sessions: Advertise and implement durable `session/load`. The agent graph
              must use a checkpointer that survives server restarts.
            agent_info: Name, title, and version reported to clients in `initialize`.
            checkpointer: The checkpointer the agent graphs share, if any. Lets the
              server read saved sessions without building an agent first.
        """
        super().__init__()
        self._agent_info = agent_info or _DEFAULT_AGENT_INFO
        self._checkpointer = checkpointer
        self._lists_sessions = load_sessions and checkpointer is not None
        self._client_capabilities = ClientCapabilities()
        self._agent_factory = agent
        self._sessions: dict[str, _Session] = {}
        self._load_sessions = load_sessions

        if isinstance(agent, CompiledStateGraph):
            if modes is not None:
                msg = "modes can only be provided when agent is a factory"
                raise ValueError(msg)
            if models is not None:
                msg = "models can only be provided when agent is a factory"
                raise ValueError(msg)
            self._modes: SessionModeState | None = None
            self._models: list[dict[str, str]] | None = None
        else:
            self._modes = modes
            self._models = models

    @property
    def client(self) -> Client:
        """The connected client, for overrides that send their own requests."""
        return self._conn

    @property
    def client_capabilities(self) -> ClientCapabilities:
        """What the client advertised in `initialize`."""
        return self._client_capabilities

    def set_models(self, models: list[dict[str, str]]) -> None:
        """Replace the model catalog, for agents that learn their models after startup."""
        self._models = models

    def on_connect(self, conn: Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn

    def _build_config_options(
        self,
        session_id: str,
    ) -> list[SessionConfigOptionSelect | SessionConfigOptionBoolean]:
        """Build the list of session configuration options.

        Returns a list combining mode and model selectors if available.
        Modes are mapped to config options with category='mode'.
        Models are exposed as config options with category='model'.
        """
        config_options: list[SessionConfigOptionSelect | SessionConfigOptionBoolean] = []
        session = self._peek(session_id)

        # Add mode selector if modes are configured
        if self._modes is not None:
            current_mode = self._modes.current_mode_id if session.mode is None else session.mode
            mode_options = [
                SessionConfigSelectOption(
                    value=mode.id,
                    name=mode.name,
                    description=mode.description,
                )
                for mode in self._modes.available_modes
            ]

            mode_select = SessionConfigOptionSelect(
                id="mode",
                name="Session Mode",
                description="Controls how the agent requests permission",
                category="mode",
                type="select",
                current_value=current_mode,
                options=mode_options,
            )
            config_options.append(mode_select)

        # Add model selector if models are configured
        current_model = session.model or self.default_model()
        if self._models and current_model is not None:
            model_options = [
                SessionConfigSelectOption(
                    value=model["value"],
                    name=model["name"],
                    description=model.get("description", ""),
                )
                for model in self._models
            ]

            model_select = SessionConfigOptionSelect(
                id="model",
                name="Model",
                description="The LLM model to use for this session",
                category="model",
                type="select",
                current_value=current_model,
                options=model_options,
            )
            config_options.append(model_select)

        config_options.extend(self.extra_config_options(self._session_context(session_id)))
        return config_options

    async def initialize(
        self,
        protocol_version: int,  # noqa: ARG002  # the server always answers with the version it speaks
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,  # noqa: ARG002  # ACP protocol interface parameter
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> InitializeResponse:
        """Return server capabilities and sign-in methods to the ACP client."""
        self._client_capabilities = client_capabilities or ClientCapabilities()
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_info=self._agent_info,
            agent_capabilities=AgentCapabilities(
                load_session=self._load_sessions,
                prompt_capabilities=PromptCapabilities(
                    image=True,
                ),
                mcp_capabilities=self.mcp_capabilities(),
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities(),
                    resume=SessionResumeCapabilities() if self._load_sessions else None,
                    list=SessionListCapabilities() if self._lists_sessions else None,
                    delete=SessionDeleteCapabilities() if self._lists_sessions else None,
                ),
            ),
            auth_methods=list(self.auth_methods(self._client_capabilities)),
        )

    def mcp_capabilities(self) -> McpCapabilities:
        """Return the MCP transports the agent connects for client-supplied servers.

        Every session gets the client's servers in `AgentSessionContext.mcp_servers`,
        but only an agent factory that connects them should advertise transports.
        """
        return McpCapabilities()

    def auth_methods(self, client_capabilities: ClientCapabilities) -> Sequence[AuthMethod]:
        """Return the sign-in methods to advertise in `initialize`.

        Override to offer sign-in. Only advertise `terminal` methods when the
        client supports them (`client_capabilities.auth.terminal` or the legacy
        `_meta["terminal-auth"]` flag).
        """
        del client_capabilities
        return []

    async def check_ready(self, context: AgentSessionContext) -> None:
        """Refuse to start a session the agent cannot run.

        Called by `session/new` and `session/load` before any session state is
        created. Override to raise `RequestError.auth_required(...)` when sign-in
        is missing, or another `RequestError` for configuration problems.
        """
        del context

    async def authenticate(
        self,
        method_id: str,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> AuthenticateResponse | None:
        """Reject `authenticate` for methods the client must run itself.

        Terminal methods run in the client's terminal, so an `authenticate`
        call for one comes from a client that ignored the method type.
        """
        methods = {method.id: method for method in self.auth_methods(self._client_capabilities)}
        method = methods.get(method_id)
        if method is None:
            raise RequestError.invalid_params({"methodId": f"unknown auth method {method_id!r}"})
        if not isinstance(method, TerminalAuthMethod):
            name = "authenticate"
            raise RequestError.method_not_found(name)
        msg = f"Auth method {method_id!r} runs in a terminal; launch it instead"
        raise RequestError.invalid_request({"methodId": method_id, "message": msg})

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | list[McpServer] | None = None,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> NewSessionResponse:
        """Create a new agent session with the given working directory."""
        mcp_servers = _session_mcp_servers(additional_directories, mcp_servers)
        await self.check_ready(self._default_context(cwd))
        session_id = uuid4().hex
        session = self._begin_session(session_id, cwd)
        try:
            session.mcp_servers = mcp_servers

            self._initialize_session_options(session)

            config_options = self._build_config_options(session_id) or None
            # Return response with both modes (for backward compatibility) and config_options
            response = NewSessionResponse(
                session_id=session_id,
                modes=self._modes if self._modes is not None else None,
                config_options=config_options,
            )
            # Last, so a session that fails to set up leaves nothing in history.
            if self._load_sessions:
                await self._persist_session(session_id)
        except BaseException:
            try:
                await self._release_session(session_id)
            finally:
                self._forget_session(session_id)
            raise
        self._spawn(session_id, self._announce_commands(session_id))
        return response

    async def load_session(  # ty: ignore[invalid-method-override]  # Keep our public positional order; the ACP router calls by keyword
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,  # noqa: ARG002  # capability is not advertised
        mcp_servers: list[McpServer] | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> LoadSessionResponse:
        """Restore and replay a persisted ACP session."""
        await self._open_saved_session("session/load", cwd, session_id, mcp_servers, replay=True)
        self._spawn(session_id, self._announce_commands(session_id))
        return LoadSessionResponse(
            modes=self._peek(session_id).mode_state,
            config_options=self._build_config_options(session_id) or None,
        )

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,  # noqa: ARG002  # capability is not advertised
        mcp_servers: list[McpServer] | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> ResumeSessionResponse:
        """Reopen a persisted ACP session without replaying its history."""
        await self._open_saved_session("session/resume", cwd, session_id, mcp_servers)
        self._spawn(session_id, self._announce_commands(session_id))
        return ResumeSessionResponse(
            modes=self._peek(session_id).mode_state,
            config_options=self._build_config_options(session_id) or None,
        )

    async def _open_saved_session(
        self,
        method: str,
        cwd: str,
        session_id: str,
        mcp_servers: list[McpServer] | None,
        *,
        replay: bool = False,
    ) -> None:
        """Restore a saved session's options, check readiness, build its agent, and replay it.

        Raises:
            RequestError: `method_not_found` without durable sessions, `invalid_request`
                for a session that is already open, or the saved-session and readiness
                errors.
        """
        if not self._load_sessions:
            raise RequestError.method_not_found(method)
        session = self._begin_session(session_id, cwd)
        # A close meanwhile cancels this task, which releases the session below.
        session.opening = asyncio.current_task()
        try:
            metadata = await self._session_metadata(session_id)
            _check_saved_session(session_id, cwd, metadata)
            session.mcp_servers = list(mcp_servers or [])
            self._initialize_session_options(session)
            self._restore_session_options(session, metadata)
            # Check the restored options, which can name another model than the launch default.
            await self.check_ready(self._session_context(session_id))
            # Claim the thread now (an ownership override takes its lease here), so a
            # session another process holds fails to open instead of on first prompt.
            self._session_config(session_id)
            agent = await self._reset_agent(session_id)
            if replay:
                await self._replay_session(session_id, agent)
        except BaseException:
            try:
                await self._release_session(session_id)
            finally:
                self._forget_session(session_id)
            raise
        session.opening = None

    async def _release_session(self, session_id: str) -> None:
        """Let an override release what it acquired for a session; never raises."""
        try:
            await self.session_closed(session_id)
        except Exception:  # noqa: BLE001  # cleanup must not mask the error being handled
            logger.warning("Releasing session %s failed", session_id, exc_info=True)

    def _begin_session(self, session_id: str, cwd: str) -> _Session:
        """Add a record for a session that is not open.

        Raises:
            RequestError: `invalid_request` when the session is already open here.
        """
        if session_id in self._sessions:
            raise RequestError.invalid_request(
                {"message": "The session is already open; close it first"}
            )
        session = self._sessions[session_id] = _Session(cwd=cwd)
        return session

    def _peek(self, session_id: str) -> _Session:
        """Return the session's record, or a blank one when the session isn't open.

        Changes to a blank record are not kept.
        """
        return self._sessions.get(session_id) or _Session(cwd="")

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> ListSessionsResponse:
        """List saved ACP sessions, newest first, a page at a time."""
        if not self._lists_sessions:
            method = "session/list"
            raise RequestError.method_not_found(method)
        sessions = await self.saved_sessions(cwd)
        start = int(cursor) if cursor and cursor.isdigit() else 0
        end = start + _SESSION_PAGE_SIZE
        return ListSessionsResponse(
            sessions=sessions[start:end],
            next_cursor=str(end) if end < len(sessions) else None,
        )

    async def saved_sessions(self, cwd: str | None) -> list[SessionInfo]:
        """Return the sessions created over ACP, newest first, optionally for one cwd.

        Only threads this server marked as ACP sessions are listed, so every
        listed session can be loaded. Override to read a faster index.
        """
        if self._checkpointer is None:
            return []
        latest: dict[str, Any] = {}
        async for saved in self._checkpointer.alist(None, filter={_ACP_SESSION_METADATA_KEY: True}):
            configurable = saved.config["configurable"]
            thread_id = configurable["thread_id"]
            if configurable.get("checkpoint_ns") or thread_id in latest:
                continue
            latest[thread_id] = saved
        sessions = [
            SessionInfo(
                session_id=thread_id,
                cwd=saved.metadata["cwd"],
                title=saved.metadata.get(_ACP_TITLE_METADATA_KEY) or None,
                updated_at=saved.checkpoint.get("ts"),
            )
            for thread_id, saved in latest.items()
            if isinstance(saved.metadata.get("cwd"), str)
            and (cwd is None or saved.metadata["cwd"] == cwd)
        ]
        return sorted(sessions, key=lambda info: info.updated_at or "", reverse=True)

    async def close_session(
        self,
        session_id: str,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> CloseSessionResponse:
        """Cancel the session's running turn and release everything it holds.

        Waits for the turn and background work to end, so nothing from this
        session runs after the close returns. A session still opening is
        closed by cancelling its load or resume, which then fails.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return CloseSessionResponse()
        if session.closing:
            raise RequestError.invalid_request({"message": "The session is closing"})
        if session.opening is not None:
            session.opening.cancel()
            await asyncio.wait({session.opening})
            return CloseSessionResponse()
        session.closing = True
        try:
            work = {task for task in (session.turn, *session.tasks) if task is not None}
            for task in work:
                task.cancel()
            if work:
                await asyncio.wait(work)
            await self._save_before_close(session_id)
        finally:
            # A cancelled release must still forget the session, or it stays "closing".
            try:
                await self._release_session(session_id)
            finally:
                self._forget_session(session_id)
        return CloseSessionResponse()

    async def _save_before_close(self, session_id: str) -> None:
        """Save option changes a closing session hasn't written yet, within the deadline."""
        session = self._peek(session_id)
        if not session.unsaved_options or session.last_agent is None:
            return
        try:
            await asyncio.wait_for(
                self._persist_session(session_id), timeout=_CLOSE_DEADLINE_SECONDS
            )
        except Exception:  # noqa: BLE001  # a failed save must not stop the close
            logger.warning("Could not save options for closing session", exc_info=True)

    async def delete_session(
        self,
        session_id: str,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> DeleteSessionResponse:
        """Close a saved ACP session and delete its history."""
        if not self._lists_sessions or self._checkpointer is None:
            method = "session/delete"
            raise RequestError.method_not_found(method)
        metadata = await self._session_metadata(session_id)
        if metadata.get(_ACP_SESSION_METADATA_KEY) is not True:
            raise RequestError.resource_not_found(session_id)
        await self.close_session(session_id)
        session = self._begin_session(session_id, str(metadata.get("cwd", "")))
        session.closing = True
        try:
            # Claim the thread as an open does, so one another process holds is not deleted.
            self._session_config(session_id)
            await self.delete_saved_session(session_id)
        finally:
            self._forget_session(session_id)
        return DeleteSessionResponse()

    async def delete_saved_session(self, session_id: str) -> None:
        """Delete a saved session's history. Override to clean up related records too."""
        if self._checkpointer is not None:
            await self._checkpointer.adelete_thread(session_id)

    async def session_closed(self, session_id: str) -> None:
        """Release resources an override holds for a session that is closing."""
        del session_id

    async def set_session_mode(  # ty: ignore[invalid-method-override]  # Keep our public positional order; the ACP router calls by keyword
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> SetSessionModeResponse:
        """Switch the session's mode from the next turn on."""
        await self._change_option(session_id, "mode", mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,  # noqa: FBT001  # signature fixed by ACP protocol interface
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> SetSessionConfigOptionResponse:
        """Change a config option from the next turn on and return all options."""
        # Only select-type options are supported; reject boolean values.
        if not isinstance(value, str):
            msg = f"Config option {config_id!r} expects a string value, got {type(value).__name__}"
            raise RequestError(-32602, msg)
        config_options = await self._change_option(session_id, config_id, value)
        return SetSessionConfigOptionResponse(config_options=config_options)

    async def _change_option(
        self, session_id: str, config_id: str, value: str
    ) -> list[SessionConfigOptionSelect | SessionConfigOptionBoolean]:
        """Validate and apply one option, then save it and tell the client.

        Raises:
            RequestError: `invalid_params` for an unknown option or a value the
                session doesn't offer, `resource_not_found` for a session that is
                not open, `invalid_request` while it opens or closes.
        """
        session = self._require_open(session_id)
        if config_id == "mode" and self._modes is not None:
            state = session.mode_state or self._modes
            if not any(mode.id == value for mode in state.available_modes):
                msg = f"Invalid mode: {value}"
                raise RequestError(-32602, msg)
            session.mode = value
            session.mode_state = SessionModeState(
                available_modes=state.available_modes, current_mode_id=value
            )
        elif config_id == "model" and self._models is not None:
            if not any(model["value"] == value for model in self._models):
                msg = f"Invalid model: {value}"
                raise RequestError(-32602, msg)
            session.model = value
        else:
            self._check_extra_option(session_id, config_id, value)
            session.options[config_id] = value
        # The next use rebuilds the agent; a turn already running keeps its own.
        session.agent = None
        await self._persist_options(session_id)
        config_options = self._build_config_options(session_id)
        with contextlib.suppress(ConnectionError):
            await self._send(
                session_id,
                ConfigOptionUpdate(
                    session_update="config_option_update", config_options=config_options
                ),
            )
            if config_id == "mode":
                await self._send(
                    session_id,
                    CurrentModeUpdate(session_update="current_mode_update", current_mode_id=value),
                )
        return config_options

    def _require_open(self, session_id: str) -> _Session:
        """Return an open session's record, refusing one that is closing or still being restored.

        Raises:
            RequestError: `resource_not_found` for a session that is not open,
                `invalid_request` while it closes or opens.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.resource_not_found(session_id)
        if session.closing:
            raise RequestError.invalid_request({"message": "The session is closing"})
        if session.opening is not None:
            raise RequestError.invalid_request({"message": "The session is still opening"})
        return session

    def extra_config_options(
        self,
        context: AgentSessionContext,  # noqa: ARG002  # used by overrides
    ) -> list[SessionConfigOptionSelect]:
        """Return config options beyond mode and model, such as reasoning effort.

        Override to add options. Each option's `current_value` is what the
        session uses now. A value the user picks reaches the agent factory as
        `AgentSessionContext.options`, and options are rebuilt after every
        change, so they can depend on the session's model.
        """
        return []

    def _check_extra_option(self, session_id: str, config_id: str, value: str) -> None:
        """Reject an unknown extra option or a value it doesn't offer.

        Raises:
            RequestError: `invalid_params` for an unknown option or value.
        """
        options = self.extra_config_options(self._session_context(session_id))
        option = next((option for option in options if option.id == config_id), None)
        if option is None:
            msg = f"Unknown config option: {config_id}"
            raise RequestError(-32602, msg)
        values = {
            choice.value
            for choice in option.options
            if isinstance(choice, SessionConfigSelectOption)
        }
        if value not in values:
            msg = f"Invalid value for {config_id}: {value}"
            raise RequestError(-32602, msg)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # ACP protocol interface parameters
        """Cancel the session's running turn; its `session/prompt` returns `cancelled`."""
        turn = self._peek(session_id).turn
        if turn is not None and not turn.done():
            turn.cancel()

    async def _send(self, session_id: str, update: Any) -> None:  # any ACP session update
        """Send one session update to the client."""
        if isinstance(update, ToolCallStart):
            # Applied here, where live turns and replay both send, so they cannot differ.
            update = self._displayed(session_id, update.name or "", update)
        await self._conn.session_update(session_id=session_id, update=update, source="DeepAgent")

    async def _log_text(self, session_id: str, text: str) -> None:
        """Send a text message update to the client."""
        update = AgentMessageChunk(
            session_update="agent_message_chunk", content=TextContentBlock(type="text", text=text)
        )
        await self._send(session_id, update)

    async def _clear_plan(self, session_id: str) -> None:
        """Clear the plan by sending an empty plan update.

        Args:
            session_id: The session ID
        """
        await self._send(session_id, _plan_update([]))
        # Clear the stored plan for this session
        self._peek(session_id).plan = []

    async def _send_all(self, session_id: str, updates: Iterable[Any]) -> None:
        """Send session updates one at a time, taking each only once the last is sent."""
        for update in updates:
            await self._send(session_id, update)

    async def _process_tool_call_chunks(
        self,
        session_id: str,
        message_chunk: Any,
        turn: _Turn,
        namespace: tuple[str, ...] = (),
    ) -> None:
        """Start streamed tool calls as soon as their arguments parse."""
        started = _start_streamed_tool_calls(turn.chunks, turn.calls, message_chunk, namespace)
        await self._send_all(session_id, started)

    async def _fail_open_tool_calls(self, session_id: str, turn: _Turn) -> None:
        """Close every tool call left open by a cancelled or failed turn."""
        for update in _fail_open_calls(turn.calls):
            with contextlib.suppress(Exception):
                await self._send(session_id, update)

    def _default_context(self, cwd: str) -> AgentSessionContext:
        """Build the context a new session starts with before any option changes."""
        return AgentSessionContext(
            cwd=cwd,
            mode=self._modes.current_mode_id if self._modes is not None else "auto",
            model=self.default_model(),
        )

    def default_model(self) -> str | None:
        """Return the model new sessions start with: the first `models` entry.

        Override when the catalog has no usable default yet, so `check_ready`
        sees `model=None` and loaded sessions can still restore a saved model.
        """
        return self._models[0]["value"] if self._models else None

    def _initialize_session_options(self, session: _Session) -> None:
        """Initialize mode and model state for a new or loaded session."""
        if self._modes is not None:
            session.mode = self._modes.current_mode_id
            session.mode_state = self._modes
        if (default := self.default_model()) is not None:
            session.model = default
        session.options = {}

    def _restore_session_options(self, session: _Session, metadata: Mapping[str, Any]) -> None:
        """Restore persisted mode and model selections."""
        saved_mode = metadata.get(_ACP_MODE_METADATA_KEY)
        if self._modes is not None and isinstance(saved_mode, str):
            offered = [mode.id for mode in self._modes.available_modes]
            # A mode no longer offered (YOLO without acknowledgement) falls back to
            # the first mode, the most restrictive by convention, not the launch default.
            saved_mode = saved_mode if saved_mode in offered else offered[0]
            session.mode = saved_mode
            session.mode_state = SessionModeState(
                available_modes=self._modes.available_modes,
                current_mode_id=saved_mode,
            )

        saved_model = metadata.get(_ACP_MODEL_METADATA_KEY)
        if (
            self._models
            and isinstance(saved_model, str)
            and any(model["value"] == saved_model for model in self._models)
        ):
            session.model = saved_model
        saved_title = metadata.get(_ACP_TITLE_METADATA_KEY)
        if isinstance(saved_title, str) and saved_title:
            session.title = saved_title
        try:
            saved_options = json.loads(metadata.get(_ACP_OPTIONS_METADATA_KEY) or "{}")
        except (TypeError, ValueError):
            saved_options = None
        if isinstance(saved_options, dict):
            session.options = {
                key: value
                for key, value in saved_options.items()
                if isinstance(key, str) and isinstance(value, str)
            }

    def _session_config(self, session_id: str) -> RunnableConfig:
        """Build the LangGraph config and durable metadata for an ACP session."""
        session = self._sessions[session_id]
        metadata = {_ACP_SESSION_METADATA_KEY: True, "cwd": session.cwd}
        if session.mode is not None:
            metadata[_ACP_MODE_METADATA_KEY] = session.mode
        if session.model is not None:
            metadata[_ACP_MODEL_METADATA_KEY] = session.model
        if session.title:
            metadata[_ACP_TITLE_METADATA_KEY] = session.title
        if session.options:
            # Checkpoint metadata keeps only primitive values, so store JSON.
            metadata[_ACP_OPTIONS_METADATA_KEY] = json.dumps(session.options)
        return {"configurable": {"thread_id": session_id}, "metadata": metadata}

    async def _checkpointed_agent(self, session_id: str) -> CompiledStateGraph:
        """Return the session agent, requiring a configured checkpointer."""
        agent = await self._agent_for(session_id)
        if getattr(agent, "checkpointer", None) is None:
            msg = "Durable sessions require an agent compiled with a checkpointer"
            raise RuntimeError(msg)
        return agent

    async def _session_metadata(self, session_id: str) -> Mapping[str, Any]:
        """Read a session's saved ACP metadata.

        With a shared `checkpointer`, this reads the checkpoint directly, so no
        agent is built for options that may not be usable yet.
        """
        if self._checkpointer is not None:
            saved = await self._checkpointer.aget_tuple(
                {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
            )
            return saved.metadata if saved is not None else {}
        agent = await self._checkpointed_agent(session_id)
        return (await agent.aget_state(self._session_config(session_id))).metadata or {}

    async def _persist_options(self, session_id: str) -> None:
        """Save option changes now, or when the session's running turn ends.

        Writing a checkpoint while a turn waits on an interrupt would move the
        thread past the interrupt that turn is about to resume.
        """
        if not self._load_sessions:
            return
        session = self._peek(session_id)
        if session.turn is not None or session.closing:
            session.unsaved_options = True
        else:
            await self._persist_session(session_id)

    async def _persist_session(self, session_id: str) -> None:
        """Write the current ACP session metadata to its checkpoint thread.

        Any agent the session built can write it, so the last one is reused
        rather than building a new one for options that just changed.
        """
        agent = self._peek(session_id).last_agent or await self._checkpointed_agent(session_id)
        await agent.aupdate_state(self._session_config(session_id), {}, as_node="__start__")

    async def _replay_session(
        self,
        session_id: str,
        agent: CompiledStateGraph,
    ) -> None:
        """Replay persisted conversation entries before `session/load` returns."""
        history = agent.aget_state_history(self._session_config(session_id))
        await self._send_all(session_id, _replay_updates(await _history_messages(history)))

    def _forget_session(self, session_id: str) -> None:
        """Discard everything the server holds for a session."""
        self._sessions.pop(session_id, None)

    def _session_context(self, session_id: str) -> AgentSessionContext:
        """Return the context a session's agent is built from."""
        session = self._peek(session_id)
        default_mode = self._modes.current_mode_id if self._modes is not None else "auto"
        return AgentSessionContext(
            cwd=session.cwd,
            mode=default_mode if session.mode is None else session.mode,
            model=session.model if self._models is not None else None,
            options=tuple(sorted(session.options.items())),
            session_id=session_id,
            mcp_servers=tuple(session.mcp_servers),
        )

    async def _agent_for(self, session_id: str) -> CompiledStateGraph:
        """Return the session's cached agent, building it on first use."""
        agent = self._peek(session_id).agent
        return agent if agent is not None else await self._reset_agent(session_id)

    async def _reset_agent(self, session_id: str) -> CompiledStateGraph:
        """Build the session's agent from its current options and cache it.

        A build that an option change overtook still serves the caller that
        asked for it, but is not cached, so the next use builds the new choice.
        """
        session = self._sessions[session_id]
        context = self._session_context(session_id)
        agent = await self._build_agent(context)
        session.last_agent = agent
        if self._session_context(session_id) == context:
            session.agent = agent
        return agent

    async def _build_agent(self, context: AgentSessionContext) -> CompiledStateGraph:
        """Run the agent factory with the session's context."""
        if isinstance(self._agent_factory, CompiledStateGraph):
            return self._agent_factory
        built = self._agent_factory(context)
        # A narrowed `Awaitable[CompiledStateGraph]` loses its type argument under ty.
        return cast("CompiledStateGraph", await built if isinstance(built, Awaitable) else built)

    async def prompt(  # ty: ignore[invalid-method-override]  # keeps public positional order; ACP calls by keyword
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        message_id: str | None = None,  # noqa: ARG002  # ACP protocol interface parameter
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> PromptResponse:
        """Run one turn and stream it to the client.

        The turn runs as its own task so `session/cancel` can stop it at any
        await, including inside a long tool call.
        """
        session = self._require_open(session_id)
        if session.turn is not None:
            raise RequestError.invalid_request(
                {"message": "A prompt is already running in this session"}
            )
        turn = session.turn = asyncio.create_task(self._run_turn(prompt, session_id))
        try:
            return await turn
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if turn.cancelled() and (current is None or not current.cancelling()):
                return PromptResponse(stop_reason="cancelled")
            raise

    async def _run_turn(self, prompt: Sequence[Any], session_id: str) -> PromptResponse:
        """Run one turn, then save option changes it held back.

        The save runs inside the turn's task, so a close that waits for the
        turn also waits for it.
        """
        try:
            return await self._stream_turn(prompt, session_id)
        finally:
            session = self._peek(session_id)
            session.turn = None
            # A closing session saves its own options before it is forgotten.
            if session.unsaved_options and not session.closing:
                await self._save_held_options(session_id)

    async def _save_held_options(self, session_id: str) -> None:
        """Save option changes held back during a turn; a failure keeps them for the next try."""
        try:
            await self._persist_session(session_id)
        except Exception:  # noqa: BLE001  # the turn's own outcome must not be replaced
            logger.warning("Could not save options for session %s", session_id, exc_info=True)
        else:
            self._peek(session_id).unsaved_options = False

    async def _stream_turn(self, prompt: Sequence[Any], session_id: str) -> PromptResponse:
        """Stream the agent until it finishes, resolving interrupts along the way."""
        agent = await self._agent_for(session_id)
        if getattr(agent, "checkpointer", None) is None:
            agent.checkpointer = MemorySaver()  # Guarded by getattr check above
        context = self._session_context(session_id)
        turn = _Turn(cwd=context.cwd, context=context)
        config = self._session_config(session_id)
        graph_input: Any = await self._command_input(session_id, prompt) or {
            "messages": [{"role": "user", "content": _prompt_content(prompt, turn.cwd)}]
        }
        try:
            while True:
                interrupts = await self._stream(session_id, agent, graph_input, config, turn)
                if not interrupts:
                    # The checkpoint backing an interrupt may only become visible after
                    # the stream closes, so look again before ending the turn.
                    state = await agent.aget_state(config)
                    interrupts = [
                        interrupt
                        for interrupt in getattr(state, "interrupts", ())
                        if not any(
                            interrupt.id == done.id and interrupt.value == done.value
                            for done in turn.resumed
                        )
                    ]
                if not interrupts:
                    await self._after_turn(session_id, turn, state)
                    return PromptResponse(stop_reason=turn.stop_reason)
                resume = {
                    interrupt.id: await self._resolve_interrupt(session_id, interrupt, turn)
                    for interrupt in interrupts
                }
                turn.resumed = interrupts
                graph_input = Command(resume=resume)
        except ModelRefusalError:
            # Deep Agents raises on a refusal block; its text has already streamed.
            await self._fail_open_tool_calls(session_id, turn)
            return PromptResponse(stop_reason="refusal")
        except BaseException:
            await self._fail_open_tool_calls(session_id, turn)
            raise

    async def _stream(
        self,
        session_id: str,
        agent: CompiledStateGraph,
        graph_input: Any,  # graph input or resume `Command`
        config: RunnableConfig,
        turn: _Turn,
    ) -> list[Interrupt]:
        """Stream one graph run to the client and return the interrupts it raised."""
        interrupts: dict[str, Interrupt] = {}
        async for chunk in agent.astream(
            graph_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            if not isinstance(chunk, tuple) or len(chunk) != 3:  # noqa: PLR2004  # (namespace, mode, data)
                continue
            namespace, stream_mode, data = chunk
            if stream_mode == "updates":
                await self._handle_updates(session_id, turn, tuple(namespace), data, interrupts)
            else:
                await self._handle_message(session_id, turn, tuple(namespace), data)
        return list(interrupts.values())

    async def _handle_updates(
        self,
        session_id: str,
        turn: _Turn,
        namespace: tuple[str, ...],
        updates: Any,  # LangGraph node updates
        interrupts: dict[str, Interrupt],
    ) -> None:
        """Collect interrupts, plans, and the tool calls of finished model messages."""
        if not isinstance(updates, dict):
            return
        for interrupt in updates.get("__interrupt__") or ():
            interrupts.setdefault(interrupt.id, interrupt)
            turn.interrupt_scopes.setdefault(interrupt.id, namespace[:1])
        for node_name, update in updates.items():
            if node_name == "__interrupt__" or not isinstance(update, dict):
                continue
            if node_name == "tools" and update.get("todos"):
                await self._send(session_id, _plan_update(update["todos"]))
            messages = update.get("messages") if node_name == "model" else None
            for message in messages if isinstance(messages, list) else ():
                if getattr(message, "type", None) != "ai":
                    continue
                started = _start_message_tool_calls(turn.calls, message, namespace[:1])
                await self._send_all(session_id, started)
                if not namespace:
                    turn.stop_reason = _stop_reason(message) or "end_turn"
                    usage = getattr(message, "usage_metadata", None)
                    if usage:
                        turn.context_tokens = usage["input_tokens"] + usage["output_tokens"]

    async def _handle_message(
        self,
        session_id: str,
        turn: _Turn,
        namespace: tuple[str, ...],
        data: Any,  # (message chunk, metadata) from the messages stream
    ) -> None:
        """Forward a streamed message chunk or tool result to the client."""
        message_chunk, metadata = data
        source = metadata.get("lc_source") if isinstance(metadata, dict) else None
        # Metadata is caller-supplied, so `lc_source` can be unhashable; only a string matches.
        if isinstance(source, str) and source in self.internal_model_sources:
            return
        if isinstance(message_chunk, str):
            if not namespace:
                await self._log_text(text=message_chunk, session_id=session_id)
            return
        if getattr(message_chunk, "type", None) == "tool":
            if message_chunk.tool_call_id in turn.calls:
                await self._send_all(session_id, _finish_tool_call(turn.calls, message_chunk))
            return
        # Only the top-level graph's AI messages report text to the client. Messages a
        # node inserts (pinned skills, local context) are not the agent speaking, and
        # subagent text and reasoning stay internal; their tool calls are still shown.
        if not namespace and getattr(message_chunk, "type", None) in {"ai", "AIMessageChunk"}:
            await self._send_all(session_id, _content_updates(message_chunk))
        # Emitted after the content above so a chunk that carries both keeps
        # the block order that session replay uses.
        await self._process_tool_call_chunks(session_id, message_chunk, turn, namespace)

    async def _resolve_interrupt(
        self, session_id: str, interrupt: Interrupt, turn: _Turn
    ) -> Any:  # resume value for this interrupt
        """Return the resume value for one interrupt."""
        value = interrupt.value
        if isinstance(value, dict) and "action_requests" in value:
            scope = turn.interrupt_scopes.get(interrupt.id)
            return {"decisions": await self._hitl_decisions(session_id, value, turn, scope)}
        return await self.handle_interrupt(session_id, interrupt)

    async def handle_interrupt(
        self, session_id: str, interrupt: Interrupt
    ) -> Any:  # resume value for this interrupt
        """Answer an interrupt that is not a human-in-the-loop permission request.

        Override to support custom interrupts, for example by asking the user
        through `elicitation/create`. The return value resumes the interrupt.

        Raises:
            RequestError: By default, since ACP has no generic way to show it.
        """
        del session_id
        del interrupt  # its value can carry private data a tool put in it
        raise RequestError(-32600, _UNSUPPORTED_INTERRUPT)

    def _execute_permission_key(
        self, session_id: str, tool_args: object, turn: _Turn
    ) -> _ExecuteKey | None:
        """Identify an exact execute request in the options it runs under.

        Uses the turn's options, so an "always" answer is keyed to the mode and
        model that actually ran the command.
        """
        arguments = _execute_arguments(tool_args)
        if session_id not in self._sessions or arguments is None:
            return None
        context = turn.context
        key = AgentSessionContext(
            cwd=turn.cwd, mode=context.mode, model=context.model, options=context.options
        )
        return key, arguments

    async def _after_turn(
        self, session_id: str, turn: _Turn, state: Any
    ) -> None:  # LangGraph StateSnapshot
        """Report context usage and cost, and name the session after its first turn."""
        values = getattr(state, "values", None) or {}
        context = turn.context
        size = self.context_window(context)
        if size is not None and turn.context_tokens is not None:
            cost = self.session_cost(values)
            await self._send(
                session_id,
                UsageUpdate(
                    session_update="usage_update",
                    used=turn.context_tokens,
                    size=size,
                    cost=Cost(amount=cost, currency="USD") if cost is not None else None,
                ),
            )
        session = self._sessions.get(session_id)
        if session is not None and session.title is None:
            session.title = ""
            messages = values.get("messages") or []
            self._spawn(session_id, self._name_session(session_id, context, messages))

    def _spawn(self, session_id: str, work: Coroutine[Any, Any, None]) -> None:
        """Run session work after the current request; closing the session cancels it."""
        task = asyncio.create_task(work)
        tasks = self._sessions[session_id].tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _announce_commands(self, session_id: str) -> None:
        """Send the session's slash commands once its `session/new` or `load` reply is out."""
        # Yield first so the response that creates the session reaches the client first.
        await asyncio.sleep(_ANNOUNCE_DELAY_SECONDS)
        commands = self.available_commands(self._session_context(session_id))
        if not commands or session_id not in self._sessions:
            return
        with contextlib.suppress(ConnectionError):
            await self._send(
                session_id,
                AvailableCommandsUpdate(
                    session_update="available_commands_update", available_commands=commands
                ),
            )

    def available_commands(
        self,
        context: AgentSessionContext,  # noqa: ARG002  # used by overrides
    ) -> list[AvailableCommand]:
        """Return the slash commands clients list for the session."""
        return []

    async def command_input(
        self,
        context: AgentSessionContext,  # noqa: ARG002  # used by overrides
        name: str,  # noqa: ARG002  # used by overrides
        args: str,  # noqa: ARG002  # used by overrides
    ) -> dict[str, Any] | None:
        """Return the graph input that runs `/name args`, or `None` to send it as text.

        The input replaces the usual `{"messages": [user message]}` for the turn,
        so it can carry extra state keys alongside the message.
        """
        return None

    async def _command_input(self, session_id: str, prompt: Sequence[Any]) -> dict[str, Any] | None:
        """Turn a prompt that is exactly one advertised `/command` into its graph input."""
        if len(prompt) != 1 or not isinstance(prompt[0], TextContentBlock):
            return None
        text = prompt[0].text.strip()
        if not text.startswith("/"):
            return None
        parts = text[1:].split(maxsplit=1)
        if not parts:
            return None
        name, args = parts[0], parts[1] if len(parts) > 1 else ""
        context = self._session_context(session_id)
        if name not in {command.name for command in self.available_commands(context)}:
            return None
        return await self.command_input(context, name, args)

    async def _name_session(
        self, session_id: str, context: AgentSessionContext, messages: Sequence[Any]
    ) -> None:
        """Generate a title in the background and tell the client."""
        try:
            title = await self.session_title(context, messages)
        except Exception:  # noqa: BLE001  # a title is cosmetic; any override failure must not surface
            logger.warning("Session title generation failed", exc_info=True)
            title = None
        session = self._sessions.get(session_id)
        if not title or session is None:
            return
        session.title = title
        await self._persist_options(session_id)
        update = SessionInfoUpdate(
            session_update="session_info_update",
            title=title,
            updated_at=datetime.now(UTC).isoformat(),
        )
        # The client may have disconnected while the title was generated.
        with contextlib.suppress(ConnectionError):
            await self._send(session_id, update)

    def tool_call_display(
        self,
        context: AgentSessionContext,  # noqa: ARG002  # used by overrides
        *,
        name: str,  # noqa: ARG002  # used by overrides
        args: Mapping[str, Any],  # noqa: ARG002  # used by overrides
    ) -> ToolCallDisplay | None:
        """Return how to show a call to the tool `name`, or `None` for the default.

        Override to give a tool with an unreadable name a title, or to attach
        ACP `_meta` that tells the client more about the call, such as where
        the tool comes from. The result applies to the call, its permission
        request, and its replay when the session is loaded, so it must not
        depend on state a reloaded session lacks.
        """
        return None

    def _displayed(self, session_id: str, name: str, update: _ShownToolCall) -> _ShownToolCall:
        """Apply `tool_call_display` to a tool call the client is about to be shown."""
        args = update.raw_input if isinstance(update.raw_input, dict) else {}
        display = self.tool_call_display(self._session_context(session_id), name=name, args=args)
        if display is None:
            return update
        meta = update.field_meta if display.meta is None else dict(display.meta)
        return update.model_copy(
            update={"title": display.title or update.title, "field_meta": meta}
        )

    def context_window(self, context: AgentSessionContext) -> int | None:  # noqa: ARG002  # used by overrides
        """Return the session model's context window in tokens, if known.

        Without it the server sends no `usage_update`, since `size` is required.
        """
        return None

    def session_cost(self, values: Mapping[str, Any]) -> float | None:  # noqa: ARG002  # used by overrides
        """Return the session's cumulative cost in USD from its graph state, if known."""
        return None

    async def session_title(
        self,
        context: AgentSessionContext,  # noqa: ARG002  # used by overrides
        messages: Sequence[Any],
    ) -> str | None:
        """Return a title for a session after its first turn.

        Defaults to the first line of the first user message. Override to
        generate a better one; it runs in the background.
        """
        for message in messages:
            if getattr(message, "type", None) == "human":
                text = " ".join(
                    block["text"] for block in message.content_blocks if block.get("type") == "text"
                )
                first_line = text.strip().splitlines()[0] if text.strip() else ""
                return first_line[:80] or None
        return None

    def _remember_decision(
        self,
        session_id: str,
        tool_name: str,
        execute_key: _ExecuteKey | None,
        *,
        allow: bool,
    ) -> None:
        """Record an "always" answer for the rest of this opening of the session."""
        self._peek(session_id).permissions.remember(tool_name, execute_key, allow=allow)

    async def _hitl_decisions(
        self,
        session_id: str,
        interrupt_value: Mapping[str, Any],
        turn: _Turn,
        scope: tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        """Ask the client about each action in a human-in-the-loop interrupt."""
        return [
            await self._hitl_decision(session_id, action, turn, scope)
            for action in interrupt_value.get("action_requests", [])
        ]

    async def _hitl_decision(
        self,
        session_id: str,
        action: Mapping[str, Any],
        turn: _Turn,
        scope: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        """Decide one action from an earlier answer, or by asking the client."""
        tool_name = action.get("name", "tool")
        tool_args = action.get("args", {})
        if tool_name == "write_todos" and self._continue_plan(session_id, tool_args):
            return {"type": "approve"}
        execute_key = (
            self._execute_permission_key(session_id, tool_args, turn)
            if tool_name == "execute"
            else None
        )
        remembered = self._peek(session_id).permissions.remembered(tool_name, execute_key)
        if remembered is not None:
            return remembered
        choice = await self._ask_permission(session_id, action, execute_key, turn, scope)
        if choice in {"approve_always", "reject_always"}:
            allow = choice == "approve_always"
            self._remember_decision(session_id, tool_name, execute_key, allow=allow)
            choice = "approve" if allow else "reject"
        if tool_name == "write_todos":
            return await self._plan_decision(session_id, tool_args, approved=choice == "approve")
        return {"type": choice}

    def _continue_plan(
        self,
        session_id: str,
        tool_args: Any,  # whatever the middleware put in the action
    ) -> bool:
        """Accept an update to a plan the user already approved and that is in progress."""
        session = self._peek(session_id)
        if (
            not isinstance(tool_args, dict)
            or session.plan is None
            or _all_tasks_completed(session.plan)
        ):
            return False
        session.plan = tool_args.get("todos", [])
        return True

    async def _ask_permission(
        self,
        session_id: str,
        action: Mapping[str, Any],
        execute_key: _ExecuteKey | None,
        turn: _Turn,
        scope: tuple[str, ...] | None,
    ) -> str:
        """Show the client one action and return the option it picked, `reject` if none."""
        tool_name = action.get("name", "tool")
        tool_args = action.get("args", {})
        if tool_name == "write_todos":
            await self._log_text(session_id=session_id, text=_plan_text(tool_args.get("todos", [])))
        options = _permission_options(tool_name, execute_key)
        tool_call_id = _permission_tool_call_id(turn.calls, action, scope)
        response = await self._conn.request_permission(
            session_id=session_id,
            tool_call=self._displayed(
                session_id, tool_name, _permission_request(tool_call_id, tool_name, tool_args)
            ),
            options=options,
        )
        outcome = response.outcome
        offered = {option.option_id for option in options}
        if outcome.outcome == "selected" and outcome.option_id in offered:
            return outcome.option_id
        return "reject"

    async def _plan_decision(
        self,
        session_id: str,
        tool_args: Any,  # `write_todos` arguments
        *,
        approved: bool,
    ) -> dict[str, Any]:
        """Keep an approved plan, or clear a rejected one and ask the agent for another."""
        if approved:
            self._peek(session_id).plan = tool_args.get("todos", [])
            return {"type": "approve"}
        await self._clear_plan(session_id)
        return {"type": "reject", "message": _PLAN_REJECTED}


async def _serve_test_agent() -> None:
    """Run test agent from the root of the repository with ACP integration."""
    from dotenv import load_dotenv  # noqa: PLC0415  # lazy import for dev-only entry point

    load_dotenv()

    checkpointer: Checkpointer = MemorySaver()

    def build_agent(context: AgentSessionContext) -> CompiledStateGraph:
        """Agent factory based in the given root directory."""
        agent_root_dir = context.cwd

        ephemeral_backend = StateBackend()
        backend = CompositeBackend(
            default=FilesystemBackend(root_dir=agent_root_dir, virtual_mode=True),
            routes={
                "/memories/": ephemeral_backend,
                "/conversation_history/": ephemeral_backend,
            },
        )

        return create_deep_agent(
            model="openai:gpt-5.2",
            checkpointer=checkpointer,
            backend=backend,
        )

    acp_agent = AgentServerACP(agent=build_agent)
    await run_acp_agent(acp_agent)
