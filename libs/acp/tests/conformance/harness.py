"""Drive `AgentServerACP` the way an ACP client does and record what the client sees.

A test scripts the model, builds a real agent graph around it, and asserts on the
sequence of session updates:

    model = scripted_model(reply("", call("echo", {"text": "hi"})), reply("Done"))
    harness = Harness.start(deep_agent(model, tools=[echo]))
    session_id = await harness.new_session()
    await harness.prompt(session_id, "Echo hi")
    start, final = check_tool_call(harness.client.tool_call(session_id, "call-echo"), "completed")

Each failure mode raises its own exception type, so a strict xfail can name the
exact defect it tracks (`raises=WrongFinalStatusError`) and any other failure, such
as a script running out or an unexpected client request, still fails the test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    CreateElicitationResponse,
    DeniedOutcome,
    PermissionOption,
    PromptResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UserMessageChunk,
)
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.chat_model import GenericFakeChatModel

AgentSource = CompiledStateGraph | Callable[[AgentSessionContext], CompiledStateGraph]
PermissionReply = str | None
"""The option id the client selects, or `None` to cancel the request."""

_FINAL_STATUSES = frozenset({"completed", "failed"})


class ScriptExhaustedError(Exception):
    """The model was called more times than its script has steps."""


class UnexpectedRequestError(Exception):
    """The server sent a client request with no scripted reply queued."""


class LifecycleError(Exception):
    """A tool call broke `tool_call` (pending) -> `in_progress`* -> one final update."""


class ToolCallNotStartedError(LifecycleError):
    """The client never received a `tool_call` for the call."""


class ToolCallNotFinishedError(LifecycleError):
    """The call started but never got a `completed` or `failed` update."""


class WrongFinalStatusError(LifecycleError):
    """The call finished with a different status than expected."""


# --- Scripted model ---------------------------------------------------------


def call(name: str, args: dict[str, Any] | None = None, *, call_id: str | None = None) -> ToolCall:
    """Build a tool call for `reply`. The id defaults to `call-<name>`."""
    return ToolCall(name=name, args=args or {}, id=call_id or f"call-{name}", type="tool_call")


def reply(text: str = "", *calls: ToolCall) -> AIMessage:
    """Build one model turn: optional text followed by tool calls."""
    return AIMessage(content=text, tool_calls=list(calls))


class ScriptedChatModel(GenericFakeChatModel):
    """Plays one scripted step per model call: an `AIMessage` to stream or an exception to raise.

    Each tool call streams as a single chunk. Empty args stream as `""`, the way
    Anthropic streams a call to a tool that takes no arguments.
    """

    messages: Iterator[AIMessage | str | Exception]

    def _next_reply(self) -> AIMessage:
        step = next(self.messages, None)
        if step is None:
            msg = "the model script ran out of steps"
            raise ScriptExhaustedError(msg)
        if isinstance(step, Exception):
            raise step
        return AIMessage(content=step) if isinstance(step, str) else step

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_reply())])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in _stream_chunks(self._next_reply()):
            yield ChatGenerationChunk(message=chunk)


def scripted_model(*steps: AIMessage | str | Exception) -> ScriptedChatModel:
    """Build a model that plays `steps` in order, one per model call."""
    return ScriptedChatModel(messages=iter(steps))


def _stream_chunks(message: AIMessage) -> list[AIMessageChunk]:
    text = [{"content": message.content}] if message.content else []
    calls = [
        {
            "content": "",
            "tool_call_chunks": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": json.dumps(tool_call["args"]) if tool_call["args"] else "",
                    "index": index,
                    "type": "tool_call_chunk",
                }
            ],
        }
        for index, tool_call in enumerate(message.tool_calls)
    ]
    parts = text + calls or [{"content": ""}]
    last = len(parts) - 1
    return [
        AIMessageChunk(
            **part,
            chunk_position="last" if index == last else None,
            response_metadata=message.response_metadata if index == last else {},
            usage_metadata=message.usage_metadata if index == last else None,
        )
        for index, part in enumerate(parts)
    ]


@tool
def echo(text: str) -> str:
    """Return `text` unchanged."""
    return text


def deep_agent(
    model: ScriptedChatModel,
    *,
    tools: Sequence[BaseTool] = (),
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_on: dict[str, bool] | None = None,
) -> CompiledStateGraph:
    """Build a real Deep Agent around a scripted model."""
    return create_deep_agent(
        model=model,
        tools=list(tools),
        checkpointer=checkpointer or MemorySaver(),
        interrupt_on=interrupt_on,
    )


# --- Recording client -------------------------------------------------------


def summarize(update: Any) -> tuple[Any, ...]:
    """Reduce a session update to the fields conformance tests assert on."""
    if isinstance(update, AgentMessageChunk | AgentThoughtChunk | UserMessageChunk):
        text = update.content.text if isinstance(update.content, TextContentBlock) else None
        return (update.session_update, text)
    if isinstance(update, ToolCallStart | ToolCallProgress):
        return (update.session_update, update.tool_call_id, update.status)
    if isinstance(update, AgentPlanUpdate):
        return ("plan", tuple((entry.content, entry.status) for entry in update.entries))
    return (update.session_update,)


class RecordingClient(Client):
    """An in-process ACP client that records everything the server sends it.

    Permission and elicitation requests are answered from scripted queues. A
    request with nothing queued raises `UnexpectedRequestError`; queue a cancel
    response to cancel deliberately.
    """

    def __init__(
        self,
        *,
        permission_replies: Sequence[PermissionReply] = (),
        elicitation_replies: Sequence[CreateElicitationResponse] = (),
    ) -> None:
        self.updates: list[tuple[str, Any]] = []
        self.permission_requests: list[RequestPermissionRequest] = []
        self.elicitations: list[dict[str, Any]] = []
        self._permission_replies = list(permission_replies)
        self._elicitation_replies = list(elicitation_replies)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        self.permission_requests.append(
            RequestPermissionRequest(session_id=session_id, tool_call=tool_call, options=options)
        )
        if not self._permission_replies:
            msg = f"unexpected permission request for {tool_call.title!r}"
            raise UnexpectedRequestError(msg)
        option_id = self._permission_replies.pop(0)
        outcome = DeniedOutcome() if option_id is None else AllowedOutcome(option_id=option_id)
        return RequestPermissionResponse(outcome=outcome)

    async def create_elicitation(
        self, message: str, mode: Any, **kwargs: Any
    ) -> CreateElicitationResponse:
        self.elicitations.append({"message": message, "mode": mode, **kwargs})
        if not self._elicitation_replies:
            msg = f"unexpected elicitation request {message!r}"
            raise UnexpectedRequestError(msg)
        return self._elicitation_replies.pop(0)

    def session_updates(self, session_id: str) -> list[Any]:
        """Every update sent for `session_id`, in order."""
        return [update for sid, update in self.updates if sid == session_id]

    def trace(self, session_id: str) -> list[tuple[Any, ...]]:
        """The `summarize`d update sequence for `session_id`."""
        return [summarize(update) for update in self.session_updates(session_id)]

    def tool_call(self, session_id: str, tool_call_id: str) -> list[Any]:
        """The `tool_call` and `tool_call_update` updates for one tool call, in order."""
        return [
            update
            for update in self.session_updates(session_id)
            if isinstance(update, ToolCallStart | ToolCallProgress)
            and update.tool_call_id == tool_call_id
        ]


def check_tool_call(
    updates: Sequence[Any], final_status: str
) -> tuple[ToolCallStart, ToolCallProgress]:
    """Check one tool call's updates follow the lifecycle and end in `final_status`.

    Returns:
        The `tool_call` start and the final `tool_call_update`.

    Raises:
        ToolCallNotStartedError: No `tool_call` came first.
        ToolCallNotFinishedError: No `completed` or `failed` update arrived.
        WrongFinalStatusError: The final update has another status.
        LifecycleError: Any other shape, such as an update after the final one.
    """
    statuses = [(update.session_update, update.status) for update in updates]
    if not updates or not isinstance(updates[0], ToolCallStart):
        raise ToolCallNotStartedError(statuses)
    start, *rest = updates
    finals = [update for update in rest if update.status in _FINAL_STATUSES]
    if not finals:
        raise ToolCallNotFinishedError(statuses)
    progress_ok = all(
        isinstance(update, ToolCallProgress) and update.status in {None, "in_progress"}
        for update in rest[:-1]
    )
    final_ok = isinstance(rest[-1], ToolCallProgress) and finals == rest[-1:]
    if start.status != "pending" or not progress_ok or not final_ok:
        raise LifecycleError(statuses)
    if rest[-1].status != final_status:
        raise WrongFinalStatusError(statuses)
    return start, rest[-1]


# --- Server -----------------------------------------------------------------


@dataclass(frozen=True)
class Harness:
    """An `AgentServerACP` connected to a `RecordingClient`."""

    server: AgentServerACP
    client: RecordingClient

    @classmethod
    def start(
        cls,
        agent: AgentSource,
        *,
        client: RecordingClient | None = None,
        load_sessions: bool = False,
        server_class: type[AgentServerACP] = AgentServerACP,
        **server_options: Any,
    ) -> Harness:
        """Connect a server to a recording client; `server_options` go to the server."""
        server = server_class(agent=agent, load_sessions=load_sessions, **server_options)
        client = client or RecordingClient()
        server.on_connect(client)
        return cls(server=server, client=client)

    async def new_session(self, cwd: str = "/tmp") -> str:
        response = await self.server.new_session(cwd=cwd, mcp_servers=[])
        return response.session_id

    async def load_session(self, session_id: str, cwd: str = "/tmp") -> None:
        await self.server.load_session(cwd=cwd, session_id=session_id, mcp_servers=[])

    async def prompt(self, session_id: str, text: str) -> PromptResponse:
        return await self.server.prompt(
            [TextContentBlock(type="text", text=text)], session_id=session_id
        )

    async def settle(self) -> None:
        """Wait for background work, such as titles and command announcements."""
        for session in list(self.server._sessions.values()):
            await asyncio.gather(*session.tasks)

    async def cancel(self, session_id: str) -> None:
        await self.server.cancel(session_id=session_id)

    def trace(self, session_id: str) -> list[tuple[Any, ...]]:
        return self.client.trace(session_id)
