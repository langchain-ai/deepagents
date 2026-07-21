"""Domain models for hook lifecycle invocations and decisions."""

from __future__ import annotations

from enum import StrEnum
from pathlib import (
    Path,  # ruff:ignore[typing-only-standard-library-import] - Pydantic resolves model annotations at runtime.
)
from typing import Annotated, Literal, TypeAlias
from uuid import (
    UUID,  # ruff:ignore[typing-only-standard-library-import] - Pydantic resolves model annotations at runtime.
)

from langchain_core.messages import (  # ruff:ignore[typing-only-third-party-import] - Pydantic runtime annotation.
    ToolMessage,
)
from langgraph.types import (
    Command,  # ruff:ignore[typing-only-third-party-import] - Pydantic runtime annotation.
)
from pydantic import BaseModel, ConfigDict, Field

from deepagents_code.approval_mode import (  # ruff:ignore[typing-only-first-party-import] - Pydantic runtime annotation.
    ApprovalMode,
)
from deepagents_code.json_types import (  # ruff:ignore[typing-only-first-party-import] - Pydantic runtime annotation.
    JsonObject,
)


class _DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class HookEvent(StrEnum):
    """Supported hook lifecycle events."""

    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    PERMISSION_REQUEST = "PermissionRequest"
    NOTIFICATION = "Notification"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"


class HookOwner(StrEnum):
    """Process responsible for originating an event."""

    CLIENT = "client"
    SERVER = "server"


class SessionStartCause(StrEnum):
    """Reason a session-start event occurred."""

    STARTUP = "startup"
    RESUME = "resume"
    CLEAR = "clear"
    COMPACT = "compact"


class SessionEndCause(StrEnum):
    """Reason a session-end event occurred."""

    CLEAR = "clear"
    RESUME = "resume"
    LOGOUT = "logout"
    PROMPT_INPUT_EXIT = "prompt_input_exit"
    BYPASS_PERMISSIONS_DISABLED = "bypass_permissions_disabled"
    OTHER = "other"


EffortLevel: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]


class ToolCallData(_DomainModel):
    """Native tool-call data used by hook lifecycle owners."""

    id: str
    name: str
    args: JsonObject


class AgentIdentity(_DomainModel):
    """Resolved subagent identity."""

    id: str
    name: str


class DcodeNotification(_DomainModel):
    """A notification emitted by a dcode lifecycle owner."""

    type: str
    message: str
    title: str | None = None


class BackgroundTaskSnapshot(_DomainModel):
    """Background task state captured for a hook invocation."""

    id: str
    type: str
    status: str
    description: str
    command: str | None = None
    agent_type: str | None = None
    server: str | None = None
    tool: str | None = None
    name: str | None = None


class SessionCronSnapshot(_DomainModel):
    """Scheduled session prompt captured for a hook invocation."""

    id: str
    schedule: str
    recurring: bool
    prompt: str


class HookContext(_DomainModel):
    """Context shared by every domain hook event."""

    thread_id: str
    cwd: Path
    prompt_id: UUID | None = None
    approval_mode: ApprovalMode
    effort: EffortLevel | None = None
    agent: AgentIdentity | None = None
    transcript_revision: str | None = None


class SessionStartEvent(_DomainModel):
    """Domain payload for `SessionStart`."""

    event: Literal[HookEvent.SESSION_START]
    cause: SessionStartCause
    model: str | None = None


class SessionEndEvent(_DomainModel):
    """Domain payload for `SessionEnd`."""

    event: Literal[HookEvent.SESSION_END]
    cause: SessionEndCause


class PermissionRequestEvent(_DomainModel):
    """Domain payload for `PermissionRequest`."""

    event: Literal[HookEvent.PERMISSION_REQUEST]
    call: ToolCallData


class NotificationEvent(_DomainModel):
    """Domain payload for `Notification`."""

    event: Literal[HookEvent.NOTIFICATION]
    notification: DcodeNotification


class PreToolUseEvent(_DomainModel):
    """Domain payload for `PreToolUse`."""

    event: Literal[HookEvent.PRE_TOOL_USE]
    call: ToolCallData


class PostToolUseEvent(_DomainModel):
    """Domain payload for `PostToolUse`."""

    event: Literal[HookEvent.POST_TOOL_USE]
    call: ToolCallData
    result: ToolMessage | Command[str]
    duration_ms: int | None = None


class StopEvent(_DomainModel):
    """Domain payload for `Stop`."""

    event: Literal[HookEvent.STOP]
    continuation_count: int
    last_assistant_message: str
    background_tasks: list[BackgroundTaskSnapshot] = Field(default_factory=list)
    session_crons: list[SessionCronSnapshot] = Field(default_factory=list)


class SubagentStartEvent(_DomainModel):
    """Domain payload for `SubagentStart`."""

    event: Literal[HookEvent.SUBAGENT_START]
    agent: AgentIdentity


class SubagentStopEvent(_DomainModel):
    """Domain payload for `SubagentStop`."""

    event: Literal[HookEvent.SUBAGENT_STOP]
    agent: AgentIdentity
    continuation_count: int
    last_assistant_message: str
    transcript_revision: str | None = None
    background_tasks: list[BackgroundTaskSnapshot] = Field(default_factory=list)
    session_crons: list[SessionCronSnapshot] = Field(default_factory=list)


HookDomainEvent: TypeAlias = Annotated[
    SessionStartEvent
    | SessionEndEvent
    | PermissionRequestEvent
    | NotificationEvent
    | PreToolUseEvent
    | PostToolUseEvent
    | StopEvent
    | SubagentStartEvent
    | SubagentStopEvent,
    Field(discriminator="event"),
]


class HookInvocation(_DomainModel):
    """A domain hook event with its invocation context."""

    context: HookContext
    event: HookDomainEvent


class HookDiagnostic(_DomainModel):
    """Structured diagnostic produced while processing a hook."""

    code: str
    severity: Literal["debug", "warning", "error"]
    message: str
    handler_id: str | None = None
    field: str | None = None


class PermissionEffect(_DomainModel):
    """Normalized permission result from hook processing."""

    behavior: Literal["allow", "deny", "ask", "none"]
    reason: str | None = None
    interrupt: bool = False


class BaseHookDecision(_DomainModel):
    """Fields common to every event-specific hook decision."""

    continue_processing: bool = True
    stop_reason: str | None = None
    user_notices: list[str] = Field(default_factory=list)
    terminal_sequences: list[str] = Field(default_factory=list)
    diagnostics: list[HookDiagnostic] = Field(default_factory=list)


class SessionStartDecision(BaseHookDecision):
    """Decision returned for `SessionStart`."""

    event: Literal[HookEvent.SESSION_START]
    context: list[str] = Field(default_factory=list)


class SessionEndDecision(BaseHookDecision):
    """Decision returned for `SessionEnd`."""

    event: Literal[HookEvent.SESSION_END]


class PermissionRequestDecision(BaseHookDecision):
    """Decision returned for `PermissionRequest`."""

    event: Literal[HookEvent.PERMISSION_REQUEST]
    permission: PermissionEffect


class NotificationDecision(BaseHookDecision):
    """Decision returned for `Notification`."""

    event: Literal[HookEvent.NOTIFICATION]


class PreToolUseDecision(BaseHookDecision):
    """Decision returned for `PreToolUse`."""

    event: Literal[HookEvent.PRE_TOOL_USE]
    permission: PermissionEffect
    context: list[str] = Field(default_factory=list)


class PostToolUseDecision(BaseHookDecision):
    """Decision returned for `PostToolUse`."""

    event: Literal[HookEvent.POST_TOOL_USE]
    feedback: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)


class StopDecision(BaseHookDecision):
    """Decision returned for `Stop`."""

    event: Literal[HookEvent.STOP]
    continue_loop: bool
    feedback: list[str] = Field(default_factory=list)


class SubagentStartDecision(BaseHookDecision):
    """Decision returned for `SubagentStart`."""

    event: Literal[HookEvent.SUBAGENT_START]
    context: list[str] = Field(default_factory=list)


class SubagentStopDecision(BaseHookDecision):
    """Decision returned for `SubagentStop`."""

    event: Literal[HookEvent.SUBAGENT_STOP]
    context: list[str] = Field(default_factory=list)


HookDecision: TypeAlias = Annotated[
    SessionStartDecision
    | SessionEndDecision
    | PermissionRequestDecision
    | NotificationDecision
    | PreToolUseDecision
    | PostToolUseDecision
    | StopDecision
    | SubagentStartDecision
    | SubagentStopDecision,
    Field(discriminator="event"),
]


class HookEffect(_DomainModel):
    """Normalized effect produced by one hook handler."""

    handler_id: str
    decision: HookDecision
