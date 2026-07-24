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
    # Domain objects are constructed by lifecycle code, not parsed from external
    # hook JSON. Keep unknown fields forbidden so typos fail loudly.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class HookEvent(StrEnum):
    """Supported hook lifecycle events."""

    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    SESSION_END = "SessionEnd"
    PERMISSION_REQUEST = "PermissionRequest"
    NOTIFICATION = "Notification"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
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


class DcodeNotificationKind(StrEnum):
    """dcode lifecycle notifications with compatible wire mappings."""

    PERMISSION_REQUIRED = "permission_required"
    AGENT_NEEDS_INPUT = "agent_needs_input"
    AGENT_COMPLETED = "agent_completed"


class CompactTrigger(StrEnum):
    """Reason context compaction was requested."""

    MANUAL = "manual"
    AUTO = "auto"


EffortLevel: TypeAlias = Literal["none", "low", "medium", "high", "xhigh", "max"]


class ToolCallData(_DomainModel):
    """Native tool-call data used by hook lifecycle owners."""

    id: str
    name: str
    args: JsonObject
    mcp_server: str | None = None


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
    """Background task state captured for a hook invocation.

    Compatible Stop/SubagentStop wire context. Omit or leave empty until a
    trustworthy background-task source exists.
    """

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
    """Scheduled session prompt captured for a hook invocation.

    Compatible Stop/SubagentStop wire context. Omit or leave empty until a
    trustworthy session-cron source exists.
    """

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


class UserPromptSubmitEvent(_DomainModel):
    """Domain payload for `UserPromptSubmit`."""

    event: Literal[HookEvent.USER_PROMPT_SUBMIT]
    prompt: str


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


class PreCompactEvent(_DomainModel):
    """Domain payload for `PreCompact`."""

    event: Literal[HookEvent.PRE_COMPACT]
    trigger: CompactTrigger
    custom_instructions: str = ""


class StopEvent(_DomainModel):
    """Domain payload for `Stop`.

    `background_tasks` and `session_crons` are optional wire-compat fields;
    omit or leave empty until they can be sourced.
    """

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
    """Domain payload for `SubagentStop`.

    `background_tasks` and `session_crons` are optional wire-compat fields.
    Blocking or resumable SubagentStop effects are not applied yet.
    """

    event: Literal[HookEvent.SUBAGENT_STOP]
    agent: AgentIdentity
    continuation_count: int
    last_assistant_message: str
    transcript_revision: str | None = None
    background_tasks: list[BackgroundTaskSnapshot] = Field(default_factory=list)
    session_crons: list[SessionCronSnapshot] = Field(default_factory=list)


HookDomainEvent: TypeAlias = Annotated[
    SessionStartEvent
    | UserPromptSubmitEvent
    | SessionEndEvent
    | PermissionRequestEvent
    | NotificationEvent
    | PreToolUseEvent
    | PostToolUseEvent
    | PreCompactEvent
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


class UserPromptSubmitDecision(BaseHookDecision):
    """Decision returned for `UserPromptSubmit`."""

    event: Literal[HookEvent.USER_PROMPT_SUBMIT]
    context: list[str] = Field(default_factory=list)
    suppress_original_prompt: bool = False


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


class PreCompactDecision(BaseHookDecision):
    """Decision returned for `PreCompact`."""

    event: Literal[HookEvent.PRE_COMPACT]


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
    | UserPromptSubmitDecision
    | SessionEndDecision
    | PermissionRequestDecision
    | NotificationDecision
    | PreToolUseDecision
    | PostToolUseDecision
    | PreCompactDecision
    | StopDecision
    | SubagentStartDecision
    | SubagentStopDecision,
    Field(discriminator="event"),
]


class HookEffect(_DomainModel):
    """Normalized effect produced by one hook handler."""

    handler_id: str
    decision: HookDecision
