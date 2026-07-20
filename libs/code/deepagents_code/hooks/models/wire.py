"""External JSON-compatible hook input and output models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias
from uuid import UUID  # noqa: TC003 - Pydantic resolves model annotations at runtime.

from pydantic import BaseModel, ConfigDict, Field

from deepagents_code.hooks.models.domain import (  # noqa: TC001 - Pydantic runtime annotation.
    EffortLevel,
    HookEvent,
    SessionEndCause,
    SessionStartCause,
)
from deepagents_code.json_types import (  # noqa: TC001 - Pydantic runtime annotation.
    JsonObject,
    JsonValue,
)


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WirePermissionMode(StrEnum):
    """Permission modes exposed by the compatible wire protocol."""

    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "acceptEdits"
    AUTO = "auto"
    DONT_ASK = "dontAsk"
    BYPASS_PERMISSIONS = "bypassPermissions"


class WireNotificationType(StrEnum):
    """Notification matcher values exposed on the wire."""

    PERMISSION_PROMPT = "permission_prompt"
    IDLE_PROMPT = "idle_prompt"
    AUTH_SUCCESS = "auth_success"
    ELICITATION_DIALOG = "elicitation_dialog"
    ELICITATION_COMPLETE = "elicitation_complete"
    ELICITATION_RESPONSE = "elicitation_response"
    AGENT_NEEDS_INPUT = "agent_needs_input"
    AGENT_COMPLETED = "agent_completed"


class Effort(_WireModel):
    """Wire representation of model effort."""

    level: EffortLevel


class PermissionRule(_WireModel):
    """Permission rule returned by a hook."""

    tool_name: str = Field(alias="toolName")
    rule_content: str | None = Field(default=None, alias="ruleContent")


class PermissionDestination(StrEnum):
    """Configuration scope targeted by a permission update."""

    SESSION = "session"
    LOCAL = "localSettings"
    PROJECT = "projectSettings"
    USER = "userSettings"


class AddRulesUpdate(_WireModel):
    """Permission update that adds rules."""

    type: Literal["addRules"]
    rules: list[PermissionRule]
    behavior: Literal["allow", "deny", "ask"]
    destination: PermissionDestination


class ReplaceRulesUpdate(_WireModel):
    """Permission update that replaces rules."""

    type: Literal["replaceRules"]
    rules: list[PermissionRule]
    behavior: Literal["allow", "deny", "ask"]
    destination: PermissionDestination


class RemoveRulesUpdate(_WireModel):
    """Permission update that removes rules."""

    type: Literal["removeRules"]
    rules: list[PermissionRule]
    behavior: Literal["allow", "deny", "ask"]
    destination: PermissionDestination


class SetModeUpdate(_WireModel):
    """Permission update that changes the active mode."""

    type: Literal["setMode"]
    mode: WirePermissionMode | Literal["manual"]
    destination: PermissionDestination


class AddDirectoriesUpdate(_WireModel):
    """Permission update that adds allowed directories."""

    type: Literal["addDirectories"]
    directories: list[str]
    destination: PermissionDestination


class RemoveDirectoriesUpdate(_WireModel):
    """Permission update that removes allowed directories."""

    type: Literal["removeDirectories"]
    directories: list[str]
    destination: PermissionDestination


PermissionUpdate: TypeAlias = Annotated[
    AddRulesUpdate
    | ReplaceRulesUpdate
    | RemoveRulesUpdate
    | SetModeUpdate
    | AddDirectoriesUpdate
    | RemoveDirectoriesUpdate,
    Field(discriminator="type"),
]


class BackgroundTaskWire(_WireModel):
    """Background task snapshot exposed to hook handlers."""

    id: str
    type: str
    status: str
    description: str
    command: str | None = None
    agent_type: str | None = None
    server: str | None = None
    tool: str | None = None
    name: str | None = None


class SessionCronWire(_WireModel):
    """Scheduled session prompt exposed to hook handlers."""

    id: str
    schedule: str
    recurring: bool
    prompt: str


class BaseHookWireInput(_WireModel):
    """Fields common to every hook input."""

    session_id: str
    transcript_path: str
    cwd: str
    hook_event_name: HookEvent
    prompt_id: UUID | None = None
    permission_mode: WirePermissionMode | None = None
    effort: Effort | None = None
    agent_id: str | None = None
    agent_type: str | None = None


class SessionStartWireInput(BaseHookWireInput):
    """Wire input for `SessionStart`."""

    hook_event_name: Literal[HookEvent.SESSION_START]
    source: SessionStartCause
    model: str | None = None
    session_title: str | None = None


class SessionEndWireInput(BaseHookWireInput):
    """Wire input for `SessionEnd`."""

    hook_event_name: Literal[HookEvent.SESSION_END]
    reason: SessionEndCause


class PermissionRequestWireInput(BaseHookWireInput):
    """Wire input for `PermissionRequest`."""

    hook_event_name: Literal[HookEvent.PERMISSION_REQUEST]
    tool_name: str
    tool_input: JsonObject
    permission_suggestions: list[PermissionUpdate] = Field(default_factory=list)


class NotificationWireInput(BaseHookWireInput):
    """Wire input for `Notification`."""

    hook_event_name: Literal[HookEvent.NOTIFICATION]
    message: str
    notification_type: WireNotificationType
    title: str | None = None


class PreToolUseWireInput(BaseHookWireInput):
    """Wire input for `PreToolUse`."""

    hook_event_name: Literal[HookEvent.PRE_TOOL_USE]
    tool_name: str
    tool_input: JsonObject
    tool_use_id: str


class PostToolUseWireInput(BaseHookWireInput):
    """Wire input for `PostToolUse`."""

    hook_event_name: Literal[HookEvent.POST_TOOL_USE]
    tool_name: str
    tool_input: JsonObject
    tool_response: JsonValue
    tool_use_id: str
    duration_ms: int | None = None


class StopWireInput(BaseHookWireInput):
    """Wire input for `Stop`."""

    hook_event_name: Literal[HookEvent.STOP]
    stop_hook_active: bool
    last_assistant_message: str
    background_tasks: list[BackgroundTaskWire] = Field(default_factory=list)
    session_crons: list[SessionCronWire] = Field(default_factory=list)


class SubagentStartWireInput(BaseHookWireInput):
    """Wire input for `SubagentStart`."""

    hook_event_name: Literal[HookEvent.SUBAGENT_START]
    agent_id: str
    agent_type: str


class SubagentStopWireInput(BaseHookWireInput):
    """Wire input for `SubagentStop`."""

    hook_event_name: Literal[HookEvent.SUBAGENT_STOP]
    stop_hook_active: bool
    agent_id: str
    agent_type: str
    agent_transcript_path: str
    last_assistant_message: str
    background_tasks: list[BackgroundTaskWire] = Field(default_factory=list)
    session_crons: list[SessionCronWire] = Field(default_factory=list)


HookWireInput: TypeAlias = Annotated[
    SessionStartWireInput
    | SessionEndWireInput
    | PermissionRequestWireInput
    | NotificationWireInput
    | PreToolUseWireInput
    | PostToolUseWireInput
    | StopWireInput
    | SubagentStartWireInput
    | SubagentStopWireInput,
    Field(discriminator="hook_event_name"),
]


class SessionStartSpecificOutput(_WireModel):
    """Event-specific output for `SessionStart`."""

    hook_event_name: Literal["SessionStart"] = Field(alias="hookEventName")
    additional_context: str | None = Field(default=None, alias="additionalContext")
    initial_user_message: str | None = Field(default=None, alias="initialUserMessage")
    session_title: str | None = Field(default=None, alias="sessionTitle")
    watch_paths: list[str] = Field(default_factory=list, alias="watchPaths")
    reload_skills: bool = Field(default=False, alias="reloadSkills")


class PreToolUseSpecificOutput(_WireModel):
    """Event-specific output for `PreToolUse`."""

    hook_event_name: Literal["PreToolUse"] = Field(alias="hookEventName")
    permission_decision: Literal["allow", "deny", "ask", "defer"] | None = Field(
        default=None,
        alias="permissionDecision",
    )
    permission_decision_reason: str | None = Field(
        default=None,
        alias="permissionDecisionReason",
    )
    updated_input: JsonObject | None = Field(default=None, alias="updatedInput")
    additional_context: str | None = Field(default=None, alias="additionalContext")


class PermissionAllow(_WireModel):
    """Permission-request output that allows a tool call."""

    behavior: Literal["allow"]
    updated_input: JsonObject | None = Field(default=None, alias="updatedInput")
    updated_permissions: list[PermissionUpdate] = Field(
        default_factory=list,
        alias="updatedPermissions",
    )


class PermissionDeny(_WireModel):
    """Permission-request output that denies a tool call."""

    behavior: Literal["deny"]
    message: str | None = None
    interrupt: bool = False


class PermissionRequestSpecificOutput(_WireModel):
    """Event-specific output for `PermissionRequest`."""

    hook_event_name: Literal["PermissionRequest"] = Field(alias="hookEventName")
    decision: Annotated[
        PermissionAllow | PermissionDeny,
        Field(discriminator="behavior"),
    ]


class PostToolUseSpecificOutput(_WireModel):
    """Event-specific output for `PostToolUse`."""

    hook_event_name: Literal["PostToolUse"] = Field(alias="hookEventName")
    additional_context: str | None = Field(default=None, alias="additionalContext")
    updated_tool_output: JsonValue = Field(
        default=None,
        alias="updatedToolOutput",
    )
    updated_mcp_tool_output: JsonValue = Field(
        default=None,
        alias="updatedMCPToolOutput",
    )


class StopSpecificOutput(_WireModel):
    """Event-specific output for `Stop`."""

    hook_event_name: Literal["Stop"] = Field(alias="hookEventName")
    additional_context: str | None = Field(default=None, alias="additionalContext")


class SubagentStartSpecificOutput(_WireModel):
    """Event-specific output for `SubagentStart`."""

    hook_event_name: Literal["SubagentStart"] = Field(alias="hookEventName")
    additional_context: str | None = Field(default=None, alias="additionalContext")


class SubagentStopSpecificOutput(_WireModel):
    """Event-specific output for `SubagentStop`."""

    hook_event_name: Literal["SubagentStop"] = Field(alias="hookEventName")
    additional_context: str | None = Field(default=None, alias="additionalContext")


HookSpecificOutput: TypeAlias = Annotated[
    SessionStartSpecificOutput
    | PreToolUseSpecificOutput
    | PermissionRequestSpecificOutput
    | PostToolUseSpecificOutput
    | StopSpecificOutput
    | SubagentStartSpecificOutput
    | SubagentStopSpecificOutput,
    Field(discriminator="hook_event_name"),
]


class HookWireOutput(BaseModel):
    """Compatible hook output with retained extension fields."""

    model_config = ConfigDict(extra="allow")

    continue_: bool = Field(default=True, alias="continue")
    stop_reason: str | None = Field(default=None, alias="stopReason")
    suppress_output: bool = Field(default=False, alias="suppressOutput")
    system_message: str | None = Field(default=None, alias="systemMessage")
    terminal_sequence: str | None = Field(default=None, alias="terminalSequence")
    decision: Literal["block"] | None = None
    reason: str | None = None
    hook_specific_output: HookSpecificOutput | None = Field(
        default=None,
        alias="hookSpecificOutput",
    )
