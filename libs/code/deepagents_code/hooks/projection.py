"""Projection from Hooks v2 domain invocations to compatible wire input."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, NotRequired, TypedDict

from langchain_core.messages import ToolMessage

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.adapters import HOOK_WIRE_INPUT_ADAPTER
from deepagents_code.hooks.models.domain import (
    DcodeNotificationKind,
    HookEvent,
    NotificationEvent,
    PermissionRequestEvent,
    PostToolUseEvent,
    PreCompactEvent,
    PreToolUseEvent,
    SessionEndEvent,
    SessionStartEvent,
    StopEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    UserPromptSubmitEvent,
)
from deepagents_code.hooks.models.wire import (
    BackgroundTaskWire,
    Effort,
    NotificationWireInput,
    PermissionRequestWireInput,
    PostToolUseWireInput,
    PreCompactWireInput,
    PreToolUseWireInput,
    SessionCronWire,
    SessionEndWireInput,
    SessionStartWireInput,
    StopWireInput,
    SubagentStartWireInput,
    SubagentStopWireInput,
    UserPromptSubmitWireInput,
    WireNotificationType,
    WirePermissionMode,
)
from deepagents_code.hooks.tools import to_wire_call
from deepagents_code.json_types import JSON_VALUE_ADAPTER, JsonValue

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from langgraph.types import Command

    from deepagents_code.hooks.models.domain import (
        AgentIdentity,
        HookDomainEvent,
        HookInvocation,
    )
    from deepagents_code.hooks.models.wire import HookWireInput


class _BaseWireFields(TypedDict):
    session_id: str
    transcript_path: str
    cwd: str
    permission_mode: NotRequired[WirePermissionMode]
    prompt_id: NotRequired[UUID]
    effort: NotRequired[Effort]
    agent_id: NotRequired[str]
    agent_type: NotRequired[str]


def project_hook_input(
    invocation: HookInvocation,
    *,
    transcript_path: Path,
    agent_transcript_path: Path | None = None,
) -> HookWireInput:
    """Project a native hook invocation into the compatible wire contract.

    Args:
        invocation: Native lifecycle invocation.
        transcript_path: Materialized client transcript path.
        agent_transcript_path: Materialized subagent transcript path.

    Returns:
        A validated event-specific wire input.
    """
    result = _project_event(
        invocation.event,
        invocation,
        transcript_path,
        agent_transcript_path,
    )

    payload = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        result,
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    return HOOK_WIRE_INPUT_ADAPTER.validate_python(payload)


@singledispatch
def _project_event(
    event: HookDomainEvent,
    _invocation: HookInvocation,
    _transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    msg = f"Unsupported hook event: {type(event).__name__}"
    raise TypeError(msg)


@_project_event.register(SessionStartEvent)
def _project_session_start(
    event: SessionStartEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return SessionStartWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.SESSION_START,
        source=event.cause,
        model=event.model,
    )


@_project_event.register(UserPromptSubmitEvent)
def _project_user_prompt_submit(
    event: UserPromptSubmitEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return UserPromptSubmitWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.USER_PROMPT_SUBMIT,
        prompt=event.prompt,
    )


@_project_event.register(SessionEndEvent)
def _project_session_end(
    event: SessionEndEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return SessionEndWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.SESSION_END,
        reason=event.cause,
    )


@_project_event.register(PermissionRequestEvent)
def _project_permission_request(
    event: PermissionRequestEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    tool_name, tool_input = to_wire_call(event.call)
    return PermissionRequestWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.PERMISSION_REQUEST,
        tool_name=tool_name,
        tool_input=tool_input,
    )


@_project_event.register(NotificationEvent)
def _project_notification(
    event: NotificationEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return NotificationWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.NOTIFICATION,
        message=event.notification.message,
        title=event.notification.title,
        notification_type=_notification_type(event.notification.type),
    )


@_project_event.register(PreToolUseEvent)
def _project_pre_tool_use(
    event: PreToolUseEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    tool_name, tool_input = to_wire_call(event.call)
    return PreToolUseWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.PRE_TOOL_USE,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=event.call.id,
    )


@_project_event.register(PostToolUseEvent)
def _project_post_tool_use(
    event: PostToolUseEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    tool_name, tool_input = to_wire_call(event.call)
    return PostToolUseWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.POST_TOOL_USE,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_response=_tool_result(event.result),
        tool_use_id=event.call.id,
        duration_ms=event.duration_ms,
    )


@_project_event.register(PreCompactEvent)
def _project_pre_compact(
    event: PreCompactEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return PreCompactWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.PRE_COMPACT,
        trigger=event.trigger,
        custom_instructions=event.custom_instructions,
    )


@_project_event.register(StopEvent)
def _project_stop(
    event: StopEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return StopWireInput(
        **_base_fields(invocation, transcript_path),
        hook_event_name=HookEvent.STOP,
        stop_hook_active=event.continuation_count > 0,
        last_assistant_message=event.last_assistant_message,
        background_tasks=[
            BackgroundTaskWire.model_validate(task.model_dump())
            for task in event.background_tasks
        ],
        session_crons=[
            SessionCronWire.model_validate(cron.model_dump())
            for cron in event.session_crons
        ],
    )


@_project_event.register(SubagentStartEvent)
def _project_subagent_start(
    event: SubagentStartEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    _agent_transcript_path: Path | None,
) -> HookWireInput:
    return SubagentStartWireInput(
        **_base_fields(invocation, transcript_path, agent=event.agent),
        hook_event_name=HookEvent.SUBAGENT_START,
    )


@_project_event.register(SubagentStopEvent)
def _project_subagent_stop(
    event: SubagentStopEvent,
    invocation: HookInvocation,
    transcript_path: Path,
    agent_transcript_path: Path | None,
) -> HookWireInput:
    if agent_transcript_path is None:
        msg = "SubagentStop requires a materialized agent transcript path"
        raise ValueError(msg)
    return SubagentStopWireInput(
        **_base_fields(invocation, transcript_path, agent=event.agent),
        hook_event_name=HookEvent.SUBAGENT_STOP,
        stop_hook_active=event.continuation_count > 0,
        agent_transcript_path=str(agent_transcript_path),
        last_assistant_message=event.last_assistant_message,
        background_tasks=[
            BackgroundTaskWire.model_validate(task.model_dump())
            for task in event.background_tasks
        ],
        session_crons=[
            SessionCronWire.model_validate(cron.model_dump())
            for cron in event.session_crons
        ],
    )


def _base_fields(
    invocation: HookInvocation,
    transcript_path: Path,
    *,
    agent: AgentIdentity | None = None,
) -> _BaseWireFields:
    context = invocation.context
    fields: _BaseWireFields = {
        "session_id": context.thread_id,
        "transcript_path": str(transcript_path),
        "cwd": str(context.cwd),
    }
    fields["permission_mode"] = _permission_mode(context.approval_mode)
    if context.prompt_id is not None:
        fields["prompt_id"] = context.prompt_id
    if context.effort is not None:
        fields["effort"] = Effort(level=context.effort)
    identity = agent or context.agent
    if identity is not None:
        fields["agent_id"] = identity.id
        fields["agent_type"] = identity.name
    return fields


def serialize_hook_input(
    invocation: HookInvocation,
    *,
    transcript_path: Path,
    agent_transcript_path: Path | None = None,
) -> bytes:
    """Serialize a hook invocation as validated compatible JSON.

    Args:
        invocation: Native lifecycle invocation.
        transcript_path: Materialized client transcript path.
        agent_transcript_path: Materialized subagent transcript path.

    Returns:
        Compact JSON bytes suitable for command stdin.
    """
    return HOOK_WIRE_INPUT_ADAPTER.dump_json(
        project_hook_input(
            invocation,
            transcript_path=transcript_path,
            agent_transcript_path=agent_transcript_path,
        ),
        by_alias=True,
        exclude_none=True,
    )


def _permission_mode(mode: ApprovalMode) -> WirePermissionMode:
    return {
        ApprovalMode.MANUAL: WirePermissionMode.DEFAULT,
        ApprovalMode.AUTO: WirePermissionMode.AUTO,
        ApprovalMode.YOLO: WirePermissionMode.BYPASS_PERMISSIONS,
    }[mode]


def _notification_type(value: str) -> WireNotificationType:
    mappings: dict[str, WireNotificationType] = {
        DcodeNotificationKind.PERMISSION_REQUIRED: (
            WireNotificationType.PERMISSION_PROMPT
        ),
        WireNotificationType.PERMISSION_PROMPT: WireNotificationType.PERMISSION_PROMPT,
        DcodeNotificationKind.AGENT_NEEDS_INPUT: WireNotificationType.AGENT_NEEDS_INPUT,
        DcodeNotificationKind.AGENT_COMPLETED: WireNotificationType.AGENT_COMPLETED,
    }
    try:
        return mappings[value]
    except KeyError as exc:
        msg = f"Unsupported notification type: {value}"
        raise ValueError(msg) from exc


def _tool_result(result: ToolMessage | Command[str]) -> JsonValue:
    if isinstance(result, ToolMessage):
        value: object = result.model_dump(mode="json")
    else:
        value = JSON_VALUE_ADAPTER.dump_python(result, mode="json", warnings=False)
    return JSON_VALUE_ADAPTER.validate_python(value)
