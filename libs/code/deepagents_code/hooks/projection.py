"""Projection from Hooks v2 domain invocations to compatible wire input."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

from langchain_core.messages import ToolMessage

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.adapters import HOOK_WIRE_INPUT_ADAPTER
from deepagents_code.hooks.models.domain import (
    HookDiagnostic,
    HookEvent,
    NotificationEvent,
    PermissionRequestEvent,
    PostToolUseEvent,
    PreToolUseEvent,
    SessionEndEvent,
    SessionStartEvent,
    StopEvent,
    SubagentStartEvent,
    SubagentStopEvent,
)
from deepagents_code.hooks.models.wire import (
    BackgroundTaskWire,
    Effort,
    NotificationWireInput,
    PermissionRequestWireInput,
    PostToolUseWireInput,
    PreToolUseWireInput,
    SessionCronWire,
    SessionEndWireInput,
    SessionStartWireInput,
    StopWireInput,
    SubagentStartWireInput,
    SubagentStopWireInput,
    WireNotificationType,
    WirePermissionMode,
)
from deepagents_code.hooks.tools import to_wire_call
from deepagents_code.json_types import JSON_VALUE_ADAPTER, JsonValue

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from langgraph.types import Command

    from deepagents_code.hooks.models.domain import HookInvocation
    from deepagents_code.hooks.models.wire import HookWireInput


class _CoreWireFields(TypedDict):
    session_id: str
    transcript_path: str
    cwd: str
    permission_mode: NotRequired[WirePermissionMode]
    prompt_id: NotRequired[UUID]
    effort: NotRequired[Effort]


class _CommonWireFields(_CoreWireFields):
    agent_id: NotRequired[str]
    agent_type: NotRequired[str]


def project_hook_input(
    invocation: HookInvocation,
    *,
    transcript_path: Path | None = None,
    agent_transcript_path: Path | None = None,
) -> HookWireInput:
    """Project a native hook invocation into the compatible wire contract.

    Args:
        invocation: Native lifecycle invocation.
        transcript_path: Materialized client transcript path.
        agent_transcript_path: Materialized subagent transcript path.

    Returns:
        A validated event-specific wire input.

    Raises:
        TypeError: If the invocation carries an unsupported event model.
        ValueError: If a required materialized transcript path is missing.
    """
    event = invocation.event
    if isinstance(event, SessionStartEvent):
        result = SessionStartWireInput(
            **_common_fields(invocation, transcript_path),
            hook_event_name=HookEvent.SESSION_START,
            source=event.cause,
            model=event.model,
        )
    elif isinstance(event, SessionEndEvent):
        result = SessionEndWireInput(
            **_common_fields(invocation, transcript_path),
            hook_event_name=HookEvent.SESSION_END,
            reason=event.cause,
        )
    elif isinstance(event, PermissionRequestEvent):
        tool_name, tool_input = to_wire_call(event.call)
        result = PermissionRequestWireInput(
            **_common_fields(invocation, transcript_path),
            hook_event_name=HookEvent.PERMISSION_REQUEST,
            tool_name=tool_name,
            tool_input=tool_input,
        )
    elif isinstance(event, NotificationEvent):
        result = NotificationWireInput(
            **_common_fields(invocation, transcript_path),
            hook_event_name=HookEvent.NOTIFICATION,
            message=event.notification.message,
            title=event.notification.title,
            notification_type=_notification_type(event.notification.type),
        )
    elif isinstance(event, PreToolUseEvent):
        tool_name, tool_input = to_wire_call(event.call)
        result = PreToolUseWireInput(
            **_common_fields(invocation, transcript_path),
            hook_event_name=HookEvent.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=event.call.id,
        )
    elif isinstance(event, PostToolUseEvent):
        tool_name, tool_input = to_wire_call(event.call)
        result = PostToolUseWireInput(
            **_common_fields(invocation, transcript_path),
            hook_event_name=HookEvent.POST_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=_tool_result(event.result),
            tool_use_id=event.call.id,
            duration_ms=event.duration_ms,
        )
    elif isinstance(event, StopEvent):
        result = StopWireInput(
            **_common_fields(invocation, transcript_path),
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
    elif isinstance(event, SubagentStartEvent):
        result = SubagentStartWireInput(
            **_core_fields(invocation, transcript_path),
            hook_event_name=HookEvent.SUBAGENT_START,
            agent_id=event.agent.id,
            agent_type=event.agent.name,
        )
    elif isinstance(event, SubagentStopEvent):
        if agent_transcript_path is None:
            msg = "SubagentStop requires a materialized agent transcript path"
            raise ValueError(msg)
        result = SubagentStopWireInput(
            **_core_fields(invocation, transcript_path),
            hook_event_name=HookEvent.SUBAGENT_STOP,
            stop_hook_active=event.continuation_count > 0,
            agent_id=event.agent.id,
            agent_type=event.agent.name,
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
    else:
        msg = f"Unsupported hook event: {type(event).__name__}"
        raise TypeError(msg)

    payload = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        result,
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    return HOOK_WIRE_INPUT_ADAPTER.validate_python(payload)


def _core_fields(
    invocation: HookInvocation,
    transcript_path: Path | None,
) -> _CoreWireFields:
    context = invocation.context
    if transcript_path is None:
        msg = "HookInvocation requires a materialized transcript path"
        raise ValueError(msg)
    fields: _CoreWireFields = {
        "session_id": context.thread_id,
        "transcript_path": str(transcript_path),
        "cwd": str(context.cwd),
    }
    permission_mode = _permission_mode(context.approval_mode)
    if permission_mode is not None:
        fields["permission_mode"] = permission_mode
    if context.prompt_id is not None:
        fields["prompt_id"] = context.prompt_id
    if context.effort is not None:
        fields["effort"] = Effort(level=context.effort)
    return fields


def _common_fields(
    invocation: HookInvocation,
    transcript_path: Path | None,
) -> _CommonWireFields:
    fields: _CommonWireFields = {**_core_fields(invocation, transcript_path)}
    agent = invocation.context.agent
    if agent is not None:
        fields["agent_id"] = agent.id
        fields["agent_type"] = agent.name
    return fields


def serialize_hook_input(
    invocation: HookInvocation,
    *,
    transcript_path: Path | None = None,
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


def projection_diagnostics(invocation: HookInvocation) -> tuple[HookDiagnostic, ...]:
    """Return visible diagnostics for lossy domain-to-wire projection."""
    if invocation.context.approval_mode is ApprovalMode.AUTO:
        return (
            HookDiagnostic(
                code="unsupported_permission_mode",
                severity="warning",
                message=(
                    "AUTO approval mode has no proven compatible hook permission "
                    "mode and was omitted"
                ),
                field="permission_mode",
            ),
        )
    return ()


def _permission_mode(mode: ApprovalMode) -> WirePermissionMode | None:
    return {
        ApprovalMode.MANUAL: WirePermissionMode.DEFAULT,
        ApprovalMode.AUTO: None,
        ApprovalMode.YOLO: WirePermissionMode.BYPASS_PERMISSIONS,
    }[mode]


def _notification_type(value: str) -> WireNotificationType:
    try:
        return WireNotificationType(value)
    except ValueError as exc:
        msg = f"Unsupported notification type: {value}"
        raise ValueError(msg) from exc


def _tool_result(result: ToolMessage | Command[str]) -> JsonValue:
    if isinstance(result, ToolMessage):
        value: object = result.model_dump(mode="json")
    else:
        value = JSON_VALUE_ADAPTER.dump_python(result, mode="json", warnings=False)
    return JSON_VALUE_ADAPTER.validate_python(value)
