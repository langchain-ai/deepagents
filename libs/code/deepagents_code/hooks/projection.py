"""Projection from Hooks v2 domain invocations to compatible wire input."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

from langchain_core.messages import ToolMessage

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.adapters import HOOK_WIRE_INPUT_ADAPTER
from deepagents_code.hooks.models.domain import (
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


class _CommonWireFields(TypedDict):
    session_id: str
    transcript_path: str
    cwd: str
    prompt_id: NotRequired[UUID | None]
    permission_mode: WirePermissionMode
    effort: NotRequired[Effort | None]


def project_hook_input(invocation: HookInvocation) -> HookWireInput:
    """Project a native hook invocation into the compatible wire contract.

    Args:
        invocation: Native lifecycle invocation.

    Returns:
        A validated event-specific wire input.

    Raises:
        TypeError: If the invocation carries an unsupported event model.
    """
    context = invocation.context
    event = invocation.event
    if isinstance(event, SessionStartEvent):
        result = SessionStartWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.SESSION_START,
            source=event.cause,
            model=event.model,
        )
    elif isinstance(event, SessionEndEvent):
        result = SessionEndWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.SESSION_END,
            reason=event.cause,
        )
    elif isinstance(event, PermissionRequestEvent):
        tool_name, tool_input = to_wire_call(event.call)
        result = PermissionRequestWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.PERMISSION_REQUEST,
            tool_name=tool_name,
            tool_input=tool_input,
        )
    elif isinstance(event, NotificationEvent):
        result = NotificationWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.NOTIFICATION,
            message=event.notification.message,
            title=event.notification.title,
            notification_type=_notification_type(event.notification.type),
        )
    elif isinstance(event, PreToolUseEvent):
        tool_name, tool_input = to_wire_call(event.call)
        result = PreToolUseWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=event.call.id,
        )
    elif isinstance(event, PostToolUseEvent):
        tool_name, tool_input = to_wire_call(event.call)
        result = PostToolUseWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.POST_TOOL_USE,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=_tool_result(event.result),
            tool_use_id=event.call.id,
            duration_ms=event.duration_ms,
        )
    elif isinstance(event, StopEvent):
        result = StopWireInput(
            **_common_fields(invocation),
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
            **_common_fields(invocation),
            hook_event_name=HookEvent.SUBAGENT_START,
            agent_id=event.agent.id,
            agent_type=event.agent.name,
        )
    elif isinstance(event, SubagentStopEvent):
        result = SubagentStopWireInput(
            **_common_fields(invocation),
            hook_event_name=HookEvent.SUBAGENT_STOP,
            stop_hook_active=event.continuation_count > 0,
            agent_id=event.agent.id,
            agent_type=event.agent.name,
            agent_transcript_path=str(_transcript_path(context.cwd, event.agent.id)),
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


def _common_fields(invocation: HookInvocation) -> _CommonWireFields:
    context = invocation.context
    return {
        "session_id": context.thread_id,
        "transcript_path": str(_transcript_path(context.cwd, context.thread_id)),
        "cwd": str(context.cwd),
        "prompt_id": context.prompt_id,
        "permission_mode": _permission_mode(context.approval_mode),
        "effort": Effort(level=context.effort) if context.effort is not None else None,
    }


def serialize_hook_input(invocation: HookInvocation) -> bytes:
    """Serialize a hook invocation as validated compatible JSON.

    Args:
        invocation: Native lifecycle invocation.

    Returns:
        Compact JSON bytes suitable for command stdin.
    """
    return HOOK_WIRE_INPUT_ADAPTER.dump_json(
        project_hook_input(invocation),
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
    try:
        return WireNotificationType(value)
    except ValueError:
        return WireNotificationType.AGENT_NEEDS_INPUT


def _transcript_path(cwd: Path, identifier: str) -> Path:
    return cwd / ".deepagents" / "transcripts" / f"{identifier}.jsonl"


def _tool_result(result: ToolMessage | Command[str]) -> JsonValue:
    if isinstance(result, ToolMessage):
        value: object = result.model_dump(mode="json")
    else:
        value = JSON_VALUE_ADAPTER.dump_python(result, mode="json", warnings=False)
    return JSON_VALUE_ADAPTER.validate_python(value)
