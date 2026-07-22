"""Capability registry for Hooks v2 events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, assert_never

from deepagents_code.hooks.models.domain import (
    HookEvent,
    HookOwner,
    NotificationDecision,
    NotificationEvent,
    PermissionRequestDecision,
    PermissionRequestEvent,
    PostToolUseDecision,
    PostToolUseEvent,
    PreToolUseDecision,
    PreToolUseEvent,
    SessionEndDecision,
    SessionEndEvent,
    SessionStartDecision,
    SessionStartEvent,
    StopDecision,
    StopEvent,
    SubagentStartDecision,
    SubagentStartEvent,
    SubagentStopDecision,
    SubagentStopEvent,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_code.hooks.models.domain import BaseHookDecision, HookDomainEvent


class HandlerType(StrEnum):
    """Supported hook handler executor kinds."""

    COMMAND = "command"


class PlainOutputPolicy(StrEnum):
    """How non-JSON stdout is treated on a successful exit."""

    IGNORE = "ignore"
    CONTEXT = "context"


class ExitCodePolicy(StrEnum):
    """How exit code 2 is interpreted for an event."""

    DENY = "deny"
    FEEDBACK = "feedback"
    CONTINUE_LOOP = "continue_loop"
    DIAGNOSE = "diagnose"
    IGNORE = "ignore"


class AggregationPolicy(StrEnum):
    """How matching handler effects are combined."""

    CONTEXT = "context"
    PERMISSION = "permission"
    FEEDBACK_AND_CONTEXT = "feedback_and_context"
    STOP_LOOP = "stop_loop"
    SIDE_EFFECT = "side_effect"


DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0
MatcherField: TypeAlias = Literal[
    "cause", "tool_name", "notification_type", "agent_name"
]


@dataclass(frozen=True, slots=True)
class HookEventSpec:
    """Immutable capability description for one hook event."""

    event: HookEvent
    owner: HookOwner
    event_model: type[HookDomainEvent]
    decision_model: type[BaseHookDecision]
    matcher_field: MatcherField | None
    default_timeout_seconds: float
    exit_code_policy: ExitCodePolicy
    plain_output_policy: PlainOutputPolicy
    aggregation_policy: AggregationPolicy
    supported_handler_types: frozenset[HandlerType]


_HOOK_EVENT_SPECS: Final[Mapping[HookEvent, HookEventSpec]] = MappingProxyType(
    {
        HookEvent.SESSION_START: HookEventSpec(
            event=HookEvent.SESSION_START,
            owner=HookOwner.CLIENT,
            event_model=SessionStartEvent,
            decision_model=SessionStartDecision,
            matcher_field="cause",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DIAGNOSE,
            plain_output_policy=PlainOutputPolicy.CONTEXT,
            aggregation_policy=AggregationPolicy.CONTEXT,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.SESSION_END: HookEventSpec(
            event=HookEvent.SESSION_END,
            owner=HookOwner.CLIENT,
            event_model=SessionEndEvent,
            decision_model=SessionEndDecision,
            matcher_field="cause",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DIAGNOSE,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.SIDE_EFFECT,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.PERMISSION_REQUEST: HookEventSpec(
            event=HookEvent.PERMISSION_REQUEST,
            owner=HookOwner.CLIENT,
            event_model=PermissionRequestEvent,
            decision_model=PermissionRequestDecision,
            matcher_field="tool_name",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DENY,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.PERMISSION,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.NOTIFICATION: HookEventSpec(
            event=HookEvent.NOTIFICATION,
            owner=HookOwner.CLIENT,
            event_model=NotificationEvent,
            decision_model=NotificationDecision,
            matcher_field="notification_type",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DIAGNOSE,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.SIDE_EFFECT,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.PRE_TOOL_USE: HookEventSpec(
            event=HookEvent.PRE_TOOL_USE,
            owner=HookOwner.SERVER,
            event_model=PreToolUseEvent,
            decision_model=PreToolUseDecision,
            matcher_field="tool_name",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DENY,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.PERMISSION,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.POST_TOOL_USE: HookEventSpec(
            event=HookEvent.POST_TOOL_USE,
            owner=HookOwner.SERVER,
            event_model=PostToolUseEvent,
            decision_model=PostToolUseDecision,
            matcher_field="tool_name",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.FEEDBACK,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.FEEDBACK_AND_CONTEXT,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.STOP: HookEventSpec(
            event=HookEvent.STOP,
            owner=HookOwner.SERVER,
            event_model=StopEvent,
            decision_model=StopDecision,
            matcher_field=None,
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.CONTINUE_LOOP,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.STOP_LOOP,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.SUBAGENT_START: HookEventSpec(
            event=HookEvent.SUBAGENT_START,
            owner=HookOwner.SERVER,
            event_model=SubagentStartEvent,
            decision_model=SubagentStartDecision,
            matcher_field="agent_name",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DIAGNOSE,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.CONTEXT,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
        HookEvent.SUBAGENT_STOP: HookEventSpec(
            event=HookEvent.SUBAGENT_STOP,
            owner=HookOwner.SERVER,
            event_model=SubagentStopEvent,
            decision_model=SubagentStopDecision,
            matcher_field="agent_name",
            default_timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            exit_code_policy=ExitCodePolicy.DIAGNOSE,
            plain_output_policy=PlainOutputPolicy.IGNORE,
            aggregation_policy=AggregationPolicy.CONTEXT,
            supported_handler_types=frozenset({HandlerType.COMMAND}),
        ),
    }
)


def get_event_spec(event: HookEvent) -> HookEventSpec:
    """Return the capability entry for `event`.

    Args:
        event: Lifecycle event name.

    Returns:
        The registered capability specification.
    """
    match event:
        case HookEvent.SESSION_START:
            return _HOOK_EVENT_SPECS[HookEvent.SESSION_START]
        case HookEvent.SESSION_END:
            return _HOOK_EVENT_SPECS[HookEvent.SESSION_END]
        case HookEvent.PERMISSION_REQUEST:
            return _HOOK_EVENT_SPECS[HookEvent.PERMISSION_REQUEST]
        case HookEvent.NOTIFICATION:
            return _HOOK_EVENT_SPECS[HookEvent.NOTIFICATION]
        case HookEvent.PRE_TOOL_USE:
            return _HOOK_EVENT_SPECS[HookEvent.PRE_TOOL_USE]
        case HookEvent.POST_TOOL_USE:
            return _HOOK_EVENT_SPECS[HookEvent.POST_TOOL_USE]
        case HookEvent.STOP:
            return _HOOK_EVENT_SPECS[HookEvent.STOP]
        case HookEvent.SUBAGENT_START:
            return _HOOK_EVENT_SPECS[HookEvent.SUBAGENT_START]
        case HookEvent.SUBAGENT_STOP:
            return _HOOK_EVENT_SPECS[HookEvent.SUBAGENT_STOP]
        case _:
            assert_never(event)
