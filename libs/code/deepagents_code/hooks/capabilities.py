"""Capability registry for Hooks v2 events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, get_args

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

    from pydantic import BaseModel


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
_SUPPORTED_MATCHER_FIELDS = frozenset(
    {"cause", "tool_name", "notification_type", "agent_name"}
)


@dataclass(frozen=True, slots=True)
class HookEventSpec:
    """Immutable capability description for one hook event."""

    event: HookEvent
    owner: HookOwner
    event_model: type[BaseModel]
    decision_model: type[BaseModel]
    matcher_field: str | None
    default_timeout_seconds: float
    exit_code_policy: ExitCodePolicy
    plain_output_policy: PlainOutputPolicy
    aggregation_policy: AggregationPolicy
    supported_handler_types: frozenset[HandlerType]


HOOK_EVENT_SPECS: Final[Mapping[HookEvent, HookEventSpec]] = MappingProxyType(
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


def assert_hook_event_registry_complete() -> None:
    """Raise if any `HookEvent` is missing from the capability registry.

    Raises:
        RuntimeError: If the registry is incomplete or internally inconsistent.
    """
    expected = frozenset(HookEvent)
    actual = frozenset(HOOK_EVENT_SPECS)
    missing = expected - actual
    extra = actual - expected
    if missing:
        msg = f"Missing HookEventSpec entries: {sorted(e.value for e in missing)}"
        raise RuntimeError(msg)
    if extra:
        msg = f"Unexpected HookEventSpec entries: {sorted(e.value for e in extra)}"
        raise RuntimeError(msg)
    for event, spec in HOOK_EVENT_SPECS.items():
        if spec.event is not event:
            msg = f"HookEventSpec key/value mismatch for {event.value}"
            raise RuntimeError(msg)
        if HandlerType.COMMAND not in spec.supported_handler_types:
            msg = f"HookEventSpec for {event.value} must support command handlers"
            raise RuntimeError(msg)
        if (
            spec.matcher_field is not None
            and spec.matcher_field not in _SUPPORTED_MATCHER_FIELDS
        ):
            msg = (
                f"HookEventSpec for {event.value} has unsupported matcher field "
                f"{spec.matcher_field!r}"
            )
            raise RuntimeError(msg)
        _assert_model_discriminator(spec.event_model, event, "event")
        _assert_model_discriminator(spec.decision_model, event, "decision")


def _assert_model_discriminator(
    model: type[BaseModel],
    event: HookEvent,
    kind: str,
) -> None:
    field = model.model_fields.get("event")
    if field is None or get_args(field.annotation) != (event,):
        msg = f"{kind.title()} model for {event.value} has an invalid discriminator"
        raise RuntimeError(msg)


assert_hook_event_registry_complete()


def get_event_spec(event: HookEvent) -> HookEventSpec:
    """Return the capability entry for `event`.

    Args:
        event: Lifecycle event name.

    Returns:
        The registered capability specification.
    """
    return HOOK_EVENT_SPECS[event]
