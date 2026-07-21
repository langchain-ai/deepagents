"""Event-aware reduction for Hooks v2 command output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepagents_code.hooks.models.domain import (
    HookDecision,
    HookDiagnostic,
    HookEvent,
    NotificationDecision,
    PermissionEffect,
    PermissionRequestDecision,
    PostToolUseDecision,
    PreToolUseDecision,
    SessionEndDecision,
    SessionStartDecision,
    StopDecision,
    StopEvent,
    SubagentStartDecision,
    SubagentStopDecision,
    SubagentStopEvent,
)
from deepagents_code.hooks.models.wire import (
    PermissionAllow,
    PermissionRequestSpecificOutput,
    PostToolUseSpecificOutput,
    PreToolUseSpecificOutput,
    SessionStartSpecificOutput,
    StopSpecificOutput,
    SubagentStartSpecificOutput,
    SubagentStopSpecificOutput,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from deepagents_code.hooks.models.domain import HookInvocation
    from deepagents_code.hooks.models.wire import HookWireOutput
    from deepagents_code.hooks.runner import HandlerResult

_PERMISSION_RANK = {"none": 0, "allow": 1, "ask": 2, "deny": 3}

# Exit 2 / decision:"block" cannot veto these lifecycle points in MVP.
_NON_BLOCKING_EVENTS = frozenset(
    {
        HookEvent.SESSION_START,
        HookEvent.SESSION_END,
        HookEvent.NOTIFICATION,
        HookEvent.SUBAGENT_START,
    }
)


@dataclass(slots=True)
class _Reduction:
    continue_processing: bool = True
    stop_reason: str | None = None
    user_notices: list[str] = field(default_factory=list)
    terminal_sequences: list[str] = field(default_factory=list)
    diagnostics: list[HookDiagnostic] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    permission: PermissionEffect = field(
        default_factory=lambda: PermissionEffect(behavior="none")
    )
    continue_loop: bool = False


def reduce_hook_results(
    invocation: HookInvocation,
    results: Iterable[HandlerResult],
    *,
    diagnostics: Iterable[HookDiagnostic] = (),
) -> HookDecision:
    """Reduce ordered handler results into an event-specific decision.

    Args:
        invocation: Native event being processed.
        results: Handler results in configuration order.
        diagnostics: Snapshot or orchestration diagnostics to retain.

    Returns:
        The normalized decision for the invocation event.
    """
    state = _Reduction(diagnostics=list(diagnostics))
    for result in results:
        state.diagnostics.extend(result.diagnostics)
        if result.output is not None:
            _merge_output(invocation, state, result.handler_id, result.output)
        if not state.continue_processing:
            break
    return _decision(invocation, state)


def _merge_output(
    invocation: HookInvocation,
    state: _Reduction,
    handler_id: str,
    output: HookWireOutput,
) -> None:
    state.continue_processing = output.continue_
    if output.stop_reason is not None:
        state.stop_reason = output.stop_reason
    if output.system_message is not None and not output.suppress_output:
        state.user_notices.append(output.system_message)
    if output.terminal_sequence is not None:
        state.terminal_sequences.append(output.terminal_sequence)
    if output.decision == "block":
        _merge_block(invocation, state, handler_id, output.reason)

    specific = output.hook_specific_output
    if specific is None:
        return
    if specific.hook_event_name != invocation.event.event.value:
        state.diagnostics.append(
            HookDiagnostic(
                code="mismatched_output",
                severity="warning",
                message="Hook-specific output does not match the invoked event",
                handler_id=handler_id,
                field="hookSpecificOutput.hookEventName",
            )
        )
        return
    _merge_specific(invocation, state, handler_id, specific)


def _merge_block(
    invocation: HookInvocation,
    state: _Reduction,
    handler_id: str,
    reason: str | None,
) -> None:
    message = reason or "Blocked by hook"
    event = invocation.event.event
    if event in {HookEvent.PERMISSION_REQUEST, HookEvent.PRE_TOOL_USE}:
        _merge_permission(state, PermissionEffect(behavior="deny", reason=message))
    elif event is HookEvent.STOP:
        if (
            isinstance(invocation.event, StopEvent)
            and invocation.event.continuation_count
        ):
            state.diagnostics.append(_loop_guard_diagnostic())
        else:
            state.continue_loop = True
            state.feedback.append(message)
    elif event is HookEvent.POST_TOOL_USE:
        state.feedback.append(message)
    elif event is HookEvent.SUBAGENT_STOP:
        if (
            isinstance(invocation.event, SubagentStopEvent)
            and invocation.event.continuation_count
        ):
            state.diagnostics.append(_loop_guard_diagnostic())
        else:
            state.context.append(message)
    elif event in _NON_BLOCKING_EVENTS:
        # Exit 2 / decision:"block" is not a veto for these events; keep going
        # and surface the attempt so configs that expect Claude blocking see why
        # dcode ignored it.
        state.diagnostics.append(
            HookDiagnostic(
                code="unsupported_block",
                severity="warning",
                message=f"Block/exit 2 is not supported for {event.value}: {message}",
                handler_id=handler_id,
                field="decision",
            )
        )
    else:
        state.stop_reason = message
        state.continue_processing = False


def _merge_specific(
    invocation: HookInvocation,
    state: _Reduction,
    handler_id: str,
    specific: object,
) -> None:
    if isinstance(specific, SessionStartSpecificOutput):
        _append(state.context, specific.additional_context)
    elif isinstance(specific, PreToolUseSpecificOutput):
        _append(state.context, specific.additional_context)
        behavior = specific.permission_decision
        if specific.updated_input is not None:
            _diagnose_unsupported_updated_input(state, handler_id)
            # Allow/ask coupled to mutation falls back to normal permission flow;
            # deny remains safe without applying the mutated input.
            if behavior in {"allow", "ask"}:
                behavior = None
        if behavior is not None:
            normalized = "none" if behavior == "defer" else behavior
            _merge_permission(
                state,
                PermissionEffect(
                    behavior=normalized,
                    reason=specific.permission_decision_reason,
                ),
            )
    elif isinstance(specific, PermissionRequestSpecificOutput):
        decision = specific.decision
        if decision.behavior == "allow":
            if (
                isinstance(decision, PermissionAllow)
                and decision.updated_input is not None
            ):
                _diagnose_unsupported_updated_input(state, handler_id)
            else:
                _merge_permission(state, PermissionEffect(behavior="allow"))
        else:
            _merge_permission(
                state,
                PermissionEffect(
                    behavior="deny",
                    reason=decision.message,
                    interrupt=decision.interrupt,
                ),
            )
    elif isinstance(specific, PostToolUseSpecificOutput):
        _append(state.context, specific.additional_context)
    elif isinstance(specific, StopSpecificOutput):
        if specific.additional_context is not None:
            if (
                isinstance(invocation.event, StopEvent)
                and invocation.event.continuation_count
            ):
                state.diagnostics.append(_loop_guard_diagnostic())
            else:
                state.continue_loop = True
                state.feedback.append(specific.additional_context)
    elif isinstance(specific, SubagentStartSpecificOutput):
        _append(state.context, specific.additional_context)
    elif (
        isinstance(specific, SubagentStopSpecificOutput)
        and specific.additional_context is not None
    ):
        if (
            isinstance(invocation.event, SubagentStopEvent)
            and invocation.event.continuation_count
        ):
            state.diagnostics.append(_loop_guard_diagnostic())
        else:
            state.context.append(specific.additional_context)


def _diagnose_unsupported_updated_input(
    state: _Reduction,
    handler_id: str,
) -> None:
    state.diagnostics.append(
        HookDiagnostic(
            code="unsupported_field",
            severity="warning",
            message=(
                "updatedInput is not supported; the mutated tool input was ignored"
            ),
            handler_id=handler_id,
            field="updatedInput",
        )
    )


def _merge_permission(state: _Reduction, effect: PermissionEffect) -> None:
    if _PERMISSION_RANK[effect.behavior] >= _PERMISSION_RANK[state.permission.behavior]:
        state.permission = effect


def _decision(invocation: HookInvocation, state: _Reduction) -> HookDecision:
    common = {
        "continue_processing": state.continue_processing,
        "stop_reason": state.stop_reason,
        "user_notices": state.user_notices,
        "terminal_sequences": state.terminal_sequences,
        "diagnostics": state.diagnostics,
    }
    event = invocation.event.event
    if event is HookEvent.SESSION_START:
        return SessionStartDecision(event=event, context=state.context, **common)
    if event is HookEvent.SESSION_END:
        return SessionEndDecision(event=event, **common)
    if event is HookEvent.PERMISSION_REQUEST:
        return PermissionRequestDecision(
            event=event,
            permission=state.permission,
            **common,
        )
    if event is HookEvent.NOTIFICATION:
        return NotificationDecision(event=event, **common)
    if event is HookEvent.PRE_TOOL_USE:
        return PreToolUseDecision(
            event=event,
            permission=state.permission,
            context=state.context,
            **common,
        )
    if event is HookEvent.POST_TOOL_USE:
        return PostToolUseDecision(
            event=event,
            feedback=state.feedback,
            context=state.context,
            **common,
        )
    if event is HookEvent.STOP:
        return StopDecision(
            event=event,
            continue_loop=state.continue_loop,
            feedback=state.feedback,
            **common,
        )
    if event is HookEvent.SUBAGENT_START:
        return SubagentStartDecision(event=event, context=state.context, **common)
    if event is HookEvent.SUBAGENT_STOP:
        return SubagentStopDecision(event=event, context=state.context, **common)
    msg = f"Unsupported hook event: {event}"
    raise ValueError(msg)


def _append(values: list[str], value: str | None) -> None:
    if value is not None:
        values.append(value)


def _loop_guard_diagnostic() -> HookDiagnostic:
    return HookDiagnostic(
        code="continuation_guard",
        severity="warning",
        message="Ignored recursive stop-hook continuation",
    )
