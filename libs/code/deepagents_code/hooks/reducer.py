"""Event-aware reduction for Hooks v2 command output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepagents_code.hooks.capabilities import ExitCodePolicy, get_event_spec
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
from deepagents_code.hooks.terminal import validate_terminal_sequence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from deepagents_code.hooks.models.domain import HookInvocation
    from deepagents_code.hooks.models.wire import HookWireOutput
    from deepagents_code.hooks.runner import HandlerResult

_PERMISSION_RANK = {"none": 0, "allow": 1, "ask": 2, "deny": 3}
MAX_STOP_CONTINUATIONS = 8

_DEFERRED_SESSION_START_FIELDS = (
    ("initial_user_message", "initialUserMessage"),
    ("session_title", "sessionTitle"),
    ("watch_paths", "watchPaths"),
    ("reload_skills", "reloadSkills"),
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
        if result.plain_output is not None:
            state.context.append(result.plain_output)
        if result.output is not None:
            _merge_output(invocation, state, result.handler_id, result.output)
    return _decision(invocation, state)


def _merge_output(
    invocation: HookInvocation,
    state: _Reduction,
    handler_id: str,
    output: HookWireOutput,
) -> None:
    state.continue_processing = state.continue_processing and output.continue_
    if output.stop_reason is not None:
        if output.continue_:
            state.diagnostics.append(
                HookDiagnostic(
                    code="ignored_stop_reason",
                    severity="warning",
                    message="stopReason is ignored while continue is true",
                    handler_id=handler_id,
                    field="stopReason",
                )
            )
        elif state.stop_reason is None:
            state.stop_reason = output.stop_reason
        else:
            state.diagnostics.append(
                HookDiagnostic(
                    code="additional_stop_reason",
                    severity="warning",
                    message="A later stopReason was ignored; the first reason wins",
                    handler_id=handler_id,
                    field="stopReason",
                )
            )
    if output.system_message is not None and not output.suppress_output:
        state.user_notices.append(output.system_message)
    if output.terminal_sequence is not None:
        validated = validate_terminal_sequence(output.terminal_sequence)
        if validated is None:
            state.diagnostics.append(
                HookDiagnostic(
                    code="invalid_terminal_sequence",
                    severity="warning",
                    message=(
                        "terminalSequence rejected; only OSC 0/1/2/9/99/777 "
                        "and BEL are allowed"
                    ),
                    handler_id=handler_id,
                    field="terminalSequence",
                )
            )
        else:
            state.terminal_sequences.append(validated)
    _diagnose_extra_fields(state, handler_id, output)
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


def _diagnose_extra_fields(
    state: _Reduction,
    handler_id: str,
    output: HookWireOutput,
) -> None:
    extras = getattr(output, "__pydantic_extra__", None)
    if not extras:
        return
    for name in sorted(extras):
        state.diagnostics.append(
            HookDiagnostic(
                code="unsupported_field",
                severity="warning",
                message=f"Unsupported hook output field ignored: {name}",
                handler_id=handler_id,
                field=name,
            )
        )


def _merge_block(
    invocation: HookInvocation,
    state: _Reduction,
    handler_id: str,
    reason: str | None,
) -> None:
    message = reason or "Blocked by hook"
    event = invocation.event.event
    policy = get_event_spec(event).exit_code_policy
    if policy is ExitCodePolicy.DENY:
        _merge_permission(state, PermissionEffect(behavior="deny", reason=message))
        return
    if policy is ExitCodePolicy.FEEDBACK:
        state.feedback.append(message)
        return
    if policy is ExitCodePolicy.CONTINUE_LOOP:
        _apply_stop_continuation(invocation, state, message)
        return
    if policy is ExitCodePolicy.IGNORE:
        return
    # DIAGNOSE: exit 2 / decision:"block" is not a veto. SubagentStop still
    # retains parent-visible context on the first attempt per the MVP matrix.
    if event is HookEvent.SUBAGENT_STOP:
        if (
            isinstance(invocation.event, SubagentStopEvent)
            and invocation.event.continuation_count
        ):
            state.diagnostics.append(_loop_guard_diagnostic())
            return
        state.diagnostics.append(
            HookDiagnostic(
                code="unsupported_block",
                severity="warning",
                message=(
                    "Blocking SubagentStop is not supported in MVP; "
                    f"retained as parent context: {message}"
                ),
                handler_id=handler_id,
                field="decision",
            )
        )
        state.context.append(message)
        return
    state.diagnostics.append(
        HookDiagnostic(
            code="unsupported_block",
            severity="warning",
            message=f"Block/exit 2 is not supported for {event.value}: {message}",
            handler_id=handler_id,
            field="decision",
        )
    )


def _apply_stop_continuation(
    invocation: HookInvocation,
    state: _Reduction,
    message: str,
) -> None:
    if not isinstance(invocation.event, StopEvent):
        return
    if invocation.event.continuation_count >= MAX_STOP_CONTINUATIONS:
        state.diagnostics.append(
            HookDiagnostic(
                code="continuation_cap",
                severity="warning",
                message=(
                    f"Ignored Stop continuation after {MAX_STOP_CONTINUATIONS} "
                    "consecutive attempts"
                ),
            )
        )
        return
    state.continue_loop = True
    state.feedback.append(message)


def _merge_specific(
    invocation: HookInvocation,
    state: _Reduction,
    handler_id: str,
    specific: object,
) -> None:
    if isinstance(specific, SessionStartSpecificOutput):
        _append(state.context, specific.additional_context)
        for attr, wire_name in _DEFERRED_SESSION_START_FIELDS:
            value = getattr(specific, attr)
            if value in (None, False, [], ""):
                continue
            state.diagnostics.append(
                HookDiagnostic(
                    code="unsupported_field",
                    severity="warning",
                    message=f"{wire_name} is not supported and was ignored",
                    handler_id=handler_id,
                    field=wire_name,
                )
            )
    elif isinstance(specific, PreToolUseSpecificOutput):
        _append(state.context, specific.additional_context)
        behavior = specific.permission_decision
        if behavior == "defer":
            state.diagnostics.append(
                HookDiagnostic(
                    code="unsupported_field",
                    severity="warning",
                    message="permissionDecision defer is not supported and was ignored",
                    handler_id=handler_id,
                    field="permissionDecision",
                )
            )
            behavior = None
        if specific.updated_input is not None:
            _diagnose_unsupported_updated_input(state, handler_id)
            # Allow/ask coupled to mutation falls back to normal permission flow;
            # deny remains safe without applying the mutated input.
            if behavior in {"allow", "ask"}:
                behavior = None
        if behavior is not None:
            _merge_permission(
                state,
                PermissionEffect(
                    behavior=behavior,
                    reason=specific.permission_decision_reason,
                ),
            )
    elif isinstance(specific, PermissionRequestSpecificOutput):
        decision = specific.decision
        if decision.behavior == "allow":
            has_updated_input = (
                isinstance(decision, PermissionAllow)
                and decision.updated_input is not None
            )
            if has_updated_input:
                _diagnose_unsupported_updated_input(state, handler_id)
            if isinstance(decision, PermissionAllow) and decision.updated_permissions:
                state.diagnostics.append(
                    HookDiagnostic(
                        code="unsupported_field",
                        severity="warning",
                        message="updatedPermissions is not supported and was ignored",
                        handler_id=handler_id,
                        field="updatedPermissions",
                    )
                )
            if not has_updated_input:
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
        if specific.updated_tool_output is not None:
            state.diagnostics.append(
                HookDiagnostic(
                    code="unsupported_field",
                    severity="warning",
                    message="updatedToolOutput is not supported and was ignored",
                    handler_id=handler_id,
                    field="updatedToolOutput",
                )
            )
        if specific.updated_mcp_tool_output is not None:
            state.diagnostics.append(
                HookDiagnostic(
                    code="unsupported_field",
                    severity="warning",
                    message="updatedMCPToolOutput is not supported and was ignored",
                    handler_id=handler_id,
                    field="updatedMCPToolOutput",
                )
            )
    elif isinstance(specific, StopSpecificOutput):
        if specific.additional_context is not None:
            _apply_stop_continuation(invocation, state, specific.additional_context)
    elif isinstance(specific, SubagentStartSpecificOutput):
        _append(state.context, specific.additional_context)
    elif (
        isinstance(specific, SubagentStopSpecificOutput)
        and specific.additional_context is not None
    ):
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
    if _PERMISSION_RANK[effect.behavior] > _PERMISSION_RANK[state.permission.behavior]:
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
