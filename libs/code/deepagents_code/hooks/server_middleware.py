"""Server-owned Hooks v2 lifecycle middleware.

Emits `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart`, and `SubagentStop`
through the LangGraph interrupt channel so the client runtime can execute
matching handlers and return typed decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
)
from uuid import UUID, uuid5

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    HITLRequest,
    ReviewConfig,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from deepagents_code.approval_mode import ApprovalMode, coerce_approval_mode
from deepagents_code.hooks.interrupt import (
    build_hook_interrupt_payload,
    parse_hook_resume_value,
)
from deepagents_code.hooks.models.domain import (
    AgentIdentity,
    BaseHookDecision,
    HookContext,
    HookDecision,
    HookEvent,
    HookInvocation,
    PermissionEffect,
    PostToolUseDecision,
    PostToolUseEvent,
    PreToolUseDecision,
    PreToolUseEvent,
    StopDecision,
    StopEvent,
    SubagentStartDecision,
    SubagentStartEvent,
    SubagentStopDecision,
    SubagentStopEvent,
    ToolCallData,
)
from deepagents_code.hooks.models.transport import HookInvocationRequest
from deepagents_code.hooks.tools import to_wire_tool_name

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages.tool import ToolCall
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from deepagents_code.json_types import JsonObject

_DEFAULT_DEADLINE = timedelta(seconds=600)
_STOP_STATE_KEY = "_hooks_stop_continuation_count"
_PRE_TOOL_STATE_KEY = "_hooks_pre_tool_outcomes"
_TASK_TOOL_NAME = "task"
_INVOCATION_NAMESPACE = UUID("f2896d18-cf2a-4e7d-b11a-d5b10fc0e335")

PreToolBehavior: TypeAlias = Literal["allow", "deny", "none"]


class _PreToolState(TypedDict):
    behavior: PreToolBehavior
    reason: str | None
    context: list[str]


class ServerHooksState(AgentState[Any]):
    """Agent state extensions for server-owned hook middleware."""

    _hooks_stop_continuation_count: NotRequired[int]
    _hooks_pre_tool_outcomes: NotRequired[dict[str, _PreToolState]]


class _HookEventsForSnapshot(TypedDict):
    snapshot_id: str
    events: frozenset[str]


class _HooksContextFields(TypedDict, total=False):
    """Hook-relevant fields projected from LangGraph run context."""

    hooks_snapshot_id: str
    hooks_server_events: list[str]
    thread_id: str
    approval_mode: str
    prompt_id: str


_HOOKS_CONTEXT_KEYS: tuple[str, ...] = (
    "hooks_snapshot_id",
    "hooks_server_events",
    "thread_id",
    "approval_mode",
    "prompt_id",
)


@dataclass(slots=True)
class _PreToolOutcome:
    """PreToolUse outcome used by the tool-call wrapper."""

    blocked: ToolMessage | None = None
    context: tuple[str, ...] = field(default_factory=tuple)


class ServerHooksMiddleware(AgentMiddleware[ServerHooksState, ContextT, ResponseT]):
    """Emit server-owned lifecycle events over the hook interrupt transport."""

    state_schema = ServerHooksState

    def __init__(
        self,
        *,
        cwd: Path,
        default_deadline: timedelta = _DEFAULT_DEADLINE,
        emit_stop: bool = True,
        mcp_tools: Sequence[BaseTool] = (),
    ) -> None:
        """Initialize middleware.

        Args:
            cwd: Session working directory projected into hook context.
            default_deadline: Client execution deadline attached to requests.
            emit_stop: Whether to emit the main-agent `Stop` event from
                `after_agent`. Subagent graphs set this to `False` so they still
                wrap tools without firing parent `Stop` handlers.
            mcp_tools: MCP tools whose server metadata is needed before tool
                execution for compatible hook projection.
        """
        super().__init__()
        self._cwd = cwd
        self._default_deadline = default_deadline
        self._emit_stop = emit_stop
        self._mcp_servers = {
            name: server
            for tool in mcp_tools
            if (name := getattr(tool, "name", None))
            and isinstance(name, str)
            and (server := _mcp_server_from_tool(tool)) is not None
        }

    def after_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        """Run `PreToolUse` before downstream HITL middleware.

        Returns:
            State update carrying per-tool hook outcomes.
        """
        return self._after_model(state, runtime)

    async def aafter_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        """Run the async graph path through the same interrupt sequence.

        Returns:
            State update carrying per-tool hook outcomes.
        """
        return self._after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run Pre/Post tool hooks around a synchronous tool call.

        Returns:
            Tool result, possibly rewritten by hook decisions.
        """
        enabled_events = _hook_events_for_snapshot(request.runtime.context)
        tool_call = _tool_call_data(request)
        pre_tool_outcome = _pre_tool_outcome(request.state, tool_call)
        hook_context = _build_hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        if pre_tool_outcome.blocked is not None:
            return _append_message_text(
                pre_tool_outcome.blocked, pre_tool_outcome.context, tool_call.id
            )
        started_or_blocked = self._maybe_subagent_start(
            request, tool_call, hook_context, enabled_events
        )
        if isinstance(started_or_blocked, ToolMessage):
            return started_or_blocked
        request = started_or_blocked
        started = time.perf_counter()
        result = handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = _append_message_text(result, pre_tool_outcome.context, tool_call.id)
        result = self._maybe_post_tool_use(
            tool_call,
            hook_context,
            enabled_events,
            request.runtime.config,
            result,
            duration_ms,
        )
        return self._maybe_subagent_stop(
            tool_call, hook_context, enabled_events, request.runtime.config, result
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run Pre/Post tool hooks around an asynchronous tool call.

        Returns:
            Tool result, possibly rewritten by hook decisions.
        """
        enabled_events = _hook_events_for_snapshot(request.runtime.context)
        tool_call = _tool_call_data(request)
        pre_tool_outcome = _pre_tool_outcome(request.state, tool_call)
        hook_context = _build_hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        if pre_tool_outcome.blocked is not None:
            return _append_message_text(
                pre_tool_outcome.blocked, pre_tool_outcome.context, tool_call.id
            )
        started_or_blocked = self._maybe_subagent_start(
            request, tool_call, hook_context, enabled_events
        )
        if isinstance(started_or_blocked, ToolMessage):
            return started_or_blocked
        request = started_or_blocked
        started = time.perf_counter()
        result = await handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = _append_message_text(result, pre_tool_outcome.context, tool_call.id)
        result = self._maybe_post_tool_use(
            tool_call,
            hook_context,
            enabled_events,
            request.runtime.config,
            result,
            duration_ms,
        )
        return self._maybe_subagent_stop(
            tool_call, hook_context, enabled_events, request.runtime.config, result
        )

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Emit `Stop` when the agent reaches a natural end.

        Returns:
            Optional state update that may jump back to the model.
        """
        return self._after_agent(state, runtime)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async `Stop` emission; mirrors `after_agent`.

        Returns:
            Optional state update that may jump back to the model.
        """
        return self._after_agent(state, runtime)

    def _maybe_subagent_start(
        self,
        request: ToolCallRequest,
        tool_call: ToolCallData,
        hook_context: HookContext,
        enabled_events: _HookEventsForSnapshot | None,
    ) -> ToolCallRequest | ToolMessage:
        if tool_call.name != _TASK_TOOL_NAME or not _event_enabled(
            enabled_events, HookEvent.SUBAGENT_START
        ):
            return request
        agent = _task_agent_identity(tool_call)
        decision = _invoke_hook(
            hook_context,
            SubagentStartEvent(event=HookEvent.SUBAGENT_START, agent=agent),
            enabled_events=enabled_events,
            config=request.runtime.config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, SubagentStartDecision)
        if not decision.continue_processing:
            return _denied_tool_message(
                tool_call,
                PermissionEffect(
                    behavior="deny",
                    reason=decision.stop_reason or "Blocked by SubagentStart hook",
                ),
            )
        return _inject_subagent_start_context(request, decision)

    def _after_model(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any]:
        enabled_events = _hook_events_for_snapshot(runtime.context)
        if not _event_enabled(enabled_events, HookEvent.PRE_TOOL_USE):
            return {_PRE_TOOL_STATE_KEY: {}}
        message = _last_ai_message(state.get("messages", ()))
        if message is None:
            return {_PRE_TOOL_STATE_KEY: {}}
        hook_context = _build_hook_context(runtime.context, None, self._cwd)
        outcomes: dict[str, _PreToolState] = {}
        for model_tool_call in message.tool_calls:
            tool_call = _tool_call_data_from_call(
                model_tool_call,
                mcp_server=self._mcp_servers.get(
                    str(model_tool_call.get("name") or "")
                ),
            )
            decision = _invoke_hook(
                hook_context,
                PreToolUseEvent(event=HookEvent.PRE_TOOL_USE, call=tool_call),
                enabled_events=enabled_events,
                config=None,
                deadline=self._default_deadline,
            )
            decision = _require_decision(decision, PreToolUseDecision)
            behavior: PreToolBehavior = "none"
            reason: str | None = None
            permission = decision.permission
            if not decision.continue_processing or permission.behavior == "deny":
                behavior = "deny"
                reason = (
                    permission.reason
                    or decision.stop_reason
                    or "Blocked by PreToolUse hook"
                )
            elif permission.behavior == "ask":
                blocked = _ask_permission_via_hitl(tool_call, permission)
                if blocked is None:
                    behavior = "allow"
                else:
                    behavior = "deny"
                    blocked_content = blocked.content
                    reason = (
                        blocked_content
                        if isinstance(blocked_content, str)
                        else str(blocked_content)
                    )
            elif permission.behavior == "allow":
                behavior = "allow"
            outcomes[tool_call.id] = {
                "behavior": behavior,
                "reason": reason,
                "context": list(decision.context),
            }
        return {_PRE_TOOL_STATE_KEY: outcomes}

    def _maybe_post_tool_use(
        self,
        tool_call: ToolCallData,
        hook_context: HookContext,
        enabled_events: _HookEventsForSnapshot | None,
        config: Mapping[str, Any] | None,
        result: ToolMessage | Command[Any],
        duration_ms: int,
    ) -> ToolMessage | Command[Any]:
        if not _event_enabled(enabled_events, HookEvent.POST_TOOL_USE):
            return result
        if _tool_result_failed(result, tool_call.id):
            return result
        decision = _invoke_hook(
            hook_context,
            PostToolUseEvent(
                event=HookEvent.POST_TOOL_USE,
                call=tool_call,
                result=result,
                duration_ms=duration_ms,
            ),
            enabled_events=enabled_events,
            config=config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, PostToolUseDecision)
        return _apply_post_tool_use(result, decision, tool_call.id)

    def _maybe_subagent_stop(
        self,
        tool_call: ToolCallData,
        hook_context: HookContext,
        enabled_events: _HookEventsForSnapshot | None,
        config: Mapping[str, Any] | None,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if tool_call.name != _TASK_TOOL_NAME or not _event_enabled(
            enabled_events, HookEvent.SUBAGENT_STOP
        ):
            return result
        agent = _task_agent_identity(tool_call)
        decision = _invoke_hook(
            hook_context,
            SubagentStopEvent(
                event=HookEvent.SUBAGENT_STOP,
                agent=agent,
                continuation_count=0,
                last_assistant_message=_tool_result_text(result, tool_call.id),
            ),
            enabled_events=enabled_events,
            config=config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, SubagentStopDecision)
        return _apply_subagent_stop(result, decision, tool_call.id)

    def _after_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        if not self._emit_stop:
            return None
        enabled_events = _hook_events_for_snapshot(runtime.context)
        if not _event_enabled(enabled_events, HookEvent.STOP):
            return None
        continuation = int(state.get(_STOP_STATE_KEY, 0) or 0)
        hook_context = _build_hook_context(runtime.context, None, self._cwd)
        decision = _invoke_hook(
            hook_context,
            StopEvent(
                event=HookEvent.STOP,
                continuation_count=continuation,
                last_assistant_message=_last_assistant_text(state.get("messages", ())),
            ),
            enabled_events=enabled_events,
            config=None,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, StopDecision)
        if not decision.continue_processing or not decision.continue_loop:
            # Reset so a later independent turn does not inherit the count.
            if continuation:
                return {_STOP_STATE_KEY: 0}
            return None
        feedback = "\n".join(decision.feedback).strip() or (
            decision.stop_reason or "Continue working."
        )
        return {
            "messages": [HumanMessage(content=feedback)],
            "jump_to": "model",
            _STOP_STATE_KEY: continuation + 1,
        }


_DecisionT = TypeVar("_DecisionT", bound=BaseHookDecision)


def _require_decision(
    decision: HookDecision,
    expected: type[_DecisionT],
) -> _DecisionT:
    if not isinstance(decision, expected):
        msg = f"Expected {expected.__name__}, got {type(decision).__name__}"
        raise TypeError(msg)
    return decision


def _hook_events_for_snapshot(runtime_context: object) -> _HookEventsForSnapshot | None:
    fields = _context_mapping(runtime_context)
    snapshot_id = fields.get("hooks_snapshot_id")
    events = fields.get("hooks_server_events")
    if snapshot_id is None or not events:
        return None
    return {
        "snapshot_id": snapshot_id,
        "events": frozenset(events),
    }


def _event_enabled(
    enabled_events: _HookEventsForSnapshot | None, event: HookEvent
) -> bool:
    return enabled_events is not None and event.value in enabled_events["events"]


def pre_tool_behavior(state: object, tool_call_id: str) -> PreToolBehavior | None:
    """Return the replayed PreToolUse permission behavior for one call."""
    outcome = _pre_tool_state(state, tool_call_id)
    if outcome is None:
        return None
    behavior = outcome.get("behavior")
    if behavior == "allow":
        return "allow"
    if behavior == "deny":
        return "deny"
    if behavior == "none":
        return "none"
    return None


def _pre_tool_state(state: object, tool_call_id: str) -> Mapping[str, object] | None:
    if not isinstance(state, Mapping):
        return None
    raw = state.get(_PRE_TOOL_STATE_KEY)
    if not isinstance(raw, Mapping):
        return None
    outcome = raw.get(tool_call_id)
    if not isinstance(outcome, Mapping):
        return None
    return {str(key): value for key, value in outcome.items()}


def _pre_tool_outcome(state: object, tool_call: ToolCallData) -> _PreToolOutcome:
    outcome = _pre_tool_state(state, tool_call.id)
    if outcome is None:
        return _PreToolOutcome()
    raw_context = outcome.get("context")
    extra_context = (
        tuple(item for item in raw_context if isinstance(item, str))
        if isinstance(raw_context, Sequence) and not isinstance(raw_context, str)
        else ()
    )
    if outcome.get("behavior") != "deny":
        return _PreToolOutcome(context=extra_context)
    raw_reason = outcome.get("reason")
    reason = raw_reason if isinstance(raw_reason, str) else None
    return _PreToolOutcome(
        blocked=_denied_tool_message(
            tool_call,
            PermissionEffect(behavior="deny", reason=reason),
        ),
        context=extra_context,
    )


def _invoke_hook(
    hook_context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    *,
    enabled_events: _HookEventsForSnapshot | None,
    config: Mapping[str, Any] | None,
    deadline: timedelta,
) -> HookDecision:
    if enabled_events is None:
        msg = "hooks_snapshot_id is required to emit server-owned hook events"
        raise RuntimeError(msg)
    run_id = _run_id(config, hook_context.thread_id)
    invocation_id = _invocation_id(
        run_id=run_id,
        snapshot_id=enabled_events["snapshot_id"],
        hook_context=hook_context,
        event=event,
    )
    request = HookInvocationRequest(
        protocol_version=1,
        invocation_id=invocation_id,
        snapshot_id=enabled_events["snapshot_id"],
        run_id=run_id,
        invocation=HookInvocation(context=hook_context, event=event),
        deadline=datetime.now(UTC) + deadline,
    )
    raw = interrupt(build_hook_interrupt_payload(request))
    response = parse_hook_resume_value(
        raw,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
    )
    return response.decision


def _build_hook_context(
    runtime_context: object,
    config: Mapping[str, Any] | None,
    cwd: Path,
) -> HookContext:
    fields = _context_mapping(runtime_context)
    thread_id = fields.get("thread_id") or _config_thread_id(config) or "unknown"
    approval = coerce_approval_mode(fields.get("approval_mode", "manual"))
    prompt_raw = fields.get("prompt_id")
    prompt_id = UUID(prompt_raw) if prompt_raw else None
    return HookContext(
        thread_id=thread_id,
        cwd=cwd,
        prompt_id=prompt_id,
        approval_mode=(
            approval if isinstance(approval, ApprovalMode) else ApprovalMode.MANUAL
        ),
    )


def _context_mapping(runtime_context: object) -> _HooksContextFields:
    """Project LangGraph run context into the typed hook field subset.

    In-process graphs coerce `context=` into `CLIContextSchema`; RemoteGraph
    delivers a plain mapping. Both shapes are accepted here. Values that are
    missing or mistyped are omitted so callers can rely on the TypedDict
    without re-checking field types.

    Returns:
        Only the recognized hook context fields with validated value types.
    """
    if runtime_context is None:
        return {}

    raw: dict[str, object]
    if isinstance(runtime_context, Mapping):
        raw = {
            key: value
            for key, value in runtime_context.items()
            if isinstance(key, str) and key in _HOOKS_CONTEXT_KEYS
        }
    else:
        raw = {}
        for key in _HOOKS_CONTEXT_KEYS:
            value = getattr(runtime_context, key, None)
            if value is not None:
                raw[key] = value

    result: _HooksContextFields = {}
    snapshot_id = raw.get("hooks_snapshot_id")
    if isinstance(snapshot_id, str) and snapshot_id:
        result["hooks_snapshot_id"] = snapshot_id
    events = raw.get("hooks_server_events")
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
        result["hooks_server_events"] = [str(item) for item in events]
    thread_id = raw.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        result["thread_id"] = thread_id
    approval_mode = raw.get("approval_mode")
    if isinstance(approval_mode, str) and approval_mode:
        result["approval_mode"] = approval_mode
    prompt_id = raw.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id:
        result["prompt_id"] = prompt_id
    return result


def _run_id(config: Mapping[str, Any] | None, thread_id: str) -> str:
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            for key in ("run_id", "thread_id"):
                value = configurable.get(key)
                if isinstance(value, UUID):
                    return str(value)
                if isinstance(value, str) and value:
                    return value
    return thread_id


def _invocation_id(
    *,
    run_id: str,
    snapshot_id: str,
    hook_context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
) -> UUID:
    identity = {
        "run_id": run_id,
        "thread_id": hook_context.thread_id,
        "snapshot_id": snapshot_id,
        "event": event.event.value,
        "logical_event": _logical_event_identity(hook_context, event),
    }
    return uuid5(
        _INVOCATION_NAMESPACE,
        json.dumps(identity, sort_keys=True, separators=(",", ":")),
    )


def _logical_event_identity(
    hook_context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
) -> str:
    if isinstance(event, PreToolUseEvent | PostToolUseEvent):
        return event.call.id
    if isinstance(event, SubagentStartEvent):
        return event.agent.id
    prompt_id = (
        str(hook_context.prompt_id) if hook_context.prompt_id is not None else ""
    )
    if isinstance(event, SubagentStopEvent):
        return f"{event.agent.id}:{event.continuation_count}:{prompt_id}"
    message_hash = hashlib.sha256(event.last_assistant_message.encode()).hexdigest()
    return f"{event.continuation_count}:{prompt_id}:{message_hash}"


def _config_thread_id(config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get("thread_id")
    return value if isinstance(value, str) and value else None


def _tool_call_data(request: ToolCallRequest) -> ToolCallData:
    return _tool_call_data_from_call(
        request.tool_call,
        mcp_server=_mcp_server_from_tool(request.tool),
    )


def _tool_call_data_from_call(
    tool_call: Mapping[str, object],
    *,
    mcp_server: str | None,
) -> ToolCallData:
    raw_args = tool_call.get("args")
    args: dict[str, Any]
    if isinstance(raw_args, dict):
        args = {str(key): value for key, value in raw_args.items()}
    else:
        args = {}
    return ToolCallData(
        id=str(tool_call.get("id") or ""),
        name=str(tool_call.get("name") or ""),
        args=cast("JsonObject", args),
        mcp_server=mcp_server,
    )


def _mcp_server_from_tool(tool: object | None) -> str | None:
    if tool is None:
        return None
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    for key in ("mcp_server", "mcp_server_name", "server_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _denied_tool_message(
    tool_call: ToolCallData,
    permission: PermissionEffect,
) -> ToolMessage:
    reason = permission.reason or "Blocked by PreToolUse hook"
    wire_name = to_wire_tool_name(tool_call.name, mcp_server=tool_call.mcp_server)
    return ToolMessage(
        content=f"{wire_name} blocked by hook: {reason}",
        name=tool_call.name,
        tool_call_id=tool_call.id,
        status="error",
    )


def _ask_permission_via_hitl(
    tool_call: ToolCallData,
    permission: PermissionEffect,
) -> ToolMessage | None:
    """Escalate PreToolUse `ask` through the existing HITL interrupt channel.

    Returns:
        A deny ToolMessage when the user rejects, otherwise `None` to proceed.
    """
    description = permission.reason or "PreToolUse hook requested approval"
    response = interrupt(
        HITLRequest(
            action_requests=[
                ActionRequest(
                    name=tool_call.name,
                    args=dict(tool_call.args),
                    description=description,
                )
            ],
            review_configs=[
                ReviewConfig(
                    action_name=tool_call.name,
                    allowed_decisions=["approve", "reject"],
                )
            ],
        )
    )
    decisions: Sequence[Any]
    if isinstance(response, Mapping):
        raw = response.get("decisions", ())
        decisions = raw if isinstance(raw, Sequence) else ()
    else:
        decisions = ()
    if not decisions:
        return _denied_tool_message(
            tool_call,
            PermissionEffect(
                behavior="deny",
                reason="PreToolUse ask was not answered",
            ),
        )
    first = decisions[0]
    decision_type = first.get("type") if isinstance(first, Mapping) else None
    if decision_type != "approve":
        reject_message = None
        if isinstance(first, Mapping):
            raw_message = first.get("message")
            if isinstance(raw_message, str) and raw_message:
                reject_message = raw_message
        return _denied_tool_message(
            tool_call,
            PermissionEffect(
                behavior="deny",
                reason=reject_message or description,
            ),
        )
    return None


def _append_message_text(
    result: ToolMessage | Command[Any],
    parts: Sequence[str],
    call_id: str,
) -> ToolMessage | Command[Any]:
    if not parts:
        return result
    return _append_tool_result_text(result, "\n".join(parts), call_id)


def _apply_post_tool_use(
    result: ToolMessage | Command[Any],
    decision: PostToolUseDecision,
    call_id: str,
) -> ToolMessage | Command[Any]:
    extras: list[str] = []
    if decision.feedback:
        extras.append("\n".join(decision.feedback))
    if decision.context:
        extras.append("\n".join(decision.context))
    if decision.stop_reason and not decision.continue_processing:
        extras.append(decision.stop_reason)
    if not extras:
        return result
    return _append_tool_result_text(
        result,
        "\n\n".join(part for part in extras if part),
        call_id,
    )


def _apply_subagent_stop(
    result: ToolMessage | Command[Any],
    decision: SubagentStopDecision,
    call_id: str,
) -> ToolMessage | Command[Any]:
    if not decision.context:
        return result
    return _append_tool_result_text(result, "\n".join(decision.context), call_id)


def _append_tool_result_text(
    result: ToolMessage | Command[Any],
    suffix: str,
    call_id: str,
) -> ToolMessage | Command[Any]:
    if isinstance(result, ToolMessage):
        return _merge_tool_message_content(result, suffix)
    update = result.update
    if not isinstance(update, Mapping):
        return result
    changed = False
    messages: list[object] = []
    for message in _command_messages(result):
        if _is_call_result(message, call_id):
            messages.append(_merge_tool_message_content(message, suffix))
            changed = True
        else:
            messages.append(message)
    if not changed:
        return result
    return replace(result, update={**update, "messages": messages})


def _tool_result_failed(result: ToolMessage | Command[Any], call_id: str) -> bool:
    if isinstance(result, ToolMessage):
        return result.status == "error"
    return any(
        _is_call_result(message, call_id) and message.status == "error"
        for message in _command_messages(result)
    )


def _command_messages(result: Command[Any]) -> Sequence[object]:
    """Return the `messages` list carried by a `Command` update.

    Returns:
        The update's messages, or an empty sequence when absent or malformed.
    """
    update = result.update
    if not isinstance(update, Mapping):
        return ()
    messages = update.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return ()
    return messages


def _is_call_result(message: object, call_id: str) -> TypeGuard[ToolMessage]:
    """Check whether a message is the `ToolMessage` for the in-flight call.

    A `Command` update may carry results for several calls, so hook context must
    only read from and write to the one this wrapper is handling.

    Returns:
        `True` when the message answers `call_id`.
    """
    return isinstance(message, ToolMessage) and message.tool_call_id == call_id


def _merge_tool_message_content(result: ToolMessage, suffix: str) -> ToolMessage:
    if not suffix:
        return result
    content = result.content
    if isinstance(content, str):
        merged = f"{content}\n\n{suffix}" if content else suffix
    # Preserve structured content blocks; append a text block.
    elif isinstance(content, list):
        merged = [*content, {"type": "text", "text": suffix}]
    else:
        merged = f"{content!s}\n\n{suffix}"
    return result.model_copy(update={"content": merged})


def _inject_subagent_start_context(
    request: ToolCallRequest,
    decision: SubagentStartDecision,
) -> ToolCallRequest:
    if not decision.context:
        return request

    original = request.tool_call
    raw_args = original.get("args")
    args: dict[str, Any]
    if isinstance(raw_args, dict):
        args = {str(key): value for key, value in raw_args.items()}
    else:
        args = {}
    description = args.get("description")
    prefix = "\n".join(decision.context)
    if isinstance(description, str) and description:
        args["description"] = f"{prefix}\n\n{description}"
    else:
        args["description"] = prefix
    tool_call = cast(
        "ToolCall",
        {
            "name": str(original.get("name") or ""),
            "args": args,
            "id": original.get("id"),
            "type": "tool_call",
        },
    )
    return request.override(tool_call=tool_call)


def _task_agent_identity(tool_call: ToolCallData) -> AgentIdentity:
    name = tool_call.args.get("subagent_type")
    if not isinstance(name, str) or not name:
        name = "unknown"
    return AgentIdentity(id=tool_call.id or name, name=name)


def _tool_result_text(result: ToolMessage | Command[Any], call_id: str) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        return content if isinstance(content, str) else str(content)
    return "\n".join(
        str(message.content)
        for message in _command_messages(result)
        if _is_call_result(message, call_id)
    )


def _last_ai_message(messages: Sequence[Any]) -> AIMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )


def _last_assistant_text(messages: Sequence[Any]) -> str:
    message = _last_ai_message(messages)
    if message is None:
        return ""
    content = message.content
    return content if isinstance(content, str) else str(content)
