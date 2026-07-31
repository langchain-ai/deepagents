"""Server-owned Hooks v2 lifecycle middleware.

Emits `PreCompact`, `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart`, and
`SubagentStop` through the LangGraph interrupt channel so the client runtime can
execute matching handlers and return typed decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Annotated,
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
    PrivateStateAttr,
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
    CompactTrigger,
    HookContext,
    HookDecision,
    HookEvent,
    HookInvocation,
    PermissionEffect,
    PostToolUseDecision,
    PostToolUseEvent,
    PreCompactDecision,
    PreCompactEvent,
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
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages.tool import ToolCall
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from deepagents_code.json_types import JsonObject

_DEFAULT_DEADLINE = timedelta(seconds=600)
_STOP_STATE_KEY = "_hooks_stop_continuation_count"
_PRE_TOOL_STATE_KEY = "_hooks_pre_tool_outcomes"
_TASK_TOOL_NAME = "task"
_COMPACT_TOOL_NAME = "compact_conversation"
_INVOCATION_NAMESPACE = UUID("f2896d18-cf2a-4e7d-b11a-d5b10fc0e335")

PreToolBehavior: TypeAlias = Literal["allow", "deny", "none"]
_DEFAULT_DENY_REASON = "Blocked by PreToolUse hook"


class _PreToolDenied(TypedDict):
    """Outcome for a call a hook refused. A denial always carries a reason."""

    behavior: Literal["deny"]
    reason: str
    context: list[str]


class _PreToolPassed(TypedDict):
    """Outcome for a call a hook allowed or had no opinion on."""

    behavior: Literal["allow", "none"]
    context: list[str]


_PreToolState: TypeAlias = _PreToolDenied | _PreToolPassed


class ServerHooksState(AgentState[Any]):
    """Agent state extensions for server-owned hook middleware.

    Both fields are per-turn bookkeeping owned by `ServerHooksMiddleware` and
    marked `PrivateStateAttr`: they are omitted from the public graph I/O schema,
    and `SubAgentMiddleware` strips them from subagent result merges so parallel
    `task` calls cannot produce two concurrent writes to these `LastValue`
    channels.

    `PrivateStateAttr` only omits the fields from the input and output schemas;
    the channels stay ordinary checkpointed `LastValue` channels visible to every
    node, so values still flow from `after_model` to `wrap_tool_call` and survive
    interrupt/resume.

    Note:
        If either field ever needs a reducer, the reducer must be placed *after*
        `PrivateStateAttr` in the `Annotated` metadata. LangGraph only inspects
        the last metadata entry when detecting reducers, so a reducer added
        before the marker is silently ignored.
    """

    _hooks_stop_continuation_count: NotRequired[Annotated[int, PrivateStateAttr]]
    """Stop-hook continuations in the current turn; reset to 0 when the loop ends."""

    _hooks_pre_tool_outcomes: NotRequired[
        Annotated[dict[str, _PreToolState], PrivateStateAttr]
    ]
    """Pre-execution hook verdicts keyed by tool-call id.

    A full snapshot of the *current* turn's calls, not an accumulator: every
    `_after_model` replaces the whole dict (including with `{}`) so stale ids
    cannot survive into a later turn.
    """


class _SessionHookGate(TypedDict):
    snapshot_id: str
    events: frozenset[str]


@dataclass(slots=True)
class _PreToolOutcome:
    """Pre-execution gate result for the tool-call wrapper."""

    blocked: ToolMessage | None = None
    context: tuple[str, ...] = field(default_factory=tuple)


@contextmanager
def _subagent_transcript_config(
    call: ToolCallData,
    config: RunnableConfig,
) -> Iterator[None]:
    if call.name != _TASK_TOOL_NAME:
        yield
        return

    from langchain_core.runnables.config import var_child_runnable_config

    from deepagents_code.hooks.transcript import (
        SUBAGENT_TRANSCRIPT_ID_METADATA_KEY,
    )

    metadata = config.get("metadata")
    child_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    child_metadata[SUBAGENT_TRANSCRIPT_ID_METADATA_KEY] = call.id
    child_config: RunnableConfig = {**config, "metadata": child_metadata}
    token = var_child_runnable_config.set(child_config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


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
        """Run pre-execution hooks before downstream HITL middleware.

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
        gate = _session_gate(request.runtime.context)
        call = _tool_call_data(request)
        pre = _pre_tool_outcome(request.state, call)
        context = _hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        if pre.blocked is not None:
            return _append_message_text(pre.blocked, pre.context, call.id)
        started_or_blocked = self._maybe_subagent_start(request, call, context, gate)
        if isinstance(started_or_blocked, ToolMessage):
            return started_or_blocked
        request = started_or_blocked
        started = time.perf_counter()
        with _subagent_transcript_config(call, request.runtime.config):
            result = handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = _append_message_text(result, pre.context, call.id)
        result = self._maybe_post_tool_use(
            call, context, gate, request.runtime.config, result, duration_ms
        )
        return self._maybe_subagent_stop(
            call, context, gate, request.runtime.config, result
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
        gate = _session_gate(request.runtime.context)
        call = _tool_call_data(request)
        pre = _pre_tool_outcome(request.state, call)
        context = _hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        if pre.blocked is not None:
            return _append_message_text(pre.blocked, pre.context, call.id)
        started_or_blocked = self._maybe_subagent_start(request, call, context, gate)
        if isinstance(started_or_blocked, ToolMessage):
            return started_or_blocked
        request = started_or_blocked
        started = time.perf_counter()
        with _subagent_transcript_config(call, request.runtime.config):
            result = await handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = _append_message_text(result, pre.context, call.id)
        result = self._maybe_post_tool_use(
            call, context, gate, request.runtime.config, result, duration_ms
        )
        return self._maybe_subagent_stop(
            call, context, gate, request.runtime.config, result
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
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
    ) -> ToolCallRequest | ToolMessage:
        if call.name != _TASK_TOOL_NAME or not _event_enabled(
            gate, HookEvent.SUBAGENT_START
        ):
            return request
        agent = _task_agent_identity(call)
        decision = _invoke_hook(
            context,
            SubagentStartEvent(event=HookEvent.SUBAGENT_START, agent=agent),
            gate=gate,
            config=request.runtime.config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, SubagentStartDecision)
        if not decision.continue_processing:
            return _denied_tool_message(
                call,
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
        gate = _session_gate(runtime.context)
        precompact_enabled = _event_enabled(gate, HookEvent.PRE_COMPACT)
        pretool_enabled = _event_enabled(gate, HookEvent.PRE_TOOL_USE)
        if not precompact_enabled and not pretool_enabled:
            return {_PRE_TOOL_STATE_KEY: {}}
        message = _last_ai_message(state.get("messages", ()))
        if message is None:
            return {_PRE_TOOL_STATE_KEY: {}}
        context = _hook_context(runtime.context, None, self._cwd)
        outcomes: dict[str, _PreToolState] = {}
        for tool_call in message.tool_calls:
            call = _tool_call_data_from_call(
                tool_call,
                mcp_server=self._mcp_servers.get(str(tool_call.get("name") or "")),
            )
            behavior: PreToolBehavior = "none"
            reason: str | None = None
            hook_context: list[str] = []
            if precompact_enabled and call.name == _COMPACT_TOOL_NAME:
                trigger = (
                    CompactTrigger.MANUAL
                    if call.args.get("force") is True
                    else CompactTrigger.AUTO
                )
                compact = _invoke_hook(
                    context,
                    PreCompactEvent(event=HookEvent.PRE_COMPACT, trigger=trigger),
                    gate=gate,
                    config=None,
                    deadline=self._default_deadline,
                    logical_event_id=call.id,
                )
                compact = _require_decision(compact, PreCompactDecision)
                if not compact.continue_processing:
                    outcomes[call.id] = {
                        "behavior": "deny",
                        "reason": compact.stop_reason or "Blocked by PreCompact hook",
                        "context": hook_context,
                    }
                    continue
            if pretool_enabled:
                decision = _invoke_hook(
                    context,
                    PreToolUseEvent(event=HookEvent.PRE_TOOL_USE, call=call),
                    gate=gate,
                    config=None,
                    deadline=self._default_deadline,
                )
                decision = _require_decision(decision, PreToolUseDecision)
                permission = decision.permission
                hook_context.extend(decision.context)
                if not decision.continue_processing or permission.behavior == "deny":
                    behavior = "deny"
                    reason = (
                        permission.reason
                        or decision.stop_reason
                        or _DEFAULT_DENY_REASON
                    )
                elif permission.behavior == "ask":
                    blocked = _ask_permission_via_hitl(call, permission)
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
            if behavior == "deny":
                outcomes[call.id] = {
                    "behavior": "deny",
                    # Every deny path above resolves a reason; the guard keeps the
                    # "a denial always explains itself" invariant checkable here.
                    "reason": reason if reason is not None else _DEFAULT_DENY_REASON,
                    "context": hook_context,
                }
            else:
                outcomes[call.id] = {
                    "behavior": behavior,
                    "context": hook_context,
                }
        return {_PRE_TOOL_STATE_KEY: outcomes}

    def _maybe_post_tool_use(
        self,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
        config: Mapping[str, Any] | None,
        result: ToolMessage | Command[Any],
        duration_ms: int,
    ) -> ToolMessage | Command[Any]:
        if not _event_enabled(gate, HookEvent.POST_TOOL_USE):
            return result
        if _tool_result_failed(result, call.id):
            return result
        decision = _invoke_hook(
            context,
            PostToolUseEvent.from_tool_result(
                result,
                call=call,
                duration_ms=duration_ms,
            ),
            gate=gate,
            config=config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, PostToolUseDecision)
        return _apply_post_tool_use(result, decision, call.id)

    def _maybe_subagent_stop(
        self,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
        config: Mapping[str, Any] | None,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        if call.name != _TASK_TOOL_NAME or not _event_enabled(
            gate, HookEvent.SUBAGENT_STOP
        ):
            return result
        agent = _task_agent_identity(call)
        decision = _invoke_hook(
            context,
            SubagentStopEvent(
                event=HookEvent.SUBAGENT_STOP,
                agent=agent,
                continuation_count=0,
                last_assistant_message=_tool_result_text(result, call.id),
            ),
            gate=gate,
            config=config,
            deadline=self._default_deadline,
        )
        decision = _require_decision(decision, SubagentStopDecision)
        return _apply_subagent_stop(result, decision, call.id)

    def _after_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        if not self._emit_stop:
            return None
        gate = _session_gate(runtime.context)
        if not _event_enabled(gate, HookEvent.STOP):
            return None
        continuation = int(state.get(_STOP_STATE_KEY, 0) or 0)
        context = _hook_context(runtime.context, None, self._cwd)
        decision = _invoke_hook(
            context,
            StopEvent(
                event=HookEvent.STOP,
                continuation_count=continuation,
                last_assistant_message=_last_assistant_text(state.get("messages", ())),
            ),
            gate=gate,
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


def _session_gate(runtime_context: object) -> _SessionHookGate | None:
    fields = _context_mapping(runtime_context)
    snapshot_id = fields.get("hooks_snapshot_id")
    events = fields.get("hooks_server_events")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        return None
    if not isinstance(events, list) or not events:
        return None
    return {
        "snapshot_id": snapshot_id,
        "events": frozenset(str(item) for item in events),
    }


def _event_enabled(gate: _SessionHookGate | None, event: HookEvent) -> bool:
    return gate is not None and event.value in gate["events"]


def hook_decided_permission(state: object, tool_call_id: str) -> bool:
    """Report whether a pre-execution hook already settled permission for a call.

    Args:
        state: Agent state carrying the current turn's hook outcomes.
        tool_call_id: Tool call to look up.

    Returns:
        `True` when a hook explicitly allowed or denied the call, so stock
        approval flows must not prompt again. `False` when no hook ran, the hook
        expressed no opinion, or no outcome was recorded -- in every one of those
        cases normal approval still applies.
    """
    outcome = _pre_tool_state(state, tool_call_id)
    if outcome is None:
        return False
    return outcome.get("behavior") in {"allow", "deny"}


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


def _pre_tool_outcome(state: object, call: ToolCallData) -> _PreToolOutcome:
    outcome = _pre_tool_state(state, call.id)
    if outcome is None:
        return _PreToolOutcome()
    raw_context = outcome.get("context")
    context = (
        tuple(item for item in raw_context if isinstance(item, str))
        if isinstance(raw_context, Sequence) and not isinstance(raw_context, str)
        else ()
    )
    if outcome.get("behavior") != "deny":
        return _PreToolOutcome(context=context)
    raw_reason = outcome.get("reason")
    reason = raw_reason if isinstance(raw_reason, str) else None
    return _PreToolOutcome(
        blocked=_denied_tool_message(
            call,
            PermissionEffect(behavior="deny", reason=reason),
        ),
        context=context,
    )


def _invoke_hook(
    context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | PreCompactEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    *,
    gate: _SessionHookGate | None,
    config: Mapping[str, Any] | None,
    deadline: timedelta,
    logical_event_id: str | None = None,
) -> HookDecision:
    if gate is None:
        msg = "hooks_snapshot_id is required to emit server-owned hook events"
        raise RuntimeError(msg)
    run_id = _run_id(config, context.thread_id)
    invocation_id = _invocation_id(
        snapshot_id=gate["snapshot_id"],
        context=context,
        event=event,
        logical_event_id=logical_event_id,
    )
    request = HookInvocationRequest(
        protocol_version=1,
        invocation_id=invocation_id,
        snapshot_id=gate["snapshot_id"],
        run_id=run_id,
        invocation=HookInvocation(context=context, event=event),
        deadline=datetime.now(UTC) + deadline,
    )
    raw = interrupt(build_hook_interrupt_payload(request))
    response = parse_hook_resume_value(
        raw,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
    )
    return response.decision


def _hook_context(
    runtime_context: object,
    config: Mapping[str, Any] | None,
    cwd: Path,
) -> HookContext:
    fields = _context_mapping(runtime_context)
    thread_id = fields.get("thread_id") or _config_thread_id(config) or "unknown"
    if not isinstance(thread_id, str):
        thread_id = "unknown"
    approval = coerce_approval_mode(fields.get("approval_mode", "manual"))
    prompt_raw = fields.get("prompt_id")
    prompt_id = UUID(prompt_raw) if isinstance(prompt_raw, str) and prompt_raw else None
    return HookContext(
        thread_id=thread_id,
        cwd=cwd,
        prompt_id=prompt_id,
        approval_mode=(
            approval if isinstance(approval, ApprovalMode) else ApprovalMode.MANUAL
        ),
    )


def _context_mapping(runtime_context: object) -> dict[str, Any]:
    """Project LangGraph run context (dataclass or mapping) into a plain dict.

    In-process graphs coerce `context=` into `CLIContextSchema`; RemoteGraph
    delivers a plain mapping. Both shapes are accepted here.

    Returns:
        A shallow string-keyed dict of the hook-relevant context fields.
    """
    if runtime_context is None:
        return {}
    if isinstance(runtime_context, Mapping):
        return {str(key): value for key, value in runtime_context.items()}
    result: dict[str, Any] = {}
    for key in (
        "hooks_snapshot_id",
        "hooks_server_events",
        "thread_id",
        "approval_mode",
        "prompt_id",
    ):
        value = getattr(runtime_context, key, None)
        if value is not None:
            result[key] = value
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
    snapshot_id: str,
    context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | PreCompactEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    logical_event_id: str | None = None,
) -> UUID:
    identity = {
        "thread_id": context.thread_id,
        "snapshot_id": snapshot_id,
        "prompt_id": str(context.prompt_id) if context.prompt_id is not None else "",
        "event": event.event.value,
        "logical_event": _logical_event_identity(
            event,
            logical_event_id=logical_event_id,
        ),
    }
    return uuid5(
        _INVOCATION_NAMESPACE,
        json.dumps(identity, sort_keys=True, separators=(",", ":")),
    )


def _logical_event_identity(
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | PreCompactEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    *,
    logical_event_id: str | None = None,
) -> str:
    if isinstance(event, PreToolUseEvent | PostToolUseEvent):
        return event.call.id
    if isinstance(event, PreCompactEvent):
        if logical_event_id:
            return logical_event_id
        msg = "PreCompact requires a stable tool-call identity"
        raise ValueError(msg)
    if isinstance(event, SubagentStartEvent):
        return event.agent.id
    if isinstance(event, SubagentStopEvent):
        return f"{event.agent.id}:{event.continuation_count}"
    message_hash = hashlib.sha256(event.last_assistant_message.encode()).hexdigest()
    return f"{event.continuation_count}:{message_hash}"


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
    call: ToolCallData,
    permission: PermissionEffect,
) -> ToolMessage:
    reason = permission.reason or "Blocked by PreToolUse hook"
    wire_name = to_wire_tool_name(call.name, mcp_server=call.mcp_server)
    return ToolMessage(
        content=f"{wire_name} blocked by hook: {reason}",
        name=call.name,
        tool_call_id=call.id,
        status="error",
    )


def _ask_permission_via_hitl(
    call: ToolCallData,
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
                    name=call.name,
                    args=dict(call.args),
                    description=description,
                )
            ],
            review_configs=[
                ReviewConfig(
                    action_name=call.name,
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
            call,
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
            call,
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


def _task_agent_identity(call: ToolCallData) -> AgentIdentity:
    name = call.args.get("subagent_type")
    if not isinstance(name, str) or not name:
        name = "unknown"
    return AgentIdentity(id=call.id or name, name=name)


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
