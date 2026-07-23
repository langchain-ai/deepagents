"""Server-owned Hooks v2 lifecycle middleware.

Emits `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart`, and `SubagentStop`
through the LangGraph interrupt channel so the client runtime can execute
matching handlers and return typed decisions.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NotRequired, cast
from uuid import UUID, uuid4

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
    from langgraph.runtime import Runtime

    from deepagents_code.json_types import JsonObject

_DEFAULT_DEADLINE = timedelta(seconds=600)
_STOP_STATE_KEY = "_hooks_stop_continuation_count"
_TASK_TOOL_NAME = "task"


class ServerHooksState(AgentState[Any]):
    """Agent state extensions for server-owned hook middleware."""

    _hooks_stop_continuation_count: NotRequired[int]


class _SessionHookGate(TypedDict):
    snapshot_id: str
    events: frozenset[str]


class ServerHooksMiddleware(AgentMiddleware[ServerHooksState, ContextT, ResponseT]):
    """Emit server-owned lifecycle events over the hook interrupt transport."""

    state_schema = ServerHooksState

    def __init__(
        self,
        *,
        cwd: Path,
        default_deadline: timedelta = _DEFAULT_DEADLINE,
    ) -> None:
        """Initialize middleware.

        Args:
            cwd: Session working directory projected into hook context.
            default_deadline: Client execution deadline attached to requests.
        """
        super().__init__()
        self._cwd = cwd
        self._default_deadline = default_deadline

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run Pre/Post tool hooks around a synchronous tool call.

        Returns:
            Tool result, possibly rewritten by hook decisions.
        """
        return self._run_tool_call(request, handler, async_handler=False)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run Pre/Post tool hooks around an asynchronous tool call.

        Returns:
            Tool result, possibly rewritten by hook decisions.
        """
        return await self._run_tool_call_async(request, handler)

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

    def _run_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        *,
        async_handler: bool,
    ) -> ToolMessage | Command[Any]:
        del async_handler
        gate = _session_gate(request.runtime.context)
        call = _tool_call_data(request)
        context = _hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        request = self._maybe_subagent_start(request, call, context, gate)
        denied = self._maybe_pre_tool_use(call, context, gate, request.runtime.config)
        if denied is not None:
            return denied
        started = time.perf_counter()
        result = handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = self._maybe_post_tool_use(
            call, context, gate, request.runtime.config, result, duration_ms
        )
        return self._maybe_subagent_stop(
            call, context, gate, request.runtime.config, result
        )

    async def _run_tool_call_async(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        gate = _session_gate(request.runtime.context)
        call = _tool_call_data(request)
        context = _hook_context(
            request.runtime.context, request.runtime.config, self._cwd
        )
        request = self._maybe_subagent_start(request, call, context, gate)
        denied = self._maybe_pre_tool_use(call, context, gate, request.runtime.config)
        if denied is not None:
            return denied
        started = time.perf_counter()
        result = await handler(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        result = self._maybe_post_tool_use(
            call, context, gate, request.runtime.config, result, duration_ms
        )
        return self._maybe_subagent_stop(
            call, context, gate, request.runtime.config, result
        )

    def _maybe_subagent_start(
        self,
        request: ToolCallRequest,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
    ) -> ToolCallRequest:
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
        if not isinstance(decision, SubagentStartDecision):
            msg = "Expected SubagentStartDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return _inject_subagent_start_context(request, decision)

    def _maybe_pre_tool_use(
        self,
        call: ToolCallData,
        context: HookContext,
        gate: _SessionHookGate | None,
        config: Mapping[str, Any] | None,
    ) -> ToolMessage | None:
        if not _event_enabled(gate, HookEvent.PRE_TOOL_USE):
            return None
        decision = _invoke_hook(
            context,
            PreToolUseEvent(event=HookEvent.PRE_TOOL_USE, call=call),
            gate=gate,
            config=config,
            deadline=self._default_deadline,
        )
        if not isinstance(decision, PreToolUseDecision):
            msg = "Expected PreToolUseDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return _denied_tool_message(call, decision.permission)

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
        if not isinstance(result, ToolMessage):
            return result
        decision = _invoke_hook(
            context,
            PostToolUseEvent(
                event=HookEvent.POST_TOOL_USE,
                call=call,
                result=result,
                duration_ms=duration_ms,
            ),
            gate=gate,
            config=config,
            deadline=self._default_deadline,
        )
        if not isinstance(decision, PostToolUseDecision):
            msg = "Expected PostToolUseDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return _apply_post_tool_use(result, decision)

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
                last_assistant_message=_tool_result_text(result),
            ),
            gate=gate,
            config=config,
            deadline=self._default_deadline,
        )
        if not isinstance(decision, SubagentStopDecision):
            msg = "Expected SubagentStopDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return _apply_subagent_stop(result, decision)

    def _after_agent(
        self,
        state: ServerHooksState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
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
        if not isinstance(decision, StopDecision):
            msg = "Expected StopDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        if not decision.continue_loop:
            return None
        feedback = "\n".join(decision.feedback).strip() or (
            decision.stop_reason or "Continue working."
        )
        return {
            "messages": [HumanMessage(content=feedback)],
            "jump_to": "model",
            _STOP_STATE_KEY: continuation + 1,
        }


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


def _invoke_hook(
    context: HookContext,
    event: (
        PreToolUseEvent
        | PostToolUseEvent
        | StopEvent
        | SubagentStartEvent
        | SubagentStopEvent
    ),
    *,
    gate: _SessionHookGate | None,
    config: Mapping[str, Any] | None,
    deadline: timedelta,
) -> HookDecision:
    if gate is None:
        msg = "hooks_snapshot_id is required to emit server-owned hook events"
        raise RuntimeError(msg)
    request = HookInvocationRequest(
        protocol_version=1,
        invocation_id=uuid4(),
        snapshot_id=gate["snapshot_id"],
        run_id=_run_id(config),
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


def _run_id(config: Mapping[str, Any] | None) -> str:
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            for key in ("run_id", "thread_id"):
                value = configurable.get(key)
                if isinstance(value, str) and value:
                    return value
    return str(uuid4())


def _config_thread_id(config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get("thread_id")
    return value if isinstance(value, str) and value else None


def _tool_call_data(request: ToolCallRequest) -> ToolCallData:
    tool_call = request.tool_call
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
        mcp_server=_mcp_server_from_tool(request.tool),
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
) -> ToolMessage | None:
    if permission.behavior not in {"deny", "ask"}:
        return None
    reason = permission.reason or (
        "Blocked by PreToolUse hook"
        if permission.behavior == "deny"
        else "PreToolUse hook requested approval (ask is not applied yet)"
    )
    wire_name = to_wire_tool_name(call.name, mcp_server=call.mcp_server)
    return ToolMessage(
        content=f"{wire_name} blocked by hook: {reason}",
        name=call.name,
        tool_call_id=call.id,
        status="error",
    )


def _apply_post_tool_use(
    result: ToolMessage,
    decision: PostToolUseDecision,
) -> ToolMessage:
    extras: list[str] = []
    if decision.feedback:
        extras.append("\n".join(decision.feedback))
    if decision.context:
        extras.append("\n".join(decision.context))
    if not extras:
        return result
    suffix = "\n\n".join(part for part in extras if part)
    content = result.content
    if isinstance(content, str):
        merged = f"{content}\n\n{suffix}" if content else suffix
    else:
        merged = f"{content!s}\n\n{suffix}"
    return result.model_copy(update={"content": merged})


def _apply_subagent_stop(
    result: ToolMessage | Command[Any],
    decision: SubagentStopDecision,
) -> ToolMessage | Command[Any]:
    if not decision.context or not isinstance(result, ToolMessage):
        return result
    suffix = "\n".join(decision.context)
    content = result.content
    merged = (
        f"{content}\n\n{suffix}" if isinstance(content, str) and content else suffix
    )
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


def _tool_result_text(result: ToolMessage | Command[Any]) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        return content if isinstance(content, str) else str(content)
    return ""


def _last_assistant_text(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""
