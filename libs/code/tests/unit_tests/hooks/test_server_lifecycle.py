"""Unit tests for Hooks v2 server-owned lifecycle integration."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from deepagents import create_deep_agent
from deepagents.middleware import CompiledSubAgent, SubAgent
from deepagents.middleware._state import private_state_field_names
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.agent import _should_interrupt_tool_call, create_cli_agent
from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.client import fulfill_hook_invocation
from deepagents_code.hooks.context import apply_hooks_context
from deepagents_code.hooks.interrupt import (
    HOOK_INVOCATION_INTERRUPT_TYPE,
    build_hook_interrupt_payload,
    build_hook_resume_value,
    is_hook_interrupt_payload,
    parse_hook_interrupt_payload,
    parse_hook_resume_value,
)
from deepagents_code.hooks.models.adapters import HOOKS_CONFIG_ADAPTER
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import (
    CompactTrigger,
    HookContext,
    HookEvent,
    HookInvocation,
    PermissionEffect,
    PostToolUseDecision,
    PreCompactDecision,
    PreCompactEvent,
    PreToolUseDecision,
    PreToolUseEvent,
    StopDecision,
    SubagentStopDecision,
    ToolCallData,
)
from deepagents_code.hooks.models.transport import (
    HookInvocationRequest,
    HookInvocationResponse,
)
from deepagents_code.hooks.presenter import HookPresenter
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.hooks.server_middleware import (
    ServerHooksMiddleware,
    ServerHooksState,
    _append_message_text,
    _append_tool_result_text,
    _apply_post_tool_use,
    _apply_subagent_stop,
    _ask_permission_via_hitl,
    _denied_tool_message,
    _invocation_id,
    _invoke_hook,
    _merge_tool_message_content,
    _session_gate,
    _tool_result_failed,
    _tool_result_text,
)
from deepagents_code.hooks.snapshot import HooksSnapshot
from deepagents_code.hooks.transcript import SUBAGENT_TRANSCRIPT_ID_METADATA_KEY

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool

    from deepagents_code._cli_context import CLIContext


class _ReplayState(BaseModel):
    completed: bool


class _ToolCallingFakeChatModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        _ = tools, tool_choice, kwargs
        return self


class _PublicHookState(AgentState[Any]):
    """Mirror of `ServerHooksState`'s keys *without* the privacy marker.

    Used only by the `CompiledSubAgent` fixtures below, which need to write the
    hook keys from outside the middleware. `_hook_state_keys` checks this mirror
    against `ServerHooksState` so it cannot silently drift.
    """

    _hooks_stop_continuation_count: NotRequired[int]
    _hooks_pre_tool_outcomes: NotRequired[dict[str, Any]]


def _hook_state_keys() -> frozenset[str]:
    """Return the private hook keys, asserting the local mirror matches."""
    private_fields = private_state_field_names(ServerHooksState)
    hook_keys = frozenset(name for name in private_fields if name.startswith("_hooks_"))
    mirrored = frozenset(_PublicHookState.__annotations__) & hook_keys
    assert mirrored == hook_keys, (
        f"_PublicHookState is missing hook keys {sorted(hook_keys - mirrored)}; "
        "update the fixture when ServerHooksState gains a private field."
    )
    return hook_keys


def _hook_state_subagent(*, name: str, content: str) -> CompiledSubAgent:
    """Subagent that writes both hook keys directly, as a bare compiled runnable.

    `CompiledSubAgent` is the weaker of the two outbound layers: a raw `SubAgent`
    also gets filtered by its own graph's output schema, so only this shape
    exercises `SubAgentMiddleware`'s explicit `private_state_keys` strip.
    """

    def finish(_state: _PublicHookState) -> dict[str, Any]:
        return {
            "_hooks_stop_continuation_count": 1,
            "_hooks_pre_tool_outcomes": {name: {"behavior": "none", "context": []}},
            "messages": [AIMessage(content=content)],
        }

    return CompiledSubAgent(
        name=name,
        description=f"Return {content}.",
        runnable=RunnableLambda(finish),
    )


def _real_hook_subagent(*, name: str, content: str, cwd: Path) -> SubAgent:
    """Subagent built the way production builds them, with its own hook middleware.

    This is the shape that actually triggered the reported crash: every real
    subagent carries `ServerHooksMiddleware`, whose `_after_model` writes
    `_hooks_pre_tool_outcomes` unconditionally -- even with no hooks configured --
    so two parallel `task` calls both write that channel in one step.
    """
    middleware: list[AgentMiddleware[Any, Any]] = [
        ServerHooksMiddleware(cwd=cwd, emit_stop=False)
    ]
    return SubAgent(
        name=name,
        description=f"Return {content}.",
        system_prompt=f"Say {content}.",
        model=_ToolCallingFakeChatModel(
            messages=iter([AIMessage(content=content)]),
        ),
        middleware=middleware,
    )


def test_server_hook_state_fields_are_private() -> None:
    private_fields = private_state_field_names(ServerHooksState)

    assert "_hooks_pre_tool_outcomes" in private_fields
    assert "_hooks_stop_continuation_count" in private_fields
    # `private_state_field_names` skips schemas whose annotations cannot be
    # resolved, which would silently return an empty set and revert the fix.
    assert _hook_state_keys() == {
        "_hooks_pre_tool_outcomes",
        "_hooks_stop_continuation_count",
    }


def test_task_omits_private_server_hook_state_from_subagent_update(
    tmp_path: Path,
) -> None:
    """A single `task` must not clobber the parent's hook state.

    One subagent cannot trip `InvalidUpdateError`, so this covers the silent half
    of the bug. Built through `create_deep_agent` so the private-key derivation in
    `deepagents.graph` is exercised rather than reimplemented.
    """
    model = _ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Run the child",
                                "subagent_type": "child",
                            },
                            "id": "call-child",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="parent complete"),
            ]
        )
    )
    agent = create_deep_agent(
        model=model,
        middleware=[ServerHooksMiddleware(cwd=tmp_path)],
        subagents=[_hook_state_subagent(name="child", content="child complete")],
    )

    result = agent.invoke({"messages": [HumanMessage(content="delegate")]})

    assert "_hooks_pre_tool_outcomes" not in result
    assert "_hooks_stop_continuation_count" not in result
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "child complete"


@pytest.mark.parametrize("subagent_kind", ["compiled", "real"])
def test_parallel_tasks_do_not_merge_subagent_server_hook_state(
    tmp_path: Path,
    subagent_kind: str,
) -> None:
    """Two `task` calls completing in one step must not both write hook channels.

    Covers both subagent shapes: `compiled` writes the keys by hand and exercises
    `SubAgentMiddleware`'s strip, while `real` carries its own
    `ServerHooksMiddleware` and reproduces the production trigger
    (`_hooks_pre_tool_outcomes`, written unconditionally by `_after_model`).
    """
    model = _ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Run the first child",
                                "subagent_type": "first",
                            },
                            "id": "call-first",
                            "type": "tool_call",
                        },
                        {
                            "name": "task",
                            "args": {
                                "description": "Run the second child",
                                "subagent_type": "second",
                            },
                            "id": "call-second",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="parent complete"),
            ]
        )
    )
    subagents: list[Any] = (
        [
            _hook_state_subagent(name="first", content="first complete"),
            _hook_state_subagent(name="second", content="second complete"),
        ]
        if subagent_kind == "compiled"
        else [
            _real_hook_subagent(name="first", content="first complete", cwd=tmp_path),
            _real_hook_subagent(name="second", content="second complete", cwd=tmp_path),
        ]
    )
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        model=model,
        middleware=[ServerHooksMiddleware(cwd=tmp_path)],
        subagents=subagents,
        checkpointer=checkpointer,
    )
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    result = agent.invoke(
        {"messages": [HumanMessage(content="run both children")]},
        config=config,
    )

    tool_messages = {
        message.tool_call_id: message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert tool_messages == {
        "call-first": "first complete",
        "call-second": "second complete",
    }
    state = agent.get_state(config).values
    assert "first" not in state.get("_hooks_pre_tool_outcomes", {})
    assert "second" not in state.get("_hooks_pre_tool_outcomes", {})
    assert "_hooks_stop_continuation_count" not in state


@pytest.mark.parametrize("resume_round_trip", [False, True])
def test_pretool_deny_blocks_tool_through_real_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume_round_trip: bool,
) -> None:
    """A `deny` must survive the real node-to-node channel and block the tool.

    The other deny tests call `_after_model`/`wrap_tool_call` directly and copy the
    update between them by hand, so none of them would notice if the outcome stopped
    reaching the tools node. This drives a compiled graph instead, which is what
    marking the state private could plausibly have broken.
    """
    executed: list[str] = []

    @tool
    def danger(target: str) -> str:
        """Do something that hooks should be able to block."""
        executed.append(target)
        return f"ran on {target}"

    model = _ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "danger",
                            "args": {"target": "prod"},
                            "id": "call-danger",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="stopped"),
            ]
        )
    )

    def _deny(*_args: object, **_kwargs: object) -> PreToolUseDecision:
        return PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="deny", reason="blocked by policy"),
        )

    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        _deny,
    )
    agent = create_agent(
        model=model,
        tools=[danger],
        middleware=[ServerHooksMiddleware(cwd=tmp_path)],
        context_schema=CLIContextSchema,
        checkpointer=InMemorySaver(),
    )
    context = CLIContextSchema(
        hooks_snapshot_id="snap",
        hooks_server_events=[HookEvent.PRE_TOOL_USE.value],
        thread_id="t1",
        approval_mode=ApprovalMode.MANUAL.value,
    )
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}

    if resume_round_trip:
        # Prove the outcome survives a checkpoint round trip, not just one step.
        agent.invoke(
            {"messages": [HumanMessage(content="go")]},
            config=config,
            context=context,
        )
        result = agent.invoke(None, config=config, context=context)
    else:
        result = agent.invoke(
            {"messages": [HumanMessage(content="go")]},
            config=config,
            context=context,
        )

    assert executed == []
    denied = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(denied) == 1
    assert denied[0].status == "error"
    assert "blocked by policy" in str(denied[0].content)


def _request(event: PreToolUseEvent | None = None) -> HookInvocationRequest:
    invocation = HookInvocation(
        context=HookContext(
            thread_id="thread-1",
            cwd=Path("/tmp"),
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=event
        or PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-1", name="execute", args={"command": "ls"}),
        ),
    )
    return HookInvocationRequest(
        protocol_version=1,
        invocation_id=uuid4(),
        snapshot_id="snapshot-1",
        run_id="run-1",
        invocation=invocation,
        deadline=datetime(2026, 7, 23, tzinfo=UTC),
    )


def test_hook_interrupt_payload_round_trip() -> None:
    request = _request()
    payload = build_hook_interrupt_payload(request)

    assert payload["type"] == HOOK_INVOCATION_INTERRUPT_TYPE
    assert is_hook_interrupt_payload(payload)
    assert parse_hook_interrupt_payload(payload) == request
    assert parse_hook_interrupt_payload({"type": "ask_user"}) is None


def test_hook_resume_value_validates_identity() -> None:
    request = _request()
    response = HookInvocationResponse(
        protocol_version=1,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
        decision=PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="allow"),
        ),
    )
    resume = build_hook_resume_value(response)
    parsed = parse_hook_resume_value(
        resume,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
    )
    assert parsed == response

    with pytest.raises(ValueError, match="invocation_id mismatch"):
        parse_hook_resume_value(
            resume,
            invocation_id=uuid4(),
            snapshot_id=request.snapshot_id,
        )


def test_real_checkpointer_resume_replays_stable_hook_identity() -> None:
    context = HookContext(
        thread_id="thread-1",
        cwd=Path("/tmp"),
        approval_mode=ApprovalMode.MANUAL,
        prompt_id=uuid4(),
    )
    event = PreToolUseEvent(
        event=HookEvent.PRE_TOOL_USE,
        call=ToolCallData(id="call-1", name="execute", args={"command": "ls"}),
    )
    gate = _session_gate(
        {
            "hooks_snapshot_id": "snapshot-1",
            "hooks_server_events": [HookEvent.PRE_TOOL_USE.value],
        }
    )
    assert gate is not None

    def invoke_hook(state: _ReplayState) -> dict[str, bool]:
        assert state.completed is False
        decision = _invoke_hook(
            context,
            event,
            gate=gate,
            config={"configurable": {"thread_id": "thread-1"}},
            deadline=timedelta(minutes=1),
        )
        assert isinstance(decision, PreToolUseDecision)
        return {"completed": decision.permission.behavior == "allow"}

    builder = StateGraph(_ReplayState)
    builder.add_node("hook", invoke_hook)
    builder.add_edge(START, "hook")
    graph = builder.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}

    interrupted = graph.invoke(_ReplayState(completed=False), config)
    pending = interrupted["__interrupt__"][0]
    request = parse_hook_interrupt_payload(pending.value)
    assert request is not None
    response = HookInvocationResponse(
        protocol_version=1,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
        decision=PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="allow"),
        ),
    )

    resumed = graph.invoke(Command(resume=build_hook_resume_value(response)), config)

    assert resumed["completed"] is True


def test_invocation_id_separates_turns_that_reuse_a_tool_call_id() -> None:
    """A tool-call id reused by a later turn must not inherit its decision.

    The fulfillment ledger caches completed responses by
    `(snapshot_id, invocation_id)`, so colliding ids would replay the earlier
    allow/block without running the hook.
    """
    event = PreToolUseEvent(
        event=HookEvent.PRE_TOOL_USE,
        call=ToolCallData(id="call-1", name="execute", args={"command": "ls"}),
    )

    def context_for_turn(prompt_id: UUID | None) -> HookContext:
        return HookContext(
            thread_id="thread-1",
            cwd=Path("/tmp"),
            approval_mode=ApprovalMode.MANUAL,
            prompt_id=prompt_id,
        )

    first_turn = context_for_turn(uuid4())
    second_turn = context_for_turn(uuid4())

    first = _invocation_id(snapshot_id="snapshot-1", context=first_turn, event=event)
    replayed = _invocation_id(snapshot_id="snapshot-1", context=first_turn, event=event)
    second = _invocation_id(snapshot_id="snapshot-1", context=second_turn, event=event)

    assert replayed == first
    assert second != first


def test_apply_hooks_context_sets_server_events(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hooks.json").write_text(
        (
            '{"hooks":{'
            '"PreCompact":[{"hooks":[{"type":"command","command":"true"}]}],'
            '"PreToolUse":[{"hooks":[{"type":"command","command":"true"}]}]'
            "}}"
        ),
        encoding="utf-8",
    )
    runtime = HooksRuntime.create(
        cwd=tmp_path,
        config_dir=config_dir,
        transcript_root=tmp_path / "transcripts",
    )
    context: CLIContext = {}
    apply_hooks_context(context, runtime, prompt_id="prompt-1")

    assert context["hooks_snapshot_id"] == runtime.snapshot_id
    assert context["hooks_server_events"] == ["PreCompact", "PreToolUse"]
    assert context["prompt_id"] == "prompt-1"
    assert runtime.configured_server_events() == ("PreCompact", "PreToolUse")


def test_session_gate_requires_snapshot_and_events() -> None:
    assert _session_gate(None) is None
    assert _session_gate({"hooks_snapshot_id": "abc"}) is None
    gate = _session_gate(
        {
            "hooks_snapshot_id": "abc",
            "hooks_server_events": ["PreToolUse", "Stop"],
        }
    )
    assert gate is not None
    assert gate["snapshot_id"] == "abc"
    assert gate["events"] == frozenset({"PreToolUse", "Stop"})


def test_denied_tool_message_for_deny() -> None:
    call = ToolCallData(id="c1", name="execute", args={})
    denied = _denied_tool_message(
        call, PermissionEffect(behavior="deny", reason="nope")
    )
    assert isinstance(denied, ToolMessage)
    assert denied.status == "error"
    assert "nope" in str(denied.content)


def test_merge_tool_message_preserves_structured_content() -> None:
    result = ToolMessage(
        content=[{"type": "text", "text": "parent result"}],
        tool_call_id="c1",
        name="task",
    )
    merged = _merge_tool_message_content(result, "hook context")
    assert isinstance(merged.content, list)
    assert merged.content[0] == {"type": "text", "text": "parent result"}
    assert merged.content[-1] == {"type": "text", "text": "hook context"}


def test_apply_subagent_stop_preserves_structured_content() -> None:
    result = ToolMessage(
        content=[{"type": "text", "text": "done"}],
        tool_call_id="c1",
        name="task",
    )
    updated = _apply_subagent_stop(
        result,
        SubagentStopDecision(
            event=HookEvent.SUBAGENT_STOP,
            context=["extra"],
        ),
        "c1",
    )
    assert isinstance(updated, ToolMessage)
    assert isinstance(updated.content, list)
    assert "extra" in str(updated.content[-1])


def test_apply_post_tool_use_appends_feedback_and_context() -> None:
    result = ToolMessage(content="ok", tool_call_id="c1", name="execute")
    updated = _apply_post_tool_use(
        result,
        PostToolUseDecision(
            event=HookEvent.POST_TOOL_USE,
            feedback=["fix it"],
            context=["note"],
        ),
        "c1",
    )
    assert "ok" in str(updated.content)
    assert "fix it" in str(updated.content)
    assert "note" in str(updated.content)


def test_post_tool_use_updates_successful_command_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    result = Command(
        update={
            "messages": [
                ToolMessage(
                    content="ok",
                    name="execute",
                    tool_call_id="c1",
                )
            ]
        }
    )
    invoke = MagicMock(
        return_value=PostToolUseDecision(
            event=HookEvent.POST_TOOL_USE,
            context=["post context"],
        )
    )
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        invoke,
    )

    updated = middleware._maybe_post_tool_use(
        ToolCallData(id="c1", name="execute", args={}),
        HookContext(
            thread_id="thread-1",
            cwd=Path("/tmp"),
            approval_mode=ApprovalMode.MANUAL,
        ),
        {"snapshot_id": "snap", "events": frozenset({"PostToolUse"})},
        {"configurable": {"thread_id": "thread-1"}},
        result,
        5,
    )

    assert isinstance(updated, Command)
    assert isinstance(updated.update, dict)
    message = updated.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert "post context" in str(message.content)
    invoke.assert_called_once()


def _multi_result_command() -> Command[Any]:
    return Command(
        update={
            "messages": [
                ToolMessage(content="mine", name="execute", tool_call_id="c1"),
                ToolMessage(
                    content="theirs",
                    name="execute",
                    tool_call_id="c2",
                    status="error",
                ),
            ]
        }
    )


def test_append_tool_result_text_only_touches_matching_call() -> None:
    updated = _append_tool_result_text(_multi_result_command(), "hook context", "c1")

    assert isinstance(updated, Command)
    assert isinstance(updated.update, dict)
    mine, theirs = updated.update["messages"]
    assert "hook context" in str(mine.content)
    assert str(theirs.content) == "theirs"


def test_append_tool_result_text_leaves_command_without_matching_call() -> None:
    result = _multi_result_command()

    assert _append_tool_result_text(result, "hook context", "c3") is result


def test_tool_result_text_reads_only_matching_call() -> None:
    assert _tool_result_text(_multi_result_command(), "c1") == "mine"


def test_tool_result_failed_ignores_unrelated_failure() -> None:
    result = _multi_result_command()

    assert _tool_result_failed(result, "c1") is False
    assert _tool_result_failed(result, "c2") is True


def test_post_tool_use_skips_failed_tool_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    result = ToolMessage(
        content="failed",
        name="execute",
        tool_call_id="c1",
        status="error",
    )
    invoke = MagicMock()
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        invoke,
    )

    updated = middleware._maybe_post_tool_use(
        ToolCallData(id="c1", name="execute", args={}),
        HookContext(
            thread_id="thread-1",
            cwd=Path("/tmp"),
            approval_mode=ApprovalMode.MANUAL,
        ),
        {"snapshot_id": "snap", "events": frozenset({"PostToolUse"})},
        {"configurable": {"thread_id": "thread-1"}},
        result,
        5,
    )

    assert updated is result
    invoke.assert_not_called()


def test_append_pretool_context_to_result() -> None:
    result = ToolMessage(content="ran", tool_call_id="c1", name="execute")
    updated = _append_message_text(result, ("pre context",), "c1")
    assert isinstance(updated, ToolMessage)
    assert "ran" in str(updated.content)
    assert "pre context" in str(updated.content)


def _pre_tool_runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.context = {
        "hooks_snapshot_id": "snap",
        "hooks_server_events": ["PreToolUse"],
        "thread_id": "thread-1",
        "approval_mode": "manual",
    }
    return runtime


def _pre_tool_state() -> ServerHooksState:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "ls"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def _tool_request(state: ServerHooksState, runtime: MagicMock) -> MagicMock:
    request = MagicMock()
    request.state = state
    request.runtime = runtime
    request.tool = None
    request.tool_call = {
        "name": "execute",
        "args": {"command": "ls"},
        "id": "call-1",
        "type": "tool_call",
    }
    return request


def test_pre_tool_allow_bypasses_hitl_and_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    runtime = _pre_tool_runtime()
    state = _pre_tool_state()
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        lambda *_args, **_kwargs: PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="allow"),
            context=["hook context"],
        ),
    )

    update = middleware._after_model(state, runtime)
    state["_hooks_pre_tool_outcomes"] = update["_hooks_pre_tool_outcomes"]
    request = _tool_request(state, runtime)
    handler = MagicMock(
        return_value=ToolMessage(
            content="ran",
            name="execute",
            tool_call_id="call-1",
        )
    )

    assert _should_interrupt_tool_call(request) is False
    result = middleware.wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert "hook context" in str(result.content)
    handler.assert_called_once_with(request)


def test_server_pre_tool_node_runs_before_stock_hitl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code._env_vars import EXPERIMENTAL

    monkeypatch.setenv(EXPERIMENTAL, "1")
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    model.profile = {"max_input_tokens": 200000}
    graph, _backend = create_cli_agent(
        model,
        "hooks-order-test",
        cwd=tmp_path,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
    )
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("model", "ServerHooksMiddleware.after_model") in edges
    assert (
        "ServerHooksMiddleware.after_model",
        "HumanInTheLoopMiddleware.after_model",
    ) in edges


def test_pre_tool_ask_reaches_hitl_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    runtime = _pre_tool_runtime()
    state = _pre_tool_state()
    order: list[str] = []

    def invoke(*_args: object, **_kwargs: object) -> PreToolUseDecision:
        order.append("hook")
        return PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="ask", reason="review"),
        )

    def ask(*_args: object, **_kwargs: object) -> None:
        order.append("hitl")

    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        invoke,
    )
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._ask_permission_via_hitl",
        ask,
    )

    update = middleware._after_model(state, runtime)
    state["_hooks_pre_tool_outcomes"] = update["_hooks_pre_tool_outcomes"]
    request = _tool_request(state, runtime)

    assert order == ["hook", "hitl"]
    assert _should_interrupt_tool_call(request) is False


def test_pre_tool_deny_skips_hitl_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    runtime = _pre_tool_runtime()
    state = _pre_tool_state()
    ask = MagicMock()
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        lambda *_args, **_kwargs: PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="deny", reason="blocked"),
        ),
    )
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._ask_permission_via_hitl",
        ask,
    )

    update = middleware._after_model(state, runtime)
    state["_hooks_pre_tool_outcomes"] = update["_hooks_pre_tool_outcomes"]
    request = _tool_request(state, runtime)
    handler = MagicMock()

    assert _should_interrupt_tool_call(request) is False
    result = middleware.wrap_tool_call(request, handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "blocked" in str(result.content)
    ask.assert_not_called()
    handler.assert_not_called()


def _compact_setup(
    *, args: dict[str, object] | None = None, events: list[str] | None = None
) -> tuple[MagicMock, ServerHooksState, dict[str, Any]]:
    runtime = MagicMock()
    runtime.context = {
        "hooks_snapshot_id": "snap",
        "hooks_server_events": events or ["PreCompact", "PreToolUse"],
        "thread_id": "t1",
        "approval_mode": "manual",
    }
    tool_call = {
        "name": "compact_conversation",
        "args": args or {},
        "id": "compact-call",
        "type": "tool_call",
    }
    return (
        runtime,
        {"messages": [AIMessage(content="", tool_calls=[tool_call])]},
        tool_call,
    )


@pytest.mark.parametrize(
    ("args", "trigger"),
    [({}, CompactTrigger.AUTO), ({"force": True}, CompactTrigger.MANUAL)],
)
def test_precompact_trigger_and_order(
    monkeypatch: pytest.MonkeyPatch,
    args: dict[str, object],
    trigger: CompactTrigger,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    runtime, state, _tool_call = _compact_setup(args=args)
    order: list[HookEvent] = []

    def invoke(
        _context: object, event: object, **kwargs: object
    ) -> PreCompactDecision | PreToolUseDecision:
        assert isinstance(event, PreCompactEvent | PreToolUseEvent)
        order.append(event.event)
        if isinstance(event, PreCompactEvent):
            assert event.trigger == trigger
            assert kwargs["logical_event_id"] == "compact-call"
            return PreCompactDecision(event=HookEvent.PRE_COMPACT)
        return PreToolUseDecision(
            event=HookEvent.PRE_TOOL_USE,
            permission=PermissionEffect(behavior="none"),
        )

    monkeypatch.setattr("deepagents_code.hooks.server_middleware._invoke_hook", invoke)
    update = middleware._after_model(state, runtime)

    assert order == [HookEvent.PRE_COMPACT, HookEvent.PRE_TOOL_USE]
    assert update["_hooks_pre_tool_outcomes"]["compact-call"]["behavior"] == "none"


def test_precompact_block_skips_hitl_and_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    runtime, state, tool_call = _compact_setup()
    runtime.config = {"configurable": {"thread_id": "t1"}}
    invoke = MagicMock(
        return_value=PreCompactDecision(
            event=HookEvent.PRE_COMPACT,
            continue_processing=False,
            stop_reason="keep full context",
        )
    )
    monkeypatch.setattr("deepagents_code.hooks.server_middleware._invoke_hook", invoke)
    state["_hooks_pre_tool_outcomes"] = middleware._after_model(state, runtime)[
        "_hooks_pre_tool_outcomes"
    ]
    request = MagicMock(state=state, runtime=runtime, tool=None, tool_call=tool_call)
    handler = MagicMock()

    assert _should_interrupt_tool_call(request) is False
    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "keep full context" in str(result.content)
    invoke.assert_called_once()
    handler.assert_not_called()


def test_ask_permission_via_hitl_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    call = ToolCallData(id="c1", name="execute", args={"command": "ls"})

    def _fake_interrupt(payload: object) -> dict[str, object]:
        assert isinstance(payload, dict)
        return {"decisions": [{"type": "approve"}]}

    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware.interrupt",
        _fake_interrupt,
    )
    assert (
        _ask_permission_via_hitl(call, PermissionEffect(behavior="ask", reason="sure?"))
        is None
    )


def test_ask_permission_via_hitl_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    call = ToolCallData(id="c1", name="execute", args={})

    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware.interrupt",
        lambda _payload: {"decisions": [{"type": "reject", "message": "no"}]},
    )
    blocked = _ask_permission_via_hitl(call, PermissionEffect(behavior="ask"))
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert "no" in str(blocked.content)


def test_stop_resets_continuation_count_when_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    state: ServerHooksState = {
        "messages": [],
        "_hooks_stop_continuation_count": 3,
    }
    runtime = MagicMock()
    runtime.context = {
        "hooks_snapshot_id": "snap",
        "hooks_server_events": ["Stop"],
        "thread_id": "t1",
        "approval_mode": "manual",
    }

    def _fake_invoke(*_args: object, **_kwargs: object) -> StopDecision:
        return StopDecision(event=HookEvent.STOP, continue_loop=False)

    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        _fake_invoke,
    )
    update = middleware._after_agent(state, runtime)
    assert update == {"_hooks_stop_continuation_count": 0}


def test_emit_stop_false_skips_after_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = ServerHooksMiddleware(cwd=Path("/tmp"), emit_stop=False)
    state: ServerHooksState = {"messages": []}
    runtime = MagicMock()
    runtime.context = {
        "hooks_snapshot_id": "snap",
        "hooks_server_events": ["Stop"],
        "thread_id": "t1",
        "approval_mode": "manual",
    }
    invoke = MagicMock()
    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        invoke,
    )
    assert middleware._after_agent(state, runtime) is None
    invoke.assert_not_called()


def test_subagent_start_deny_returns_error_tool_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.hooks.models.domain import SubagentStartDecision

    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    request = MagicMock()
    request.tool_call = {
        "name": "task",
        "args": {"subagent_type": "researcher", "description": "go"},
        "id": "call-1",
        "type": "tool_call",
    }
    request.tool = None
    request.runtime.context = {
        "hooks_snapshot_id": "snap",
        "hooks_server_events": ["SubagentStart"],
        "thread_id": "t1",
        "approval_mode": "manual",
    }
    request.runtime.config = {"configurable": {"thread_id": "t1"}}

    monkeypatch.setattr(
        "deepagents_code.hooks.server_middleware._invoke_hook",
        lambda *_args, **_kwargs: SubagentStartDecision(
            event=HookEvent.SUBAGENT_START,
            continue_processing=False,
            stop_reason="no subagents",
        ),
    )

    handler = MagicMock()
    blocked = middleware.wrap_tool_call(request, handler)
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert "no subagents" in str(blocked.content)
    handler.assert_not_called()


async def test_async_task_tool_scopes_subagent_transcript_identity() -> None:
    from langchain_core.runnables.config import ensure_config

    middleware = ServerHooksMiddleware(cwd=Path("/tmp"))
    request = MagicMock()
    request.state = {}
    request.tool = None
    request.tool_call = {
        "name": "task",
        "args": {"subagent_type": "researcher", "description": "go"},
        "id": "call-1",
        "type": "tool_call",
    }
    request.runtime.context = {
        "thread_id": "t1",
        "approval_mode": "manual",
    }
    request.runtime.config = {"configurable": {"thread_id": "t1"}}
    observed: list[str | None] = []

    async def handler(_request: object) -> ToolMessage:
        await asyncio.sleep(0)
        nested = ensure_config({"configurable": {"ls_agent_type": "subagent"}})
        observed.append(nested["metadata"].get(SUBAGENT_TRANSCRIPT_ID_METADATA_KEY))
        return ToolMessage(content="done", name="task", tool_call_id="call-1")

    result = await middleware.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert observed == ["call-1"]
    assert SUBAGENT_TRANSCRIPT_ID_METADATA_KEY not in ensure_config()["metadata"]


async def test_fulfill_hook_invocation_runs_engine(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hooks.json").write_text('{"hooks":{}}', encoding="utf-8")
    runtime = HooksRuntime.create(
        cwd=tmp_path,
        config_dir=config_dir,
        transcript_root=tmp_path / "transcripts",
    )
    request = _request()
    request = request.model_copy(update={"snapshot_id": runtime.snapshot_id})

    resume = await fulfill_hook_invocation(runtime, request)
    response = parse_hook_resume_value(
        resume,
        invocation_id=request.invocation_id,
        snapshot_id=runtime.snapshot_id,
    )
    assert isinstance(response.decision, PreToolUseDecision)
    assert response.decision.permission.behavior in {"allow", "none"}


async def test_fulfillment_is_idempotent_in_flight_and_after_completion(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    marker = tmp_path / "marker.txt"
    script = (
        "import json,pathlib,time; "
        f"pathlib.Path({str(marker)!r}).write_text('x'); "
        "time.sleep(0.05); "
        "print(json.dumps({'systemMessage':'once'}))"
    )
    (config_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        f"{sys.executable} -c {json.dumps(script)}"
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    notices: list[tuple[str, str]] = []
    runtime = HooksRuntime.create(
        cwd=tmp_path,
        config_dir=config_dir,
        presenter=HookPresenter(
            notice=lambda message, severity: notices.append((message, severity))
        ),
    )
    request = _request().model_copy(update={"snapshot_id": runtime.snapshot_id})

    first, second = await asyncio.gather(
        fulfill_hook_invocation(runtime, request),
        fulfill_hook_invocation(runtime, request),
    )
    third = await fulfill_hook_invocation(runtime, request)

    assert first == second == third
    assert marker.read_text() == "x"
    assert notices == [("once", "information")]


def test_snapshot_configured_server_events() -> None:
    config = HOOKS_CONFIG_ADAPTER.validate_python(
        {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "echo client"}]}
                ],
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "echo compact"}]}
                ],
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "echo server"}]}
                ],
            }
        }
    )
    assert isinstance(config, HooksConfig)
    snapshot = HooksSnapshot.from_config(config)
    assert snapshot.configured_events() == {
        HookEvent.SESSION_START,
        HookEvent.PRE_COMPACT,
        HookEvent.PRE_TOOL_USE,
    }
    assert snapshot.configured_server_events() == {
        HookEvent.PRE_COMPACT,
        HookEvent.PRE_TOOL_USE,
    }
