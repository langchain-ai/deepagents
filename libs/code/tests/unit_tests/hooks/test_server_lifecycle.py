"""Unit tests for Hooks v2 server-owned lifecycle integration."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

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
    HookContext,
    HookEvent,
    HookInvocation,
    PermissionEffect,
    PostToolUseDecision,
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
    _hook_events_for_snapshot,
    _invoke_hook,
    _merge_tool_message_content,
    _tool_result_failed,
    _tool_result_text,
)
from deepagents_code.hooks.snapshot import HooksSnapshot

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from deepagents_code._cli_context import CLIContext


class _ReplayState(BaseModel):
    completed: bool


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
    )
    event = PreToolUseEvent(
        event=HookEvent.PRE_TOOL_USE,
        call=ToolCallData(id="call-1", name="execute", args={"command": "ls"}),
    )
    enabled_events = _hook_events_for_snapshot(
        {
            "hooks_snapshot_id": "snapshot-1",
            "hooks_server_events": [HookEvent.PRE_TOOL_USE.value],
        }
    )
    assert enabled_events is not None

    def invoke_hook(state: _ReplayState) -> dict[str, bool]:
        assert state.completed is False
        decision = _invoke_hook(
            context,
            event,
            enabled_events=enabled_events,
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


def test_apply_hooks_context_sets_server_events(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hooks.json").write_text(
        '{"hooks":{"PreToolUse":[{"hooks":[{"type":"command","command":"true"}]}]}}',
        encoding="utf-8",
    )
    runtime = HooksRuntime.create(cwd=tmp_path, config_dir=config_dir)
    context: CLIContext = {}
    apply_hooks_context(context, runtime, prompt_id="prompt-1")

    assert context["hooks_snapshot_id"] == runtime.snapshot_id
    assert context["hooks_server_events"] == ["PreToolUse"]
    assert context["prompt_id"] == "prompt-1"
    assert runtime.configured_server_events() == ("PreToolUse",)


def test_hook_events_for_snapshot_requires_snapshot_and_events() -> None:
    assert _hook_events_for_snapshot(None) is None
    assert _hook_events_for_snapshot({"hooks_snapshot_id": "abc"}) is None
    enabled_events = _hook_events_for_snapshot(
        {
            "hooks_snapshot_id": "abc",
            "hooks_server_events": ["PreToolUse", "Stop"],
        }
    )
    assert enabled_events is not None
    assert enabled_events["snapshot_id"] == "abc"
    assert enabled_events["events"] == frozenset({"PreToolUse", "Stop"})


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


def test_server_pre_tool_node_runs_before_stock_hitl(tmp_path: Path) -> None:
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


async def test_fulfill_hook_invocation_runs_engine(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hooks.json").write_text('{"hooks":{}}', encoding="utf-8")
    runtime = HooksRuntime.create(cwd=tmp_path, config_dir=config_dir)
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
    caplog: pytest.LogCaptureFixture,
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
    runtime = HooksRuntime.create(cwd=tmp_path, config_dir=config_dir)
    request = _request().model_copy(update={"snapshot_id": runtime.snapshot_id})

    with caplog.at_level("WARNING", logger="deepagents_code.hooks.client"):
        first, second = await asyncio.gather(
            fulfill_hook_invocation(runtime, request),
            fulfill_hook_invocation(runtime, request),
        )
        third = await fulfill_hook_invocation(runtime, request)

    assert first == second == third
    assert marker.read_text() == "x"
    assert [record.message for record in caplog.records].count(
        "Hook user notice: once"
    ) == 1


def test_snapshot_configured_server_events() -> None:
    config = HOOKS_CONFIG_ADAPTER.validate_python(
        {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "echo client"}]}
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
        HookEvent.PRE_TOOL_USE,
    }
    assert snapshot.configured_server_events() == {HookEvent.PRE_TOOL_USE}
