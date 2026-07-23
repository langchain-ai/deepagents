"""Unit tests for Hooks v2 server-owned lifecycle integration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage

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
    _append_message_text,
    _apply_post_tool_use,
    _apply_subagent_stop,
    _ask_permission_via_hitl,
    _denied_tool_message,
    _merge_tool_message_content,
    _session_gate,
)
from deepagents_code.hooks.snapshot import HooksSnapshot

if TYPE_CHECKING:
    from deepagents_code._cli_context import CLIContext


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
    )
    assert "ok" in str(updated.content)
    assert "fix it" in str(updated.content)
    assert "note" in str(updated.content)


def test_append_pretool_context_to_result() -> None:
    result = ToolMessage(content="ran", tool_call_id="c1", name="execute")
    updated = _append_message_text(result, ("pre context",))
    assert isinstance(updated, ToolMessage)
    assert "ran" in str(updated.content)
    assert "pre context" in str(updated.content)


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
    state: dict[str, Any] = {
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
