"""Tests for `RambleMiddleware` (nudge to act when the model rambles)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_code.agent import create_cli_agent
from deepagents_code.ramble_middleware import (
    _DEFAULT_RAMBLE_OUTPUT_TOKENS,
    _NUDGE_TEXT,
    RambleMiddleware,
)

# Reuse the fake-model harness from the end-to-end suite (same test package).
from .test_end_to_end import FixedGenericFakeChatModel, mock_settings  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pytest


def _ai(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    output_tokens: int | None = None,
) -> AIMessage:
    """Build an ``AIMessage`` with optional finish reason / output-token usage."""
    response_metadata = {"finish_reason": finish_reason} if finish_reason else {}
    usage_metadata = (
        {
            "input_tokens": 1,
            "output_tokens": output_tokens,
            "total_tokens": output_tokens + 1,
        }
        if output_tokens is not None
        else None
    )
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
    )


def _call(mw: RambleMiddleware, **state: object) -> dict[str, Any] | None:
    """Invoke ``after_model`` with a minimal state dict (runtime is unused)."""
    return mw.after_model(state, None)  # type: ignore[arg-type]


# --- detection logic -------------------------------------------------------------


def test_truncated_no_toolcall_triggers_nudge() -> None:
    mw = RambleMiddleware()
    result = _call(mw, messages=[_ai("rambling...", finish_reason="length")])
    assert result is not None
    assert result["jump_to"] == "model"
    assert result["ramble_nudged"] is True
    [msg] = result["messages"]
    assert isinstance(msg, HumanMessage)
    assert "tool call" in msg.content


def test_large_output_no_toolcall_triggers_nudge() -> None:
    mw = RambleMiddleware(output_tokens=8000)
    result = _call(mw, messages=[_ai("x", output_tokens=8000)])
    assert result is not None
    assert result["jump_to"] == "model"


def test_message_with_tool_call_is_left_alone() -> None:
    # A huge output is fine as long as the model actually called a tool.
    mw = RambleMiddleware(output_tokens=10)
    tool_call = {"name": "ls", "args": {"path": "."}, "id": "1", "type": "tool_call"}
    result = _call(
        mw, messages=[_ai("", tool_calls=[tool_call], output_tokens=999_999)]
    )
    assert result is None


def test_short_no_toolcall_message_does_not_misfire() -> None:
    # A brief no-tool-call "done" message must not be treated as rambling.
    mw = RambleMiddleware(output_tokens=8000)
    result = _call(mw, messages=[_ai("done", output_tokens=42)])
    assert result is None


def test_nudge_fires_only_once() -> None:
    mw = RambleMiddleware()
    result = _call(
        mw,
        messages=[_ai("rambling...", finish_reason="length")],
        ramble_nudged=True,
    )
    assert result is None


def test_non_ai_last_message_is_ignored() -> None:
    mw = RambleMiddleware(output_tokens=10)
    result = _call(mw, messages=[ToolMessage(content="x" * 9999, tool_call_id="1")])
    assert result is None


def test_empty_messages_ignored() -> None:
    mw = RambleMiddleware()
    assert _call(mw, messages=[]) is None
    assert _call(mw) is None


# --- configurable boundary -------------------------------------------------------


def test_boundary_editable_via_constructor() -> None:
    mw = RambleMiddleware(output_tokens=5000)
    assert _call(mw, messages=[_ai("x", output_tokens=4999)]) is None
    above = _call(mw, messages=[_ai("x", output_tokens=5000)])
    assert above is not None
    assert above["jump_to"] == "model"


def test_boundary_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_RAMBLE_OUTPUT_TOKENS", "3000")
    mw = RambleMiddleware()
    assert _call(mw, messages=[_ai("x", output_tokens=2999)]) is None
    assert _call(mw, messages=[_ai("x", output_tokens=3000)]) is not None


def test_invalid_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_RAMBLE_OUTPUT_TOKENS", "not-an-int")
    mw = RambleMiddleware()
    boundary = _DEFAULT_RAMBLE_OUTPUT_TOKENS
    assert _call(mw, messages=[_ai("x", output_tokens=boundary - 1)]) is None
    assert _call(mw, messages=[_ai("x", output_tokens=boundary)]) is not None


# --- integration: the nudge must actually re-invoke the model in a real graph ----


def _ramble_then_act_messages() -> Iterator[AIMessage]:
    """Fake-model stream: ramble once (no tool call), then act on the next call."""
    # 1) Long, length-truncated turn with no tool call -> should trigger the nudge.
    yield _ai(
        "thinking out loud " * 50,
        finish_reason="length",
        output_tokens=9000,
    )
    # 2) After the nudge loops back to the model, it calls a tool (proof of re-invoke).
    yield _ai(
        "",
        tool_calls=[
            {"name": "ls", "args": {"path": "."}, "id": "call_1", "type": "tool_call"}
        ],
    )
    # 3) Then finish cleanly (short, no tool call -> not a ramble).
    while True:
        yield _ai("done", output_tokens=5)


def test_ramble_nudge_reinvokes_model_in_real_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a rambling first turn is nudged and the model is re-invoked.

    Without the middleware, the no-tool-call first message would end the run with
    nothing on disk; with it, the agent is pushed back into the model node.
    """
    monkeypatch.delenv("DEEPAGENTS_RAMBLE_OUTPUT_TOKENS", raising=False)
    with mock_settings(tmp_path):
        model = FixedGenericFakeChatModel(messages=_ramble_then_act_messages())
        agent, _ = create_cli_agent(
            model=model,
            assistant_id="test-agent",
            tools=[],
            enable_anti_ramble=True,
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="go")]},
            {"configurable": {"thread_id": "ramble-1"}, "recursion_limit": 50},
        )

    messages = result["messages"]
    # The one-time nudge was injected...
    assert any(
        isinstance(m, HumanMessage) and _NUDGE_TEXT in m.content for m in messages
    )
    # ...and the model was re-invoked: the post-nudge tool call is present.
    assert any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
