"""Tests for the GLM-5.2 harness profile's behavioral middlewares.

Covers ``RambleMiddleware`` detection/boundary logic, ``FinalizeMiddleware``
turn-budget behavior, and that the built-in GLM-5.2 profile actually wires both
middlewares in via ``extra_middleware``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.profiles.harness._fireworks_glm_5p2_middleware import (
    _DEFAULT_RAMBLE_OUTPUT_TOKENS,
    FinalizeMiddleware,
    RambleMiddleware,
)
from deepagents.profiles.harness.harness_profiles import _get_harness_profile

if TYPE_CHECKING:
    import pytest

_GLM_5P2_SPEC = "fireworks:accounts/fireworks/models/glm-5p2"


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


def _ramble(mw: RambleMiddleware, **state: object) -> dict[str, Any] | None:
    """Invoke ``after_model`` with a minimal state dict (runtime is unused)."""
    return mw.after_model(state, None)  # type: ignore[arg-type]


# --- RambleMiddleware: detection logic -------------------------------------------


def test_truncated_no_toolcall_triggers_nudge() -> None:
    mw = RambleMiddleware()
    result = _ramble(mw, messages=[_ai("rambling...", finish_reason="length")])
    assert result is not None
    assert result["jump_to"] == "model"
    assert result["ramble_nudged"] is True
    [msg] = result["messages"]
    assert isinstance(msg, HumanMessage)
    assert "tool call" in msg.content


def test_large_output_no_toolcall_triggers_nudge() -> None:
    mw = RambleMiddleware(output_tokens=8000)
    result = _ramble(mw, messages=[_ai("x", output_tokens=8000)])
    assert result is not None
    assert result["jump_to"] == "model"


def test_message_with_tool_call_is_left_alone() -> None:
    # A huge output is fine as long as the model actually called a tool.
    mw = RambleMiddleware(output_tokens=10)
    tool_call = {"name": "ls", "args": {"path": "."}, "id": "1", "type": "tool_call"}
    result = _ramble(mw, messages=[_ai("", tool_calls=[tool_call], output_tokens=999_999)])
    assert result is None


def test_short_no_toolcall_message_does_not_misfire() -> None:
    # A brief no-tool-call "done" message must not be treated as rambling.
    mw = RambleMiddleware(output_tokens=8000)
    result = _ramble(mw, messages=[_ai("done", output_tokens=42)])
    assert result is None


def test_no_renudge_while_ramble_persists() -> None:
    # Already nudged this episode and still rambling back-to-back: no re-nudge (loop-safe).
    mw = RambleMiddleware()
    result = _ramble(
        mw,
        messages=[_ai("rambling...", finish_reason="length")],
        ramble_nudged=True,
    )
    assert result is None


def test_acting_turn_rearms_after_nudge() -> None:
    # After a nudge, a turn that actually acts (tool call) clears the flag so the NEXT
    # ramble episode is nudged again instead of being ignored for the rest of the run.
    mw = RambleMiddleware(output_tokens=10)
    tool_call = {"name": "ls", "args": {}, "id": "1", "type": "tool_call"}
    result = _ramble(
        mw,
        messages=[_ai("", tool_calls=[tool_call], output_tokens=5)],
        ramble_nudged=True,
    )
    assert result == {"ramble_nudged": False}


def test_non_ai_last_message_is_ignored() -> None:
    mw = RambleMiddleware(output_tokens=10)
    result = _ramble(mw, messages=[ToolMessage(content="x" * 9999, tool_call_id="1")])
    assert result is None


def test_empty_messages_ignored() -> None:
    mw = RambleMiddleware()
    assert _ramble(mw, messages=[]) is None
    assert _ramble(mw) is None


# --- RambleMiddleware: configurable boundary -------------------------------------


def test_boundary_editable_via_constructor() -> None:
    mw = RambleMiddleware(output_tokens=5000)
    assert _ramble(mw, messages=[_ai("x", output_tokens=4999)]) is None
    above = _ramble(mw, messages=[_ai("x", output_tokens=5000)])
    assert above is not None
    assert above["jump_to"] == "model"


def test_boundary_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_RAMBLE_OUTPUT_TOKENS", "3000")
    mw = RambleMiddleware()
    assert _ramble(mw, messages=[_ai("x", output_tokens=2999)]) is None
    assert _ramble(mw, messages=[_ai("x", output_tokens=3000)]) is not None


def test_invalid_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_RAMBLE_OUTPUT_TOKENS", "not-an-int")
    mw = RambleMiddleware()
    boundary = _DEFAULT_RAMBLE_OUTPUT_TOKENS
    assert _ramble(mw, messages=[_ai("x", output_tokens=boundary - 1)]) is None
    assert _ramble(mw, messages=[_ai("x", output_tokens=boundary)]) is not None


# --- FinalizeMiddleware: turn-budget behavior ------------------------------------


def test_finalize_counts_turns_below_soft() -> None:
    mw = FinalizeMiddleware(soft_turns=3, hard_turns=5)
    result = mw.before_model({}, None)  # type: ignore[arg-type]
    assert result == {"finalize_turns": 1}


def test_finalize_soft_nudge_fires_once() -> None:
    mw = FinalizeMiddleware(soft_turns=3, hard_turns=5)
    # Reaching the soft threshold injects a one-time HumanMessage nudge.
    result = mw.before_model({"finalize_turns": 2}, None)  # type: ignore[arg-type]
    assert result["finalize_nudged"] is True
    assert result["finalize_turns"] == 3
    [msg] = result["messages"]
    assert isinstance(msg, HumanMessage)
    # Already-nudged state does not re-nudge, just keeps counting.
    again = mw.before_model(  # type: ignore[arg-type]
        {"finalize_turns": 3, "finalize_nudged": True}, None
    )
    assert "messages" not in again
    assert again["finalize_turns"] == 4


def test_finalize_hard_cap_ends_run() -> None:
    mw = FinalizeMiddleware(soft_turns=3, hard_turns=5)
    result = mw.before_model({"finalize_turns": 4}, None)  # type: ignore[arg-type]
    assert result["jump_to"] == "end"
    [msg] = result["messages"]
    assert isinstance(msg, AIMessage)


def test_finalize_misconfigured_thresholds_fall_back() -> None:
    # soft >= hard is invalid; thresholds revert to module defaults (soft < hard).
    mw = FinalizeMiddleware(soft_turns=10, hard_turns=5)
    soft, hard = mw._thresholds()
    assert soft < hard


# --- profile wiring --------------------------------------------------------------


def test_glm_5p2_profile_wires_both_middlewares() -> None:
    """The built-in GLM-5.2 profile attaches Finalize + Ramble via extra_middleware."""
    profile = _get_harness_profile(_GLM_5P2_SPEC)
    assert profile is not None
    middlewares = profile.materialize_extra_middleware()
    assert [type(m) for m in middlewares] == [FinalizeMiddleware, RambleMiddleware]
    # Factory returns fresh instances each call (not shared across stacks).
    assert profile.materialize_extra_middleware()[0] is not middlewares[0]
