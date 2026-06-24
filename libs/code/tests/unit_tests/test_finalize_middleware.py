"""Tests for `FinalizeMiddleware` (turn-budget finalize nudge + graceful end)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_code.agent import create_cli_agent
from deepagents_code.finalize_middleware import (
    _DEFAULT_HARD_TURNS,
    _DEFAULT_SOFT_TURNS,
    _HARD_TEXT,
    _NUDGE_TEXT,
    FinalizeMiddleware,
)

# Reuse the fake-model harness from the end-to-end suite (same test package).
from .test_end_to_end import FixedGenericFakeChatModel, mock_settings

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _call(mw: FinalizeMiddleware, **state: object) -> dict[str, Any]:
    """Invoke ``before_model`` with a minimal state dict (runtime is unused)."""
    return mw.before_model(state, None)  # type: ignore[arg-type]


# --- threshold logic (now over the real `finalize_turns` field) ------------------


def test_below_soft_threshold_only_counts() -> None:
    mw = FinalizeMiddleware(soft_turns=36, hard_turns=42)
    result = _call(mw, finalize_turns=10)
    assert result == {"finalize_turns": 11}


def test_first_turn_only_counts() -> None:
    mw = FinalizeMiddleware(soft_turns=36, hard_turns=42)
    assert _call(mw) == {"finalize_turns": 1}


def test_soft_threshold_injects_one_nudge_and_continues() -> None:
    mw = FinalizeMiddleware(soft_turns=36, hard_turns=42)

    result = _call(mw, finalize_turns=35)

    assert "jump_to" not in result
    assert result["finalize_nudged"] is True
    assert result["finalize_turns"] == 36
    [msg] = result["messages"]
    assert isinstance(msg, HumanMessage)
    assert "best-effort" in msg.content


def test_soft_nudge_fires_only_once() -> None:
    mw = FinalizeMiddleware(soft_turns=36, hard_turns=42)
    # Already nudged and still in the soft band -> just keep counting.
    result = _call(mw, finalize_turns=36, finalize_nudged=True)
    assert result == {"finalize_turns": 37}


def test_hard_threshold_jumps_to_end() -> None:
    mw = FinalizeMiddleware(soft_turns=36, hard_turns=42)

    result = _call(mw, finalize_turns=41, finalize_nudged=True)

    assert result["jump_to"] == "end"
    assert result["finalize_turns"] == 42
    [msg] = result["messages"]
    assert isinstance(msg, AIMessage)


def test_hard_threshold_wins_even_if_not_yet_nudged() -> None:
    mw = FinalizeMiddleware(soft_turns=36, hard_turns=42)
    result = _call(mw, finalize_turns=50)
    assert result["jump_to"] == "end"


def test_env_overrides_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_SOFT_TURNS", "5")
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_HARD_TURNS", "10")
    mw = FinalizeMiddleware()

    # turns=5 hits env soft (5), below env hard (10) -> nudge.
    nudge = _call(mw, finalize_turns=4)
    assert "jump_to" not in nudge
    assert nudge["finalize_nudged"] is True
    # turns=10 hits env hard (10) -> end.
    end = _call(mw, finalize_turns=9, finalize_nudged=True)
    assert end["jump_to"] == "end"


def test_invalid_env_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_SOFT_TURNS", "not-an-int")
    monkeypatch.delenv("DEEPAGENTS_FINALIZE_HARD_TURNS", raising=False)
    mw = FinalizeMiddleware()

    # Just reaches the default soft band, below default hard -> nudge.
    result = _call(mw, finalize_turns=_DEFAULT_SOFT_TURNS - 1)
    assert "jump_to" not in result
    assert result["finalize_nudged"] is True


def test_misconfigured_thresholds_fall_back_to_defaults() -> None:
    # soft >= hard is invalid; thresholds revert to module defaults.
    mw = FinalizeMiddleware(soft_turns=50, hard_turns=10)

    nudge = _call(mw, finalize_turns=_DEFAULT_SOFT_TURNS - 1)
    assert "jump_to" not in nudge
    assert nudge["finalize_nudged"] is True
    end = _call(mw, finalize_turns=_DEFAULT_HARD_TURNS - 1, finalize_nudged=True)
    assert end["jump_to"] == "end"


# --- integration: the counter must actually persist + fire in a real graph -------


def _looping_ls_messages() -> Any:
    """Infinite fake-model stream that always calls `ls` (agent never self-stops)."""
    i = 0
    while True:
        i += 1
        yield AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ls",
                    "args": {"path": "."},
                    "id": f"call_{i}",
                    "type": "tool_call",
                }
            ],
        )


def test_finalize_ends_runaway_agent_in_real_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a looping agent is gracefully ended by the turn budget.

    This exercises the real compiled graph (not a mocked state dict), so it would
    fail against the old `remaining_steps` implementation, which never populated.
    """
    # Small budget so the test is fast; the middleware reads these at run time.
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_SOFT_TURNS", "2")
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_HARD_TURNS", "3")
    with mock_settings(tmp_path):
        model = FixedGenericFakeChatModel(messages=_looping_ls_messages())
        agent, _ = create_cli_agent(
            model=model,
            assistant_id="test-agent",
            tools=[],
            enable_finalize=True,
            checkpointer=InMemorySaver(),
        )

        # recursion_limit well above the ~3-turn budget so that, if the budget
        # failed to fire, the run would crash with GraphRecursionError instead.
        result = agent.invoke(
            {"messages": [HumanMessage(content="loop forever")]},
            {"configurable": {"thread_id": "finalize-1"}, "recursion_limit": 200},
        )

    messages = result["messages"]
    # The one-time finalize nudge was injected...
    assert any(
        isinstance(m, HumanMessage) and _NUDGE_TEXT in m.content for m in messages
    )
    # ...and the run ended gracefully on the hard stop, not a recursion crash.
    assert any(
        isinstance(m, AIMessage) and _HARD_TEXT in (m.content or "") for m in messages
    )
