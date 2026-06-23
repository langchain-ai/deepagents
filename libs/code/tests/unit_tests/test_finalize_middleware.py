"""Unit tests for `FinalizeMiddleware` (step-budget finalize nudge + graceful end)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.finalize_middleware import (
    _DEFAULT_HARD_STEPS_LEFT,
    _DEFAULT_SOFT_STEPS_LEFT,
    FinalizeMiddleware,
)

if TYPE_CHECKING:
    import pytest


def _call(mw: FinalizeMiddleware, **state: object) -> dict[str, Any] | None:
    """Invoke ``before_model`` with a minimal state dict (runtime is unused)."""
    return mw.before_model(state, None)  # type: ignore[arg-type]


def test_no_action_above_soft_threshold() -> None:
    mw = FinalizeMiddleware(soft_steps_left=120, hard_steps_left=40)
    assert _call(mw, remaining_steps=500) is None


def test_missing_remaining_steps_is_noop() -> None:
    mw = FinalizeMiddleware(soft_steps_left=120, hard_steps_left=40)
    assert _call(mw) is None


def test_soft_threshold_injects_one_nudge_and_continues() -> None:
    mw = FinalizeMiddleware(soft_steps_left=120, hard_steps_left=40)

    result = _call(mw, remaining_steps=100)

    assert result is not None
    assert "jump_to" not in result
    assert result["finalize_nudged"] is True
    [msg] = result["messages"]
    assert isinstance(msg, HumanMessage)
    assert "best-effort" in msg.content


def test_soft_nudge_fires_only_once() -> None:
    mw = FinalizeMiddleware(soft_steps_left=120, hard_steps_left=40)
    # Once nudged, a later call still in the soft band does nothing.
    assert _call(mw, remaining_steps=90, finalize_nudged=True) is None


def test_hard_threshold_jumps_to_end() -> None:
    mw = FinalizeMiddleware(soft_steps_left=120, hard_steps_left=40)

    result = _call(mw, remaining_steps=30, finalize_nudged=True)

    assert result is not None
    assert result["jump_to"] == "end"
    [msg] = result["messages"]
    assert isinstance(msg, AIMessage)


def test_hard_threshold_wins_even_if_not_yet_nudged() -> None:
    mw = FinalizeMiddleware(soft_steps_left=120, hard_steps_left=40)
    result = _call(mw, remaining_steps=10)
    assert result is not None
    assert result["jump_to"] == "end"


def test_env_overrides_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_SOFT_STEPS_LEFT", "300")
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_HARD_STEPS_LEFT", "100")
    mw = FinalizeMiddleware()

    # 250 is below the env soft (300) but above env hard (100) -> nudge.
    nudge = _call(mw, remaining_steps=250)
    assert nudge is not None
    assert "jump_to" not in nudge
    # 80 is below env hard (100) -> end.
    end = _call(mw, remaining_steps=80, finalize_nudged=True)
    assert end is not None
    assert end["jump_to"] == "end"


def test_invalid_env_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_FINALIZE_SOFT_STEPS_LEFT", "not-an-int")
    monkeypatch.delenv("DEEPAGENTS_FINALIZE_HARD_STEPS_LEFT", raising=False)
    mw = FinalizeMiddleware()

    # Just inside the default soft band, above default hard -> nudge.
    result = _call(mw, remaining_steps=_DEFAULT_SOFT_STEPS_LEFT - 1)
    assert result is not None
    assert "jump_to" not in result


def test_misconfigured_thresholds_fall_back_to_defaults() -> None:
    # hard >= soft is invalid; thresholds revert to module defaults.
    mw = FinalizeMiddleware(soft_steps_left=10, hard_steps_left=50)

    nudge = _call(mw, remaining_steps=_DEFAULT_SOFT_STEPS_LEFT - 1)
    assert nudge is not None
    end = _call(
        mw, remaining_steps=_DEFAULT_HARD_STEPS_LEFT - 1, finalize_nudged=True
    )
    assert end is not None
    assert end["jump_to"] == "end"
