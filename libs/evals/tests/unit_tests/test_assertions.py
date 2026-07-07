"""Deterministic unit tests for the tool-call trajectory assertions.

Covers the `ToolNotCalled` hard-fail assertion (the negation of `ToolCall`) and
the shared construction-time validation on both `ToolCall` and `ToolNotCalled`.
These run without a model against hand-built `AgentTrajectory` objects, mirroring
`test_external_benchmark_helpers.py`. They are the fast-suite guard for logic
that the eval tier only exercises behind `--model` + `LANGSMITH_TRACING`.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tests.evals.utils import (
    AgentStep,
    AgentTrajectory,
    ToolCall,
    ToolNotCalled,
    tool_call,
    tool_not_called,
)


def _step(index: int, *tool_calls: dict[str, object]) -> AgentStep:
    """Build a single agent step whose AI message emits the given tool calls."""
    return AgentStep(
        index=index,
        action=AIMessage(content="", tool_calls=list(tool_calls)),
        observations=[],
    )


def _tc(name: str, **args: object) -> dict[str, object]:
    """Build a normalized tool-call dict for an `AIMessage`."""
    return {"name": name, "args": dict(args), "id": name}


def _traj(*steps: AgentStep) -> AgentTrajectory:
    return AgentTrajectory(steps=list(steps), files={})


# ---------------------------------------------------------------------------
# ToolNotCalled — the behavior the eval tier depends on
# ---------------------------------------------------------------------------


class TestToolNotCalled:
    def test_absent_passes(self) -> None:
        traj = _traj(_step(1, _tc("lookup_population", city="tokyo")))
        assert tool_not_called("get_rubric").check(traj) is True

    def test_present_fails(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")))
        assert tool_not_called("get_rubric").check(traj) is False

    def test_describe_failure_names_tool_and_count(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")), _step(2, _tc("get_rubric")))
        msg = tool_not_called("get_rubric").describe_failure(traj)
        assert "get_rubric" in msg
        # Two forbidden calls were found; the count must surface.
        assert "2" in msg

    def test_step_scoped_match(self) -> None:
        traj = _traj(_step(1, _tc("lookup_population")), _step(2, _tc("get_rubric")))
        # Forbidden only in step 1 (where it is absent) → passes.
        assert tool_not_called("get_rubric", step=1).check(traj) is True
        # Forbidden in step 2 (where it is present) → fails.
        assert tool_not_called("get_rubric", step=2).check(traj) is False

    def test_step_out_of_range_passes(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")))
        assert tool_not_called("get_rubric", step=5).check(traj) is True

    def test_args_contains_narrows_the_forbidden_match(self) -> None:
        traj = _traj(_step(1, _tc("write_file", file_path="/keep.md")))
        # Same tool, different args → not the forbidden call → passes.
        assert (
            tool_not_called("write_file", args_contains={"file_path": "/secret.md"}).check(traj)
            is True
        )
        # Matching args → the forbidden call is present → fails.
        assert (
            tool_not_called("write_file", args_contains={"file_path": "/keep.md"}).check(traj)
            is False
        )

    def test_factory_equals_class(self) -> None:
        assert tool_not_called("get_goal", step=2) == ToolNotCalled(name="get_goal", step=2)


# ---------------------------------------------------------------------------
# Shared selector validation (fail fast at construction)
# ---------------------------------------------------------------------------


class TestSelectorValidation:
    @pytest.mark.parametrize("bad_step", [0, -1])
    def test_tool_not_called_nonpositive_step_raises(self, bad_step: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            tool_not_called("get_rubric", step=bad_step)

    def test_tool_not_called_both_arg_filters_raise(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            tool_not_called("write_file", args_contains={"a": 1}, args_equals={"a": 1})

    @pytest.mark.parametrize("bad_step", [0, -1])
    def test_tool_call_nonpositive_step_raises(self, bad_step: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            tool_call(name="write_file", step=bad_step)

    def test_tool_call_both_arg_filters_raise(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ToolCall(name="write_file", args_contains={"a": 1}, args_equals={"a": 1})
