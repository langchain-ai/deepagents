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
    ToolCalled,
    ToolNotCalled,
    tool_call,
    tool_called,
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

    def test_step_out_of_range_fails(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")))
        assertion = tool_not_called("get_rubric", step=5)
        assert assertion.check(traj) is False
        assert "trajectory has 1 step" in assertion.describe_failure(traj)

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

    def test_args_contains_none_requires_the_key(self) -> None:
        """A missing arg must not match an arg explicitly set to `None`."""
        missing = _traj(_step(1, _tc("write_file")))
        explicit = _traj(_step(1, _tc("write_file", reason=None)))

        assertion = tool_not_called("write_file", args_contains={"reason": None})
        assert assertion.check(missing)
        assert not assertion.check(explicit)

    def test_args_equals_requires_exact_args(self) -> None:
        """`args_equals` matches only on a whole-dict exact match."""
        traj = _traj(_step(1, _tc("write_file", file_path="/a.md", mode="w")))
        # Exact match → the forbidden call is present → fails.
        assert (
            tool_not_called("write_file", args_equals={"file_path": "/a.md", "mode": "w"}).check(
                traj
            )
            is False
        )
        # A subset is not an exact match → not forbidden → passes. This is the
        # branch that distinguishes `args_equals` from `args_contains`.
        assert tool_not_called("write_file", args_equals={"file_path": "/a.md"}).check(traj) is True

    def test_describe_failure_names_the_scoped_step(self) -> None:
        """A step-scoped failure surfaces the step in its description."""
        traj = _traj(_step(1, _tc("lookup_population")), _step(2, _tc("get_rubric")))
        msg = tool_not_called("get_rubric", step=2).describe_failure(traj)
        assert "step 2" in msg

    def test_factory_equals_class(self) -> None:
        assert tool_not_called("get_goal", step=2) == ToolNotCalled(name="get_goal", step=2)


# ---------------------------------------------------------------------------
# ToolCalled — hard-fail presence assertion
# ---------------------------------------------------------------------------


class TestToolCalled:
    def test_present_passes(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")))
        assert tool_called("get_rubric").check(traj) is True

    def test_absent_fails(self) -> None:
        traj = _traj(_step(1, _tc("lookup_population")))
        assert tool_called("get_rubric").check(traj) is False

    def test_out_of_range_step_fails(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")))
        assert tool_called("get_rubric", step=2).check(traj) is False

    def test_step_and_args_matching(self) -> None:
        traj = _traj(
            _step(1, _tc("lookup_population", city="tokyo")),
            _step(2, _tc("lookup_population", city="delhi")),
        )
        assert tool_called(
            "lookup_population",
            step=2,
            args_contains={"city": "delhi"},
        ).check(traj)
        assert not tool_called(
            "lookup_population",
            step=1,
            args_equals={"city": "delhi"},
        ).check(traj)

    def test_describe_failure_names_tool_and_step(self) -> None:
        traj = _traj(_step(1, _tc("lookup_population")))
        message = tool_called("get_rubric", step=1).describe_failure(traj)
        assert "get_rubric" in message
        assert "step 1" in message

    def test_factory_equals_class(self) -> None:
        assert tool_called("get_goal", step=2) == ToolCalled(
            name="get_goal",
            step=2,
        )


# ---------------------------------------------------------------------------
# ToolCall — informational presence counterpart
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_present_true(self) -> None:
        traj = _traj(_step(1, _tc("get_rubric")))
        assert tool_call(name="get_rubric").check(traj) is True

    def test_absent_false(self) -> None:
        traj = _traj(_step(1, _tc("lookup_population")))
        assert tool_call(name="get_rubric").check(traj) is False

    def test_combined_arg_filters_preserve_existing_behavior(self) -> None:
        traj = _traj(_step(1, _tc("write_file", a=1, b=2)))
        assertion = ToolCall(
            name="write_file",
            args_contains={"a": 1},
            args_equals={"a": 1, "b": 2},
        )
        assert assertion.check(traj)


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
    def test_tool_called_nonpositive_step_raises(self, bad_step: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            tool_called("get_rubric", step=bad_step)

    def test_tool_called_both_arg_filters_raise(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ToolCalled(
                name="write_file",
                args_contains={"a": 1},
                args_equals={"a": 1},
            )

    @pytest.mark.parametrize("bad_step", [0, -1])
    def test_tool_call_nonpositive_step_raises(self, bad_step: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            tool_call(name="write_file", step=bad_step)
