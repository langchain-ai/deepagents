from __future__ import annotations

from langchain_core.messages import AIMessage

from tests.evals.external_benchmarks import (
    _fix_bfcl_gt_call,
    _NormalizedSubstringsPresent,
)
from tests.evals.utils import AgentStep, AgentTrajectory


def _make_trajectory(answer: str) -> AgentTrajectory:
    """Build a minimal trajectory with the given final answer text."""
    return AgentTrajectory(
        steps=[AgentStep(index=1, action=AIMessage(content=answer), observations=[])],
        files={},
    )


# ---------------------------------------------------------------------------
# _NormalizedSubstringsPresent
# ---------------------------------------------------------------------------


class TestNormalizedSubstringsPresent:
    def test_matching_snippets(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=("hello", "world"))
        trajectory = _make_trajectory("Hello, World!")
        assert assertion.check(trajectory) is True

    def test_non_matching_snippet(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=("hello", "missing"))
        trajectory = _make_trajectory("Hello, World!")
        assert assertion.check(trajectory) is False

    def test_whitespace_normalization(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=("foobar",))
        trajectory = _make_trajectory("foo   bar")
        assert assertion.check(trajectory) is True

    def test_quote_stripping(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=("topics=[food]",))
        trajectory = _make_trajectory("topics=['food']")
        assert assertion.check(trajectory) is True

    def test_backtick_stripping(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=("myvar",))
        trajectory = _make_trajectory("`my_var`")
        # underscore is not stripped, but backticks are
        assert assertion.check(trajectory) is False

        assertion2 = _NormalizedSubstringsPresent(snippets=("myvar",))
        trajectory2 = _make_trajectory("`myvar`")
        assert assertion2.check(trajectory2) is True

    def test_empty_snippets_passes(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=())
        trajectory = _make_trajectory("anything")
        assert assertion.check(trajectory) is True

    def test_describe_failure(self) -> None:
        assertion = _NormalizedSubstringsPresent(snippets=("missing",))
        trajectory = _make_trajectory("answer text")
        msg = assertion.describe_failure(trajectory)
        assert "missing" in msg
        assert "answer text" in msg


# ---------------------------------------------------------------------------
# _fix_bfcl_gt_call
# ---------------------------------------------------------------------------


class TestFixBfclGtCall:
    def test_sender_id_middle(self) -> None:
        call = "send_message(sender_id='USR001', receiver_id='USR002', message='hi')"
        result = _fix_bfcl_gt_call(call)
        assert "sender_id" not in result
        assert "receiver_id='USR002'" in result
        assert "message='hi'" in result

    def test_sender_id_last(self) -> None:
        call = "send_message(receiver_id='USR002', message='hi', sender_id='USR001')"
        result = _fix_bfcl_gt_call(call)
        assert "sender_id" not in result
        assert "receiver_id='USR002'" in result

    def test_sender_id_only_is_noop(self) -> None:
        """sender_id as the sole argument is not stripped (never occurs in real data)."""
        call = "send_message(sender_id='USR001')"
        result = _fix_bfcl_gt_call(call)
        assert result == call

    def test_no_sender_id(self) -> None:
        call = "get_user_id(user='Alice')"
        result = _fix_bfcl_gt_call(call)
        assert result == call

    def test_double_quoted_sender_id(self) -> None:
        call = 'send_message(sender_id="USR001", receiver_id="USR002")'
        result = _fix_bfcl_gt_call(call)
        assert "sender_id" not in result
        assert 'receiver_id="USR002"' in result
