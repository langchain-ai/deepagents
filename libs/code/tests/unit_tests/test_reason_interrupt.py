"""Unit tests for `ReasonInterruptMiddleware`."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall

from deepagents_code.reason_interrupt import (
    _MAX_REASON_CHARS,
    ReasonInterruptMiddleware,
    _clean_reason,
)


def _model(text: str) -> GenericFakeChatModel:
    """Return a fake chat model that replies once with `text`."""
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def _tool_call(name: str = "execute", **args: object) -> ToolCall:
    """Build a `ToolCall` for tests."""
    return ToolCall(name=name, args=args or {"command": "ls"}, id="1", type="tool_call")


class TestReasonAttachment:
    """Tests for attaching reasons to action requests."""

    def test_reason_attached_for_configured_tool(self) -> None:
        """A tool with `reason: True` gets a model-generated reason."""
        mw = ReasonInterruptMiddleware(
            {"execute": {"allowed_decisions": ["approve", "reject"], "reason": True}},
            model=_model("Run the suite to validate the change."),
        )
        state = {"messages": [HumanMessage(content="run the tests")]}
        action, _ = mw._create_action_and_config(
            _tool_call(), mw.interrupt_on["execute"], state, None
        )
        assert action["reason"] == "Run the suite to validate the change."

    def test_no_reason_for_unflagged_tool(self) -> None:
        """A tool without `reason: True` gets no reason."""
        mw = ReasonInterruptMiddleware(
            {"write_file": {"allowed_decisions": ["approve", "reject"]}},
            model=_model("should not be used"),
        )
        state = {"messages": [HumanMessage(content="write it")]}
        action, _ = mw._create_action_and_config(
            _tool_call("write_file", file_path="x"),
            mw.interrupt_on["write_file"],
            state,
            None,
        )
        assert "reason" not in action

    def test_model_failure_does_not_block_approval(self) -> None:
        """A model error leaves the request without a reason rather than raising."""

        class _Boom(GenericFakeChatModel):
            def invoke(self, *args: object, **kwargs: object) -> AIMessage:  # noqa: ARG002
                msg = "model down"
                raise RuntimeError(msg)

        mw = ReasonInterruptMiddleware(
            {"execute": {"allowed_decisions": ["approve", "reject"], "reason": True}},
            model=_Boom(messages=iter([])),
        )
        state = {"messages": [HumanMessage(content="go")]}
        action, _ = mw._create_action_and_config(
            _tool_call(), mw.interrupt_on["execute"], state, None
        )
        assert "reason" not in action

    def test_empty_reason_is_omitted(self) -> None:
        """A blank model reply does not set a reason key."""
        mw = ReasonInterruptMiddleware(
            {"execute": {"allowed_decisions": ["approve", "reject"], "reason": True}},
            model=_model("   "),
        )
        state = {"messages": [HumanMessage(content="go")]}
        action, _ = mw._create_action_and_config(
            _tool_call(), mw.interrupt_on["execute"], state, None
        )
        assert "reason" not in action


class TestCleanReason:
    """Tests for `_clean_reason`."""

    def test_collapses_whitespace(self) -> None:
        """Newlines and runs of spaces collapse to single spaces."""
        assert _clean_reason(AIMessage(content="a\n  b   c")) == "a b c"

    def test_blank_returns_none(self) -> None:
        """Whitespace-only text returns `None`."""
        assert _clean_reason(AIMessage(content="  \n ")) is None

    def test_truncates_long_text(self) -> None:
        """Reasons longer than the cap are truncated with an ellipsis."""
        cleaned = _clean_reason(AIMessage(content="x" * (_MAX_REASON_CHARS + 50)))
        assert cleaned is not None
        assert len(cleaned) == _MAX_REASON_CHARS
        assert cleaned.endswith("\u2026")
