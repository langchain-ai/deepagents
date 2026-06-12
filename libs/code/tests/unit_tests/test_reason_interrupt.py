"""Unit tests for `ReasonInterruptMiddleware`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolCall

from deepagents_code.reason_interrupt import (
    _MAX_REASON_CHARS,
    ReasonInterruptMiddleware,
    _clean_reason,
)

if TYPE_CHECKING:
    import pytest
    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
    from langchain.agents.middleware.types import AgentState
    from langchain_core.language_models import BaseChatModel
    from langgraph.runtime import Runtime


def _model(text: str) -> GenericFakeChatModel:
    """Return a fake chat model that replies once with `text`."""
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def _interrupt_on(
    *names: str, reason: bool = True
) -> dict[str, bool | InterruptOnConfig]:
    """Build an interrupt map with optional reason generation enabled."""
    config: dict[str, Any] = {"allowed_decisions": ["approve", "reject"]}
    if reason:
        config["reason"] = True
    return cast(
        "dict[str, bool | InterruptOnConfig]",
        {name: config.copy() for name in names},
    )


def _state(messages: list[Any]) -> AgentState[Any]:
    """Cast a minimal test state to the middleware state type."""
    return cast("AgentState[Any]", {"messages": messages})


def _runtime() -> Runtime[Any]:
    """Return a runtime placeholder for tests that do not use runtime fields."""
    return cast("Runtime[Any]", None)


def _tool_call(name: str = "execute", call_id: str = "1", **args: object) -> ToolCall:
    """Build a `ToolCall` for tests."""
    return ToolCall(
        name=name, args=args or {"command": "ls"}, id=call_id, type="tool_call"
    )


class _CountingModel:
    """Minimal model double that records reason-generation calls."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        """Return a fixed response while tracking the invocation."""
        self.calls += 1
        self.messages = messages
        self.kwargs = kwargs
        return AIMessage(content=self.text)


class TestReasonAttachment:
    """Tests for attaching reasons to action requests."""

    def test_reason_attached_for_configured_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool with `reason: True` gets a model-generated reason."""
        mw = ReasonInterruptMiddleware(
            _interrupt_on("execute"),
            model=_model("Run the suite to validate the change."),
        )
        state = _state(
            [
                HumanMessage(content="run the tests"),
                AIMessage(content="", tool_calls=[_tool_call()]),
            ]
        )
        captured: dict[str, Any] = {}

        def interrupt(request: dict[str, Any]) -> dict[str, Any]:
            captured["request"] = request
            return {"decisions": [{"type": "approve"}]}

        monkeypatch.setattr("deepagents_code.reason_interrupt.interrupt", interrupt)
        mw.after_model(state, _runtime())

        actions = captured["request"]["action_requests"]
        assert actions[0]["reason"] == "Run the suite to validate the change."

    def test_reason_generated_once_for_approval_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple reason-enabled tools share one reason-generation call."""
        model = _CountingModel("Inspect both risky actions before running them.")
        mw = ReasonInterruptMiddleware(
            _interrupt_on("execute", "task"),
            model=cast("BaseChatModel", model),
        )
        state = _state(
            [
                HumanMessage(content="run tests and inspect the issue"),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("execute", call_id="1", command="pytest"),
                        _tool_call("task", call_id="2", description="inspect"),
                    ],
                ),
            ]
        )
        captured: dict[str, Any] = {}

        def interrupt(request: dict[str, Any]) -> dict[str, Any]:
            captured["request"] = request
            return {"decisions": [{"type": "approve"}, {"type": "approve"}]}

        monkeypatch.setattr("deepagents_code.reason_interrupt.interrupt", interrupt)
        mw.after_model(state, _runtime())

        actions = captured["request"]["action_requests"]
        assert model.calls == 1
        assert actions[0]["reason"] == "Inspect both risky actions before running them."
        assert actions[1]["reason"] == "Inspect both risky actions before running them."

    def test_no_reason_for_unflagged_tool(self) -> None:
        """A tool without `reason: True` gets no reason."""
        mw = ReasonInterruptMiddleware(
            _interrupt_on("write_file", reason=False),
            model=_model("should not be used"),
        )
        state = _state([HumanMessage(content="write it")])
        action, _ = mw._create_action_and_config(
            _tool_call("write_file", file_path="x"),
            mw.interrupt_on["write_file"],
            state,
            _runtime(),
        )
        assert "reason" not in action

    def test_model_failure_does_not_block_approval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model error leaves the request without a reason rather than raising."""

        class _Boom(GenericFakeChatModel):
            def invoke(self, *args: object, **kwargs: object) -> AIMessage:  # noqa: ARG002
                msg = "model down"
                raise RuntimeError(msg)

        mw = ReasonInterruptMiddleware(
            _interrupt_on("execute"),
            model=_Boom(messages=iter([])),
        )
        state = _state(
            [
                HumanMessage(content="go"),
                AIMessage(content="", tool_calls=[_tool_call()]),
            ]
        )
        captured: dict[str, Any] = {}

        def interrupt(request: dict[str, Any]) -> dict[str, Any]:
            captured["request"] = request
            return {"decisions": [{"type": "approve"}]}

        monkeypatch.setattr("deepagents_code.reason_interrupt.interrupt", interrupt)
        mw.after_model(state, _runtime())

        actions = captured["request"]["action_requests"]
        assert "reason" not in actions[0]

    def test_empty_reason_is_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank model reply does not set a reason key."""
        mw = ReasonInterruptMiddleware(
            _interrupt_on("execute"),
            model=_model("   "),
        )
        state = _state(
            [
                HumanMessage(content="go"),
                AIMessage(content="", tool_calls=[_tool_call()]),
            ]
        )
        captured: dict[str, Any] = {}

        def interrupt(request: dict[str, Any]) -> dict[str, Any]:
            captured["request"] = request
            return {"decisions": [{"type": "approve"}]}

        monkeypatch.setattr("deepagents_code.reason_interrupt.interrupt", interrupt)
        mw.after_model(state, _runtime())

        actions = captured["request"]["action_requests"]
        assert "reason" not in actions[0]


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
