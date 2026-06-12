"""Unit tests for `ReasonInterruptMiddleware`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolCall

from deepagents_code.reason_interrupt import (
    _MAX_REASON_CHARS,
    _REASON_SYSTEM_PROMPT,
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

    def test_only_flagged_tool_in_mixed_batch_gets_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In a mixed batch, only the reason-flagged tool receives a reason."""
        interrupt_on = cast(
            "dict[str, bool | InterruptOnConfig]",
            {
                "execute": {"allowed_decisions": ["approve", "reject"], "reason": True},
                "write_file": {"allowed_decisions": ["approve", "reject"]},
            },
        )
        mw = ReasonInterruptMiddleware(
            interrupt_on, model=_model("Run the command to apply the change.")
        )
        state = _state(
            [
                HumanMessage(content="run it and save"),
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("execute", call_id="1", command="pytest"),
                        _tool_call("write_file", call_id="2", file_path="x"),
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
        # Action requests preserve tool-call order: execute first, write_file second.
        assert actions[0]["name"] == "execute"
        assert actions[0]["reason"] == "Run the command to apply the change."
        assert "reason" not in actions[1]


class TestBuildPrompt:
    """Tests for `ReasonInterruptMiddleware._build_prompt`."""

    def test_includes_system_prompt_and_most_recent_human(self) -> None:
        """The prompt leads with the system message and uses the latest human."""
        state = _state(
            [
                HumanMessage(content="first, stale request"),
                AIMessage(content="working on it"),
                HumanMessage(content="actually, run the tests"),
            ]
        )
        messages = ReasonInterruptMiddleware._build_prompt([_tool_call()], state)

        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == _REASON_SYSTEM_PROMPT
        joined = "\n".join(str(m.content) for m in messages)
        assert "actually, run the tests" in joined
        assert "first, stale request" not in joined

    def test_handles_state_without_human_message(self) -> None:
        """A state with no `HumanMessage` yields system + instruction, no raise."""
        state = _state([AIMessage(content="no human turn yet")])
        messages = ReasonInterruptMiddleware._build_prompt([_tool_call()], state)

        assert len(messages) == 2  # system prompt + final instruction
        assert isinstance(messages[0], SystemMessage)

    def test_lists_every_tool_call(self) -> None:
        """Each tool call in the batch appears in the final instruction."""
        state = _state([HumanMessage(content="go")])
        tool_calls = [
            _tool_call("execute", call_id="1", command="pytest"),
            _tool_call("task", call_id="2", description="inspect"),
        ]
        messages = ReasonInterruptMiddleware._build_prompt(tool_calls, state)

        final = str(messages[-1].content)
        assert "execute" in final
        assert "task" in final


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

    def test_exact_max_length_not_truncated(self) -> None:
        """Text of exactly the cap length is kept verbatim, no ellipsis."""
        cleaned = _clean_reason(AIMessage(content="x" * _MAX_REASON_CHARS))
        assert cleaned == "x" * _MAX_REASON_CHARS
        assert not cleaned.endswith("\u2026")

    def test_one_over_max_is_truncated(self) -> None:
        """A single char over the cap triggers truncation to the cap length."""
        cleaned = _clean_reason(AIMessage(content="x" * (_MAX_REASON_CHARS + 1)))
        assert cleaned is not None
        assert len(cleaned) == _MAX_REASON_CHARS
        assert cleaned.endswith("\u2026")

    def test_truncation_rstrips_before_ellipsis(self) -> None:
        """Trailing whitespace at the cut point is stripped before the ellipsis."""
        # The space lands inside the kept prefix so the cut would otherwise leave
        # "<...> \u2026"; rstrip must remove it first.
        text = "a" * (_MAX_REASON_CHARS - 2) + " " + "b" * 10
        cleaned = _clean_reason(AIMessage(content=text))
        assert cleaned is not None
        assert cleaned.endswith("\u2026")
        assert not cleaned.endswith(" \u2026")
        assert len(cleaned) < _MAX_REASON_CHARS
