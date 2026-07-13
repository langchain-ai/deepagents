"""Tests for `TimeBudgetMiddleware`.

Coverage is split into three layers:

- **Unit** tests exercise the middleware's methods directly (budget math,
    request transformation, tool-delta accounting, the model hooks) with
    hand-built inputs so assertions do not depend on agent-loop or
    fake-model quirks.
- **End-to-end** tests drive `create_agent` with a scripted fake model and
    a deterministic step clock, covering consumed accounting, wind-down,
    hard-stop, the event callback, and the async path.
- **Integration** tests confirm the middleware composes inside the full
    `create_deep_agent` stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents import TimeBudgetMiddleware, create_deep_agent
from deepagents.middleware.time_budget import TIME_BUDGET_EVENT_SOURCE
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.filterwarnings("ignore:.*is in beta")


class StepClock:
    """Deterministic clock that advances a fixed step on every read.

    Because the middleware reads the clock twice per timed action (start and
    end), a fixed step makes every model call and every tool call consume
    exactly ``step`` seconds, regardless of interleaving.
    """

    def __init__(self, step: float = 1.0, start: float = 0.0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


@tool
def slow_tool(x: int) -> int:
    """Return x unchanged (stands in for a time-consuming tool)."""
    return x


def _make_request(
    *,
    consumed: float,
    tools: list | None = None,
    system_text: str | None = "base system",
) -> ModelRequest:
    """Build a `ModelRequest` with a given consumed-budget state."""
    system_message = SystemMessage(content=system_text) if system_text is not None else None
    return ModelRequest(
        model=GenericFakeChatModel(messages=iter([])),
        messages=[HumanMessage(content="hello")],
        system_message=system_message,
        tools=list(tools) if tools is not None else [slow_tool],
        state={"messages": [], "_time_budget_consumed": consumed},
    )


class TestValidation:
    def test_rejects_nonpositive_total(self) -> None:
        with pytest.raises(ValueError, match="total_seconds must be positive"):
            TimeBudgetMiddleware(total_seconds=0)
        with pytest.raises(ValueError, match="total_seconds must be positive"):
            TimeBudgetMiddleware(total_seconds=-5)

    def test_rejects_bad_warn_fraction(self) -> None:
        with pytest.raises(ValueError, match="warn_fraction must be in"):
            TimeBudgetMiddleware(total_seconds=10, warn_fraction=0)
        with pytest.raises(ValueError, match="warn_fraction must be in"):
            TimeBudgetMiddleware(total_seconds=10, warn_fraction=1.5)

    def test_defaults(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=360)
        assert mw.total_seconds == 360.0
        assert mw.warn_fraction == 0.8
        assert mw.on_exceed == "wind_down"
        assert mw.inject_awareness is True
        assert mw._warn_threshold == pytest.approx(288.0)


class TestStatus:
    def test_status_ok(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100)
        assert mw._status_for(0.0) == "ok"
        assert mw._status_for(79.0) == "ok"

    def test_status_warning(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, warn_fraction=0.8)
        assert mw._status_for(80.0) == "warning"
        assert mw._status_for(99.0) == "warning"

    def test_status_wind_down_when_exhausted(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, on_exceed="wind_down")
        assert mw._status_for(100.0) == "wind_down"
        assert mw._status_for(150.0) == "wind_down"

    def test_status_hard_stop_when_exhausted(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, on_exceed="hard_stop")
        assert mw._status_for(100.0) == "hard_stop"


class TestModifyRequest:
    def test_awareness_injected_under_budget(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100)
        req = _make_request(consumed=10.0)
        out = mw._modify_request(req)
        assert out is not req
        assert len(out.tools) == 1
        assert "remaining" in out.system_message.text
        assert "base system" in out.system_message.text

    def test_awareness_warning_text_in_warn_zone(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, warn_fraction=0.8)
        out = mw._modify_request(_make_request(consumed=90.0))
        assert "Running low" in out.system_message.text
        assert len(out.tools) == 1

    def test_awareness_disabled_returns_request_unchanged(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, inject_awareness=False)
        req = _make_request(consumed=10.0)
        assert mw._modify_request(req) is req

    def test_wind_down_strips_tools_and_injects_finalize(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, on_exceed="wind_down")
        out = mw._modify_request(_make_request(consumed=120.0))
        assert out.tools == []
        assert "Time is up" in out.system_message.text

    def test_wind_down_preserves_original_system_message(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100)
        out = mw._modify_request(_make_request(consumed=120.0, system_text="IMPORTANT RULES"))
        assert "IMPORTANT RULES" in out.system_message.text


class TestAttachToolDelta:
    def test_toolmessage_wrapped_in_command_with_delta(self) -> None:
        tm = ToolMessage(content="result", tool_call_id="c1")
        out = TimeBudgetMiddleware._attach_tool_delta(tm, 3.0)
        assert isinstance(out, Command)
        assert out.update["_time_budget_consumed"] == 3.0
        assert out.update["messages"] == [tm]

    def test_command_result_gets_delta_merged(self) -> None:
        tm = ToolMessage(content="fs", tool_call_id="c2")
        cmd = Command(update={"messages": [tm], "files": {"a.txt": "x"}})
        out = TimeBudgetMiddleware._attach_tool_delta(cmd, 2.5)
        assert out is cmd
        assert out.update["_time_budget_consumed"] == 2.5
        assert out.update["files"] == {"a.txt": "x"}
        assert out.update["messages"] == [tm]

    def test_command_with_existing_delta_accumulates(self) -> None:
        cmd = Command(update={"_time_budget_consumed": 1.0})
        out = TimeBudgetMiddleware._attach_tool_delta(cmd, 2.0)
        assert out.update["_time_budget_consumed"] == 3.0


class TestModelHooks:
    def test_before_model_stamps_start_under_budget(self) -> None:
        clock = StepClock(step=1.0, start=42.0)
        mw = TimeBudgetMiddleware(total_seconds=100, clock=clock)
        update = mw._before_model({"_time_budget_consumed": 10.0})
        assert update == {"_time_budget_model_start": 42.0}

    def test_before_model_hard_stop_when_over(self) -> None:
        events: list[dict[str, Any]] = []
        mw = TimeBudgetMiddleware(total_seconds=100, on_exceed="hard_stop", on_event=events.append)
        update = mw._before_model({"_time_budget_consumed": 150.0})
        assert update["jump_to"] == "end"
        assert update["_time_budget_status"] == "hard_stop"
        (msg,) = update["messages"]
        assert isinstance(msg, AIMessage)
        assert "time budget" in msg.content.lower()
        assert msg.additional_kwargs["lc_source"] == TIME_BUDGET_EVENT_SOURCE
        assert events and events[-1]["status"] == "hard_stop"

    def test_before_model_wind_down_does_not_hard_stop(self) -> None:
        clock = StepClock(step=1.0, start=7.0)
        mw = TimeBudgetMiddleware(total_seconds=100, on_exceed="wind_down", clock=clock)
        update = mw._before_model({"_time_budget_consumed": 150.0})
        assert update == {"_time_budget_model_start": 7.0}

    def test_after_model_records_delta_and_status(self) -> None:
        clock = StepClock(step=1.0, start=50.0)
        mw = TimeBudgetMiddleware(total_seconds=100, clock=clock)
        update = mw._after_model({"_time_budget_model_start": 45.0, "_time_budget_consumed": 10.0})
        assert update["_time_budget_consumed"] == pytest.approx(5.0)
        assert update["_time_budget_status"] == "ok"
        assert update["_time_budget_model_start"] is None

    def test_after_model_noop_without_start(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=100, clock=StepClock())
        assert mw._after_model({"_time_budget_consumed": 10.0}) is None

    def test_after_model_status_warning(self) -> None:
        clock = StepClock(step=1.0, start=100.0)
        mw = TimeBudgetMiddleware(total_seconds=100, warn_fraction=0.8, clock=clock)
        update = mw._after_model({"_time_budget_model_start": 99.0, "_time_budget_consumed": 90.0})
        assert update["_time_budget_status"] == "warning"


class _ToolSpy(AgentMiddleware):
    """Records how many tools the (inner) handler sees on each model call."""

    def __init__(self) -> None:
        super().__init__()
        self.tools_seen: list[int] = []

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        self.tools_seen.append(len(request.tools))
        return handler(request)


def _tool_then_final_model() -> GenericFakeChatModel:
    """Model that calls a tool once, then returns a final answer."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "slow_tool", "args": {"x": 1}, "id": "c1", "type": "tool_call"}],
                ),
                AIMessage(content="Final answer."),
            ]
        )
    )


class TestEndToEnd:
    def test_consumed_accounting_model_and_tool(self) -> None:
        events: list[dict[str, Any]] = []
        mw = TimeBudgetMiddleware(total_seconds=1000, clock=StepClock(step=100.0), on_event=events.append)
        agent = create_agent(model=_tool_then_final_model(), tools=[slow_tool], middleware=[mw])
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        assert result["messages"][-1].content == "Final answer."
        assert events[-1]["consumed"] == pytest.approx(300.0)
        assert events[-1]["remaining"] == pytest.approx(700.0)

    def test_wind_down_forces_final_answer_and_strips_tools(self) -> None:
        events: list[dict[str, Any]] = []
        mw = TimeBudgetMiddleware(total_seconds=150, on_exceed="wind_down", clock=StepClock(step=100.0), on_event=events.append)
        spy = _ToolSpy()
        agent = create_agent(model=_tool_then_final_model(), tools=[slow_tool], middleware=[mw, spy])
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        assert result["messages"][-1].content == "Final answer."
        assert spy.tools_seen[-1] == 0
        assert spy.tools_seen[0] == 1
        assert any(e["status"] == "wind_down" for e in events)

    def test_hard_stop_ends_run_with_notice(self) -> None:
        events: list[dict[str, Any]] = []
        model = _tool_then_final_model()
        mw = TimeBudgetMiddleware(total_seconds=150, on_exceed="hard_stop", clock=StepClock(step=100.0), on_event=events.append)
        agent = create_agent(model=model, tools=[slow_tool], middleware=[mw])
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        last = result["messages"][-1]
        assert isinstance(last, AIMessage)
        assert "time budget" in last.content.lower()
        assert "Final answer." not in last.content
        assert len(model.call_history) == 1
        assert events[-1]["status"] == "hard_stop"

    async def test_async_wind_down(self) -> None:
        events: list[dict[str, Any]] = []
        mw = TimeBudgetMiddleware(total_seconds=150, on_exceed="wind_down", clock=StepClock(step=100.0), on_event=events.append)
        agent = create_agent(model=_tool_then_final_model(), tools=[slow_tool], middleware=[mw])
        result = await agent.ainvoke({"messages": [HumanMessage(content="go")]})
        assert result["messages"][-1].content == "Final answer."
        assert any(e["status"] == "wind_down" for e in events)

    def test_no_awareness_leaves_system_prompt_clean(self) -> None:
        recorded: list[str] = []

        class _SysSpy(AgentMiddleware):
            def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
                recorded.append(request.system_message.text if request.system_message else "")
                return handler(request)

        mw = TimeBudgetMiddleware(total_seconds=1000, inject_awareness=False, clock=StepClock(step=1.0))
        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
        agent = create_agent(model=model, middleware=[mw, _SysSpy()], system_prompt="ROOT")
        agent.invoke({"messages": [HumanMessage(content="go")]})
        assert all("time budget" not in s for s in recorded)


class TestDeepAgentIntegration:
    def test_composes_in_deep_agent_stack(self) -> None:
        events: list[dict[str, Any]] = []
        mw = TimeBudgetMiddleware(total_seconds=1000, clock=StepClock(step=100.0), on_event=events.append)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Done, no tools needed.")]))
        agent = create_deep_agent(model=model, middleware=[mw])
        result = agent.invoke({"messages": [HumanMessage(content="say hi")]})
        assert result["messages"][-1].content == "Done, no tools needed."
        assert events[-1]["consumed"] == pytest.approx(100.0)
        assert events[-1]["status"] == "ok"

    def test_state_schema_readable_via_get_state(self) -> None:
        mw = TimeBudgetMiddleware(total_seconds=1000, clock=StepClock(step=100.0))
        model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
        agent = create_deep_agent(model=model, middleware=[mw], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        agent.invoke({"messages": [HumanMessage(content="go")]}, config)
        values = agent.get_state(config).values
        assert values["_time_budget_consumed"] == pytest.approx(100.0)
