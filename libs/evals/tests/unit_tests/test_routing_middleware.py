"""Exercise routing across user turns, tool loops, and checkpoint restores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai.decisions import DecisionResponse, OpenAIDecisions, ScoreAnswer
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from deepagents_evals.routing import config as routing_config
from deepagents_evals.routing.middleware import Granularity, LadderRoutingMiddleware
from deepagents_evals.routing.turns import read_human_turn
from deepagents_harbor.langgraph_project import langgraph_agent
from deepagents_harbor.langgraph_project.langgraph_agent import _tau3_turn

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph


class ScriptedModel(FakeMessagesListChatModel):
    model_name: str
    calls: list[int] = Field(default_factory=list)

    def bind_tools(self, tools: object, **kwargs: object) -> Self:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        self.calls.append((self.metadata or {})["routing_turn"])
        message = self.responses.pop(0).model_copy(
            update={"usage_metadata": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}}
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture(autouse=True)
def _untraced(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(name, "false")


def _answer(tier: int) -> DecisionResponse:
    return DecisionResponse(
        model="test-decision-model",
        answers={
            "min_tier": ScoreAnswer(
                type="score",
                score=tier,
                legend={0: "cheap", 1: "strong"},
                probabilities={0: float(tier == 0), 1: float(tier == 1)},
                confidence=1.0,
            )
        },
    )


def _decisions(*responses: DecisionResponse | Exception) -> Mock:
    decisions = Mock(spec=OpenAIDecisions)
    decisions.invoke.side_effect = list(responses)
    decisions.ainvoke = AsyncMock(side_effect=list(responses))
    return decisions


@tool
def start_conversation() -> str:
    """Receive the simulated user's first message."""
    return "Please check my account."


@tool
def lookup_account() -> str:
    """Look up the simulated user's account."""
    return "Account found."


@tool
def send_message_to_user(message: str) -> str:
    """Send a reply and receive the simulated user's next message."""
    assert message
    return "Now help me dispute this charge."


def _call(name: str, **args: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("granularity", ["per_turn", "per_thread"])
async def test_simulator_turns_route_inside_one_invocation(
    async_mode: bool, granularity: Granularity
) -> None:
    cheap = ScriptedModel(
        model_name="cheap",
        responses=[
            _call("lookup_account"),
            _call("send_message_to_user", message="I found it."),
            AIMessage(content="Done"),
        ],
    )
    strong = ScriptedModel(
        model_name="strong", responses=[_call("start_conversation"), AIMessage(content="Done")]
    )
    decisions = _decisions(_answer(0), _answer(1))
    router = LadderRoutingMiddleware(
        [cheap, strong], decision_model=decisions, granularity=granularity, turn_reader=_tau3_turn
    )
    agent = create_agent(
        model=strong,
        tools=[start_conversation, lookup_account, send_message_to_user],
        middleware=[router],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "simulator"}}
    inputs = {"messages": [HumanMessage(content="Follow the conversation protocol.")]}
    result = await agent.ainvoke(inputs, config) if async_mode else agent.invoke(inputs, config)
    state = agent.get_state(config).values
    calls = decisions.ainvoke.call_args_list if async_mode else decisions.invoke.call_args_list

    assert result["messages"][-1].content == "Done"
    assert strong.calls == ([0, 2] if granularity == "per_turn" else [0])
    assert cheap.calls == ([1, 1] if granularity == "per_turn" else [1, 1, 1])
    assert len(calls) == (2 if granularity == "per_turn" else 1)
    assert len(state["decision_log"]) == len(calls)
    assert calls[0].args[0]["input"][-1].content == "Please check my account."
    if granularity == "per_turn":
        context = calls[1].args[0]["input"]
        assert isinstance(context[-1], HumanMessage)
        assert context[-1].content == "Now help me dispute this charge."
        assert any(m.content == "Account found." for m in context)
    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in replies] == [
        "start_conversation",
        "lookup_account",
        "send_message_to_user",
    ]
    assert len([m for m in result["messages"] if isinstance(m, HumanMessage)]) == 1


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("granularity", ["per_turn", "per_thread"])
async def test_human_turns_survive_checkpoint_restore(
    async_mode: bool, granularity: Granularity
) -> None:
    cheap = ScriptedModel(
        model_name="cheap",
        responses=[
            AIMessage(content="First"),
            AIMessage(content="Continued"),
            AIMessage(content="Second"),
        ],
    )
    strong = ScriptedModel(model_name="strong", responses=[AIMessage(content="Second")])
    decisions = _decisions(_answer(0), _answer(1))
    router = LadderRoutingMiddleware(
        [cheap, strong], decision_model=decisions, granularity=granularity
    )
    agent = create_agent(model=strong, middleware=[router], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "human"}}
    for messages in [
        [HumanMessage(content="Help", id="first")],
        [],
        [HumanMessage(content="Help", id="second")],
    ]:
        if async_mode:
            await agent.ainvoke({"messages": messages}, config)
        else:
            agent.invoke({"messages": messages}, config)
    state = agent.get_state(config).values
    calls = decisions.ainvoke.call_args_list if async_mode else decisions.invoke.call_args_list

    assert [r["turn_id"] for r in state["decision_log"]] == (
        ["first", "second"] if granularity == "per_turn" else ["first"]
    )
    assert state["current_tier"] == (1 if granularity == "per_turn" else 0)
    assert len(calls) == (2 if granularity == "per_turn" else 1)
    if granularity == "per_turn":
        assert [m.content for m in calls[1].args[0]["input"]] == [
            "Help",
            "First",
            "Continued",
            "Help",
        ]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_decision_failure_is_held_until_a_new_user_turn(async_mode: bool) -> None:
    cheap = ScriptedModel(model_name="cheap", responses=[AIMessage(content="Recovered")])
    strong = ScriptedModel(
        model_name="strong",
        responses=[AIMessage(content="Fallback"), AIMessage(content="Continued")],
    )
    decisions = _decisions(
        openai.APIConnectionError(request=httpx.Request("POST", "https://example.com")), _answer(0)
    )
    agent = create_agent(
        model=strong,
        middleware=[LadderRoutingMiddleware([cheap, strong], decision_model=decisions)],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "fallback"}}
    for messages in [
        [HumanMessage(content="First", id="first")],
        [],
        [HumanMessage(content="Next", id="next")],
    ]:
        if async_mode:
            await agent.ainvoke({"messages": messages}, config)
        else:
            agent.invoke({"messages": messages}, config)
    state = agent.get_state(config).values

    assert strong.calls == [1, 1]
    assert cheap.calls == [2]
    assert [r["to_tier"] for r in state["decision_log"]] == [1, 0]


def test_readers_ignore_assistant_and_unrelated_or_failed_tool_results() -> None:
    human = HumanMessage(content="Initial instruction", id="human")
    reply = ToolMessage(
        content="User reply", name="start_conversation", tool_call_id="start", id="reply"
    )
    messages = [
        human,
        reply,
        AIMessage(content="Thinking"),
        ToolMessage(content="Lookup", name="lookup_account", tool_call_id="lookup"),
        ToolMessage(
            content="Unavailable",
            name="send_message_to_user",
            tool_call_id="failed",
            status="error",
        ),
    ]
    turn = _tau3_turn(messages)

    assert turn is not None
    assert turn.key == "reply"
    assert [m.content for m in turn.messages] == ["Initial instruction", "User reply"]
    assert _tau3_turn([human]) is None
    human_turn = read_human_turn(messages)
    assert human_turn is not None
    assert human_turn.key == "human"
    assert read_human_turn([]) is None


async def test_decision_failure_keeps_the_previously_selected_tier() -> None:
    cheap = ScriptedModel(
        model_name="cheap",
        responses=[
            AIMessage(content="First"),
            AIMessage(content="Fallback"),
            AIMessage(content="Continued"),
        ],
    )
    strong = ScriptedModel(model_name="strong", responses=[])
    decisions = _decisions(
        _answer(0), openai.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
    )
    agent = create_agent(
        model=strong,
        middleware=[LadderRoutingMiddleware([cheap, strong], decision_model=decisions)],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "retain"}}
    for messages in [[HumanMessage(content="First")], [HumanMessage(content="Next")], []]:
        await agent.ainvoke({"messages": messages}, config)

    assert cheap.calls == [1, 2, 2]
    assert strong.calls == []
    assert decisions.ainvoke.await_count == 2


@pytest.mark.parametrize("simulated", [False, True])
async def test_harbor_factory_selects_the_turn_reader_for_its_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, simulated: bool
) -> None:
    cheap = ScriptedModel(
        model_name="cheap",
        responses=[
            _call("send_message_to_user", message="Hello")
            if simulated
            else AIMessage(content="Done")
        ],
    )
    strong = ScriptedModel(
        model_name="strong",
        responses=[_call("start_conversation"), AIMessage(content="Done")],
    )
    decisions = _decisions(_answer(0), _answer(1))
    monkeypatch.setattr(
        routing_config,
        "load_config",
        lambda: routing_config.RoutingConfig(
            anchor=strong, models=[cheap, strong], decision_model="test-decision-model"
        ),
    )
    monkeypatch.setattr("langchain_openai.decisions.OpenAIDecisions", lambda **_kwargs: decisions)
    monkeypatch.setattr(
        langgraph_agent,
        "_mcp_tools",
        AsyncMock(
            return_value=[start_conversation, send_message_to_user]
            if simulated
            else [lookup_account]
        ),
    )
    agent = cast(
        "CompiledStateGraph",
        await langgraph_agent.make_routed_graph({"configurable": {"cwd": str(tmp_path)}}),
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content="Follow the protocol")]})

    assert result["messages"][-1].content == "Done"
    assert cheap.calls == [1]
    assert strong.calls == ([0, 2] if simulated else [])
    assert decisions.ainvoke.await_count == (2 if simulated else 1)
