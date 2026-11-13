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
from langgraph.runtime import Runtime
from pydantic import Field

from deepagents_evals.routing import middleware as routing_config
from deepagents_evals.routing.middleware import (
    Granularity,
    LadderRoutingMiddleware,
    ModelDetails,
    ModelPricing,
    RouterState,
    read_human_turn,
)
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
        message = self.responses.pop(0)
        assert isinstance(message, AIMessage)
        message = message.model_copy(
            update={
                "usage_metadata": message.usage_metadata
                or {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
            }
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
            "task_difficulty": ScoreAnswer(
                type="score",
                score=tier * 4,
                legend={0: "routine", 4: "expert"},
                probabilities={level: float(level == tier * 4) for level in range(5)},
                confidence=1.0,
            )
        },
    )


def _decisions(*responses: DecisionResponse | Exception) -> Mock:
    decisions = Mock(spec=OpenAIDecisions)
    decisions.invoke.side_effect = list(responses)
    decisions.ainvoke = AsyncMock(side_effect=list(responses))
    return decisions


def _details(*prices: float) -> list[ModelDetails]:
    return [
        ModelDetails(
            pricing=ModelPricing(
                input=price, output=5 * price, cache_read=0.1 * price, cache_write=1.25 * price
            )
        )
        for price in prices
    ]


@pytest.mark.parametrize(
    ("probabilities", "limits", "selected"),
    [
        ([0, 0, 1, 0, 0], [2, 4], 0),
        ([0.5, 0, 0, 0, 0.5], [2, 4], 1),
        ([0, 0, 1, 0, 0], [1, 4], 1),
    ],
)
def test_difficulty_distribution_and_model_limits_control_routing(
    probabilities: list[float], limits: list[float], selected: int
) -> None:
    answer = _answer(0)
    answer.scores["task_difficulty"].probabilities = dict(enumerate(probabilities))
    answer.scores["task_difficulty"].score = sum(i * p for i, p in enumerate(probabilities))
    decisions = _decisions(answer)
    models = [
        ScriptedModel(model_name=name, responses=[AIMessage(content=name)])
        for name in ("cheap", "strong")
    ]
    router = LadderRoutingMiddleware(
        models, decision_model=decisions, difficulty_limits=limits, model_details=_details(1, 2)
    )
    agent = create_agent(model=models[0], middleware=[router])
    result = agent.invoke({"messages": [HumanMessage(content="Help", id="turn")]})

    assert result["messages"][-1].content == models[selected].model_name
    assert result["decision_log"][0]["to_tier"] == selected
    assert result["decision_log"][0]["probabilities"] == probabilities
    assert result["decision_log"][0]["difficulty"] == 2
    assert result["decision_log"][0]["difficulty_limits"] == limits


def test_difficulty_question_does_not_depend_on_model_ladder() -> None:
    requests = []
    for names in (("cheap", "strong"), ("first", "second", "third")):
        decisions = _decisions(_answer(0))
        router = LadderRoutingMiddleware(
            [ScriptedModel(model_name=name, responses=[]) for name in names],
            decision_model=decisions,
        )
        router.before_model(
            RouterState(messages=[HumanMessage(content="Help", id="turn")], decision_log=[]),
            Runtime(context=None),
        )
        requests.append(decisions.invoke.call_args.args[0])

    assert requests[0]["questions"] == requests[1]["questions"]
    levels = requests[0]["questions"]["task_difficulty"].levels
    assert len(levels) == 5
    assert all(level.description for level in levels)


@pytest.mark.parametrize("limits", [[2], [-1, 4], [2, 5], [float("nan"), 4]])
def test_invalid_difficulty_limits_are_rejected(limits: list[float]) -> None:
    models = [ScriptedModel(model_name=name, responses=[]) for name in ("cheap", "strong")]
    with pytest.raises(ValueError, match="difficulty_limits"):
        LadderRoutingMiddleware(models, decision_model=_decisions(), difficulty_limits=limits)


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
        [cheap, strong],
        decision_model=decisions,
        granularity=granularity,
        turn_reader=_tau3_turn,
        model_details=_details(1, 2),
        difficulty_limits=[2, 4],
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
    assert result["decision_log"] == state["decision_log"]
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
        [cheap, strong],
        decision_model=decisions,
        granularity=granularity,
        model_details=_details(1, 2),
        difficulty_limits=[2, 4],
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
        middleware=[
            LadderRoutingMiddleware(
                [cheap, strong],
                decision_model=decisions,
                model_details=_details(1, 2),
                difficulty_limits=[2, 4],
            )
        ],
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
        middleware=[
            LadderRoutingMiddleware(
                [cheap, strong],
                decision_model=decisions,
                model_details=_details(1, 2),
                difficulty_limits=[2, 4],
            )
        ],
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
            anchor=strong,
            models=[cheap, strong],
            decision_model="test-decision-model",
            model_details=_details(1, 2),
            difficulty_limits=[2, 4],
        ),
    )
    monkeypatch.setattr(
        "deepagents_evals.routing.middleware.make_decisions", lambda _model: decisions
    )
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
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="Follow the protocol")],
            "decision_log": [{"to_tier": 999}],
        }
    )

    assert result["messages"][-1].content == "Done"
    assert cheap.calls == [1]
    assert strong.calls == ([0, 2] if simulated else [])
    assert decisions.ainvoke.await_count == (2 if simulated else 1)
    assert [entry["to_tier"] for entry in result["decision_log"]] == ([0, 1] if simulated else [0])
    assert {
        "current_tier",
        "turn",
        "last_turn_id",
        "cache_expires_at",
        "prefix_anchor",
        "last_usage",
        "pending_routing",
    }.isdisjoint(result)


@pytest.mark.parametrize("reverse", [False, True])
def test_prices_and_suitability_are_independent_of_candidate_order(reverse: bool) -> None:
    models = [
        ScriptedModel(model_name="capable", responses=[AIMessage(content="capable")]),
        ScriptedModel(model_name="expensive", responses=[AIMessage(content="expensive")]),
    ]
    details, limits = _details(1, 5), [4, 2]
    if reverse:
        models.reverse()
        details.reverse()
        limits.reverse()
    decisions = _decisions(_answer(1))
    router = LadderRoutingMiddleware(
        models, decision_model=decisions, model_details=details, difficulty_limits=limits
    )
    result = create_agent(model=models[0], middleware=[router]).invoke(
        {"messages": [HumanMessage(content="Solve this")]}
    )

    assert result["messages"][-1].content == "capable"
    assert result["decision_log"][0]["fallback_reason"] is None


def test_default_suitability_does_not_depend_on_price_or_context_window() -> None:
    expensive = ScriptedModel(
        model_name="expensive", responses=[], profile={"max_input_tokens": 10000}
    )
    cheap = ScriptedModel(
        model_name="cheap",
        responses=[AIMessage(content="cheap")],
        profile={"max_input_tokens": 1000000},
    )
    router = LadderRoutingMiddleware(
        [expensive, cheap], decision_model=_decisions(_answer(1)), model_details=_details(5, 1)
    )
    result = create_agent(model=expensive, middleware=[router]).invoke(
        {"messages": [HumanMessage(content="Solve this")]}
    )

    assert result["messages"][-1].content == "cheap"


@pytest.mark.parametrize("decision_fails", [False, True])
def test_incompatible_candidate_cannot_win_selection_or_fallback(decision_fails: bool) -> None:
    cheap = ScriptedModel(model_name="cheap", responses=[], profile={"tool_calling": False})
    eligible = ScriptedModel(model_name="eligible", responses=[AIMessage(content="eligible")])
    response = (
        openai.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        if decision_fails
        else _answer(0)
    )
    router = LadderRoutingMiddleware(
        [cheap, eligible],
        decision_model=_decisions(response),
        model_details=_details(1, 5),
        difficulty_limits=[4, 2],
    )
    result = create_agent(model=eligible, tools=[lookup_account], middleware=[router]).invoke(
        {"messages": [HumanMessage(content="Look up my account")]}
    )

    assert result["messages"][-1].content == "eligible"
    assert result["decision_log"][0]["excluded_tiers"] == {0: ["tool_calling"]}
    assert cheap.calls == []


def test_no_eligible_model_fails_before_calling_an_incompatible_model() -> None:
    model = ScriptedModel(model_name="text-only", responses=[], profile={"image_inputs": False})
    decisions = _decisions()
    router = LadderRoutingMiddleware([model], decision_model=decisions, model_details=_details(1))
    agent = create_agent(model=model, middleware=[router])

    with pytest.raises(ValueError, match=r"No eligible routing model.*image_inputs"):
        agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=[{"type": "image", "url": "https://example.com/image.png"}]
                    )
                ]
            }
        )
    assert model.calls == []
    decisions.invoke.assert_not_called()


def test_missing_prices_fall_back_using_explicit_suitability() -> None:
    strong = ScriptedModel(model_name="strong", responses=[AIMessage(content="strong")])
    weak = ScriptedModel(model_name="weak", responses=[])
    router = LadderRoutingMiddleware(
        [strong, weak], decision_model=_decisions(_answer(0)), difficulty_limits=[4, 1]
    )
    result = create_agent(model=strong, middleware=[router]).invoke(
        {"messages": [HumanMessage(content="Hello")]}
    )

    assert result["messages"][-1].content == "strong"
    assert result["decision_log"][0]["fallback_reason"] == "missing_pricing"
    assert result["decision_log"][0]["expected_costs"] == [None, None]


def test_checkpoint_resume_reuses_assessment_before_first_model_call() -> None:
    cheap = ScriptedModel(model_name="cheap", responses=[AIMessage(content="cheap")])
    strong = ScriptedModel(model_name="strong", responses=[])
    decisions = _decisions(_answer(0))
    router = LadderRoutingMiddleware(
        [cheap, strong],
        decision_model=decisions,
        model_details=_details(1, 2),
        difficulty_limits=[2, 4],
    )
    agent = create_agent(
        model=strong,
        middleware=[router],
        checkpointer=InMemorySaver(),
        interrupt_before=["model"],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "pending-assessment"}}
    agent.invoke({"messages": [HumanMessage(content="Hello")]}, config)
    assert cheap.calls == strong.calls == []

    result = agent.invoke(None, config)

    assert result["messages"][-1].content == "cheap"
    assert decisions.invoke.call_count == 1
    assert len(result["decision_log"]) == 1


def test_warm_cache_can_win_next_turn_and_expiration_restores_cold_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    monkeypatch.setattr("deepagents_evals.routing.middleware.time.time", lambda: now[0])
    usage = {
        "input_tokens": 100000,
        "output_tokens": 1,
        "total_tokens": 100001,
        "input_token_details": {"cache_creation": 100000},
    }
    cheap = ScriptedModel(model_name="cheap", responses=[AIMessage(content="cheap")])
    strong = ScriptedModel(
        model_name="strong",
        responses=[
            AIMessage(content="strong", usage_metadata=usage),
            AIMessage(content="warm", usage_metadata=usage),
        ],
    )
    decisions = _decisions(_answer(1), _answer(0), _answer(0))
    router = LadderRoutingMiddleware(
        [cheap, strong],
        decision_model=decisions,
        model_details=_details(1, 2),
        difficulty_limits=[2, 4],
    )
    agent = create_agent(model=strong, middleware=[router], checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "cache"}}
    first = agent.invoke({"messages": [HumanMessage(content="Solve this")]}, config)
    now[0] += 1
    second = agent.invoke({"messages": [HumanMessage(content="Thanks")]}, config)
    now[0] += 301
    third = agent.invoke({"messages": [HumanMessage(content="Hello")]}, config)

    assert [result["messages"][-1].content for result in (first, second, third)] == [
        "strong",
        "warm",
        "cheap",
    ]
    assert [record["to_tier"] for record in third["decision_log"]] == [1, 1, 0]
    assert decisions.invoke.call_count == 3
