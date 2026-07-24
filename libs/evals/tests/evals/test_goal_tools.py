"""Behavioral evals for the static goal-tool prompt and state notices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents_code.goal_state_notice import build_goal_state_notice
from deepagents_code.goal_tools import GOAL_TOOL_NAMES, GoalToolsMiddleware
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.evals.utils import (
    ToolNotCalled,
    TrajectoryScorer,
    final_text_contains,
    final_text_contains_any,
    run_agent,
    tool_call,
    tool_called,
    tool_not_called,
)

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

pytestmark = [pytest.mark.eval_category("tool_use")]


@tool
def lookup_population(city: str) -> str:
    """Return the population of a city as a string."""
    data = {
        "tokyo": "13,960,000",
        "delhi": "32,900,000",
        "shanghai": "29,200,000",
    }
    return data.get(city.lower(), "unknown")


def _make_agent(
    model: BaseChatModel,
    *,
    tools: list[Any] | None = None,
) -> CompiledStateGraph[Any, Any]:
    """Build a bare agent wired with the production goal-tool middleware."""
    return create_agent(
        model=model,
        tools=tools or [],
        middleware=[GoalToolsMiddleware()],
        checkpointer=InMemorySaver(),
    )


def _goal_tools_not_called() -> tuple[ToolNotCalled, ...]:
    return tuple(tool_not_called(name) for name in sorted(GOAL_TOOL_NAMES))


def _seed_private_state(
    agent: CompiledStateGraph[Any, Any],
    state: dict[str, object],
    *,
    thread_id: str,
) -> None:
    agent.update_state(
        {"configurable": {"thread_id": thread_id}},
        state,
        as_node="model",
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_no_goal_trivial_task_skips_goal_tools(model: BaseChatModel) -> None:
    """A fresh trivial task must not touch goal tools."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="What is 12 * 4?",
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("48"),
            *_goal_tools_not_called(),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_no_goal_multistep_task_skips_goal_tools(model: BaseChatModel) -> None:
    """Legitimate domain-tool work must use its tool without goal-tool calls."""
    agent = _make_agent(model, tools=[lookup_population])
    run_agent(
        agent,
        model=model,
        query=(
            'Call `lookup_population` with city="tokyo" and city="delhi". Then '
            "tell me which has more people and by how much."
        ),
        scorer=TrajectoryScorer()
        .expect(tool_calls=[tool_call(name="lookup_population")])
        .success(
            tool_called("lookup_population", args_contains={"city": "tokyo"}),
            tool_called("lookup_population", args_contains={"city": "delhi"}),
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains_any(
                "18,940,000",
                "18940000",
                "18.94 million",
                "18.9 million",
                "19 million",
                case_insensitive=True,
            ),
            *_goal_tools_not_called(),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_latest_inactive_notice_supersedes_stale_active_notice(
    model: BaseChatModel,
) -> None:
    """The newest canonical notice controls whether goal tools are relevant."""
    stale = build_goal_state_notice(
        {"rubric": "STALE-RUBRIC-SHOULD-NOT-BE-READ"},
        event_id="goal-state-stale-active",
    )
    inactive = build_goal_state_notice(
        {},
        event_id="goal-state-current-inactive",
    )
    messages = [
        stale,
        inactive,
        HumanMessage(content="What is 7 + 5?"),
    ]

    run_agent(
        _make_agent(model),
        model=model,
        query=messages,
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("12"),
            *_goal_tools_not_called(),
        ),
    )


@pytest.mark.parametrize("status", ["paused", "complete"])
@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_inactive_goal_status_skips_goal_tools(
    model: BaseChatModel,
    status: str,
) -> None:
    """Retained goal data must not drive unrelated work while inactive."""
    state = {
        "_goal_objective": "STALE-GOAL-SHOULD-NOT-DRIVE-WORK",
        "_goal_status": status,
        "_goal_rubric": "STALE-RUBRIC-SHOULD-NOT-BE-READ",
        "_sticky_rubric": "STALE-RUBRIC-SHOULD-NOT-BE-READ",
    }
    messages = [HumanMessage(content="What is 8 + 7?")]

    agent = _make_agent(model)
    thread_id = f"inactive-goal-{status}"
    _seed_private_state(agent, state, thread_id=thread_id)
    run_agent(
        agent,
        model=model,
        query=messages,
        thread_id=thread_id,
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("15"),
            *_goal_tools_not_called(),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_active_goal_followed_without_read_tools(
    model: BaseChatModel,
) -> None:
    """An actionable goal is followed straight from the injected notice."""
    marker = "ACTIVE-GOAL-2A6D"
    objective = f"Answer 8 * 7 and include the exact marker {marker}."
    rubric = f"- The final response includes `{marker}` and the correct result."
    state = {
        "_goal_objective": objective,
        "_goal_status": "active",
        "_goal_rubric": rubric,
        "_sticky_rubric": rubric,
    }
    messages = [HumanMessage(content="Continue the saved goal from its objective.")]

    agent = _make_agent(model)
    thread_id = "active-goal"
    _seed_private_state(agent, state, thread_id=thread_id)
    run_agent(
        agent,
        model=model,
        query=messages,
        thread_id=thread_id,
        scorer=TrajectoryScorer()
        .expect(tool_call_requests=0)
        .success(
            final_text_contains("56"),
            final_text_contains(marker),
            *_goal_tools_not_called(),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_active_rubric_followed_without_read_tools(model: BaseChatModel) -> None:
    """An active rubric is followed from the injected notice without a tool call."""
    marker = "ACTIVE-RUBRIC-7C91"
    rubric = f"- Include the exact marker `{marker}` in the final response."
    messages = [
        HumanMessage(content=("Follow the active acceptance criteria and answer: What is 9 * 6?")),
    ]

    run_agent(
        _make_agent(model),
        model=model,
        query=messages,
        extra_state={"rubric": rubric},
        scorer=TrajectoryScorer()
        .expect(tool_call_requests=0)
        .success(
            final_text_contains("54"),
            final_text_contains(marker),
            *_goal_tools_not_called(),
        ),
    )
