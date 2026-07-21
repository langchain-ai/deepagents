"""Behavioral evals for the static goal-tool prompt and state notices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents_code.goal_state_notice import build_goal_state_notice
from deepagents_code.goal_tools import GoalToolsMiddleware
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from tests.evals.utils import (
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


@tool
def lookup_area_km2(city: str) -> str:
    """Return the area of a city in square kilometers as a string."""
    data = {
        "tokyo": "2,194",
        "delhi": "1,484",
        "shanghai": "6,341",
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
            tool_not_called("get_rubric"),
            tool_not_called("get_goal"),
            tool_not_called("update_goal"),
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
            "Which has more people, Tokyo or Delhi? Look up the population for "
            "each and tell me which has more and by how much."
        ),
        scorer=TrajectoryScorer()
        .expect(tool_calls=[tool_call(name="lookup_population")])
        .success(
            tool_called("lookup_population"),
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains_any(
                "18,940,000",
                "18940000",
                "18.94 million",
                "18.9 million",
                "19 million",
                case_insensitive=True,
            ),
            tool_not_called("get_rubric"),
            tool_not_called("get_goal"),
            tool_not_called("update_goal"),
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
            tool_not_called("get_rubric"),
            tool_not_called("get_goal"),
            tool_not_called("update_goal"),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_active_rubric_requires_get_rubric_and_marker(model: BaseChatModel) -> None:
    """An active matching notice must lead to rubric retrieval and use."""
    marker = "ACTIVE-RUBRIC-7C91"
    rubric = f"- Include the exact marker `{marker}` in the final response."
    notice = build_goal_state_notice(
        {"rubric": rubric},
        event_id="goal-state-active-rubric",
    )
    messages = [notice, HumanMessage(content="What is 9 * 6?")]

    run_agent(
        _make_agent(model),
        model=model,
        query=messages,
        extra_state={"rubric": rubric},
        scorer=TrajectoryScorer().success(
            tool_called("get_rubric"),
            final_text_contains("54"),
            final_text_contains(marker),
            tool_not_called("get_goal"),
            tool_not_called("update_goal"),
        ),
    )
