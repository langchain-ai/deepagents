"""Eval tests for `deepagents_code`'s goal-tools prompt (`dcode`).

These tests probe the behavioral properties of `GOAL_TOOLS_SYSTEM_PROMPT` and
the `get_rubric` / `get_goal` / `update_goal` tool descriptions directly — using
`create_agent` + the real `GoalToolsMiddleware` (not `create_deep_agent`) — so
they exercise exactly the guidance that ships in
`deepagents_code.goal_tools` without any other deepagents-side prompt running in
front of it. This mirrors `test_langchain_middleware_todo.py`, which probes
`langchain`'s `TodoListMiddleware` the same way.

The failure mode under test: models over-eagerly call `get_rubric` / `get_goal`
even when *no goal or rubric was ever set* earlier in the conversation. When
nothing is set those tools return an inactive snapshot and add nothing, so a
well-behaved agent should not touch them. The baseline tests here are the
regression gate for that behavior; the hillclimb test confirms the guidance
does not over-correct into never consulting the tools when a rubric *is* active.

Seeding note: the goal channels (`_goal_objective`, ...) are `PrivateStateAttr`
and are not part of the public graph input in this isolated `create_agent`
harness (only `messages` / `rubric` are exposed). The active-context hillclimb
test therefore seeds the public `rubric` input — the same channel
`RubricMiddleware` grades — to make a rubric active, rather than trying to seed
a goal directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents_code.goal_tools import GoalToolsMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
    tool_not_called,
)

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

pytestmark = [pytest.mark.eval_category("tool_use")]
"""Apply tool_use category to all tests in this module. Tier is set per-test."""


# ---------------------------------------------------------------------------
# Mock tools — lightweight stubs so the agent has real work to do
# ---------------------------------------------------------------------------


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
    """Build a bare `create_agent` wired with the real `GoalToolsMiddleware`."""
    return create_agent(
        model=model,
        tools=tools or [],
        middleware=[GoalToolsMiddleware()],
    )


# ---------------------------------------------------------------------------
# Baseline tier — regression gates for over-eager goal-tool calls
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_no_goal_trivial_task_skips_goal_tools(model: BaseChatModel) -> None:
    """Trivial one-shot task with no goal/rubric must not touch the goal tools.

    No goal or rubric is set, so `get_rubric` / `get_goal` would only return an
    inactive snapshot. A model that reflexively "inspects acceptance criteria
    before deciding whether work is complete" fires them anyway — this test
    hard-fails in that regime. It is the case the current copy fails and the
    rewritten, precondition-gated copy should pass.
    """
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
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_no_goal_multistep_task_skips_goal_tools(model: BaseChatModel) -> None:
    """Real multi-step tool use with no goal/rubric must still skip goal tools.

    Over-eagerness is not just a trivial-task artifact: even when the agent
    legitimately calls domain tools, it should not reach for `get_rubric` /
    `get_goal` when nothing was ever set. The agent looks up two populations
    and reports which city is larger; the goal tools must stay untouched.
    """
    agent = _make_agent(model, tools=[lookup_population])
    run_agent(
        agent,
        model=model,
        query=(
            "Which has more people, Tokyo or Delhi? Look up the population for "
            "each and tell me which is larger."
        ),
        scorer=TrajectoryScorer()
        .expect(tool_calls=[tool_call(name="lookup_population")])
        .success(
            final_text_contains("delhi", case_insensitive=True),
            tool_not_called("get_rubric"),
            tool_not_called("get_goal"),
        ),
    )


# ---------------------------------------------------------------------------
# Hillclimb tier — the guidance should not over-correct
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_active_rubric_may_be_consulted(model: BaseChatModel) -> None:
    """When a rubric IS active, consulting `get_rubric` is allowed, not banned.

    This guards against the rewrite over-correcting into "never call these
    tools." A rubric is seeded via the public `rubric` input (the channel
    `RubricMiddleware` grades), so `get_rubric` returns active criteria. The
    hard requirement is only that the substantive answer lands; whether the
    model consults `get_rubric` is logged as an efficiency signal, since
    "should use" is inherently noisier than "should not."
    """
    agent = _make_agent(model, tools=[lookup_population, lookup_area_km2])
    run_agent(
        agent,
        model=model,
        query=(
            "Rank Tokyo, Delhi, and Shanghai by population density (people per "
            "km²) from highest to lowest. Look up the population and area for "
            "each, compute density, and present the ranking."
        ),
        extra_state={
            "rubric": (
                "- Every city is ranked by population density.\n"
                "- Each density value is shown with its units."
            )
        },
        scorer=TrajectoryScorer()
        .expect(tool_calls=[tool_call(name="get_rubric")])
        .success(
            final_text_contains("tokyo", case_insensitive=True),
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains("shanghai", case_insensitive=True),
        ),
    )
