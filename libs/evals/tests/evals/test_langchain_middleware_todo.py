"""Eval tests for `langchain`'s `TodoListMiddleware` against bare `create_agent`.

These tests probe the behavioral properties of `WRITE_TODOS_SYSTEM_PROMPT` and
`WRITE_TODOS_TOOL_DESCRIPTION` directly — using `create_agent` +
`TodoListMiddleware` (not `create_deep_agent`) — so they exercise the
prompt content that ships in `langchain.agents.middleware.todo` without
any Deep Agents system prompt running in front of it.

The companion langchain PR landed a fix for the "wasted post-tool turn"
pattern that fhuang originally reported on Sonnet 4.6: when the model
finishes work with a `write_todos(all completed)` call, the agent loop
forces one more model turn, and that turn was producing empty / recap
messages instead of the substantive answer. The fix lives in
`WRITE_TODOS_SYSTEM_PROMPT` (a "Finishing a task" section), in
`WRITE_TODOS_TOOL_DESCRIPTION` (a "When You Finish" section), and in the
`write_todos` tool message (neutralized from a verbose status dump to
`"Todo list updated."`).

These tests verify the fix holds on real models, with the same
`TrajectoryScorer` framework deepagents already uses internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool

from tests.evals.utils import (
    AgentTrajectory,
    MaxToolCallRequests,
    SuccessAssertion,
    TrajectoryScorer,
    final_text_contains,
    final_text_contains_any,
    final_text_min_length,
    run_agent,
    tool_call,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [pytest.mark.eval_category("langchain/middleware")]


# ---------------------------------------------------------------------------
# Tools used by the tests
# ---------------------------------------------------------------------------


@tool
def lookup_population(city: str) -> str:
    """Return the population of a city as a string."""
    data = {
        "tokyo": "13,960,000",
        "delhi": "32,900,000",
        "shanghai": "29,200,000",
        "cairo": "21,800,000",
    }
    return data.get(city.lower(), "unknown")


@tool
def lookup_area_km2(city: str) -> str:
    """Return the area of a city in square kilometers as a string."""
    data = {
        "tokyo": "2,194",
        "delhi": "1,484",
        "shanghai": "6,341",
        "cairo": "606",
    }
    return data.get(city.lower(), "unknown")


def _max_tool_calls_success(n: int) -> SuccessAssertion:
    """Wrap `MaxToolCallRequests` as a hard-fail success assertion.

    `MaxToolCallRequests` is normally an `EfficiencyAssertion` (logged but
    never fails). For the trivial-skip test where over-using `write_todos`
    *is* the failure mode, promote it to hard-fail.
    """
    eff = MaxToolCallRequests(n=n)

    class _AsSuccess(SuccessAssertion):
        def check(self, trajectory: AgentTrajectory) -> bool:
            return eff.check(trajectory)

        def describe_failure(self, trajectory: AgentTrajectory) -> str:
            return eff.describe_failure(trajectory)

    return _AsSuccess()


# ---------------------------------------------------------------------------
# Baseline tier — regression gates
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_density_rank_lands_in_final_message(model: BaseChatModel) -> None:
    """Substantive ranked answer must land in the last AIMessage.

    Without the loop-contract fix the model puts the ranked output in the
    same turn as the final `write_todos(completed)` call, and the
    loop-terminating message becomes a 20-30 char wrap-up that omits the
    cities by name. This test fails in that regime.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population, lookup_area_km2],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Rank Tokyo, Delhi, and Shanghai by population density (people per "
            "km²) from highest to lowest. Look up the population and area for "
            "each city, compute density for each, and present the ranking. Use "
            "a todo list to plan and track your work."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=4,
            tool_call_requests=8,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("tokyo", case_insensitive=True),
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains("shanghai", case_insensitive=True),
            final_text_min_length(80),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_population_compare_lands_in_final_message(model: BaseChatModel) -> None:
    """Binary comparison answer must land in the last AIMessage.

    Delhi (32,900,000) - Tokyo (13,960,000) = 18,940,000 difference. The
    model needs to communicate that gap — formatted any reasonable way.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Which has more people: Tokyo or Delhi? Use a todo list to plan, "
            "look up the population for each city, and tell me which has more "
            "and by how much."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains_any(
                "18,940,000",
                "18940000",
                "18.94 million",
                "18.9 million",
                "19 million",
                "18 million",
                case_insensitive=True,
            ),
            final_text_min_length(50),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_trivial_arithmetic_skips_write_todos(model: BaseChatModel) -> None:
    """One-shot arithmetic must NOT invoke `write_todos`.

    Cross-model baseline check for the "skip for simple" guidance in
    `WRITE_TODOS_SYSTEM_PROMPT`. Pure single-step arithmetic with no list-
    shape or planning bait, so every model with a working skip-for-simple
    disposition should answer directly. If a future prompt change
    accidentally removes the "skip for simple" line, every model would
    start cargo-culting `write_todos` and this test catches it.
    """
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query="What is 47 + 18?",
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("65"),
            _max_tool_calls_success(0),
        ),
    )


# ---------------------------------------------------------------------------
# Hillclimb tier — progress signals (not regression gates)
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_rank_with_unknown_lookup_lands_in_final_message(model: BaseChatModel) -> None:
    """Substantive answer must land in the last AIMessage when a lookup fails.

    Atlantis is intentionally absent from the lookup data — both lookups
    return `"unknown"` for it. The agent has to revise its plan, present
    a partial ranking for the cities it could look up, and surface the
    missing data.

    Hillclimb tier: this is the hardest baseline-style probe — multi-city
    ranking + mid-flow data gap + revise-and-report. Some models flake on
    it across consecutive runs even when other baseline tests pass
    deterministically. Useful as a progress signal but not a hard gate.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population, lookup_area_km2],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Rank Tokyo, Atlantis, and Cairo by population density (people "
            "per km²) from highest to lowest. Look up the population and area "
            "for each city, compute density for each, and present the "
            "ranking. Use a todo list to plan and track your work."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=4,
            tool_call_requests=8,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("tokyo", case_insensitive=True),
            final_text_contains("cairo", case_insensitive=True),
            final_text_contains("atlantis", case_insensitive=True),
            # Any reasonable acknowledgement of the missing data counts.
            # A hallucinating model that invents stats for Atlantis would
            # not include any of these phrases.
            final_text_contains_any(
                "unknown",
                "no data",
                "n/a",
                "unable",
                "cannot be ranked",
                "not available",
                "no information",
                "missing",
                "mythical",
                "fictional",
                "legendary",
                case_insensitive=True,
            ),
            final_text_min_length(150),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_design_api_lands_in_final_message(model: BaseChatModel) -> None:
    """Design/synthesis answer must land in the last AIMessage.

    No external tools; the agent has only `write_todos`. The substantive
    output is a small API design that must mention endpoints,
    authentication, and at least one HTTP method.
    """
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Design a small REST API for a todo-list application. Use a todo "
            "list to plan and track your work. Cover at minimum: endpoints "
            "with their HTTP methods, request/response shape, and "
            "authentication approach."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("endpoint", case_insensitive=True),
            final_text_contains("authentication", case_insensitive=True),
            final_text_contains("POST", case_insensitive=True),
            final_text_min_length(200),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_density_cairo_lands_in_final_message(model: BaseChatModel) -> None:
    """Single-density answer must land in the last AIMessage.

    Softer than density_rank because the recap-summary often happens to
    include the looked-up values + density terms; partially passable under
    a model that doesn't fully respect the loop contract. Useful directional
    metric, not a regression gate.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population, lookup_area_km2],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "What is the approximate population density (people per km²) of "
            "Cairo? Look up the population, look up the area, then compute "
            "and report the density. Use a todo list to plan and track your "
            "work."
        ),
        scorer=TrajectoryScorer()
        .expect(agent_steps=4, tool_call_requests=4)
        .success(
            final_text_contains("21,800,000"),
            final_text_contains("606"),
            final_text_min_length(80),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_trivial_plan_skips_write_todos(model: BaseChatModel) -> None:
    """Trivial planning request should NOT invoke `write_todos`.

    Hillclimb tier: the "Plan a 3-course dinner menu" shape (the word
    "Plan" plus 3 explicit items) gets interpreted literally enough by
    some models (Llama, Gemini in our observations) to invoke
    `write_todos` despite the "skip for simple" guidance.
    Claude/GPT-5/DeepSeek pass it. Useful progress signal but not a
    hard regression gate.
    """
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query="Plan a simple 3-course dinner menu (appetizer, main, dessert).",
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("appetizer", case_insensitive=True),
            final_text_contains("main", case_insensitive=True),
            final_text_contains("dessert", case_insensitive=True),
            _max_tool_calls_success(0),
        ),
    )
