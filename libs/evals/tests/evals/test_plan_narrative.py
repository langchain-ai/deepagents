"""Eval tests for plan narrative quality.

Tests whether the agent emits human-readable narrative text alongside or
before its first `write_todos` call, asks for plan confirmation, and
calibrates plan depth to task scope.

Background: trace `019de4e4-b965-73f1-9bf7-0bcd7c1ca7a4` showed
`openai:gpt-5.5` calling `write_todos` with zero accompanying text, then
emitting only "Plan looks good?" in the next turn — leaving the user with
no human-readable plan to approve. These evals catch that regression and
measure plan-quality across models and harness profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.llm_judge import llm_judge
from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent,
)

pytestmark = [pytest.mark.eval_category("conversation")]
"""Apply conversation category to all tests in this module. Tier is set per-test."""

_PLANNING_TOOLS: tuple[str, ...] = ("write_todos",)
"""Tool calls that count as 'committing to a plan' for narrative-presence checks."""

_MIN_NARRATIVE_CHARS = 80
"""Minimum text length to count as substantive narrative.

Tuned to exclude bare confirmation prompts like "Plan looks good?" (17
chars) while accepting a short two-sentence summary.
"""


# ---------------------------------------------------------------------------
# Custom success assertion: narrative must appear before/alongside the first
# planning tool call.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NarrativeBeforePlan(SuccessAssertion):
    """Assert the agent emits substantive narrative before/alongside planning.

    Walks the trajectory and finds the first `AIMessage` containing a
    planning tool call (e.g. `write_todos`). Passes if any `AIMessage` at
    or before that step contains text of at least `min_chars`.

    Attributes:
        planning_tools: Tool names that count as committing to a plan.
        min_chars: Minimum stripped-text length to count as narrative.
    """

    planning_tools: tuple[str, ...] = _PLANNING_TOOLS
    min_chars: int = _MIN_NARRATIVE_CHARS

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return ``True`` when narrative precedes or accompanies planning.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the trajectory satisfies the narrative-before-plan rule.
        """
        plan_step = self._first_planning_step(trajectory)
        if plan_step is None:
            # Agent did not commit to a plan — let the LLM judge or other
            # assertions decide whether that is correct for this prompt.
            return True
        for step in trajectory.steps[:plan_step]:
            if step.action.text and len(step.action.text.strip()) >= self.min_chars:
                return True
        return False

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the narrative-before-plan check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        plan_step = self._first_planning_step(trajectory)
        if plan_step is None:
            return "internal error: no planning tool call found"
        joined = " | ".join(self.planning_tools)
        return (
            f"Agent committed to a plan ({joined}) at step {plan_step} "
            f"without first emitting narrative text "
            f"(>= {self.min_chars} chars). The user has no human-readable "
            f"plan to approve."
        )

    def _first_planning_step(self, trajectory: AgentTrajectory) -> int | None:
        """Return the 1-indexed step containing the first planning tool call.

        Args:
            trajectory: The agent trajectory to search.

        Returns:
            The 1-indexed step number, or `None` if no planning call exists.
        """
        for step in trajectory.steps:
            for tc in step.action.tool_calls:
                if tc.get("name") in self.planning_tools:
                    return step.index
        return None


def narrative_before_plan(
    *,
    planning_tools: tuple[str, ...] = _PLANNING_TOOLS,
    min_chars: int = _MIN_NARRATIVE_CHARS,
) -> NarrativeBeforePlan:
    """Create a `NarrativeBeforePlan` success assertion.

    Args:
        planning_tools: Tool names that count as committing to a plan.
        min_chars: Minimum stripped-text length to count as narrative.

    Returns:
        A `NarrativeBeforePlan` assertion instance.
    """
    return NarrativeBeforePlan(planning_tools=planning_tools, min_chars=min_chars)


# ---------------------------------------------------------------------------
# Test cases — multi-step prompts where the agent should emit a narrative
# plan and a checklist together.
# ---------------------------------------------------------------------------

PLAN_NARRATIVE_CASES: list[dict[str, Any]] = [
    {
        "id": "feature_provider_profile",
        "query": (
            "Add a builtin harness profile for openai:gpt-5.5 to the "
            "deepagents SDK. Match the conventions used by existing "
            "OpenAI Codex and Anthropic profiles in "
            "libs/deepagents/deepagents/profiles/harness/."
        ),
        "criteria": (
            "The agent emits at least two sentences of narrative text "
            "explaining its planned approach before or alongside any "
            "write_todos call.",
            "The narrative explains WHY this approach was chosen "
            "(e.g. mentions matching existing profile conventions, or "
            "naming the files/modules to be created), not just WHAT will "
            "be done.",
            "The narrative content is consistent with the items in the "
            "todo list — the prose and the checklist describe the same "
            "plan, not contradictory or unrelated work.",
        ),
    },
    {
        "id": "bug_investigation_open_ended",
        "query": (
            "Users report that on Windows the deepagents CLI hangs for "
            "around 8 seconds at startup before showing the splash "
            "screen. We don't see this on macOS. Figure out what's "
            "happening and propose a fix."
        ),
        "criteria": (
            "The agent describes an investigation strategy in narrative "
            "form before committing to a fix — e.g. naming what it will "
            "look at first (imports, subprocess calls, filesystem "
            "checks).",
            "The agent acknowledges that the root cause is unknown and "
            "frames the plan as investigation rather than prescribing a "
            "specific fix prematurely.",
            "The agent does not skip narrative entirely and dive "
            "straight into a write_todos checklist with no explanatory "
            "text.",
        ),
    },
    {
        "id": "refactor_cross_package",
        "query": (
            "Move the format_tool_display function out of "
            "libs/cli/deepagents_cli/tool_display.py into the deepagents "
            "SDK so libs/evals can import it too. Keep the CLI's "
            "existing import path working with a re-export shim."
        ),
        "criteria": (
            "The agent emits narrative text explaining the refactor's "
            "shape — at minimum naming the source and destination "
            "packages and the re-export shim — before or alongside "
            "write_todos.",
            "The narrative addresses backwards compatibility (the "
            "existing CLI import path) rather than treating it as an "
            "afterthought.",
            "The plan explicitly accounts for tests covering both import paths.",
        ),
    },
    {
        "id": "performance_profile_then_fix",
        "query": (
            "create_deep_agent() is too slow on cold start — it's the "
            "dominant TTFT cost when the CLI launches. Profile it, "
            "identify the bottleneck, and reduce wall-clock cold start "
            "by at least 30% without breaking the public API."
        ),
        "criteria": (
            "The agent's narrative makes clear that profiling comes "
            "before fixing — it does not jump to optimization without "
            "measurement.",
            "The narrative names a specific profiling tool or approach "
            "(e.g. cProfile, py-spy, pytest-codspeed, importtime) "
            "rather than waving at 'profile it'.",
            "The narrative acknowledges the public-API constraint as a "
            "design boundary, not a footnote.",
        ),
    },
    {
        "id": "migration_settings_format",
        "query": (
            "Migrate the deepagents CLI from .deepagents/config.toml to "
            ".deepagents/settings.json. Existing user configs should be "
            "auto-migrated on first run. Keep the old file readable for "
            "one release with a deprecation warning. Update the docs."
        ),
        "criteria": (
            "The narrative covers all four scopes mentioned in the "
            "request — new file format, auto-migration, deprecation "
            "window, docs — without dropping any.",
            "The narrative explains the deprecation strategy in at least "
            "one sentence (when the warning fires, when the old file "
            "stops being read).",
            "The agent emits this narrative before its first write_todos call, not after.",
        ),
    },
    {
        "id": "docs_only_rewrite",
        "query": (
            "The Profiles page at "
            "docs.langchain.com/oss/python/deepagents/profiles is hard "
            "to follow. Restructure it so a new user can pick the right "
            "profile in under 30 seconds. Don't change any code — only "
            "the docs."
        ),
        "criteria": (
            "The agent emits narrative explaining its restructuring "
            "approach (e.g. decision tree, table-of-contents, quick-pick "
            "section) before any write_todos call.",
            "The narrative respects the 'no code changes' constraint and "
            "does not propose code changes as part of the plan.",
            "The agent does not skip planning entirely just because the "
            "task is non-code; it still produces a structured plan.",
        ),
    },
]


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    PLAN_NARRATIVE_CASES,
    ids=[c["id"] for c in PLAN_NARRATIVE_CASES],
)
def test_plan_narrative_quality(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Agent emits substantive plan narrative for multi-step requests."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=case["query"],
        scorer=(
            TrajectoryScorer().success(
                narrative_before_plan(),
                llm_judge(*case["criteria"], include_tool_calls=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Calibration cases — the agent should NOT scaffold a plan for trivial
# requests, and should ask before planning when the request is ambiguous.
# ---------------------------------------------------------------------------

PLAN_CALIBRATION_CASES: list[dict[str, Any]] = [
    {
        "id": "trivial_one_liner",
        "query": "What's the capital of France?",
        "criteria": (
            "The agent answers the question directly without scaffolding a multi-step plan.",
            "The agent does NOT call write_todos for this request — it "
            "is a single-step factual question that does not warrant a "
            "checklist.",
        ),
    },
    {
        "id": "trivial_format_change",
        "query": (
            "Rename the variable `x` to `count` in this snippet:\n"
            "    x = 0\n"
            "    for item in items:\n"
            "        x += 1\n"
            "    return x"
        ),
        "criteria": (
            "The agent performs the rename and returns the updated snippet directly.",
            "The agent does NOT call write_todos — a single mechanical "
            "rename does not warrant a multi-step plan.",
        ),
    },
    {
        "id": "ambiguous_no_scope",
        "query": "Make the CLI faster.",
        "criteria": (
            "The agent asks at least one clarifying question before "
            "committing to a plan — at minimum about which slowness the "
            "user means (startup, response latency, specific command).",
            "The agent does NOT call write_todos with assumed scope — "
            "the request is too ambiguous to plan against without more "
            "information.",
        ),
    },
    {
        "id": "explicit_no_plan_request",
        "query": (
            "Quickly — without making a plan or asking questions — add "
            "a one-line print statement to libs/cli/deepagents_cli/"
            "main.py that says 'hello' at the very start of main()."
        ),
        "criteria": (
            "The agent respects the explicit 'no plan' instruction and does NOT call write_todos.",
            "The agent proceeds directly to the requested edit (or its "
            "best approximation given available tools).",
        ),
    },
]


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    PLAN_CALIBRATION_CASES,
    ids=[c["id"] for c in PLAN_CALIBRATION_CASES],
)
def test_plan_calibration(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Agent calibrates plan depth to request scope (no over-planning)."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=case["query"],
        scorer=(
            TrajectoryScorer().success(
                llm_judge(*case["criteria"], include_tool_calls=True),
            )
        ),
    )
