"""Eval tests for multi-turn memory behavior.

Tests whether the agent:
1. Picks up on implicit user preferences revealed through conversation (not
    explicit "remember this" instructions).
2. Records explicit user instructions given during multi-turn exchanges.
3. Does NOT persist transient or one-off information from multi-turn exchanges.

Ported from the agent-builder-graphs eval_remember_info suite and adapted for
deepagents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage

from tests.evals.llm_judge import llm_judge
from tests.evals.utils import (
    TrajectoryScorer,
    file_contains,
    file_excludes,
    final_text_contains,
    run_agent,
    tool_call,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [pytest.mark.eval_category("memory"), pytest.mark.eval_tier("baseline")]
"""Apply memory category and baseline tier to all tests in this module."""

MEMORY_PATH = "/project/AGENTS.md"
"""Virtual path to the agent's memory file used across all memory multi-turn evals."""

MEMORY_SEED = "# Project Memory\n\nUser preferences and project facts.\n"
"""Initial content seeded into the memory file before each test run."""


# ---------------------------------------------------------------------------
# Implicit preference extraction — user reveals preference through
# conversation rather than explicitly asking the agent to remember.
# ---------------------------------------------------------------------------

IMPLICIT_REMEMBER_CASES = [
    {
        "id": "implicit_language_preference",
        "messages": [
            HumanMessage(content="How can I create a list?"),
            AIMessage(content="In Python, you define a list like this: nums = [1, 2, 3]"),
            HumanMessage(
                content="Sorry I only know how to write C++, can you show me in C++ instead?"
            ),
        ],
        "should_contain": "C++",
        "criteria": (
            "The agent should write a note to its memory file recording that the user prefers or uses C++.",
            "The memory update should NOT contain a full C++ code tutorial — just the preference.",
        ),
    },
    {
        "id": "implicit_working_hours",
        "messages": [
            HumanMessage(content="When is Alice free to meet for a call about the project?"),
            AIMessage(
                content="Alice is only free before 10am and after 8pm Eastern Time. Should I suggest one of those slots?"
            ),
            HumanMessage(
                content="I only take meetings between 9am and 5pm Eastern Time. Let's find another time."
            ),
        ],
        "should_contain": "9am",
        "criteria": (
            "The agent should record the user's meeting availability (9am-5pm Eastern) in its memory file.",
            "The memory update should capture the timezone (Eastern or ET or EST).",
        ),
    },
    {
        "id": "implicit_domain_expertise",
        "messages": [
            HumanMessage(content="Can you review the pull request for the JavaScript SDK?"),
            AIMessage(content="Sure, I'll take a look at the JavaScript SDK PR."),
            HumanMessage(
                content="Actually, I'm not familiar with JavaScript at all. I only review Python PRs. Please don't assign JS work to me."
            ),
        ],
        "should_contain": "Python",
        "criteria": (
            "The agent should record that the user only reviews Python PRs in its memory file.",
            "The memory should note that the user is not familiar with JavaScript.",
        ),
    },
]


# ---------------------------------------------------------------------------
# Explicit memory update — user directly asks agent to remember something.
# ---------------------------------------------------------------------------

EXPLICIT_REMEMBER_CASES = [
    {
        "id": "explicit_formal_language",
        "messages": [
            HumanMessage(
                content="Update your instructions to always use formal language when responding to me."
            ),
        ],
        "should_contain": "formal",
        "criteria": (
            "The agent should write an instruction about using formal language into its memory file.",
        ),
    },
    {
        "id": "explicit_no_emojis",
        "messages": [
            HumanMessage(content="Write me a short congratulations message for a teammate."),
            AIMessage(content="Congrats on the launch! 🎉🚀 Amazing work from the whole team! 🙌"),
            HumanMessage(
                content="No emojis please. Remember: never use emojis in anything you write for me."
            ),
        ],
        "should_contain": "emoji",
        "criteria": (
            "The agent should record the no-emojis preference in its memory file.",
            "The memory update should be a durable instruction, not a one-off correction.",
        ),
    },
    {
        "id": "explicit_timezone_preference",
        "messages": [
            HumanMessage(content="What time is the standup meeting?"),
            AIMessage(content="The standup is at 9:00 AM UTC every weekday."),
            HumanMessage(content="Always show times in Pacific Time for me. Remember that."),
        ],
        "should_contain": "Pacific",
        "criteria": (
            "The agent should record the user's timezone preference (Pacific Time) in its memory file.",
            "The memory update should be a durable preference, not a one-off conversion.",
        ),
    },
]


# ---------------------------------------------------------------------------
# Should NOT remember — transient info in multi-turn context.
# ---------------------------------------------------------------------------

# expected_answer: optional substring the agent's final text must contain,
# verifying it actually responded. None when the answer is open-ended.
SHOULD_NOT_REMEMBER_CASES = [
    {
        "id": "transient_mood",
        "messages": [
            HumanMessage(content="I'm exhausted today, barely slept last night."),
            AIMessage(content="Sorry to hear that! Hope you can rest soon. How can I help?"),
            HumanMessage(
                content="Just help me rename 'processData' to 'process_data' in the codebase."
            ),
        ],
        "should_not_contain": "exhausted",
        "expected_answer": "process_data",
    },
    {
        "id": "transient_one_off_search",
        "messages": [
            HumanMessage(
                content="I need to find a good Italian restaurant near downtown for tonight."
            ),
            AIMessage(content="I can help with that! Do you have a preference for price range?"),
            HumanMessage(content="Mid-range. Just give me a recommendation."),
        ],
        "should_not_contain": "Italian",
        "expected_answer": None,
    },
    {
        "id": "transient_weather_question",
        "messages": [
            HumanMessage(content="What's the weather like in San Francisco today?"),
            AIMessage(
                content="I don't have real-time weather data, but SF is typically foggy this time of year."
            ),
            HumanMessage(content="Good to know, thanks."),
        ],
        "should_not_contain": "San Francisco",
        "expected_answer": None,
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    IMPLICIT_REMEMBER_CASES,
    ids=[c["id"] for c in IMPLICIT_REMEMBER_CASES],
)
def test_implicit_preference_remembered(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Agent picks up on an implicit user preference from multi-turn conversation and writes it to memory."""
    agent = create_deep_agent(model=model, memory=[MEMORY_PATH])
    run_agent(
        agent,
        model=model,
        query=case["messages"],
        initial_files={MEMORY_PATH: MEMORY_SEED},
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[tool_call(name="edit_file", args_contains={"file_path": MEMORY_PATH})],
            )
            .success(
                file_contains(MEMORY_PATH, case["should_contain"]),
                llm_judge(*case["criteria"], include_tool_calls=True),
            )
        ),
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    EXPLICIT_REMEMBER_CASES,
    ids=[c["id"] for c in EXPLICIT_REMEMBER_CASES],
)
def test_explicit_preference_remembered(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Agent writes an explicit user instruction into memory."""
    agent = create_deep_agent(model=model, memory=[MEMORY_PATH])
    run_agent(
        agent,
        model=model,
        query=case["messages"],
        initial_files={MEMORY_PATH: MEMORY_SEED},
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[tool_call(name="edit_file", args_contains={"file_path": MEMORY_PATH})],
            )
            .success(
                file_contains(MEMORY_PATH, case["should_contain"]),
                llm_judge(*case["criteria"], include_tool_calls=True),
            )
        ),
    )


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    SHOULD_NOT_REMEMBER_CASES,
    ids=[c["id"] for c in SHOULD_NOT_REMEMBER_CASES],
)
def test_transient_info_not_persisted(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Agent does NOT write transient conversational info into durable memory."""
    agent = create_deep_agent(model=model, memory=[MEMORY_PATH])
    assertions = [
        file_contains(MEMORY_PATH, "Project Memory"),
        file_excludes(MEMORY_PATH, case["should_not_contain"]),
    ]
    if case.get("expected_answer"):
        assertions.append(final_text_contains(case["expected_answer"]))
    run_agent(
        agent,
        model=model,
        query=case["messages"],
        initial_files={MEMORY_PATH: MEMORY_SEED},
        scorer=TrajectoryScorer().success(*assertions),
    )
