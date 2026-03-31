"""Eval tests for followup question quality.

Tests whether the agent asks relevant, non-redundant followup questions when
given underspecified requests. Uses the LLM-as-judge to evaluate semantic
quality of the questions.

Ported from the agent-builder-graphs followup eval suite and adapted for
deepagents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.llm_judge import llm_judge
from tests.evals.utils import TrajectoryScorer, run_agent

pytestmark = [pytest.mark.eval_category("conversation")]

# ---------------------------------------------------------------------------
# Test cases — each describes a user request (varying in specificity) and
# criteria for evaluating the agent's followup questions.
# ---------------------------------------------------------------------------

FOLLOWUP_CASES = [
    {
        "id": "vague_data_analysis",
        "query": "Analyze my data",
        "criteria": (
            "The agent should ask what data source or file the user wants analyzed.",
            "The agent should ask what kind of analysis the user needs (summary stats, trends, anomalies, etc.).",
            "The agent should NOT assume a specific file format or tool without asking.",
        ),
    },
    {
        "id": "vague_send_report",
        "query": "Send a report to my team every week",
        "criteria": (
            "The agent should ask what the report should contain or what data to include.",
            "The agent should ask how the report should be delivered (email, Slack, etc.).",
            "The agent should NOT ask about scheduling details since the user already specified 'every week'.",
        ),
    },
    {
        "id": "vague_monitor_system",
        "query": "Monitor our production system and alert me if something goes wrong",
        "criteria": (
            "The agent should ask what metrics or signals define 'something going wrong'.",
            "The agent should ask how the user wants to be alerted (Slack, email, PagerDuty, etc.).",
            "The agent should NOT assume specific thresholds without asking.",
        ),
    },
    {
        "id": "vague_summarize_emails",
        "query": "I want you to summarize my email every day",
        "criteria": (
            "The agent should ask about the preferred summary format or level of detail.",
            "The agent should assume summaries apply to all emails and should NOT ask which emails to summarize.",
            "The followup questions should remain concise and directly relevant.",
        ),
    },
    {
        "id": "vague_customer_support",
        "query": "Help me respond to customer questions faster",
        "criteria": (
            "The agent should ask where customer questions come from (email, Slack, support tool, etc.).",
            "The agent should ask about the domain or product to understand what kinds of questions to expect.",
            "The agent should NOT ask whether responses should be automated vs. drafted unless the distinction is unclear from context.",
        ),
    },
    {
        "id": "detailed_calendar_brief",
        "query": "Every morning at 5am, look at my Google Calendar and send me a brief of what's upcoming for the day",
        "criteria": (
            "The agent should ask exactly one followup question about how the user wants to receive the brief (email, Slack, SMS, etc.).",
            "The agent should NOT ask about schedule timing or scope because those were already provided.",
        ),
    },
]


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
@pytest.mark.parametrize(
    "case",
    FOLLOWUP_CASES,
    ids=[c["id"] for c in FOLLOWUP_CASES],
)
def test_followup_question_quality(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Agent asks relevant followup questions for an underspecified request."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=case["query"],
        scorer=(
            TrajectoryScorer().success(
                llm_judge(*case["criteria"]),
            )
        ),
    )
