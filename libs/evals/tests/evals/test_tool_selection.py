"""Eval tests for tool selection behavior.

Tests whether the agent selects the correct tool(s) from a set of available
tools given direct, indirect, and multi-step user requests. Ported from the
agent-builder-graphs tool-discovery eval suite and adapted for deepagents.

The agent is given a pool of mock tools and a user query. Assertions check that
the agent called the right tool(s) and avoided calling wrong ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain_core.tools import tool

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
)

pytestmark = [pytest.mark.eval_category("tool_use")]
"""Apply tool_use category to all tests in this module. Tier is set per-test."""

# ---------------------------------------------------------------------------
# Mock tools — lightweight stubs that return a fixed string
# ---------------------------------------------------------------------------


@tool
def slack_send_dm(user_id: str, message: str) -> str:
    """Send a direct message to a user on Slack."""
    return f"Sent DM to {user_id}: {message}"


@tool
def slack_post_channel(channel: str, message: str) -> str:
    """Post a message to a Slack channel."""
    return f"Posted to #{channel}: {message}"


@tool
def github_create_issue(repo: str, title: str, body: str) -> str:
    """Create a new GitHub issue."""
    return f"Created issue '{title}' in {repo} — {body}"


@tool
def github_create_pr(repo: str, title: str, head: str, base: str) -> str:
    """Create a pull request on GitHub."""
    return f"Created PR '{title}' in {repo} ({head} -> {base})"


@tool
def linear_create_issue(team: str, title: str, description: str) -> str:
    """Create a new issue in Linear."""
    return f"Created Linear issue '{title}' in {team} — {description}"


@tool
def gmail_send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail."""
    return f"Sent email to {to}: {subject} — {body}"


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


@tool
def calendar_create_event(title: str, date: str, attendees: list[str]) -> str:
    """Create a calendar event."""
    return f"Created event '{title}' on {date} with {', '.join(attendees)}"


ALL_TOOLS = [
    slack_send_dm,
    slack_post_channel,
    github_create_issue,
    github_create_pr,
    linear_create_issue,
    gmail_send_email,
    web_search,
    calendar_create_event,
]


def _make_agent(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Build a deep agent with the standard tool pool for these evals."""
    return create_deep_agent(model=model, tools=ALL_TOOLS)


# ---------------------------------------------------------------------------
# Test cases — direct requests (user names the tool explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_direct_request_slack_dm(model: BaseChatModel) -> None:
    """Agent uses the Slack DM tool when explicitly asked."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Send a Slack DM to user U12345 saying 'Hello from evals'",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="slack_send_dm", args_contains={"user_id": "U12345"})],
            )
            .success(
                final_text_contains("U12345", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_direct_request_github_pr(model: BaseChatModel) -> None:
    """Agent uses the GitHub PR tool when explicitly asked."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Create a pull request in repo langchain-ai/deepagents with title 'fix: typo' from branch fix-typo to main",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="github_create_pr",
                        args_contains={"repo": "langchain-ai/deepagents"},
                    )
                ],
            )
            .success(
                final_text_contains("fix-typo", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_direct_request_multiple_tools(model: BaseChatModel) -> None:
    """Agent uses both Linear and GitHub issue tools when asked for both."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Create an issue titled 'Bug: crash on login' in the Linear team 'engineering' "
            "and also create a GitHub issue in repo org/app with the same title and body 'Tracking in Linear'"
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[
                    tool_call(name="linear_create_issue"),
                    tool_call(name="github_create_issue"),
                ],
            )
            .success(
                final_text_contains("crash on login", case_insensitive=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Indirect requests (user describes intent, agent infers tool)
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_indirect_schedule_meeting(model: BaseChatModel) -> None:
    """Agent infers the calendar tool from a scheduling request."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Schedule a team standup for tomorrow at 10am with alice@co.com and bob@co.com",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="calendar_create_event")],
            )
            .success(
                final_text_contains("standup", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_indirect_notify_team(model: BaseChatModel) -> None:
    """Agent infers the Slack channel post tool from a 'notify the team' request."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Notify the #deployments channel that v2.0 has been released",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="slack_post_channel",
                        args_contains={"channel": "deployments"},
                    )
                ],
            )
            .success(
                final_text_contains("v2.0", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_indirect_email_report(model: BaseChatModel) -> None:
    """Agent infers the Gmail tool from 'email a report' request."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Email the weekly status report to manager@company.com with subject 'Week 10 Status'",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="gmail_send_email",
                        args_contains={"to": "manager@company.com"},
                    )
                ],
            )
            .success(
                final_text_contains("Week 10", case_insensitive=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Multi-step: tool chaining
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_chain_search_then_email(model: BaseChatModel) -> None:
    """Agent searches the web then emails results — two tools in sequence."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Search for 'LangGraph 0.3 release notes' and email a summary to team@co.com with subject 'LangGraph Update'",
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[
                    tool_call(name="web_search"),
                    tool_call(name="gmail_send_email", args_contains={"to": "team@co.com"}),
                ],
            )
            .success(
                final_text_contains("team@co.com", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_chain_create_issue_then_notify(model: BaseChatModel) -> None:
    """Agent creates a GitHub issue then notifies Slack — two tools in sequence."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Create a GitHub issue in org/backend titled 'Fix memory leak' with body 'OOM in prod', "
            "then post a message to #incidents saying the issue was created"
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[
                    tool_call(name="github_create_issue"),
                    tool_call(
                        name="slack_post_channel",
                        args_contains={"channel": "incidents"},
                    ),
                ],
            )
            .success(
                final_text_contains("memory leak", case_insensitive=True),
            )
        ),
    )
