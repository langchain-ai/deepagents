"""Mock SaaS tools for the tool-selection eval suite.

Lightweight stubs that return a fixed string, used to test whether the agent
selects the correct tool(s) from a pool given direct, indirect, and multi-step
requests.

Extracted from `tests/evals/test_tool_selection.py` so both the pytest suite
and the Harbor sandbox dispatcher share the same tool definitions.
"""

from __future__ import annotations

from langchain_core.tools import tool


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
    return (
        f"Top 3 results for {query!r}:\n"
        f"1. Official documentation page covering the topic.\n"
        f"2. Recent blog post discussing updates and features.\n"
        f"3. Community discussion thread with examples."
    )


@tool
def calendar_create_event(title: str, date: str, attendees: list[str]) -> str:
    """Create a calendar event."""
    return f"Created event '{title}' on {date} with {', '.join(attendees)}"


TOOL_SELECTION_TOOLS = [
    slack_send_dm,
    slack_post_channel,
    github_create_issue,
    github_create_pr,
    linear_create_issue,
    gmail_send_email,
    web_search,
    calendar_create_event,
]
