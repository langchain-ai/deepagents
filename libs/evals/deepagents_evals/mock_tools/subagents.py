"""Mock tools for the subagent delegation eval suite.

Extracted from `tests/evals/test_subagents.py` so both the pytest suite and
the Harbor sandbox dispatcher share the same tool definitions.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def get_weather_fake(location: str) -> str:  # noqa: ARG001
    """Return a fixed weather response for eval scenarios."""
    return "It's sunny at 89 degrees F"
