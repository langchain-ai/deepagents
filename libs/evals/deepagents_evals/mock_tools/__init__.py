"""Shared mock tool definitions for the Deep Agents eval suite.

All mock tools used by the eval test suite live here so that both the pytest
tests and the Harbor sandbox dispatcher import from the same source. Each
submodule corresponds to one eval category's tool set:

- ``tool_selection`` — 8 mock SaaS tools (Slack, GitHub, Linear, Gmail, etc.)
- ``relational`` — 16 relational-data lookup/search tools
- ``incident_graph`` — 34 incident-management graph tools + error middleware
- ``subagents`` — weather stub tool for subagent delegation evals
- ``constraints`` — word-count tool for the constraint-satisfaction eval
"""

from deepagents_evals.mock_tools.constraints import count_words
from deepagents_evals.mock_tools.incident_graph import (
    INCIDENT_GRAPH_TOOLS,
    incident_graph_tool_error_middleware,
)
from deepagents_evals.mock_tools.relational import (
    RELATIONAL_TOOL_IMPLEMENTATIONS,
    RELATIONAL_TOOL_NAMES,
    RELATIONAL_TOOLS,
)
from deepagents_evals.mock_tools.subagents import get_weather_fake
from deepagents_evals.mock_tools.tool_selection import TOOL_SELECTION_TOOLS

__all__ = [
    "INCIDENT_GRAPH_TOOLS",
    "RELATIONAL_TOOLS",
    "RELATIONAL_TOOL_IMPLEMENTATIONS",
    "RELATIONAL_TOOL_NAMES",
    "TOOL_SELECTION_TOOLS",
    "count_words",
    "get_weather_fake",
    "incident_graph_tool_error_middleware",
]
