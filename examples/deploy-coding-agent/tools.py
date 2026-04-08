"""Local tools for the coding agent example.

Demonstrates the ``[tools].python_file`` plumbing in ``deepagents deploy``.
The bundler copies this file into the build dir and the generated graph
auto-discovers ``BaseTool`` instances and ``@tool``-decorated callables
at module level.

Tools exposed
-------------

- ``tavily_search`` — web search via Tavily, useful for the coding agent
  when it needs to look up library docs, error messages, or framework
  conventions. Requires ``TAVILY_API_KEY`` in the deployment env.
- ``deepagents_smoke_marker`` — a fixed-string smoke test used by
  ``test_deployment.py`` to verify the local tools file actually loaded
  in the deployed image.
"""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# Auto-discovered by the bundler's tools loader because it's a ``BaseTool``
# instance at module level. Requires ``TAVILY_API_KEY`` in the env.
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
)


@tool
def deepagents_smoke_marker() -> str:
    """Return a fixed marker string proving the local tools file loaded.

    The deployment test asserts the agent can call this and round-trip
    the marker, which proves ``[tools].python_file`` is wired correctly
    in the hub-backed deploy template.
    """
    return "deepagents-tools-smoke-ok"
