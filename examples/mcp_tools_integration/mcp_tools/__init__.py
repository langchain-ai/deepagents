"""MCP Tools Package for DeepAgents.

This package provides custom MCP-based tools that can be registered with DeepAgents:

Internal Tools (local/private services):
- rag_search: RAG-based document search

External Tools (third-party services):
- google_search_and_summarize: Google search with web page fetching
- weather_forecast: 5-day weather forecast using OpenWeatherMap
- sentinel_search: Sentinel satellite imagery search
- arxiv_search: arXiv paper search
- arxiv_download_paper: Download arXiv papers
- arxiv_read_paper: Read downloaded arXiv papers
- arxiv_list_papers: List downloaded arXiv papers

Usage:
    from mcp_tools import (
        google_search_and_summarize,
        rag_search,
        weather_forecast,
        sentinel_search,
        arxiv_search,
        ALL_MCP_TOOLS,
    )

    agent = create_deep_agent(
        model="openai:gpt-4o",
        tools=ALL_MCP_TOOLS,
    )
"""

# Internal tools
from .internal import rag_search

# External tools
from .external import (
    arxiv_download_paper,
    arxiv_list_papers,
    arxiv_read_paper,
    arxiv_search,
    google_search_and_summarize,
    sentinel_search,
    weather_forecast,
)

# Tool collections
ALL_MCP_TOOLS = [
    google_search_and_summarize,
    rag_search,
    weather_forecast,
    sentinel_search,
    arxiv_search,
]

ARXIV_TOOLS = [
    arxiv_search,
    arxiv_download_paper,
    arxiv_read_paper,
    arxiv_list_papers,
]

INTERNAL_TOOLS = [
    rag_search,
]

EXTERNAL_TOOLS = [
    google_search_and_summarize,
    weather_forecast,
    sentinel_search,
    arxiv_search,
    arxiv_download_paper,
    arxiv_read_paper,
    arxiv_list_papers,
]

__all__ = [
    # Internal tools
    "rag_search",
    # External tools
    "google_search_and_summarize",
    "weather_forecast",
    "sentinel_search",
    "arxiv_search",
    "arxiv_download_paper",
    "arxiv_read_paper",
    "arxiv_list_papers",
    # Tool collections
    "ALL_MCP_TOOLS",
    "ARXIV_TOOLS",
    "INTERNAL_TOOLS",
    "EXTERNAL_TOOLS",
]
