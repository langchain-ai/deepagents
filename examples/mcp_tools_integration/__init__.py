"""MCP Tools Integration Example for DeepAgents.

This package provides custom MCP-based tools that can be registered with DeepAgents:

Internal Tools:
- rag_search: RAG-based document search

External Tools:
- google_search_and_summarize: Google search with web page fetching
- weather_forecast: 5-day weather forecast
- sentinel_search: Sentinel satellite imagery search
- arxiv_search: arXiv paper search
- arxiv_download_paper: Download arXiv papers
- arxiv_read_paper: Read downloaded arXiv papers
- arxiv_list_papers: List downloaded arXiv papers

Usage:
    from mcp_tools import google_search_and_summarize, ALL_MCP_TOOLS

    # Single tool
    result = google_search_and_summarize("Python tutorial")

    # With DeepAgents
    agent = create_deep_agent(model="openai:gpt-4o", tools=ALL_MCP_TOOLS)
"""

from mcp_tools import (
    ALL_MCP_TOOLS,
    ARXIV_TOOLS,
    EXTERNAL_TOOLS,
    INTERNAL_TOOLS,
    arxiv_download_paper,
    arxiv_list_papers,
    arxiv_read_paper,
    arxiv_search,
    google_search_and_summarize,
    rag_search,
    sentinel_search,
    weather_forecast,
)

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
