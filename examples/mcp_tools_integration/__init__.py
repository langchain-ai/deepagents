"""MCP Tools Integration Example for DeepAgents.

This package provides custom MCP-based tools that can be registered with DeepAgents:
- google_search_and_summarize: Google search with web page fetching
- rag_search: RAG-based document search
- weather_forecast: Weather forecast using OpenWeatherMap
- sentinel_search: Sentinel satellite imagery search
- arxiv_search: arXiv paper search and download
"""

from mcp_tools import (
    ALL_MCP_TOOLS,
    ARXIV_TOOLS,
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
    "google_search_and_summarize",
    "rag_search",
    "weather_forecast",
    "sentinel_search",
    "arxiv_search",
    "arxiv_download_paper",
    "arxiv_read_paper",
    "arxiv_list_papers",
    "ALL_MCP_TOOLS",
    "ARXIV_TOOLS",
]
