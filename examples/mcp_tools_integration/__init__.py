"""MCP Tools Integration Example for DeepAgents.

This package provides custom MCP-based tools that can be registered with DeepAgents:
- google_search_and_summarize: Google search with web page fetching
- rag_search: RAG-based document search
- weather_forecast: Weather forecast using OpenWeatherMap
- sentinel_search: Sentinel satellite imagery search
"""

from mcp_tools import (
    ALL_MCP_TOOLS,
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
    "ALL_MCP_TOOLS",
]
