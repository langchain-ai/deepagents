"""Google Search and Summarize tool.

This tool searches Google and fetches web page contents for summarization.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .._utils import (
    GOOGLE_SEARCH_SERVER,
    WEB_FETCH_SERVER,
    extract_payload,
    run_async,
)


async def _google_search_and_summarize_async(
    query: str,
    num_results: int = 5,
    fetch_top_n: int = 3,
    max_chars_per_page: int = 2500,
) -> dict[str, Any]:
    """Async implementation of Google search with web fetching."""
    if not os.path.exists(GOOGLE_SEARCH_SERVER):
        return {
            "success": False,
            "query": query,
            "error": f"Google search MCP server not found: {GOOGLE_SEARCH_SERVER}. "
                     f"Set MCP_GOOGLE_SEARCH_SERVER environment variable.",
            "sources": [],
        }
    if not os.path.exists(WEB_FETCH_SERVER):
        return {
            "success": False,
            "query": query,
            "error": f"Web fetch MCP server not found: {WEB_FETCH_SERVER}. "
                     f"Set MCP_WEB_FETCH_SERVER environment variable.",
            "sources": [],
        }

    try:
        client = MultiServerMCPClient({
            "gsearch": {
                "transport": "stdio",
                "command": "python",
                "args": [GOOGLE_SEARCH_SERVER],
            },
            "fetcher": {
                "transport": "stdio",
                "command": "python",
                "args": [WEB_FETCH_SERVER],
            }
        })

        async with AsyncExitStack() as stack:
            gsession = await stack.enter_async_context(client.session("gsearch"))
            fsession = await stack.enter_async_context(client.session("fetcher"))

            gtools = await load_mcp_tools(gsession)
            ftools = await load_mcp_tools(fsession)

            google_tool = next(
                (t for t in gtools if "google" in t.name.lower() or "search" in t.name.lower()),
                None
            )
            fetch_tool = next(
                (t for t in ftools if "fetch" in t.name.lower() or "extract" in t.name.lower()),
                None
            )

            if google_tool is None:
                return {
                    "success": False,
                    "query": query,
                    "error": f"No google search tool found. Available tools: {[t.name for t in gtools]}",
                    "sources": [],
                }

            # Step 1: Search
            search_raw = await google_tool.ainvoke({
                "query": query,
                "num_results": num_results,
                "lang": "lang_ko",
                "country": "KR",
                "safe": "active",
            })
            search_payload = extract_payload(search_raw)
            items = search_payload.get("items", [])[:fetch_top_n]

            if not items:
                return {
                    "success": False,
                    "query": query,
                    "error": "No search results found",
                    "sources": [],
                }

            # Step 2: Fetch pages concurrently (only if fetch tool available)
            if fetch_tool is None:
                sources = [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "text": "",
                    }
                    for item in items
                ]
                return {
                    "success": True,
                    "query": query,
                    "num_sources": len(sources),
                    "sources": sources,
                }

            sem = asyncio.Semaphore(3)

            async def fetch_one(item: dict[str, Any]) -> dict[str, Any]:
                url = item.get("link", "")
                if not url:
                    return {
                        "url": "",
                        "title": item.get("title", ""),
                        "text": "",
                        "error": "missing url"
                    }
                async with sem:
                    raw = await fetch_tool.ainvoke({
                        "url": url,
                        "max_chars": max_chars_per_page,
                        "timeout": 20
                    })
                payload = extract_payload(raw)
                return {
                    "title": item.get("title", "") or payload.get("title", ""),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "text": payload.get("text", ""),
                    "content_type": payload.get("content_type", ""),
                }

            results = await asyncio.gather(
                *[fetch_one(item) for item in items],
                return_exceptions=True
            )

            sources = []
            for item, result in zip(items, results):
                if isinstance(result, Exception):
                    continue
                if isinstance(result, dict):
                    sources.append(result)

            return {
                "success": True,
                "query": query,
                "num_sources": len(sources),
                "sources": sources,
            }

    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "sources": [],
        }


def google_search_and_summarize(
    query: str,
    num_results: int = 5,
    fetch_top_n: int = 3,
    max_chars_per_page: int = 2500,
) -> dict[str, Any]:
    """Search Google and fetch web page contents for summarization.

    This tool performs a Google search, fetches the top results, and extracts
    their content for analysis. Use this for finding current information,
    news, documentation, or any web-based research.

    Args:
        query: The search query (be specific and detailed for better results)
        num_results: Number of search results to retrieve (default: 5)
        fetch_top_n: Number of top results to fetch full content from (default: 3)
        max_chars_per_page: Maximum characters to extract per page (default: 2500)

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - num_sources: Number of sources successfully fetched
        - sources: List of source dictionaries, each with:
            - title: Page title
            - url: Page URL
            - snippet: Search result snippet
            - text: Extracted page content
            - content_type: Content MIME type

    Example:
        result = google_search_and_summarize("트럼프 임기 기간")
        for source in result["sources"]:
            print(f"Title: {source['title']}")
            print(f"Content: {source['text'][:500]}...")
    """
    return run_async(_google_search_and_summarize_async(
        query=query,
        num_results=num_results,
        fetch_top_n=fetch_top_n,
        max_chars_per_page=max_chars_per_page,
    ))
