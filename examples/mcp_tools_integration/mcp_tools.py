"""Custom MCP-based tools for deepagents.

This module provides 4 MCP tools that can be registered with deepagents:
1. google_search_and_summarize - Google search with web page fetching and summarization
2. rag_search - RAG-based document/paper search
3. weather_forecast - Weather forecast using OpenWeatherMap
4. sentinel_search - Sentinel satellite imagery search

Usage:
    from mcp_tools import (
        google_search_and_summarize,
        rag_search,
        weather_forecast,
        sentinel_search,
    )

    agent = create_deep_agent(
        model="openai:gpt-4o",
        tools=[google_search_and_summarize, rag_search, weather_forecast, sentinel_search],
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


# =============================================================================
# Configuration - MCP Server Paths
# =============================================================================

# Default MCP server paths (override via environment variables)
GOOGLE_SEARCH_SERVER = os.getenv(
    "MCP_GOOGLE_SEARCH_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/google_search_server.py"
)
WEB_FETCH_SERVER = os.getenv(
    "MCP_WEB_FETCH_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/web_fetch_server.py"
)
RAG_SERVER = os.getenv(
    "MCP_RAG_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/spaceops_rag.py"
)
WEATHER_SERVER = os.getenv(
    "MCP_WEATHER_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/weather.py"
)
SENTINEL_SERVER = os.getenv(
    "MCP_SENTINEL_SERVER",
    "/workspace/nas/users/hcnoh/LLM/MCP_server/sentinel_server.py"
)


# =============================================================================
# Helper Functions
# =============================================================================

def _try_parse_json(s: str) -> dict | None:
    """Try to parse a JSON string, handling BOM and wrapped content."""
    s = (s or "").strip().lstrip("\ufeff")

    # Direct parse attempt
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract first JSON object from text
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _extract_payload(mcp_output: Any) -> dict[str, Any]:
    """
    Extract payload from MCP output wrapper.

    MCP outputs can be:
    - list[{"text": "<json-string>"}]
    - dict{"text": "<json-string>"}
    - dict (direct payload)
    """
    if mcp_output is None:
        return {}

    # Handle list - recurse on first element
    if isinstance(mcp_output, list):
        if not mcp_output:
            return {}
        return _extract_payload(mcp_output[0])

    # Handle dict
    if isinstance(mcp_output, dict):
        # Already a payload with items
        if "items" in mcp_output and isinstance(mcp_output.get("items"), list):
            return mcp_output

        # Wrapper with text field - parse and recurse
        if "text" in mcp_output:
            t = mcp_output.get("text", "")
            if isinstance(t, str):
                parsed = _try_parse_json(t)
                if parsed is None:
                    return {"text": t}
                return _extract_payload(parsed)
            return mcp_output

        return mcp_output

    return {}


def _run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, create a task
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)


# =============================================================================
# Tool 1: Google Search and Summarize
# =============================================================================

async def _google_search_and_summarize_async(
    query: str,
    num_results: int = 5,
    fetch_top_n: int = 3,
    max_chars_per_page: int = 2500,
) -> dict[str, Any]:
    """Async implementation of Google search with web fetching."""
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

        google_tool = next(t for t in gtools if t.name.endswith("google_search"))
        fetch_tool = next(t for t in ftools if t.name.endswith("fetch_and_extract"))

        # Step 1: Search
        search_raw = await google_tool.ainvoke({
            "query": query,
            "num_results": num_results,
            "lang": "lang_ko",
            "country": "KR",
            "safe": "active",
        })
        search_payload = _extract_payload(search_raw)
        items = search_payload.get("items", [])[:fetch_top_n]

        if not items:
            return {
                "success": False,
                "query": query,
                "error": "No search results found",
                "sources": [],
            }

        # Step 2: Fetch pages concurrently
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
            payload = _extract_payload(raw)
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
    return _run_async(_google_search_and_summarize_async(
        query=query,
        num_results=num_results,
        fetch_top_n=fetch_top_n,
        max_chars_per_page=max_chars_per_page,
    ))


# =============================================================================
# Tool 2: RAG Search
# =============================================================================

async def _rag_search_async(
    query: str,
    k: int = 4,
    vectorstore_dir: str = "vectorstore.db",
) -> dict[str, Any]:
    """Async implementation of RAG search."""
    client = MultiServerMCPClient({
        "local_rag": {
            "transport": "stdio",
            "command": "python",
            "args": [RAG_SERVER],
            "env": {"RAG_VECTORSTORE_DIR": vectorstore_dir},
        }
    })

    async with client.session("local_rag") as session:
        tools = await load_mcp_tools(session)
        retrieve_tool = next(t for t in tools if t.name.endswith("rag_retrieve"))

        raw = await retrieve_tool.ainvoke({"query": query, "k": k})
        payload = _extract_payload(raw)

    # Convert sources to standardized format
    sources = payload.get("sources", [])
    documents = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        documents.append({
            "content": s.get("content", ""),
            "metadata": s.get("metadata", {}),
            "rank": s.get("rank", 0),
        })

    return {
        "success": True,
        "query": query,
        "k": k,
        "num_documents": len(documents),
        "documents": documents,
    }


def rag_search(
    query: str,
    k: int = 4,
    vectorstore_dir: str = "vectorstore.db",
) -> dict[str, Any]:
    """Search documents using RAG (Retrieval-Augmented Generation).

    This tool searches a local vector store for relevant documents based on
    semantic similarity. Use this for searching papers, research documents,
    technical documentation, or any pre-indexed content.

    Args:
        query: The search query (natural language question or keywords)
        k: Number of top documents to retrieve (default: 4)
        vectorstore_dir: Path to the vector store directory (default: "vectorstore.db")

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - k: Number of documents requested
        - num_documents: Number of documents actually retrieved
        - documents: List of document dictionaries, each with:
            - content: Document text content
            - metadata: Document metadata (source, page, etc.)
            - rank: Relevance ranking

    Example:
        result = rag_search("SpaceOps 논문 중 SCIENCE GOAL DRIVEN OBSERVING")
        for doc in result["documents"]:
            print(f"Content: {doc['content'][:300]}...")
            print(f"Source: {doc['metadata'].get('source', 'Unknown')}")
    """
    return _run_async(_rag_search_async(
        query=query,
        k=k,
        vectorstore_dir=vectorstore_dir,
    ))


# =============================================================================
# Tool 3: Weather Forecast
# =============================================================================

async def _weather_forecast_async(
    city: str,
    units: str = "c",
    lang: str = "kr",
) -> dict[str, Any]:
    """Async implementation of weather forecast."""
    if not city.strip():
        return {
            "success": False,
            "error": "City name is required",
            "city": city,
        }

    client = MultiServerMCPClient({
        "openweather": {
            "transport": "stdio",
            "command": "python",
            "args": [WEATHER_SERVER],
            "env": {"OWM_API_KEY": os.environ.get("OPENWEATHER_API_KEY", "")},
        }
    })

    tools = await client.get_tools()
    weather_tool = next(t for t in tools if t.name.endswith("get_weather_5day"))

    result = await weather_tool.ainvoke({
        "city": city,
        "units": units,
        "lang": lang,
    })

    payload = _extract_payload(result)

    return {
        "success": True,
        "city": city,
        "units": units,
        "forecast": payload,
    }


def weather_forecast(
    city: str,
    units: str = "c",
    lang: str = "kr",
) -> dict[str, Any]:
    """Get 5-day weather forecast for a city.

    This tool retrieves weather forecast data from OpenWeatherMap API.
    Use this for weather-related queries, outdoor activity planning,
    or any task requiring weather information.

    Args:
        city: City name in English (e.g., "Seoul", "Busan", "Daejeon")
              Korean city names will be automatically translated:
              - 서울 -> Seoul
              - 부산 -> Busan
              - 대전 -> Daejeon
              - 인천 -> Incheon
              - 대구 -> Daegu
        units: Temperature units - "c" (Celsius), "f" (Fahrenheit), "k" (Kelvin)
        lang: Language for descriptions - "kr" (Korean), "en" (English)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - city: The city queried
        - units: Temperature units used
        - forecast: Weather forecast data including:
            - Temperature
            - Weather conditions
            - Humidity
            - Wind speed
            - Cloud coverage
            - Precipitation probability

    Example:
        result = weather_forecast("Seoul", units="c", lang="kr")
        if result["success"]:
            print(f"Weather in {result['city']}: {result['forecast']}")

    Note:
        Requires OPENWEATHER_API_KEY environment variable to be set.
    """
    # Translate common Korean city names
    city_translations = {
        "서울": "Seoul",
        "부산": "Busan",
        "대전": "Daejeon",
        "인천": "Incheon",
        "대구": "Daegu",
        "광주": "Gwangju",
        "울산": "Ulsan",
        "세종": "Sejong",
        "제주": "Jeju",
        "경기": "Gyeonggi",
        "수원": "Suwon",
        "의왕": "Uiwang",
        "유성": "Yuseong",
    }

    city_normalized = city.strip()
    for kr, en in city_translations.items():
        if kr in city_normalized:
            city_normalized = city_normalized.replace(kr, en)
            break

    return _run_async(_weather_forecast_async(
        city=city_normalized,
        units=units,
        lang=lang,
    ))


# =============================================================================
# Tool 4: Sentinel Satellite Search
# =============================================================================

async def _sentinel_search_async(
    query: str,
    sensor: str | None = None,
    aoi: str | None = None,
    date_range: str | None = None,
    cloud_cover_max: int | None = None,
) -> dict[str, Any]:
    """Async implementation of Sentinel satellite search."""
    client = MultiServerMCPClient({
        "sentinel": {
            "transport": "stdio",
            "command": "python",
            "args": [SENTINEL_SERVER],
        }
    })

    async with client.session("sentinel") as session:
        tools = await load_mcp_tools(session)

        # Find the sentinel search tool
        search_tool = None
        for t in tools:
            if "sentinel" in t.name.lower() and ("search" in t.name.lower() or "query" in t.name.lower()):
                search_tool = t
                break

        if search_tool is None:
            # Fallback: use first available tool
            search_tool = tools[0] if tools else None

        if search_tool is None:
            return {
                "success": False,
                "error": "No Sentinel search tool available",
                "query": query,
            }

        # Build search parameters
        params = {"query": query}
        if sensor:
            params["sensor"] = sensor
        if aoi:
            params["aoi"] = aoi
        if date_range:
            params["date_range"] = date_range
        if cloud_cover_max is not None:
            params["cloud_cover_max"] = cloud_cover_max

        raw = await search_tool.ainvoke(params)
        payload = _extract_payload(raw)

    return {
        "success": True,
        "query": query,
        "results": payload,
    }


def sentinel_search(
    query: str,
    sensor: str | None = None,
    aoi: str | None = None,
    date_range: str | None = None,
    cloud_cover_max: int | None = None,
) -> dict[str, Any]:
    """Search for Sentinel satellite imagery.

    This tool searches for Sentinel-1, Sentinel-2, or Sentinel-3 satellite
    imagery from ESA/Copernicus. Use this for finding satellite scenes,
    checking acquisition dates, or planning satellite observations.

    Args:
        query: Natural language search query describing what you're looking for
        sensor: Sensor type - "S1" (Sentinel-1 SAR), "S2" (Sentinel-2 optical),
                "S3" (Sentinel-3)
        aoi: Area of interest - can be:
             - City/region name: "Daejeon", "Seoul"
             - Bounding box: "126.5,36.0,127.5,37.0" (min_lon,min_lat,max_lon,max_lat)
        date_range: Date range in format "YYYY-MM-DD/YYYY-MM-DD"
                    e.g., "2024-01-01/2024-01-31"
        cloud_cover_max: Maximum cloud coverage percentage (0-100),
                         only applicable to optical sensors (S2, S3)

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - results: Search results including:
            - scenes: List of available satellite scenes
            - acquisition_dates: Acquisition timestamps
            - orbits: Orbit information
            - footprints: Geographic coverage
            - cloud_coverage: Cloud percentage (for optical)
            - polarization: Polarization mode (for SAR)

    Example:
        # Search for Sentinel-2 imagery of Daejeon with low cloud cover
        result = sentinel_search(
            query="대전 유성구 위성 영상",
            sensor="S2",
            aoi="Daejeon",
            date_range="2024-01-01/2024-01-31",
            cloud_cover_max=20
        )

    Note:
        - Sentinel-1 (SAR) works regardless of weather/clouds
        - Sentinel-2 (optical) is affected by cloud cover
        - Use weather_forecast to check cloud conditions before optical imaging
    """
    return _run_async(_sentinel_search_async(
        query=query,
        sensor=sensor,
        aoi=aoi,
        date_range=date_range,
        cloud_cover_max=cloud_cover_max,
    ))


# =============================================================================
# All Tools Export
# =============================================================================

# List of all available MCP tools
ALL_MCP_TOOLS = [
    google_search_and_summarize,
    rag_search,
    weather_forecast,
    sentinel_search,
]

__all__ = [
    "google_search_and_summarize",
    "rag_search",
    "weather_forecast",
    "sentinel_search",
    "ALL_MCP_TOOLS",
]
