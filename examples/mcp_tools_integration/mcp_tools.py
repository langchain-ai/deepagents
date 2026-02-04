"""Custom MCP-based tools for deepagents.

This module provides 5 MCP tools that can be registered with deepagents:
1. google_search_and_summarize - Google search with web page fetching and summarization
2. rag_search - RAG-based document/paper search
3. weather_forecast - Weather forecast using OpenWeatherMap
4. sentinel_search - Sentinel satellite imagery search
5. arxiv_search - Search and download arXiv papers

Usage:
    from mcp_tools import (
        google_search_and_summarize,
        rag_search,
        weather_forecast,
        sentinel_search,
        arxiv_search,
    )

    agent = create_deep_agent(
        model="openai:gpt-4o",
        tools=[google_search_and_summarize, rag_search, weather_forecast, sentinel_search, arxiv_search],
    )
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import threading
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

# arXiv MCP server - uses 'uv tool run' command
# Install with: uv tool install arxiv-mcp-server
ARXIV_STORAGE_PATH = os.getenv(
    "ARXIV_STORAGE_PATH",
    os.path.expanduser("~/.arxiv-mcp-storage")
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


def _run_in_new_loop(coro):
    """Run async coroutine in a new event loop in a separate thread.

    This is necessary because MCP clients require their own clean event loop
    and cannot be nested inside another running loop.
    """
    result = None
    exception = None

    def run():
        nonlocal result, exception
        try:
            # Create a completely new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
            finally:
                # Clean up the loop
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()
        except Exception as e:
            exception = e

    # Run in a separate thread to avoid event loop conflicts
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=120)  # 2 minute timeout

    if thread.is_alive():
        raise TimeoutError("MCP operation timed out after 120 seconds")

    if exception is not None:
        raise exception

    return result


def _run_async(coro):
    """Run async coroutine safely from sync context.

    Handles the case where we might already be in an async context
    by running the coroutine in a separate thread with its own event loop.
    """
    try:
        # Check if we're in an async context
        asyncio.get_running_loop()
        # We are in an async context - must run in separate thread
        return _run_in_new_loop(coro)
    except RuntimeError:
        # No running loop - we can use asyncio.run directly
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
    # Validate MCP server paths
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

            google_tool = next((t for t in gtools if "google" in t.name.lower() or "search" in t.name.lower()), None)
            fetch_tool = next((t for t in ftools if "fetch" in t.name.lower() or "extract" in t.name.lower()), None)

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
            search_payload = _extract_payload(search_raw)
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
                # Return search results without fetching
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
    # Validate MCP server path
    if not os.path.exists(RAG_SERVER):
        return {
            "success": False,
            "query": query,
            "error": f"RAG MCP server not found: {RAG_SERVER}. "
                     f"Set MCP_RAG_SERVER environment variable.",
            "documents": [],
        }

    try:
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
            retrieve_tool = next(
                (t for t in tools if "rag" in t.name.lower() or "retrieve" in t.name.lower()),
                tools[0] if tools else None
            )

            if retrieve_tool is None:
                return {
                    "success": False,
                    "query": query,
                    "error": "No RAG retrieve tool found",
                    "documents": [],
                }

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

    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "documents": [],
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
            "forecast": {},
        }

    # Validate MCP server path
    if not os.path.exists(WEATHER_SERVER):
        return {
            "success": False,
            "city": city,
            "error": f"Weather MCP server not found: {WEATHER_SERVER}. "
                     f"Set MCP_WEATHER_SERVER environment variable.",
            "forecast": {},
        }

    # Check API key
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "city": city,
            "error": "OPENWEATHER_API_KEY environment variable not set.",
            "forecast": {},
        }

    try:
        client = MultiServerMCPClient({
            "openweather": {
                "transport": "stdio",
                "command": "python",
                "args": [WEATHER_SERVER],
                "env": {"OWM_API_KEY": api_key},
            }
        })

        tools = await client.get_tools()
        weather_tool = next(
            (t for t in tools if "weather" in t.name.lower()),
            tools[0] if tools else None
        )

        if weather_tool is None:
            return {
                "success": False,
                "city": city,
                "error": "No weather tool found in MCP server",
                "forecast": {},
            }

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

    except Exception as e:
        return {
            "success": False,
            "city": city,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "forecast": {},
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
    # Validate MCP server path
    if not os.path.exists(SENTINEL_SERVER):
        return {
            "success": False,
            "query": query,
            "error": f"Sentinel MCP server not found: {SENTINEL_SERVER}. "
                     f"Set MCP_SENTINEL_SERVER environment variable.",
            "results": {},
        }

    try:
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
                if "sentinel" in t.name.lower() or "search" in t.name.lower() or "query" in t.name.lower():
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
                    "results": {},
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

    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"MCP connection failed: {type(e).__name__}: {str(e)}",
            "results": {},
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
# Tool 5: arXiv Paper Search
# =============================================================================

async def _arxiv_search_async(
    query: str,
    max_results: int = 10,
    date_from: str | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Async implementation of arXiv paper search."""
    try:
        # arXiv MCP server uses 'uv tool run' command
        client = MultiServerMCPClient({
            "arxiv": {
                "transport": "stdio",
                "command": "uv",
                "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", ARXIV_STORAGE_PATH],
            }
        })

        async with client.session("arxiv") as session:
            tools = await load_mcp_tools(session)

            # Find search_papers tool
            search_tool = next(
                (t for t in tools if "search" in t.name.lower()),
                None
            )

            if search_tool is None:
                return {
                    "success": False,
                    "query": query,
                    "error": f"No search tool found. Available tools: {[t.name for t in tools]}",
                    "papers": [],
                }

            # Build search parameters
            params = {
                "query": query,
                "max_results": max_results,
            }
            if date_from:
                params["date_from"] = date_from
            if categories:
                params["categories"] = categories

            raw = await search_tool.ainvoke(params)
            payload = _extract_payload(raw)

        # Extract papers from payload
        papers = payload.get("papers", payload.get("results", []))
        if isinstance(papers, list):
            paper_list = []
            for p in papers:
                if isinstance(p, dict):
                    paper_list.append({
                        "paper_id": p.get("id", p.get("paper_id", "")),
                        "title": p.get("title", ""),
                        "authors": p.get("authors", []),
                        "summary": p.get("summary", p.get("abstract", "")),
                        "published": p.get("published", ""),
                        "updated": p.get("updated", ""),
                        "categories": p.get("categories", []),
                        "pdf_url": p.get("pdf_url", ""),
                    })
            papers = paper_list

        return {
            "success": True,
            "query": query,
            "num_papers": len(papers) if isinstance(papers, list) else 0,
            "papers": papers,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "query": query,
            "error": "arXiv MCP server not installed. Install with: uv tool install arxiv-mcp-server",
            "papers": [],
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"arXiv search failed: {type(e).__name__}: {str(e)}",
            "papers": [],
        }


def arxiv_search(
    query: str,
    max_results: int = 10,
    date_from: str | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Search for papers on arXiv.

    This tool searches the arXiv preprint repository for academic papers.
    Use this for finding research papers, preprints, and scientific literature.

    Args:
        query: Search query - can include:
               - Keywords: "transformer attention mechanism"
               - Author: "au:Vaswani"
               - Title: "ti:attention is all you need"
               - Abstract: "abs:deep learning"
               - arXiv ID: "2301.00001"
        max_results: Maximum number of papers to return (default: 10, max: 100)
        date_from: Filter papers from this date (format: "YYYY-MM-DD")
        categories: List of arXiv categories to filter by, e.g.:
                   - "cs.AI" (Artificial Intelligence)
                   - "cs.LG" (Machine Learning)
                   - "cs.CL" (Computation and Language)
                   - "cs.CV" (Computer Vision)
                   - "stat.ML" (Statistics - Machine Learning)
                   - "physics.space-ph" (Space Physics)

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - num_papers: Number of papers found
        - papers: List of paper dictionaries, each with:
            - paper_id: arXiv paper ID (e.g., "2301.00001")
            - title: Paper title
            - authors: List of author names
            - summary: Paper abstract
            - published: Publication date
            - updated: Last update date
            - categories: List of arXiv categories
            - pdf_url: URL to download PDF

    Example:
        # Search for transformer papers in AI category
        result = arxiv_search(
            query="transformer neural network",
            max_results=5,
            categories=["cs.AI", "cs.LG"]
        )
        for paper in result["papers"]:
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Abstract: {paper['summary'][:200]}...")

    Note:
        Requires arxiv-mcp-server to be installed:
        $ uv tool install arxiv-mcp-server
    """
    return _run_async(_arxiv_search_async(
        query=query,
        max_results=max_results,
        date_from=date_from,
        categories=categories,
    ))


async def _arxiv_download_paper_async(paper_id: str) -> dict[str, Any]:
    """Async implementation of arXiv paper download."""
    try:
        client = MultiServerMCPClient({
            "arxiv": {
                "transport": "stdio",
                "command": "uv",
                "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", ARXIV_STORAGE_PATH],
            }
        })

        async with client.session("arxiv") as session:
            tools = await load_mcp_tools(session)

            # Find download_paper tool
            download_tool = next(
                (t for t in tools if "download" in t.name.lower()),
                None
            )

            if download_tool is None:
                return {
                    "success": False,
                    "paper_id": paper_id,
                    "error": f"No download tool found. Available tools: {[t.name for t in tools]}",
                }

            raw = await download_tool.ainvoke({"paper_id": paper_id})
            payload = _extract_payload(raw)

        return {
            "success": True,
            "paper_id": paper_id,
            "message": payload.get("message", "Paper downloaded successfully"),
            "storage_path": payload.get("path", ARXIV_STORAGE_PATH),
        }

    except Exception as e:
        return {
            "success": False,
            "paper_id": paper_id,
            "error": f"Download failed: {type(e).__name__}: {str(e)}",
        }


def arxiv_download_paper(paper_id: str) -> dict[str, Any]:
    """Download an arXiv paper by its ID.

    Downloads the PDF and extracts text content for later reading.

    Args:
        paper_id: arXiv paper ID (e.g., "2301.00001" or "2301.00001v2")

    Returns:
        Dictionary containing:
        - success: Whether the download succeeded
        - paper_id: The paper ID
        - message: Status message
        - storage_path: Where the paper is stored

    Example:
        result = arxiv_download_paper("2301.00001")
        if result["success"]:
            print(f"Downloaded to: {result['storage_path']}")
    """
    return _run_async(_arxiv_download_paper_async(paper_id))


async def _arxiv_read_paper_async(paper_id: str) -> dict[str, Any]:
    """Async implementation of reading downloaded arXiv paper."""
    try:
        client = MultiServerMCPClient({
            "arxiv": {
                "transport": "stdio",
                "command": "uv",
                "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", ARXIV_STORAGE_PATH],
            }
        })

        async with client.session("arxiv") as session:
            tools = await load_mcp_tools(session)

            # Find read_paper tool
            read_tool = next(
                (t for t in tools if "read" in t.name.lower()),
                None
            )

            if read_tool is None:
                return {
                    "success": False,
                    "paper_id": paper_id,
                    "error": f"No read tool found. Available tools: {[t.name for t in tools]}",
                    "content": "",
                }

            raw = await read_tool.ainvoke({"paper_id": paper_id})
            payload = _extract_payload(raw)

        content = payload.get("content", payload.get("text", ""))
        if isinstance(content, str):
            pass
        elif isinstance(content, dict):
            content = content.get("text", str(content))
        else:
            content = str(content)

        return {
            "success": True,
            "paper_id": paper_id,
            "title": payload.get("title", ""),
            "content": content,
        }

    except Exception as e:
        return {
            "success": False,
            "paper_id": paper_id,
            "error": f"Read failed: {type(e).__name__}: {str(e)}",
            "content": "",
        }


def arxiv_read_paper(paper_id: str) -> dict[str, Any]:
    """Read the content of a downloaded arXiv paper.

    Returns the extracted text content of a previously downloaded paper.
    You must download the paper first using arxiv_download_paper().

    Args:
        paper_id: arXiv paper ID (e.g., "2301.00001")

    Returns:
        Dictionary containing:
        - success: Whether reading succeeded
        - paper_id: The paper ID
        - title: Paper title
        - content: Extracted text content of the paper

    Example:
        # First download
        arxiv_download_paper("2301.00001")

        # Then read
        result = arxiv_read_paper("2301.00001")
        if result["success"]:
            print(f"Title: {result['title']}")
            print(f"Content: {result['content'][:1000]}...")
    """
    return _run_async(_arxiv_read_paper_async(paper_id))


async def _arxiv_list_papers_async() -> dict[str, Any]:
    """Async implementation of listing downloaded arXiv papers."""
    try:
        client = MultiServerMCPClient({
            "arxiv": {
                "transport": "stdio",
                "command": "uv",
                "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", ARXIV_STORAGE_PATH],
            }
        })

        async with client.session("arxiv") as session:
            tools = await load_mcp_tools(session)

            # Find list_papers tool
            list_tool = next(
                (t for t in tools if "list" in t.name.lower()),
                None
            )

            if list_tool is None:
                return {
                    "success": False,
                    "error": f"No list tool found. Available tools: {[t.name for t in tools]}",
                    "papers": [],
                }

            raw = await list_tool.ainvoke({})
            payload = _extract_payload(raw)

        papers = payload.get("papers", payload.get("items", []))

        return {
            "success": True,
            "num_papers": len(papers) if isinstance(papers, list) else 0,
            "papers": papers,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"List failed: {type(e).__name__}: {str(e)}",
            "papers": [],
        }


def arxiv_list_papers() -> dict[str, Any]:
    """List all downloaded arXiv papers.

    Returns a list of papers that have been downloaded and are available
    for reading locally.

    Returns:
        Dictionary containing:
        - success: Whether the operation succeeded
        - num_papers: Number of downloaded papers
        - papers: List of paper info (id, title, etc.)

    Example:
        result = arxiv_list_papers()
        for paper in result["papers"]:
            print(f"- {paper['paper_id']}: {paper['title']}")
    """
    return _run_async(_arxiv_list_papers_async())


# =============================================================================
# All Tools Export
# =============================================================================

# List of all available MCP tools
ALL_MCP_TOOLS = [
    google_search_and_summarize,
    rag_search,
    weather_forecast,
    sentinel_search,
    arxiv_search,
]

# Additional arXiv tools (not in ALL_MCP_TOOLS by default, add as needed)
ARXIV_TOOLS = [
    arxiv_search,
    arxiv_download_paper,
    arxiv_read_paper,
    arxiv_list_papers,
]

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
