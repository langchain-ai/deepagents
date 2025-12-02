"""Custom tools for the CLI agent."""

import os
from typing import Any, Literal, Sequence

import requests
from markdownify import markdownify
from tavily import TavilyClient
from langchain.tools import tool

from deepagents_cli.config import settings

# Initialize Tavily client if API key is available
tavily_client = TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None


# --- AviationBot helpers and tools -----------------------------------------------------------

AVIATIONBOT_BASE_URL = os.getenv("AVIATION_BOT_BASE_URL", "https://beta.aviation.bot/api/v1")
AVIATIONBOT_DEFAULT_TIMEOUT = int(os.getenv("AVIATION_BOT_TIMEOUT", "60"))


def _build_auth_headers() -> dict[str, str]:
    """Return Authorization header using AVIATION_BOT_API_KEY env if present."""
    api_key = os.getenv("AVIATION_BOT_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _aviationbot_request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json: Any = None,
    timeout: int | None = None,
):
    """Internal shared request helper for AviationBot endpoints."""
    url = f"{AVIATIONBOT_BASE_URL}{path}"
    effective_timeout = timeout or AVIATIONBOT_DEFAULT_TIMEOUT
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers={"Accept": "application/json", **_build_auth_headers()},
            timeout=effective_timeout,
        )
        response.raise_for_status()
        # Prefer JSON, fallback to text
        try:
            return {"success": True, "status_code": response.status_code, "data": response.json()}
        except ValueError:
            return {"success": True, "status_code": response.status_code, "data": response.text}
    except requests.HTTPError as exc:  # includes status >=400
        payload: dict[str, Any] = {"success": False, "status_code": exc.response.status_code}
        try:
            payload["error"] = exc.response.json()
        except Exception:
            payload["error"] = exc.response.text
        return payload
    except requests.RequestException as exc:  # network/timeout etc
        return {"success": False, "status_code": 0, "error": str(exc)}


@tool(
    "aviationbot_document_retrieval",
    description="Semantic retrieval of EASA Easy Access Rules by query (AviationBot /tool/EASA/document-retrieval)",
)
def aviationbot_document_retrieval(
    query: str,
    erules_ids: Sequence[str] | None = None,
) -> Any:
    """Fetch EASA aviation regulations matching a natural-language query.

    Args:
        query: Search phrase or question.
        erules_ids: Optional list of ERules IDs to restrict results.
    Returns:
        Dict with success flag, status_code, and data/error from AviationBot.
    """

    params: dict[str, Any] = {"query": query}
    if erules_ids:
        params["erules_ids"] = list(erules_ids)
    return _aviationbot_request(
        "GET",
        "/tool/EASA/document-retrieval",
        params=params,
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


@tool(
    "aviationbot_meta_model",
    description="Retrieve EASA regulatory metamodel info (AviationBot /tool/EASA/meta-model)",
)
def aviationbot_meta_model(query: str) -> Any:
    """Get structured information about the EASA regulatory framework."""

    return _aviationbot_request(
        "GET",
        "/tool/EASA/meta-model",
        params={"query": query},
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for current information and documentation.

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except Exception as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}
