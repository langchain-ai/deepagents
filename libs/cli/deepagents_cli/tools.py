"""Custom tools for the CLI agent."""

from typing import Any, Literal

import requests
from markdownify import markdownify
from tavily import TavilyClient

from deepagents_cli.config import settings

# Initialize Tavily client if API key is available
tavily_client = TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    **When to Use:**
    - Calling REST APIs with specific endpoints
    - POST/PUT/DELETE operations
    - Requests requiring custom headers or authentication
    - Interacting with known API endpoints

    **When NOT to Use:**
    - General web searches (use `web_search` instead)
    - Fetching web page content to read (use `fetch_url` instead)
    - When you don't know the exact API endpoint

    Args:
        url: Target URL (must be exact API endpoint)
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include (for auth tokens, content-type, etc.)
        data: Request body data (string or dict - dict will be sent as JSON)
        params: URL query parameters
        timeout: Request timeout in seconds (default: 30)

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
    """Search the web for current information, documentation, and answers.

    **When to Use:**
    - Finding documentation or tutorials
    - Researching error messages or solutions
    - Looking up current events or recent information
    - Finding code examples or best practices
    - Any general information search

    **When NOT to Use:**
    - Fetching a specific URL you already know (use `fetch_url` instead)
    - Calling a specific API endpoint (use `http_request` instead)
    - Searching local files (use `grep` instead)

    Args:
        query: The search query - be specific and detailed for better results
        max_results: Number of results to return (default: 5)
        topic: Search type - "general" (default), "news" for current events, "finance" for financial data
        include_raw_content: Include full page content (uses more tokens, usually not needed)

    Returns:
        Dictionary with results containing title, url, content excerpt, and relevance score

    **IMPORTANT - After receiving results:**
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize into a clear, natural language response
    4. Cite sources by mentioning page titles or URLs
    5. NEVER show raw JSON to the user - always provide a formatted response
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
    """Fetch content from a specific URL and convert HTML to markdown.

    **When to Use:**
    - Reading documentation from a known URL
    - Fetching content from a specific webpage
    - Getting content from URLs found via web_search
    - Reading GitHub READMEs, blog posts, or articles

    **When NOT to Use:**
    - General web searches (use `web_search` instead - it's faster and returns multiple results)
    - Calling REST APIs (use `http_request` instead)
    - When you don't have a specific URL

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - url: The final URL after redirects
        - markdown_content: The page content converted to clean markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content

    **IMPORTANT - After receiving content:**
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize into a clear, natural language response
    4. NEVER show raw markdown to the user unless specifically requested
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
