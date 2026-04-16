"""Web tools for the WhatsApp channel agent.

Extracted from deepagents_cli.tools to avoid a dependency on the CLI
package (which pins a specific deepagents version).
"""

from __future__ import annotations

from typing import Any, Literal


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
    import requests as req

    try:
        kwargs: dict[str, Any] = {}
        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = req.request(method.upper(), url, timeout=timeout, **kwargs)

        try:
            content = response.json()
        except (ValueError, req.exceptions.JSONDecodeError):
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }
    except req.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except req.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict[str, Any]:
    """Search the web using Tavily for current information and documentation.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type
        include_raw_content: Include full page content (uses more tokens)

    Returns:
        Dictionary with search results including title, url, content, and score.
    """
    try:
        import requests as req
        from tavily import TavilyClient
    except ImportError as exc:
        return {
            "error": f"Required package not installed: {exc.name}. "
            "Install with: pip install tavily-python",
            "query": query,
        }

    import os

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": "Tavily API key not configured. Set TAVILY_API_KEY.",
            "query": query,
        }

    try:
        client = TavilyClient(api_key=api_key)
        return client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (req.exceptions.RequestException, ValueError, TypeError) as e:
        return {"error": f"Web search error: {e!s}", "query": query}
    except Exception as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary with markdown_content, status_code, and url.
    """
    try:
        import requests as req
        from markdownify import markdownify
    except ImportError as exc:
        return {
            "error": f"Required package not installed: {exc.name}. "
            "Install with: pip install requests markdownify",
            "url": url,
        }

    try:
        response = req.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()
        markdown_content = markdownify(response.text)
        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except req.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}
