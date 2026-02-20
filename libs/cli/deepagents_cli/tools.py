"""Custom tools for the CLI agent."""

import ipaddress
import re
import socket
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

import requests
from markdownify import markdownify
from tavily import (
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyClient,
    UsageLimitExceededError,
)
from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError

from deepagents_cli.config import settings

# Initialize Tavily client if API key is available
tavily_client = (
    TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None
)


_HTTP_URL_PATTERN = re.compile(r"https?://[^\s)\]}>\"']+")
_METADATA_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.goog.internal",
}


def _extract_first_http_url(text: str) -> str | None:
    """Extract the first HTTP(S) URL from text.

    Returns:
        First detected HTTP(S) URL, or `None` when no URL is present.
    """
    match = _HTTP_URL_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
        or ip == ipaddress.ip_address("169.254.169.254")
    )


def _validate_direct_fetch_url(url: str) -> tuple[bool, str | None]:
    """Validate direct-fetch URL to reduce SSRF risk for local/internal hosts.

    Returns:
        A tuple of `(is_safe, reason)`. `reason` is set when validation fails.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return (
            False,
            f"Blocked direct URL fetch for unsupported scheme: {parsed.scheme}",
        )

    hostname = parsed.hostname
    if not hostname:
        return False, "Blocked direct URL fetch for invalid URL host."

    normalized_host = hostname.rstrip(".").lower()

    if (
        normalized_host == "localhost"
        or normalized_host.endswith(".localhost")
        or normalized_host == "localhost.localdomain"
    ):
        return (
            False,
            f"Blocked direct URL fetch for local/internal host: {hostname}",
        )

    if normalized_host in _METADATA_HOSTNAMES:
        return (
            False,
            f"Blocked direct URL fetch for cloud metadata host: {hostname}",
        )

    try:
        literal_ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            msg = (
                "Blocked direct URL fetch because host could not be safely "
                f"resolved: {hostname}"
            )
            return (
                False,
                msg,
            )

        for info in resolved:
            if not info or len(info) < 5 or not info[4]:
                continue
            ip_text = str(info[4][0]).split("%", 1)[0]
            try:
                resolved_ip = ipaddress.ip_address(ip_text)
            except ValueError:
                continue
            if _is_blocked_ip(resolved_ip):
                return (
                    False,
                    f"Blocked direct URL fetch for local/internal host: {hostname}",
                )
    else:
        if _is_blocked_ip(literal_ip):
            return (
                False,
                f"Blocked direct URL fetch for local/internal host: {hostname}",
            )

    return True, None


def _fetch_url_with_safe_redirects(
    url: str,
    timeout: int = 30,
    max_redirects: int = 5,
) -> dict[str, Any]:
    """Fetch URL content while validating each redirect hop for SSRF safety.

    Returns:
        Fetch response payload, or an error payload when blocked/failed.
    """
    current_url = url

    for _ in range(max_redirects + 1):
        is_safe, block_reason = _validate_direct_fetch_url(current_url)
        if not is_safe:
            return {
                "error": block_reason,
                "url": current_url,
                "blocked_url": current_url,
            }

        try:
            response = requests.get(
                current_url,
                timeout=timeout,
                allow_redirects=False,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
            )
        except requests.exceptions.RequestException as e:
            return {"error": f"Fetch URL error: {e!s}", "url": current_url}

        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            if not location:
                return {
                    "error": (
                        "Fetch URL error: redirect response missing "
                        "Location header"
                    ),
                    "url": current_url,
                }
            current_url = urljoin(current_url, location)
            continue

        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {"error": f"Fetch URL error: {e!s}", "url": current_url}

        markdown_content = markdownify(response.text)
        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }

    return {
        "error": f"Fetch URL error: too many redirects (>{max_redirects})",
        "url": current_url,
    }


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

        response = requests.request(method.upper(), url, timeout=timeout, **kwargs)

        try:
            content = response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
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
    direct_url = _extract_first_http_url(query)
    if direct_url:
        fetched = _fetch_url_with_safe_redirects(direct_url)
        if "error" in fetched:
            return {
                "error": fetched["error"],
                "query": query,
                "blocked_url": fetched.get("blocked_url", direct_url),
                "direct_fetch": False,
                "source_tool": "fetch_url",
            }
        return {
            "query": query,
            "direct_fetch": True,
            "source_tool": "fetch_url",
            **fetched,
        }

    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        # Tavily-specific exceptions
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
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
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}
