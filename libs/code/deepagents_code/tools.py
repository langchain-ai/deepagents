"""Custom tools for the agent."""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import socket
import threading
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tavily import TavilyClient

logger = logging.getLogger(__name__)

_UNSET = object()
_tavily_client: TavilyClient | object | None = _UNSET

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_MAX_FETCH_REDIRECTS = 5

# Module-level lock guarding the urllib3 connection-factory monkeypatch used by
# `_pinned_dns`. The patch is process-global, so serializing fetches keeps
# concurrent calls from clobbering each other's pinned IP set.
_dns_pin_lock = threading.Lock()


class _UrlValidationError(ValueError):
    """Raised by `_validate_url` for scheme/DNS/SSRF-blocked URLs.

    Distinguishes intentional SSRF-guard rejections from incidental
    `ValueError`s raised elsewhere in the fetch path (e.g., markdown
    conversion).
    """


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if `ip` belongs to a non-publicly-routable range.

    Rejects: private (RFC1918/ULA), loopback, link-local (including cloud
    IMDS at `169.254.169.254`), reserved, multicast, unspecified
    (`0.0.0.0`/`::`), and anything `ipaddress` does not consider globally
    routable (catches benchmarking, documentation, and similar ranges the
    explicit predicates miss).

    IPv4-mapped IPv6 (`::ffff:a.b.c.d`) and 6to4 (`2002::/16`) are unwrapped
    to their underlying IPv4 address before the checks so that private
    space tunneled inside an IPv6 wrapper is still caught — e.g.,
    `::ffff:127.0.0.1` and `2002:a9fe:a9fe::1` (6to4 over IMDS) both
    evaluate as blocked.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_url(url: str) -> list[str]:
    """Reject URLs that target private/internal/metadata addresses.

    Resolves the URL's hostname and rejects any URL whose hostname resolves
    to a private, loopback, link-local (includes cloud IMDS at
    `169.254.169.254`), reserved, multicast, or unspecified IP — including
    such addresses wrapped in IPv4-mapped IPv6 (`::ffff:...`) or 6to4
    (`2002::/16`). This is the SSRF guard required because the URL is
    supplied by an LLM agent and may originate from prompt-injected content.

    Note:
        This function resolves DNS once. The HTTP client must be pinned to
        the returned IP list (see `_pinned_dns`) to close the TOCTOU window
        against attacker-controlled DNS (rebinding).

    Args:
        url: Candidate URL to validate.

    Returns:
        The list of validated IP strings the hostname resolves to.

            Callers should pin the outgoing connection to one of these IPs.

    Raises:
        _UrlValidationError: If the URL is malformed, uses a disallowed
            scheme, fails DNS resolution, or resolves to a blocked address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        msg = f"URL scheme not allowed: {parsed.scheme!r} (must be http or https)"
        raise _UrlValidationError(msg)

    hostname = parsed.hostname
    if not hostname:
        msg = "URL is missing a hostname"
        raise _UrlValidationError(msg)

    try:
        encoded_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        msg = f"Could not encode hostname {hostname!r} as IDNA: {exc}"
        raise _UrlValidationError(msg) from exc

    try:
        infos = socket.getaddrinfo(
            encoded_hostname,
            None,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        msg = f"Could not resolve hostname {hostname!r}: {exc}"
        raise _UrlValidationError(msg) from exc

    validated_ips: list[str] = []
    for info in infos:
        # `sockaddr[0]` may include an IPv6 scope id (`fe80::1%eth0`); strip
        # it before parsing so `ipaddress.ip_address` never raises.
        raw_ip = str(info[4][0]).split("%", 1)[0]
        ip = ipaddress.ip_address(raw_ip)
        if _is_blocked_ip(ip):
            logger.warning(
                "SSRF guard blocked URL %r: hostname %r resolves to %s",
                url,
                hostname,
                ip,
            )
            msg = (
                f"URL hostname {hostname!r} resolves to blocked address {ip} "
                "(private, loopback, link-local, reserved, or non-global range)"
            )
            raise _UrlValidationError(msg)
        validated_ips.append(raw_ip)

    if not validated_ips:
        msg = f"Hostname {hostname!r} resolved to no addresses"
        raise _UrlValidationError(msg)

    return validated_ips


@contextlib.contextmanager
def _pinned_dns(hostname: str, allowed_ips: list[str]) -> Iterator[None]:
    """Force outgoing urllib3 connections for `hostname` to use `allowed_ips`.

    Patches `urllib3.util.connection.create_connection` for the duration of
    the context so that `requests` cannot re-resolve `hostname` to a
    different IP than the one `_validate_url` vetted (defends against DNS
    rebinding TOCTOU). The patch is process-global, so the module lock
    serializes concurrent fetches.

    Args:
        hostname: The exact hostname (already IDNA-encoded by the caller)
            whose resolution must be pinned.
        allowed_ips: The IPs `_validate_url` confirmed are safe to connect
            to. Tried in order; the first that accepts the connection wins.
    """
    from urllib3.util import connection as urllib3_connection

    with _dns_pin_lock:
        original = urllib3_connection.create_connection

        def patched(
            address: tuple[str, int], *args: Any, **kwargs: Any
        ) -> socket.socket:
            host, port = address[0], address[1]
            if host != hostname:
                return original(address, *args, **kwargs)
            last_exc: OSError | None = None
            for ip in allowed_ips:
                try:
                    return original((ip, port), *args, **kwargs)
                except OSError as exc:
                    last_exc = exc
            assert last_exc is not None  # noqa: S101  # loop body guarantees this
            raise last_exc

        urllib3_connection.create_connection = patched  # type: ignore[assignment]  # signature matches at runtime
        try:
            yield
        finally:
            urllib3_connection.create_connection = original


def _get_tavily_client() -> TavilyClient | None:
    """Get or initialize the lazy Tavily client singleton.

    Returns:
        TavilyClient instance, or None if API key is not configured.
    """
    global _tavily_client  # noqa: PLW0603  # Module-level cache requires global statement
    if _tavily_client is not _UNSET:
        return _tavily_client  # type: ignore[return-value]  # narrowed by sentinel check

    from deepagents_code.config import settings

    if settings.has_tavily:
        from tavily import TavilyClient as _TavilyClient

        _tavily_client = _TavilyClient(api_key=settings.tavily_api_key)
    else:
        _tavily_client = None
    return _tavily_client


def web_search(  # noqa: ANN201  # Return type depends on dynamic tool configuration
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
    try:
        import requests
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return {"error": f"Required package not installed: {exc.name}."}

    client = _get_tavily_client()
    if client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return client.search(
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
        import requests
        from markdownify import markdownify
    except ImportError as exc:
        return {"error": f"Required package not installed: {exc.name}."}

    try:
        response = _fetch_with_redirects(url, timeout=timeout)
    except _UrlValidationError as e:
        return {
            "error": f"Fetch URL error: {e!s}",
            "url": url,
            "category": "validation",
        }
    except requests.exceptions.TooManyRedirects as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url, "category": "redirects"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url, "category": "network"}

    markdown_content = markdownify(response.text)
    return {
        "url": str(response.url),
        "markdown_content": markdown_content,
        "status_code": response.status_code,
        "content_length": len(markdown_content),
    }


def _fetch_with_redirects(url: str, *, timeout: int) -> Any:  # noqa: ANN401  # requests.Response, but kept dynamic to avoid eager import
    """Fetch `url`, re-validating each redirect hop against the SSRF guard.

    Each hop is validated by `_validate_url` and its connection pinned to
    the validated IP via `_pinned_dns`. Caps at `_MAX_FETCH_REDIRECTS`
    redirects (so up to `_MAX_FETCH_REDIRECTS + 1` total hops counting the
    initial request). Network/HTTP errors propagate as
    `requests.exceptions.RequestException` (or its subclasses).

    Args:
        url: Initial URL to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        The final `requests.Response` for the non-redirect terminal hop.

    Raises:
        _UrlValidationError: If any hop fails SSRF validation or returns a
            3xx without a `Location` header.
        requests.exceptions.TooManyRedirects: If the redirect cap is exceeded.
    """
    import requests

    current_url = url
    session = requests.Session()
    # DNS pinning only protects the direct target connection. Environment
    # proxies resolve the target separately, so they must be disabled here.
    session.trust_env = False
    for _hop in range(_MAX_FETCH_REDIRECTS + 1):
        validated_ips = _validate_url(current_url)
        hostname = urlparse(current_url).hostname
        # `_validate_url` raises if hostname is missing, so this is non-None.
        assert hostname is not None  # noqa: S101  # invariant from _validate_url
        encoded_hostname = hostname.encode("idna").decode("ascii")

        with _pinned_dns(encoded_hostname, validated_ips):
            response = session.get(
                current_url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
                allow_redirects=False,
            )

        # 300-399 covers every redirect class. `requests.Response.is_redirect`
        # also checks for a `Location` header, which would hide malformed 3xx
        # responses — so we check the raw status code instead.
        if 300 <= response.status_code < 400:  # noqa: PLR2004  # HTTP redirect class
            location = response.headers.get("Location")
            if not location:
                msg = (
                    f"Redirect response (status {response.status_code}) at "
                    f"{current_url!r} is missing a Location header"
                )
                raise _UrlValidationError(msg)
            current_url = urljoin(current_url, location)
            continue

        response.raise_for_status()
        return response

    msg = f"Exceeded {_MAX_FETCH_REDIRECTS} redirects starting from {url!r}"
    raise requests.exceptions.TooManyRedirects(msg)
