"""Custom tools for the agent."""

from __future__ import annotations

import contextlib
import functools
import ipaddress
import logging
import socket
import threading
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool
from langgraph.config import get_config
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from langchain_core.tools import BaseTool
    from tavily import TavilyClient

logger = logging.getLogger(__name__)

_UNSET = object()
_tavily_client: TavilyClient | object | None = _UNSET

_OLLAMA_WEB_SEARCH_URL = "https://ollama.com/api/web_search"
"""Ollama Cloud web search endpoint (`Authorization: Bearer OLLAMA_API_KEY`)."""

_OLLAMA_WEB_FETCH_URL = "https://ollama.com/api/web_fetch"
"""Ollama Cloud web fetch endpoint (`Authorization: Bearer OLLAMA_API_KEY`)."""

_OLLAMA_MAX_RESULTS = 10
"""Hard cap the Ollama web search API accepts for `max_results`."""

_TAVILY_PROVIDER = "tavily"
_OLLAMA_PROVIDER = "ollama"

_WEB_SEARCH_MARKER = "deepagents_web_search"
"""Tool-metadata key marking a workspace-bound `web_search` variant.

Read by `is_web_search_tool`, the same way MCP read-only hints are read off
tool metadata, so a variant does not have to be registered anywhere.
"""

_WEB_SEARCH_TOKEN = object()
"""Value `is_web_search_tool` requires under `_WEB_SEARCH_MARKER`.

A module-private object rather than `True` so the marker cannot be forged: MCP
tool metadata is deserialized JSON, which can carry the key but never this
identity. Callers therefore need no separate "is this tool remote?" guard.
"""

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_MAX_FETCH_REDIRECTS = 5

# Maintainer note: `deepagents-talon` imports `web_search` and `fetch_url`
# directly from this module. Keep their names, signatures, and return/error dict
# shapes stable unless `deepagents-talon` is migrated in the same change.

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

        urllib3_connection.create_connection = patched  # ty: ignore[invalid-assignment]  # signature matches at runtime
        try:
            yield
        finally:
            urllib3_connection.create_connection = original


class _TextExtractor(HTMLParser):
    """Extract text content from HTML as a markdownify fallback.

    The character data inside raw-text elements (`script`, `style`,
    `noscript`, `template`) is skipped so the fallback never emits
    JavaScript or CSS source from the fetched (untrusted) page as page
    content.
    """

    # Tags whose character data is never page content. Suppressed via an
    # explicit allowlist of skipped tags rather than trying to detect script
    # payloads after the fact.
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],  # noqa: ARG002  # required by HTMLParser override
    ) -> None:
        """Enter a raw-text element so its data is skipped."""
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave a raw-text element."""
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect non-empty, whitespace-collapsed text outside skipped tags."""
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        """Return extracted text fragments separated by blank lines."""
        return "\n\n".join(self.parts)


def _html_to_markdown_content(html: str, markdownify: Callable[[str], str]) -> str:
    """Convert HTML to markdown, falling back to plain text on recursion.

    Args:
        html: Raw HTML to convert.
        markdownify: The `markdownify.markdownify` callable, injected so this
            module avoids an eager top-level import of the optional dependency.

    Returns:
        Markdown content, or text extracted from the HTML if markdown
        conversion exceeds the recursion limit. Returns an empty string if
        the text-extraction fallback itself fails.
    """
    try:
        return markdownify(html)
    except RecursionError:
        logger.warning(
            "markdownify hit recursion depth; falling back to text extraction",
            exc_info=True,
        )

    # Best-effort plain-text extraction. Guard it so a failure here (e.g. the
    # same pathological input that exhausted markdownify's recursion) cannot
    # re-introduce the uncaught crash this fallback exists to prevent.
    try:
        parser = _TextExtractor()
        parser.feed(html)
        parser.close()
    except Exception:  # fallback is best-effort; must never propagate
        logger.warning("text-extraction fallback failed", exc_info=True)
        return ""
    return parser.get_text()


def _active_web_provider() -> str | None:
    """Return the provider backing the web tools, or `None` when unconfigured.

    Tavily keeps priority so existing Tavily workspaces are unaffected; Ollama
    Cloud (`OLLAMA_API_KEY`) is the fallback. The same Ollama key backs cloud
    models and the web search/fetch APIs, so no separate credential is needed.

    Returns:
        `"tavily"`, `"ollama"`, or `None`.
    """
    from deepagents_code.config import credentials

    if credentials.has_tavily:
        return _TAVILY_PROVIDER
    if credentials.has_ollama:
        return _OLLAMA_PROVIDER
    return None


def _missing_key_error(provider: str | None, query: object) -> dict[str, object]:
    """Return the payload the model sees when no web search key is configured.

    Shared by the built-in and workspace-bound variants: `is_web_search_tool`
    treats them as one tool, so they have to fail identically.

    Args:
        provider: Selected web provider, or `None` when no key is set. `None`
            keeps the historical Tavily-only message because Tavily is still
            the documented primary and existing deployments rely on it.

    Returns:
        Error payload naming the env var to set.
    """
    if provider == _OLLAMA_PROVIDER:
        return {
            "error": "Ollama API key not configured. "
            "Please set OLLAMA_API_KEY environment variable.",
            "query": query,
        }
    return {
        "error": "Tavily API key not configured. "
        "Please set TAVILY_API_KEY environment variable.",
        "query": query,
    }


def _missing_package_error(exc: ImportError) -> dict[str, str]:
    """Return the payload the model sees when an optional package is absent.

    Returns:
        Error payload naming the missing package.
    """
    return {"error": f"Required package not installed: {exc.name}."}


def _get_tavily_client() -> TavilyClient | None:
    """Get or initialize the lazy Tavily client singleton.

    Returns:
        TavilyClient instance, or None if API key is not configured.
    """
    global _tavily_client  # noqa: PLW0603  # Module-level cache requires global statement
    if _tavily_client is not _UNSET:
        return _tavily_client  # ty: ignore[invalid-return-type]  # narrowed by sentinel check

    from deepagents_code.config import credentials

    if credentials.has_tavily:
        from tavily import TavilyClient as _TavilyClient

        _tavily_client = _TavilyClient(api_key=credentials.tavily_api_key)
    else:
        _tavily_client = None
    return _tavily_client


def create_web_search_tool(
    api_key: str,
    *,
    provider: str = _TAVILY_PROVIDER,
) -> BaseTool:
    """Bind web search to one workspace credential.

    The schema is taken from `web_search` via `functools.wraps` so the built-in
    and workspace-bound variants can never present different arguments. The two
    also have to fail the same way: `is_web_search_tool` treats them as one, so
    a missing package or an unusable key must return the payload the model can
    act on rather than raising.

    Args:
        api_key: The workspace provider credential (`""` reports unconfigured).
        provider: Which backend the key belongs to (`"tavily"` or `"ollama"`).

    Returns:
        Workspace-bound web search tool.
    """
    # Built on first use and reused: a per-call client would open a fresh
    # connection pool and repeat the TLS handshake for every search.
    client: TavilyClient | None = None

    @tool("web_search")
    @functools.wraps(web_search)
    def workspace_web_search(**kwargs: Any) -> object:
        nonlocal client
        if not api_key:
            return _missing_key_error(provider, kwargs.get("query"))
        if provider == _OLLAMA_PROVIDER:
            return _search_with_ollama(
                api_key,
                query=kwargs.get("query", ""),
                max_results=kwargs.get("max_results", 5),
            )
        if client is None:
            try:
                from tavily import TavilyClient as _TavilyClient

                client = _TavilyClient(api_key=api_key)
            except ImportError as exc:
                return _missing_package_error(exc)
        return _search_with_tavily(client, **kwargs)

    workspace_web_search.metadata = {
        **(workspace_web_search.metadata or {}),
        _WEB_SEARCH_MARKER: _WEB_SEARCH_TOKEN,
    }
    return workspace_web_search


def is_web_search_tool(candidate: object) -> bool:
    """Return whether `candidate` is a built-in or workspace-bound search tool.

    Returns:
        `True` for the module-level tool or any variant the factory marked.
    """
    if candidate is web_search:
        return True
    metadata = getattr(candidate, "metadata", None) or {}
    return metadata.get(_WEB_SEARCH_MARKER) is _WEB_SEARCH_TOKEN


@tool
def get_current_thread_id() -> str:
    """Get the current Deep Agents thread ID for LangSmith or MCP tooling.

    Returns:
        The current `configurable.thread_id`, or an explanatory message if missing.
    """
    thread_id = get_config().get("configurable", {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return "No current thread ID is available."


def web_search(  # noqa: ANN201  # Return type depends on dynamic tool configuration
    query: Annotated[
        str,
        Field(description="The search query (be specific and detailed)."),
    ],
    max_results: Annotated[
        int,
        Field(description="Number of results to return."),
    ] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"],
        Field(
            description=(
                'Search topic type: "general" for most queries, "news" for '
                'current events, or "finance".'
            )
        ),
    ] = "general",
    include_raw_content: Annotated[
        bool,
        Field(
            description=(
                "Include full page content (uses more tokens). Prefer `fetch_url` "
                "for a single URL."
            )
        ),
    ] = False,
):
    """Search the web for current information.

    Backed by Tavily when `TAVILY_API_KEY` is set, otherwise by Ollama Cloud
    (`OLLAMA_API_KEY`). The `topic` and `include_raw_content` arguments only
    apply to the Tavily backend; Ollama ignores them.

    Returns:
        Search hits with title, URL, snippet, and score.
    """
    provider = _active_web_provider()
    if provider is None:
        return _missing_key_error(None, query)
    if provider == _OLLAMA_PROVIDER:
        from deepagents_code.config import credentials

        return _search_with_ollama(
            credentials.ollama_api_key or "",
            query=query,
            max_results=max_results,
        )
    client = _get_tavily_client()
    if client is None:
        return _missing_key_error(provider, query)
    return _search_with_tavily(
        client,
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


def _search_with_ollama(
    api_key: str,
    *,
    query: str,
    max_results: int,
) -> object:
    """Execute an Ollama Cloud web search with the standard error translation.

    Args:
        api_key: Ollama Cloud API key (Bearer credential).
        query: The search query.
        max_results: Requested result count, capped at the API's own limit.

    Returns:
        Search hits (`{"query", "results"}` with `title`, `url`, and `content`
        per hit) or a translated error payload.
    """
    if not api_key:
        # Mirrors the workspace variant's contract: an empty key reports the
        # configuration problem instead of sending an unusable request.
        return _missing_key_error(_OLLAMA_PROVIDER, query)

    try:
        import requests
    except ImportError as exc:
        return _missing_package_error(exc)

    try:
        response = requests.post(
            _OLLAMA_WEB_SEARCH_URL,
            json={"query": query, "max_results": min(max_results, _OLLAMA_MAX_RESULTS)},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        return {"error": f"Web search error: {e!s}", "query": query}

    results = payload.get("results", []) if isinstance(payload, dict) else []
    return {"query": query, "results": results}


def _search_with_tavily(
    client: TavilyClient,
    *,
    query: str,
    max_results: int,
    topic: Literal["general", "news", "finance"],
    include_raw_content: bool,
) -> object:
    """Execute a Tavily search with the standard error translation.

    Returns:
        Search hits or a translated error payload.
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
        return _missing_package_error(exc)

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


def fetch_url(
    url: Annotated[
        str,
        Field(description="The URL to fetch (must be a valid HTTP/HTTPS URL)."),
    ],
    timeout: Annotated[
        int,
        Field(description="Request timeout in seconds."),
    ] = 30,
) -> dict[str, Any]:
    """Fetch a URL and return the page content as markdown.

    Fetches directly unless Ollama Cloud is the active web provider (Tavily
    unconfigured, `OLLAMA_API_KEY` set), in which case the Ollama web fetch
    API retrieves the page server-side and no SSRF guard applies — the local
    process never connects to the target host.

    Returns:
        Fetched page markdown plus status metadata. The Ollama backend omits
        `status_code` and adds `title` and `links`.
    """
    try:
        import requests
        from markdownify import markdownify
    except ImportError as exc:
        return _missing_package_error(exc)

    if _active_web_provider() == _OLLAMA_PROVIDER:
        from deepagents_code.config import credentials

        return _fetch_with_ollama(url, credentials.ollama_api_key or "")

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

    markdown_content = _html_to_markdown_content(response.text, markdownify)
    if not markdown_content:
        logger.warning(
            "fetch_url produced empty content for %s (status %s)",
            response.url,
            response.status_code,
        )
    return {
        "url": str(response.url),
        "markdown_content": markdown_content,
        "status_code": response.status_code,
        "content_length": len(markdown_content),
    }


def _fetch_with_ollama(url: str, api_key: str) -> dict[str, Any]:
    """Fetch `url` through the Ollama Cloud web fetch API.

    Args:
        url: The URL Ollama should retrieve server-side.
        api_key: Ollama Cloud API key (Bearer credential).

    Returns:
        Fetched page content and metadata, or an error payload using the same
        shape the direct fetch path returns. No SSRF guard applies here: the
        local process only connects to `ollama.com`, so fetching internal
        addresses does not leak the network the agent runs in.
    """
    import requests

    try:
        response = requests.post(
            _OLLAMA_WEB_FETCH_URL,
            json={"url": url},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url, "category": "network"}

    content = ""
    links: list[str] = []
    if isinstance(payload, dict):
        content = str(payload.get("content", ""))
        raw_links = payload.get("links", [])
        links = [str(link) for link in raw_links] if isinstance(raw_links, list) else []

    result: dict[str, Any] = {
        "url": url,
        "markdown_content": content,
        "content_length": len(content),
    }
    if isinstance(payload, dict) and payload.get("title"):
        result["title"] = payload["title"]
    if links:
        result["links"] = links
    return result


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
