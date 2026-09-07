"""Custom tools for the agent."""

from __future__ import annotations

import functools
import logging
import weakref
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.tools import tool
from langgraph.config import get_config
from pydantic import Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from tavily import TavilyClient

logger = logging.getLogger(__name__)

_UNSET = object()
_tavily_client: TavilyClient | object | None = _UNSET
_workspace_web_search_tools: weakref.WeakValueDictionary[int, object] = (
    weakref.WeakValueDictionary()
)

# Maintainer note: `deepagents-talon` imports `web_search` and `fetch_url`
# directly from this module. Keep their names, signatures, and return/error dict
# shapes stable unless `deepagents-talon` is migrated in the same change.


def __getattr__(name: str) -> object:
    """Lazily re-export SDK-owned tools without slowing dcode startup.

    Returns:
        The requested SDK-owned tool.

    Raises:
        AttributeError: If `name` is not a compatibility re-export.
    """
    if name == "fetch_url":
        from deepagents.tools import fetch_url

        return fetch_url
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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


def create_web_search_tool(api_key: str) -> BaseTool:
    """Bind web search to one workspace credential.

    The schema is taken from `web_search` via `functools.wraps` so the built-in
    and workspace-bound variants can never present different arguments.

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
        if client is None:
            from tavily import TavilyClient as _TavilyClient

            client = _TavilyClient(api_key=api_key)
        return _search_with_tavily(client, **kwargs)

    _workspace_web_search_tools[id(workspace_web_search)] = workspace_web_search
    return workspace_web_search


def is_web_search_tool(candidate: object) -> bool:
    """Return whether `candidate` is a built-in or workspace-bound search tool."""
    return (
        candidate is web_search
        or _workspace_web_search_tools.get(id(candidate)) is candidate
    )


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

    Returns:
        Search hits with title, URL, snippet, and score.
    """
    client = _get_tavily_client()
    if client is None:
        return {
            "error": "Tavily API key not configured. "
            "Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }
    return _search_with_tavily(
        client,
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


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
        return {"error": f"Required package not installed: {exc.name}."}

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
