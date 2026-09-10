"""Web search and cheap page fetch, via Browserbase's hosted services.

Both run server-side at Browserbase and do NOT consume a browser session, so
they never compete with the Stagehand tools for the project's (small)
concurrency budget. That is the main reason to prefer `fetch_page` over
driving the browser when all you need is page text.

Uses BROWSERBASE_API_KEY — the same key the browser tools use — so there is no
separate search provider or secret to manage.
"""

from __future__ import annotations

import inspect
import os

from langchain.tools import tool
from stagehand.browserbase_services import fetch_browserbase, search_browserbase
from stagehand.client_models import (
    _BrowserbaseFetchOptions as FetchOptions,
)
from stagehand.client_models import (
    _BrowserbaseSearchOptions as SearchOptions,
)


def _api_key() -> str:
    key = os.environ.get("BROWSERBASE_API_KEY")
    if not key:
        msg = "BROWSERBASE_API_KEY is required for web search and fetch."
        raise RuntimeError(msg)
    return key


async def _maybe_await(value):
    """Support both sync and async variants of the Browserbase helpers."""
    return await value if inspect.isawaitable(value) else value


@tool
async def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web for products, prices, merchants, and other current info.

    Returns ranked results with title and URL. Cheap and fast — use this to
    FIND candidate pages, then `fetch_page` to read one, and only open the
    browser when you actually need to interact with a page.
    """
    result = await _maybe_await(
        search_browserbase(
            SearchOptions(api_key=_api_key(), query=query, num_results=num_results)
        )
    )
    return [
        {
            "title": item.title,
            "url": item.url,
            "published_date": item.published_date,
        }
        for item in result.results
    ]


@tool
async def fetch_page(url: str) -> str:
    """Fetch a page's content WITHOUT opening a browser.

    Prefer this over `navigate` + `extract` for read-only lookups (product
    pages, prices, policies). It costs no browser session, so it leaves the
    browser free for checkout. Use the browser only when you must click, type,
    or otherwise change page state.
    """
    result = await _maybe_await(
        fetch_browserbase(FetchOptions(api_key=_api_key(), url=url))
    )
    content = result.content
    return content if isinstance(content, str) else str(content)
