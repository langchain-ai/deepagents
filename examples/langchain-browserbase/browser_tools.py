from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable
from typing import Any

from browserbase import Browserbase
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from stagehand import Stagehand, StagehandConfig

DEFAULT_STAGEHAND_MODEL = os.getenv(
    "STAGEHAND_MODEL",
    "google/gemini-3-flash-preview",
)
DEFAULT_STAGEHAND_AGENT_MODEL = os.getenv(
    "STAGEHAND_AGENT_MODEL",
    "anthropic/claude-sonnet-4-6",
)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        msg = f"Missing required environment variable: {name}"
        raise ValueError(msg)
    return value


def _browserbase_client() -> Browserbase:
    return Browserbase(api_key=_require_env("BROWSERBASE_API_KEY"))


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump())
    if hasattr(value, "dict"):
        return _normalize(value.dict())
    if hasattr(value, "__dict__"):
        public = {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_") and not callable(val)
        }
        if public:
            return _normalize(public)
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(_normalize(value), indent=2, default=str)


def _html_to_text(html: str, max_chars: int) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text[: max(1, max_chars)]


def _stagehand_client() -> Stagehand:
    _require_env("BROWSERBASE_API_KEY")
    return Stagehand()


def _stagehand_config(model_name: str) -> StagehandConfig:
    return StagehandConfig(
        env="BROWSERBASE",
        api_key=_require_env("BROWSERBASE_API_KEY"),
        project_id=os.getenv("BROWSERBASE_PROJECT_ID"),
        model_name=model_name,
    )


def _session_id(response: Any) -> str:
    data = getattr(response, "data", None)
    session_id = getattr(data, "session_id", None) or getattr(response, "session_id", None)
    if not session_id:
        msg = f"Could not extract session id from Stagehand response: {_json(response)}"
        raise RuntimeError(msg)
    return session_id


def _extract_result_payload(result: Any) -> Any:
    data = getattr(result, "data", None)
    extracted = getattr(data, "result", None)
    if extracted is not None:
        return _normalize(extracted)
    return _normalize(result)


def _close_stagehand(client: Any) -> None:
    closer = getattr(client, "close", None)
    if callable(closer):
        closer()


def _run_async(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


@tool
def browserbase_search(query: str, num_results: int = 5) -> str:
    """Search the web with Browserbase.

    Args:
        query: Search query.
        num_results: Number of search results to return.

    Returns:
        JSON string with search results and request metadata.

    """
    bb = _browserbase_client()
    response = bb.search.web(query=query, num_results=max(1, min(num_results, 10)))
    results = []
    for result in getattr(response, "results", []):
        results.append(
            {
                "title": getattr(result, "title", ""),
                "url": getattr(result, "url", ""),
                "author": getattr(result, "author", None),
                "published_date": (
                    getattr(result, "published_date", None)
                    or getattr(result, "publishedDate", None)
                ),
            }
        )
    return _json(
        {
            "query": query,
            "request_id": getattr(response, "request_id", None)
            or getattr(response, "requestId", None),
            "results": results,
        }
    )


@tool
def browserbase_fetch(url: str, use_proxy: bool = False, max_chars: int = 12000) -> str:
    """Fetch page content without a browser session.

    Args:
        url: Page URL to fetch.
        use_proxy: Whether to fetch through Browserbase proxies.
        max_chars: Maximum number of text characters to return.

    Returns:
        JSON string with status metadata and extracted text.

    """
    bb = _browserbase_client()
    response = bb.fetch_api.create(url=url, proxies=use_proxy)
    content = getattr(response, "content", "")
    content_type = (
        getattr(response, "content_type", None) or getattr(response, "contentType", "") or ""
    ).lower()

    title = ""
    text = str(content)[: max(1, max_chars)]
    if "html" in content_type:
        title, text = _html_to_text(str(content), max_chars=max_chars)

    return _json(
        {
            "url": url,
            "status_code": getattr(response, "status_code", None)
            or getattr(response, "statusCode", None),
            "content_type": getattr(response, "content_type", None)
            or getattr(response, "contentType", None),
            "encoding": getattr(response, "encoding", None),
            "title": title,
            "text": text,
        }
    )


@tool
def browserbase_rendered_extract(start_url: str, instruction: str) -> str:
    """Extract rendered content from a page with Stagehand.

    Args:
        start_url: Initial URL for the hosted browser session.
        instruction: Natural language extraction instruction.

    Returns:
        JSON string with extraction results and Browserbase session metadata.

    """
    client = _stagehand_client()
    response = client.sessions.start(model_name=DEFAULT_STAGEHAND_MODEL)
    session_id = _session_id(response)

    try:
        client.sessions.navigate(id=session_id, url=start_url)
        result = client.sessions.extract(id=session_id, instruction=instruction)
        return _json(
            {
                "start_url": start_url,
                "session_id": session_id,
                "session_url": f"https://browserbase.com/sessions/{session_id}",
                "instruction": instruction,
                "result": _extract_result_payload(result),
            }
        )
    finally:
        try:
            client.sessions.end(id=session_id)
        finally:
            _close_stagehand(client)


@tool
def browserbase_interactive_task(start_url: str, task: str) -> str:
    """Run a multi-step browser task in a Browserbase-hosted Stagehand session.

    Args:
        start_url: Initial URL for the hosted browser session.
        task: Natural language instruction for the Stagehand agent.

    Returns:
        JSON string with the Stagehand agent result.

    """
    return _run_async(_browserbase_interactive_task_async(start_url=start_url, task=task))


async def _browserbase_interactive_task_async(start_url: str, task: str) -> str:
    config = _stagehand_config(model_name=DEFAULT_STAGEHAND_AGENT_MODEL)
    async with Stagehand(config) as stagehand:
        page = stagehand.page
        await page.goto(start_url)

        agent = stagehand.agent(
            model=DEFAULT_STAGEHAND_AGENT_MODEL,
            instructions=(
                "You are executing a browser task on behalf of a LangChain tool. "
                "Be precise, avoid unnecessary actions, and stop once the requested "
                "task is complete."
            ),
        )
        agent_result = await agent.execute(task)
        return _json(_normalize(agent_result))
