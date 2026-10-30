"""The Stagehand facade contract -- `snapshot`, `run`, `screenshot` -- as native tools.

Fourth arm of the browsing comparison. The other three hand the model either a
shell (`browser-mda`), a JavaScript REPL (`browser-mda-repl`), or Browserbase's
hosted MCP (`browser-mda-mcp`). This one implements the contract Stagehand ships
for agent frameworks: three tools, where `snapshot` returns an accessibility
tree with bracketed IDs and `run` either evaluates JavaScript or replays a batch
of actions against those IDs.

What distinguishes it from the MCP arm is where the thinking happens. `act`,
`extract` and `observe` each spend a Stagehand-side model call that never enters
our trace, so that arm's cost is understated. Nothing here calls a second model:
the agent reads the tree and decides for itself, so every token is ours and
visible.

Stagehand's reference facade is a TypeScript stdio MCP server, and `define_mcp`
accepts only remote http/sse transports. Native Python tools are the path the
Stagehand docs name for Managed Deep Agents, so that is what this is -- the same
contract, not the same process.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Any

from langchain.tools import tool
from managed_deepagents import ManagedDeepAgentRuntime
from stagehand import Stagehand, browserbase

_MODEL = os.environ.get("STAGEHAND_MODEL", "google/gemini-2.5-flash-lite")
# Backstop only -- the session is closed explicitly on `after_agent`. A long
# TTL is what let a finished suite keep all 75 concurrency slots occupied,
# so if a run dies before teardown the slot should come back quickly.
_SESSION_TIMEOUT = int(os.environ.get("BROWSERBASE_SESSION_TIMEOUT", "300"))
_MAX_CHARS = 8_000

logger = logging.getLogger(__name__)

_sessions: dict[str, Any] = {}
# Snapshot IDs are only meaningful against the snapshot that minted them, so the
# map is replaced wholesale on every `snapshot` call -- same rule the reference
# facade states: "Every call replaces the active page's ID map."
_id_maps: dict[str, dict[str, str]] = {}
_lock = asyncio.Lock()

_DEAD_SESSION = ("rpc client is closed", "connection closed", "session expired",
                 "websocket", "connection reset", "not connected", "target closed")
_PAGE_LEVEL = ("uncaught", "syntaxerror", "referenceerror", "typeerror")


def _thread_key(runtime: ManagedDeepAgentRuntime) -> str:
    execution = getattr(runtime, "execution_runtime", None)
    for holder in (execution, runtime):
        for attr in ("thread_id", "run_id"):
            value = getattr(holder, attr, None) if holder is not None else None
            if value:
                return str(value)
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if isinstance(configurable, dict):
        for attr in ("thread_id", "run_id"):
            if configurable.get(attr):
                return str(configurable[attr])
    # No shared fallback: one dead session must never be handed to other threads.
    raise RuntimeError("no thread_id on the runtime; cannot key a browser session")


def _is_dead_session(err: BaseException) -> bool:
    text = str(err).lower()
    if any(marker in text for marker in _PAGE_LEVEL):
        return False
    return any(marker in text for marker in _DEAD_SESSION)


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n[truncated]"


def _error(err: BaseException) -> str:
    return _clip(f"error: {type(err).__name__}: {err}")


async def _launch() -> tuple[Any, Any]:
    api_key = os.environ.get("BROWSERBASE_API_KEY", "")
    if not api_key.startswith("bb_"):
        raise RuntimeError("BROWSERBASE_API_KEY is required")
    browser = await browserbase.launch(
        api_key=api_key,
        browser_settings={"viewport": {"width": 1280, "height": 800}},
        timeout=_SESSION_TIMEOUT,
    )
    stagehand = await Stagehand.create(browser=browser, api_key=api_key, model=_MODEL)
    return browser, stagehand


async def _session(runtime: ManagedDeepAgentRuntime):
    key = _thread_key(runtime)
    async with _lock:
        if key not in _sessions:
            _sessions[key] = await _launch()
        return _sessions[key]


async def _evict(runtime: ManagedDeepAgentRuntime) -> None:
    key = _thread_key(runtime)
    async with _lock:
        browser, _ = _sessions.pop(key, (None, None))
        _id_maps.pop(key, None)
    if browser is not None:
        try:
            await browser.close()
        except Exception:  # noqa: BLE001 - the session is already gone
            logger.debug("closing a dead browser failed", exc_info=True)


async def _with_session(runtime: ManagedDeepAgentRuntime, fn):
    """Run `fn(browser, stagehand)`, relaunching once if the session is dead.

    Always returns rather than raising: a browser failure is something the agent
    can route around, and an exception out of a tool ends the whole graph run.
    """
    try:
        browser, stagehand = await _session(runtime)
    except Exception as err:  # noqa: BLE001 - report, do not kill the run
        logger.debug("browser session launch failed", exc_info=True)
        return _error(err)
    try:
        return await fn(browser, stagehand)
    except Exception as err:  # noqa: BLE001 - classify, then recover or report
        if not _is_dead_session(err):
            logger.debug("browser call failed", exc_info=True)
            return _error(err)
        logger.debug("browser session is dead, relaunching", exc_info=True)
        await _evict(runtime)
    try:
        browser, stagehand = await _session(runtime)
        return await fn(browser, stagehand)
    except Exception as err:  # noqa: BLE001 - report, do not kill the run
        logger.debug("browser call failed after relaunch", exc_info=True)
        return _error(err)


def _as_expression(code: str) -> str:
    """Make a snippet evaluable.

    `page.evaluate` takes an *expression*, so a snippet with a top-level
    `return` -- the form every model writes, and the form the facade contract
    documents -- fails with `RPCError: Uncaught`. Wrapping in an async IIFE
    accepts both `return` and `await`; a bare expression is passed through so
    its value still comes back.
    """
    body = code.strip()
    if "return" not in body:
        return body
    return f"(async () => {{ {body} }})()"


async def _page(browser):
    return (await browser.context.pages())[0]


@tool(parse_docstring=True)
async def snapshot(runtime: ManagedDeepAgentRuntime) -> str:
    """Capture the page's accessibility tree, with an ID on every interactive element.

    Call this before acting. The bracketed IDs it returns are what `run`'s
    `actions` refer to, and they are only valid until the next snapshot.
    """
    key = _thread_key(runtime)

    async def go(browser, _stagehand):
        page = await _page(browser)
        result = await page.snapshot()
        _id_maps[key] = dict(getattr(result, "xpath_map", {}) or {})
        return _clip(str(result.formatted_tree))

    return await _with_session(runtime, go)


@tool(parse_docstring=True)
async def run(
    runtime: ManagedDeepAgentRuntime,
    code: str | None = None,
    actions: list[dict] | None = None,
) -> str:
    """Drive the browser, either by running JavaScript or by replaying snapshot actions.

    Provide exactly one of `code` or `actions`.

    `code` runs **in the page**, like a devtools console: `document`, `window`
    and `location` are in scope. There is no `page`, `context` or `browser`
    object -- navigate with an action instead:
    `{"actions": [{"op": "goto", "url": "https://example.com"}]}`.

    Args:
        code: JavaScript evaluated in the page. Return a JSON-serializable value,
            e.g. `return [...document.querySelectorAll('h3')].map(e => e.innerText)`.
        actions: Batch of steps. `{"op": "goto", "url": ...}` navigates; every
            other op takes an `id` from the most recent snapshot --
            click, hover, fill, type, press, select.
    """
    if (code is None) == (actions is None):
        return "error: provide exactly one of `code` or `actions`."

    key = _thread_key(runtime)

    async def go(browser, _stagehand):
        page = await _page(browser)
        if code is not None:
            if "page." in code or "context." in code or "browser." in code:
                return ("error: `code` runs in the page, so there is no `page`, "
                        "`context` or `browser` object. Use `document`/`window`, "
                        'or navigate with {"actions": [{"op": "goto", "url": ...}]}.')
            return _clip(str(await page.evaluate(_as_expression(code))))

        id_map = _id_maps.get(key) or {}
        done = []
        for step in actions:
            op = str(step.get("op", "")).lower()
            if op == "goto":
                url = str(step.get("url") or step.get("value") or "")
                if not url.startswith(("http://", "https://")):
                    return f"error: goto needs an absolute http(s) url, got {url!r}."
                await page.goto(url)
                _id_maps.pop(key, None)  # the page changed; old ids are stale
                id_map = {}
                done.append(f"goto {url}")
                continue
            if not id_map:
                return "error: no snapshot yet. Call `snapshot` before acting on ids."
            ref = str(step.get("id", ""))
            # IDs index a map this process built from the last snapshot; a model
            # string never becomes a selector.
            xpath = id_map.get(ref)
            if xpath is None:
                return (f"error: unknown id {ref!r}. Re-snapshot; "
                        f"{len(id_map)} ids are currently valid.")
            locator = page.locator(xpath)
            if op == "click":
                await locator.click()
            elif op == "hover":
                await locator.hover()
            elif op in ("fill", "type"):
                value = str(step.get("value", ""))
                await (locator.fill(value) if op == "fill" else locator.type(value))
            elif op == "select":
                await locator.select_option(step.get("values") or step.get("value"))
            elif op == "press":
                await locator.type(str(step.get("value", "")))
            else:
                return (f"error: unsupported op {op!r}. "
                        "Use click, hover, fill, type, press or select.")
            done.append(f"{op} {ref}")
        return _clip("ok: " + ", ".join(done))

    return await _with_session(runtime, go)


@tool(parse_docstring=True)
async def screenshot(runtime: ManagedDeepAgentRuntime, full_page: bool = False) -> Any:
    """Photograph the page when the accessibility tree is not enough to judge it.

    Expensive in tokens -- prefer `snapshot`. Useful for charts, maps, canvas
    and visual layout questions.

    Args:
        full_page: Capture the whole scrollable page rather than the viewport.
    """

    async def go(browser, _stagehand):
        page = await _page(browser)
        raw = await page.screenshot(type="jpeg", quality=60, full_page=full_page)
        return [{
            "type": "image",
            "source_type": "base64",
            "mime_type": "image/jpeg",
            "data": base64.b64encode(raw).decode(),
        }]

    return await _with_session(runtime, go)


async def release(runtime: ManagedDeepAgentRuntime) -> None:
    """Close this thread's Browserbase session.

    Nothing else releases it. `browserbase.launch(timeout=...)` sets a session
    TTL, so a leaked session lives out that TTL doing nothing but occupying a
    concurrency slot: one 100-task suite at concurrency 8 pinned the project at
    its full 75 live sessions and starved every other arm. Closing on
    `after_agent` keeps the live count near the actual concurrency.
    """
    await _evict(runtime)


FACADE_TOOLS = [snapshot, run, screenshot]
