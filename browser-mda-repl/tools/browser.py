"""Stagehand browser tools, exposed only inside the QuickJS REPL.

These are never handed to the model directly. `CodeInterpreterMiddleware(ptc=...)`
surfaces them as `tools.<camelCaseName>()` inside JavaScript, so the agent has to
write code to browse -- which is the whole point: one snippet can navigate,
extract and filter in a single model call, and return only the fields it wants.

The browser is a Browserbase session, so inference for act/extract/observe runs
on their Model Gateway against the session and needs no provider key. One session
per thread, reused across calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain.tools import tool
from managed_deepagents import ManagedDeepAgentRuntime
from stagehand import Stagehand, browserbase

# Held equal to the hosted Browserbase MCP endpoint, which is pinned to
# `google/gemini-2.5-flash-lite` and confirmed so from its own responses --
# `"modelId": "gemini-2.5-flash-lite"` appears in 72 of 274 sampled
# `browserbase_act` results from the 2026-10-09 run. act/observe/extract are the
# same Stagehand verbs in both arms, so leaving this on a different model meant
# the MCP-vs-REPL comparison varied model and interface at once.
#
# Matching downward rather than pinning MCP upward is deliberate: overriding
# `modelName` on the hosted endpoint also requires a `modelApiKey` Browserbase
# can use directly, and the gateway key in ANTHROPIC_API_KEY is not one. This
# direction needs no new credential.
_MODEL = os.environ.get("STAGEHAND_MODEL", "google/gemini-2.5-flash-lite")
_SESSION_TIMEOUT = int(os.environ.get("BROWSERBASE_SESSION_TIMEOUT", "900"))
_MAX_CHARS = 8_000

logger = logging.getLogger(__name__)

_sessions: dict[str, Any] = {}
_lock = asyncio.Lock()


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
    # No shared fallback key. The server process is long-lived, so a literal
    # "default" would hand one thread's Browserbase session to every other
    # thread -- and once that session hit the project timeout, every later run
    # in the deployment failed with "RPC client is closed".
    raise RuntimeError("no thread_id on the runtime; cannot key a browser session")


# A Browserbase session outlives neither the project timeout nor a dropped
# websocket, and Stagehand surfaces both the same way.
_DEAD_SESSION = ("rpc client is closed", "connection closed", "session expired",
                 "websocket", "connection reset", "not connected", "target closed")

# A page-level throw is NOT a dead session. `page.evaluate` on a snippet that
# raises comes back as RPCError("Uncaught"), and relaunching on that both
# destroys a healthy browser and loses the page the agent was working on.
_PAGE_LEVEL = ("uncaught", "syntaxerror", "referenceerror", "typeerror")


def _is_dead_session(err: BaseException) -> bool:
    text = str(err).lower()
    if any(marker in text for marker in _PAGE_LEVEL):
        return False
    return any(marker in text for marker in _DEAD_SESSION)


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
    # Which model actually served act/extract/observe is otherwise unknowable from
    # the outside: Stagehand runs them on Browserbase's Model Gateway, so the
    # choice shows up in neither the LangSmith trace (unlike the MCP variant,
    # whose raw responses carry `"modelId"`) nor the Browserbase session logs.
    # Log it so a run can be attributed to a model after the fact.
    logger.info("stagehand session up on model=%s", _MODEL)
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
    if browser is not None:
        try:
            await browser.close()
        except Exception:  # noqa: BLE001 - the session is already gone
            pass


async def _with_session(runtime: ManagedDeepAgentRuntime, fn) -> str:
    """Run `fn(browser, stagehand)`, relaunching once if the session is dead.

    Always returns a string. A browser failure is something the agent can work
    around -- rewrite the selector, try a different page -- so it comes back as
    text for the model to read. Raising instead ends the whole run: 6 of the
    first 17 runs died that way, most of them on a snippet whose JavaScript
    threw in the page.
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


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n[truncated]"


def _error(err: BaseException) -> str:
    """Render a browser failure as something the agent can act on."""
    return _clip(f"error: {type(err).__name__}: {err}")


@tool(parse_docstring=True)
async def navigate(url: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Open a URL. Returns the final URL and page title only.

    Args:
        url: Absolute URL including the scheme.
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    async def go(browser, _stagehand):
        page = (await browser.context.pages())[0]
        await page.goto(url)
        # Deliberately narrow: the MCP variant's navigate returned a serialized
        # CDP object graph averaging 11,160 tokens, the single largest cost in
        # that harness. There is nothing here the agent did not ask for.
        return f"{await page.url()} | {await page.title()}"

    return await _with_session(runtime, go)


@tool(parse_docstring=True)
async def extract(instruction: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Pull information off the current page, described in plain English.

    Args:
        instruction: What to extract, e.g. "the top three story titles and links".
    """
    async def run(_browser, stagehand):
        result = await stagehand.extract(instruction)
        return _clip(str(getattr(result, "data", result)))

    return await _with_session(runtime, run)


@tool(parse_docstring=True)
async def act(instruction: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Perform ONE action on the page, described in plain English.

    Args:
        instruction: A single action, e.g. "click the Sign in link".
    """
    async def run(_browser, stagehand):
        return _clip(str(await stagehand.act(instruction)))

    return await _with_session(runtime, run)


@tool(parse_docstring=True)
async def observe(instruction: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Ask what is on the page and what can be acted on.

    Args:
        instruction: What to look for, e.g. "what can I click here?".
    """
    async def run(_browser, stagehand):
        return _clip(str(await stagehand.observe(instruction)))

    return await _with_session(runtime, run)


@tool(parse_docstring=True)
async def evaluateJs(expression: str, runtime: ManagedDeepAgentRuntime) -> str:
    """Evaluate a JavaScript expression in the PAGE and return its result.

    The cheap read path: pull out exactly the fields you need rather than an
    extraction over the whole page.

    Args:
        expression: A JavaScript expression evaluated in the page context.
    """
    async def run(browser, _stagehand):
        page = (await browser.context.pages())[0]
        return _clip(str(await page.evaluate(expression)))

    return await _with_session(runtime, run)


@tool(parse_docstring=True)
async def currentUrl(runtime: ManagedDeepAgentRuntime) -> str:
    """Report the URL the browser is on."""
    async def run(browser, _stagehand):
        return str(await (await browser.context.pages())[0].url())

    return await _with_session(runtime, run)


# Held equal to the Browserbase MCP server's surface by default.
#
# That server exposes exactly six tools -- start, end, navigate, act, observe,
# extract -- and has no JavaScript-evaluation tool at all. `evaluateJs` is
# therefore a capability the MCP variant cannot match, and `currentUrl` reads a
# field its `navigate` buries in a serialized CDP object. Shipping both here
# meant a three-way run measured capability as well as interface, which is the
# one thing the comparison is supposed to hold still. `start`/`end` have no
# counterpart to drop: session lifecycle is handled per-thread in `_session`.
#
# Set `REPL_BROWSER_EXTRAS=1` to restore them and measure code mode's actual
# ceiling rather than its MCP-matched floor. `instructions.md` documents only
# the parity set, so the extras are available but unadvertised.
_PARITY = [navigate, extract, act, observe]
_EXTRAS = [evaluateJs, currentUrl]

BROWSER_TOOLS = _PARITY + (_EXTRAS if os.environ.get("REPL_BROWSER_EXTRAS") == "1" else [])
