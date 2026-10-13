"""Browser tools for a deep agent: `snapshot`, `run`, `screenshot`.

The tool surface is Stagehand's facade contract. The browser behind it is a
`BrowserProvider` -- hosted Browserbase, a Chrome you launched, or anything
reachable over CDP -- so the vocabulary the model learns does not change when
the infrastructure does.

That split is not invented here. Anthropic's `browser_toolset_20260801` ships
thirty members, and Stagehand's adapter executes all of them against four
primitives: snapshot, run, run-actions, screenshot. The vocabulary and the
connection are genuinely separate concerns, so they are separate types.

    from deepagents.middleware.browser import BrowserMiddleware, Browserbase

    agent = create_deep_agent(middleware=[BrowserMiddleware(Browserbase())])
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool, ToolRuntime, tool

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from langchain.agents.middleware.types import AgentState
    from langgraph.runtime import Runtime
    from stagehand import Stagehand
    from stagehand.browser import BrowserbaseRegion, StagehandBrowser
    from stagehand.page import Page

    AnyRuntime = ToolRuntime | Runtime

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RESULT_CHARS = 8_000

# A transport that died. The session is gone; reopen and try once more.
_DEAD_SESSION = (
    "rpc client is closed",
    "connection closed",
    "connection reset",
    "session expired",
    "target closed",
    "websocket",
)
# A page that threw. The browser is fine -- reopening would destroy a healthy
# session and lose the page the agent was working on.
_PAGE_ERROR = ("uncaught", "syntaxerror", "referenceerror", "typeerror")


@dataclass(frozen=True)
class BrowserContext:
    """What a provider is told when it is asked for a browser."""

    thread_id: str
    runtime: AnyRuntime
    """The agent's `ToolRuntime`, so a provider can reach the agent's own
    sandbox. The sandbox is per-thread, so its CDP endpoint is too."""


class BrowserProvider(Protocol):
    """Where the browser comes from, and who is responsible for closing it."""

    async def open(self, ctx: BrowserContext) -> StagehandBrowser:
        """Return a connected Stagehand browser."""
        ...

    async def close(self, browser: StagehandBrowser, ctx: BrowserContext) -> None:
        """Release it. Close only what you opened."""
        ...


@dataclass
class Browserbase:
    """A hosted Browserbase session. The Stagehand extension is preinstalled."""

    api_key: str | None = None
    region: BrowserbaseRegion | None = None
    # A backstop, not the release mechanism -- the middleware closes on
    # `after_agent`. A long TTL means a crashed run holds a concurrency slot
    # until it expires.
    session_timeout: int = 300

    async def open(self, ctx: BrowserContext) -> StagehandBrowser:
        """Start a session, labelled with the thread that owns it."""
        from stagehand import browserbase  # noqa: PLC0415 - optional dependency

        return await browserbase.launch(
            api_key=self.api_key or os.environ["BROWSERBASE_API_KEY"],
            region=self.region,
            timeout=self.session_timeout,
            # Makes a leaked session traceable to the run that leaked it.
            user_metadata={"thread_id": ctx.thread_id},
        )

    async def close(self, browser: StagehandBrowser, ctx: BrowserContext) -> None:  # noqa: ARG002
        """Shut the browser we launched."""
        await browser.close()


@dataclass
class LocalChrome:
    """Chrome launched on this machine, torn down with the run."""

    headless: bool = True
    executable_path: str | None = None

    async def open(self, ctx: BrowserContext) -> StagehandBrowser:  # noqa: ARG002
        """Launch Chrome."""
        from stagehand import local_browser  # noqa: PLC0415 - optional dependency

        return await local_browser.launch(headless=self.headless, executable_path=self.executable_path)

    async def close(self, browser: StagehandBrowser, ctx: BrowserContext) -> None:  # noqa: ARG002
        """Shut the browser we launched."""
        await browser.close()


@dataclass
class CDP:
    """A browser someone else is running, reached over CDP.

    Stagehand loads its extension with `Extensions.loadUnpacked`, which reads a
    path on the *browser's* filesystem -- so this works for a Chrome on this
    machine or in a sandbox that shares it, and not for a remote provider
    unless that provider preinstalls the extension.
    """

    url: str

    async def open(self, ctx: BrowserContext) -> StagehandBrowser:  # noqa: ARG002
        """Attach to the browser already running at `url`."""
        from stagehand import local_browser  # noqa: PLC0415 - optional dependency

        return await local_browser.connect(cdp_url=self.url)

    async def close(self, browser: StagehandBrowser, ctx: BrowserContext) -> None:
        """Leave it running."""
        # Deliberately nothing. `browser.close()` on a connected browser sends
        # `Browser.close`, which would kill a browser we never owned.


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n[truncated]"


def _ambient_config() -> dict | None:
    from langgraph.config import get_config  # noqa: PLC0415 - avoids an import cycle

    try:
        return dict(get_config())
    except Exception:  # noqa: BLE001 - no graph context (e.g. a direct unit test)
        return None


def _thread_id(runtime: AnyRuntime) -> str:
    """The key for this run's browser.

    Tools get a `ToolRuntime` carrying `config`; `after_agent` gets the agent
    `Runtime`, which does not. The ambient LangGraph config covers both.
    """
    for source in (getattr(runtime, "config", None), _ambient_config()):
        configurable = source.get("configurable") if isinstance(source, dict) else None
        if isinstance(configurable, dict) and configurable.get("thread_id"):
            return str(configurable["thread_id"])
    for attr in ("thread_id", "run_id"):
        if value := getattr(runtime, attr, None):
            return str(value)
    # Never fall back to a shared key: in a long-lived process that hands one
    # thread's dead browser to every thread that follows it.
    msg = "no thread_id on the runtime; cannot key a browser session"
    raise RuntimeError(msg)


# `experimental_batch` hands the callback a worker-local batch context whose
# `page` and `context` are already Playwright-shaped, so a snippet can call
# `await page.goto(...)` and `page.locator(sel).click()` the way the facade
# contract documents. Wrapping the body in an async IIFE is what makes a
# top-level `return` legal.
_RUN_SOURCE = """async (batch, input) => {
  const page = batch.page;
  const context = batch.context;
  return await (async () => { %s })();
}"""

# One round trip for a whole batch, rather than one per step.
_ACTIONS_SOURCE = """async (batch, input) => {
  for (const step of input.steps) {
    const target = batch.page.locator(step.selector);
    if (step.op === "click") await target.click();
    else if (step.op === "hover") await target.hover();
    else if (step.op === "fill") await target.fill(step.value);
    else if (step.op === "type" || step.op === "press") await target.type(step.value);
    else if (step.op === "select") await target.selectOption(step.value);
  }
  return { completed: input.steps.length, url: await batch.page.url() };
}"""

RUN_TIMEOUT_MS = 60_000
MAX_QUALITY = 100

# Tool descriptions and argument names are copied verbatim from Stagehand's
# facade contract (`packages/integrations/core/src/facade/contract.ts`), which
# pins them string-exact on purpose: "models are prompted against these
# descriptions". Reword these and the agent is running a different contract
# from every published benchmark of it.
RUN_DESCRIPTION = (
    "Browse and automate websites in the persistent browser by executing JavaScript "
    "against a Playwright-shaped API. The code runs inside an async function with "
    "page, context, and browser in scope (Playwright Page, BrowserContext, and "
    "Browser); use await directly and return a JSON-serializable value when useful. "
    'Navigate with await page.goto("https://example.com"); there is no separate '
    "navigate or start tool. Alternatively, provide a batch of actions using IDs from "
    "the latest snapshot. Provide exactly one of code or actions. Each action must use "
    '"op" (never "kind") and "id" (never "ref"). Copy the bracketed snapshot ID as a '
    'string. Examples: {"actions":[{"op":"click","id":"1-42"}]}, '
    '{"actions":[{"op":"fill","id":"2-14","value":"Miami"}]}, '
    '{"actions":[{"op":"select","id":"3-9","values":"Lowest price"}]}.'
)

SNAPSHOT_DESCRIPTION = (
    "Capture the active page's Stagehand accessibility tree and hydrate its displayed "
    "IDs for subsequent run actions. Every call replaces the active page's ID map."
)

SCREENSHOT_DESCRIPTION = (
    "Capture a screenshot of the active page. For size-constrained MCP clients, prefer "
    'a viewport JPEG: {"type":"jpeg","quality":40,"fullPage":false}.'
)

# Per-op required fields, also from the contract: `type` carries `text`, not
# `value`, and `press` carries `key`.
_ACTION_FIELDS: dict[str, str] = {"fill": "value", "type": "text", "press": "key"}


@dataclass
class _Session:
    browser: StagehandBrowser
    stagehand: Stagehand
    """Attaching is what puts `browser.context` in service; holding the handle
    is what keeps it there."""
    # Snapshot ids are only valid against the snapshot that minted them.
    ids: dict[str, str] = field(default_factory=dict)


class BrowserMiddleware(AgentMiddleware):
    """Give the agent a browser: `snapshot`, `run`, and `screenshot`."""

    def __init__(
        self,
        provider: BrowserProvider | None = None,
        *,
        allow_javascript: bool = True,
    ) -> None:
        """Build the browser tools.

        Args:
            provider: Where the browser comes from. Defaults to `Browserbase()`.
            allow_javascript: Whether `run` accepts a `code` snippet.
        """
        super().__init__()
        self.provider = provider or Browserbase()
        # Anthropic's browser toolset ships `javascript_exec` disabled by
        # default. Same capability, same reasoning -- but on here, because
        # letting a snippet pick what to return is the whole cost advantage.
        self.allow_javascript = allow_javascript
        self._sessions: dict[str, _Session] = {}
        self.tools = self._build_tools()

    # -- session lifecycle ----------------------------------------------

    async def _session(self, runtime: AnyRuntime) -> _Session:
        key = _thread_id(runtime)
        if key not in self._sessions:
            from stagehand import Stagehand  # noqa: PLC0415 - optional dependency

            ctx = BrowserContext(thread_id=key, runtime=runtime)
            browser = await self.provider.open(ctx)
            # Providers hand back a browser; attaching Stagehand to it is the
            # same step for every provider, so it belongs here.
            self._sessions[key] = _Session(browser=browser, stagehand=await Stagehand.create(browser=browser))
        return self._sessions[key]

    async def _release(self, runtime: AnyRuntime) -> None:
        key = _thread_id(runtime)
        session = self._sessions.pop(key, None)
        if session is None:
            return
        ctx = BrowserContext(thread_id=key, runtime=runtime)
        try:
            await self.provider.close(session.browser, ctx)
        except Exception:  # noqa: BLE001 - teardown must not fail a finished run
            logger.debug("closing the browser failed", exc_info=True)

    async def _call(self, runtime: AnyRuntime, fn: Callable[[_Session, Page], Coroutine[object, object, T]]) -> T | str:
        """Run `fn(session, page)`, reopening once if the session died.

        Always returns. An exception out of a tool ends the whole graph run,
        and a browsing failure is something the next turn can route around.
        """
        for attempt in (1, 2):
            try:
                session = await self._session(runtime)
                page = (await session.browser.context.pages())[0]
                return await fn(session, page)
            except Exception as err:  # noqa: BLE001 - classify, then recover or report
                text = str(err).lower()
                dead = any(m in text for m in _DEAD_SESSION) and not any(m in text for m in _PAGE_ERROR)
                if dead and attempt == 1:
                    logger.debug("browser session is dead, reopening", exc_info=True)
                    await self._release(runtime)
                    continue
                logger.debug("browser call failed", exc_info=True)
                return _clip(f"error: {type(err).__name__}: {err}")
        return "error: browser unavailable"

    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> None:  # noqa: ARG002
        """Release the browser. Nothing else does."""
        await self._release(runtime)

    # -- tools ----------------------------------------------------------

    def _build_tools(self) -> list[BaseTool]:  # noqa: C901 - three tool closures, each flat
        middleware = self

        @tool(description=SNAPSHOT_DESCRIPTION)
        async def snapshot(runtime: ToolRuntime, *, includeIframes: bool = True) -> str:  # noqa: N803 - wire contract name
            async def go(session: _Session, page: Page) -> str:
                result = await page.snapshot(include_iframes=includeIframes)
                session.ids = dict(getattr(result, "xpath_map", {}) or {})
                return _clip(str(result.formatted_tree))

            return await middleware._call(runtime, go)

        @tool(description=RUN_DESCRIPTION)
        async def run(
            runtime: ToolRuntime,
            code: str | None = None,
            actions: list[dict] | None = None,
        ) -> str:
            if (code is None) == (actions is None):
                return "error: provide exactly one of `code` or `actions`."
            if code is not None and not middleware.allow_javascript:
                return "error: JavaScript is disabled. Use `actions`."

            async def go(session: _Session, page: Page) -> str:
                if code is not None:
                    result = await session.stagehand.experimental_batch(_RUN_SOURCE % code, {}, timeout=RUN_TIMEOUT_MS, page=page)
                    return _clip(str(result))
                return await _apply(session, page, actions or [])

            return await middleware._call(runtime, go)

        tools = [snapshot, run]

        @tool(description=SCREENSHOT_DESCRIPTION)
        async def screenshot(
            runtime: ToolRuntime,
            *,
            fullPage: bool = False,  # noqa: N803 - wire contract name
            type: str = "jpeg",  # noqa: A002 - wire contract name
            quality: int | None = 60,
        ) -> list[dict[str, str]] | str:
            problem = _check_image(type, quality)
            if problem is not None:
                return problem
            if type == "png":
                quality = None  # meaningless for png

            image_type: Literal["png", "jpeg"] = "png" if type == "png" else "jpeg"

            async def go(_session: _Session, page: Page) -> list[dict[str, str]]:
                raw = await page.screenshot(type=image_type, quality=quality, full_page=fullPage)
                return [
                    {
                        "type": "image",
                        "source_type": "base64",
                        "mime_type": f"image/{image_type}",
                        "data": base64.b64encode(raw).decode(),
                    }
                ]

            return await middleware._call(runtime, go)

        tools.append(screenshot)
        return tools


def _check_image(image_type: str, quality: int | None) -> str | None:
    """Validate the screenshot arguments the contract advertises."""
    if image_type not in ("png", "jpeg"):
        return "error: type must be png or jpeg."
    if image_type == "jpeg" and quality is not None and not 0 <= quality <= MAX_QUALITY:
        return f"error: quality must be between 0 and {MAX_QUALITY}."
    return None


def _hydrate(session: _Session, step: dict) -> tuple[dict | None, str | None]:
    """Turn one contract action into a selector step, or explain why not."""
    op = str(step.get("op", "")).lower()
    if op not in {"click", "hover", "fill", "type", "press", "select"}:
        return None, (f"error: unsupported op {op!r}. Use click, hover, fill, type, press or select; navigate with `code` and `page.goto`.")
    ref = step.get("id")
    if not isinstance(ref, str) or not ref:
        return None, 'error: each action requires a non-empty string "id".'
    xpath = session.ids.get(ref)
    if xpath is None:
        return None, (f"error: snapshot ID {ref!r} is stale or not actionable; snapshot again. {len(session.ids)} ids are currently valid.")

    value: str | list[str] = ""
    if field := _ACTION_FIELDS.get(op):
        raw = step.get(field)
        if not isinstance(raw, str):
            return None, f'error: {op} requires string field "{field}".'
        value = raw
    elif op == "select":
        values = step.get("values")
        if isinstance(values, str):
            value = values
        elif isinstance(values, list) and values and all(isinstance(v, str) for v in values):
            value = values[0] if len(values) == 1 else values
        else:
            return None, 'error: select requires "values" as a string or non-empty string list.'

    return {"op": op, "selector": f"xpath={xpath}", "value": value}, None


async def _apply(session: _Session, page: Page, actions: list[dict]) -> str:
    """Replay snapshot actions in one round trip.

    Field names are the contract's, not conveniences: `type` carries `text`,
    `press` carries `key`, `select` carries `values`. A model prompted on the
    reference description sends exactly these, and quietly accepting a
    different key would hide its mistakes.
    """
    if not actions:
        return "error: actions must not be empty."

    steps = []
    for action in actions:
        step, problem = _hydrate(session, action)
        if problem is not None:
            return problem
        steps.append(step)

    result = await session.stagehand.experimental_batch(_ACTIONS_SOURCE, {"steps": steps}, timeout=RUN_TIMEOUT_MS, page=page)
    return _clip(str(result))
