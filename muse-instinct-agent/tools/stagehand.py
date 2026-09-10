from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, ConfigDict, Field
from stagehand import Page, Stagehand, StagehandBrowser, browserbase

_ACTION_SOURCE = """async (stagehand, input) => {
  let completed = 0;
  for (const action of input.actions) {
    const locator = stagehand.page.locator(action.selector);
    switch (action.op) {
      case "click": await locator.click(); break;
      case "hover": await locator.hover(); break;
      case "fill": await locator.fill(action.value); break;
      case "type": await locator.type(
        action.text,
        action.delay === undefined ? undefined : { delay: action.delay },
      ); break;
      case "press": await locator.click(); await stagehand.page.keyPress(action.key); break;
      case "select": await locator.selectOption(action.values); break;
      default: throw new Error("Unsupported ref action: " + String(action.op));
    }
    completed += 1;
  }
  return { completed };
}"""

logger = logging.getLogger(__name__)

_TEXT_NODE_SUFFIX = re.compile(r"/text\(\)(?:\[\d+\])?$")


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sanitize_error(message: str) -> str:
    """Redact credential-bearing fragments before errors reach the model."""
    message = re.sub(
        r"([?&](?:signingKey|apiKey|api_key|token|key)=)[^&\s\"']+",
        r"\1[redacted]",
        message,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\b(sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]+", r"\1[redacted]", message)


@lru_cache(maxsize=1)
def _facade_source() -> str:
    return Path(__file__).with_name("_assets").joinpath("playwright_facade.js").read_text()


async def _release_session(api_key: str, api_url: str, session_id: str) -> None:
    """Best-effort release of a keep-alive session so expiry doesn't strand it."""
    from browserbase import AsyncBrowserbase

    try:
        async with AsyncBrowserbase(api_key=api_key, base_url=api_url) as client:
            await client.sessions.update(session_id, status="REQUEST_RELEASE")
    except Exception:
        pass


class ClickAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["click"]
    id: str = Field(min_length=1)


class HoverAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["hover"]
    id: str = Field(min_length=1)


class FillAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["fill"]
    id: str = Field(min_length=1)
    value: str


class TypeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["type"]
    id: str = Field(min_length=1)
    text: str
    delay: float | None = Field(default=None, ge=0)


class PressAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["press"]
    id: str = Field(min_length=1)
    key: str = Field(min_length=1)


class SelectAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["select"]
    id: str = Field(min_length=1)
    values: str | list[str]


Action = ClickAction | HoverAction | FillAction | TypeAction | PressAction | SelectAction


@dataclass(frozen=True)
class _Snapshot:
    url: str
    xpath_by_id: dict[str, str]


class _BrowserRuntime:
    def __init__(
        self,
        stagehand: Stagehand,
        browser: StagehandBrowser,
        *,
        browserbase_api_key: str,
        browserbase_api_url: str,
    ) -> None:
        self.stagehand = stagehand
        self.browser = browser
        self.browserbase_api_key = browserbase_api_key
        self.browserbase_api_url = browserbase_api_url
        self.session_id = browser.session_id
        self.snapshots: dict[str, _Snapshot] = {}
        self.lock = asyncio.Lock()

    @classmethod
    async def start(cls, *, session_id: str | None = None) -> _BrowserRuntime:
        # LOCAL PATCH (not upstream): honour STAGEHAND_BROWSER=local so `mda dev`
        # can drive free local Chrome instead of metered Browserbase minutes.
        # Upstream's local example uses the same variable name. The deployed
        # runtime has no Chrome and no display, so this only applies locally —
        # deployments must stay on Browserbase.
        # Double-gated on purpose. `.env` is forwarded to the deployment as
        # secrets, so STAGEHAND_BROWSER=local WILL reach the cloud — where there
        # is no Chrome and no display, and every browser call would fail.
        # MDA_LOCAL_DEV is set only by `mda dev`, so production ignores the flag
        # entirely and falls through to Browserbase below.
        _local_dev = os.environ.get("MDA_LOCAL_DEV") == "1"
        _want_local = os.environ.get("STAGEHAND_BROWSER", "browserbase").lower() == "local"
        if _want_local and not _local_dev:
            logger.warning(
                "STAGEHAND_BROWSER=local ignored: no local Chrome outside `mda dev`; "
                "using Browserbase."
            )
        if _want_local and _local_dev:
            from stagehand import local_browser

            # Chrome discovery stats the filesystem (os.access), which the
            # LangGraph dev server's blockbuster flags as blocking IO. It is a
            # handful of stat calls at launch only, so skip the check for this
            # scope rather than lose the dev server. No-op when blockbuster is
            # absent, and unreachable in deployments (Browserbase branch below).
            try:
                from blockbuster.blockbuster import blockbuster_skip

                skip_token = blockbuster_skip.set(True)
            except ImportError:
                blockbuster_skip = skip_token = None

            try:
                browser = await local_browser.launch(
                    headless=os.environ.get("STAGEHAND_HEADLESS", "true").lower() != "false",
                    viewport_width=1280,
                    viewport_height=720,
                    keep_alive=True,
                )
            finally:
                if blockbuster_skip is not None and skip_token is not None:
                    blockbuster_skip.reset(skip_token)
            return await cls._attach(browser, browserbase_api_key="", browserbase_api_url="")

        api_key = os.environ.get("BROWSERBASE_API_KEY", "")
        if not api_key:
            raise RuntimeError("BROWSERBASE_API_KEY is required")
        api_url = os.environ.get("BROWSERBASE_API_URL", "https://api.browserbase.com")
        if session_id is not None:
            # Reconnect to a keep-alive session that already carries the
            # extension from its original launch.
            browser = await browserbase.connect(
                api_key=api_key,
                base_url=api_url,
                session_id=session_id,
            )
            return await cls._attach(
                browser,
                browserbase_api_key=api_key,
                browserbase_api_url=api_url,
            )
        # The published stagehand wheel bundles the extension;
        # browserbase.launch provisions it automatically. STAGEHAND_EXTENSION_ID
        # remains available for pre-uploaded extensions.
        configured_extension_id = os.environ.get("STAGEHAND_EXTENSION_ID") or None
        browser = await browserbase.launch(
            api_key=api_key,
            base_url=api_url,
            browser_settings={"viewport": {"width": 1280.0, "height": 720.0}},
            extension_id=configured_extension_id,
            keep_alive=True,
        )
        return await cls._attach(
            browser,
            browserbase_api_key=api_key,
            browserbase_api_url=api_url,
        )

    @classmethod
    async def _attach(
        cls,
        browser: StagehandBrowser,
        *,
        browserbase_api_key: str,
        browserbase_api_url: str,
    ) -> _BrowserRuntime:
        try:
            model = os.environ.get("STAGEHAND_MODEL") or None
            model_api_key = os.environ.get("STAGEHAND_MODEL_API_KEY") or None
            if model_api_key and not model:
                raise RuntimeError("STAGEHAND_MODEL_API_KEY requires STAGEHAND_MODEL")
            create_options: dict[str, object] = {}
            if model:
                create_options["model"] = model
            if model_api_key:
                create_options["model_api_key"] = model_api_key
            stagehand_api_url = os.environ.get("STAGEHAND_API_URL") or None
            if stagehand_api_url:
                create_options["api_url"] = stagehand_api_url
            stagehand = await Stagehand.create(browser=browser, **create_options)
            return cls(
                stagehand,
                browser,
                browserbase_api_key=browserbase_api_key,
                browserbase_api_url=browserbase_api_url,
            )
        except BaseException:
            await browser.close()
            raise

    async def close(self) -> None:
        try:
            try:
                await self.stagehand.close()
            finally:
                await self.browser.close()
        finally:
            if self.session_id is not None:
                # keep-alive sessions outlive close(); release explicitly
                # so expired registry entries don't strand paid sessions.
                await _release_session(
                    self.browserbase_api_key, self.browserbase_api_url, self.session_id
                )

    async def active_page(self) -> Page:
        pages = await self.browser.context.pages()
        if not pages:
            return await self.browser.context.new_page()
        return pages[-1]

    async def execute(
        self,
        *,
        code: str | None,
        actions: list[Action] | None,
    ) -> object:
        if (code is None) == (actions is None):
            raise ValueError("run requires exactly one of code or actions")
        async with self.lock:
            page = await self.active_page()
            if code is not None:
                return await self._execute_code(page, code)
            return await self._execute_actions(page, actions or [])

    async def snapshot(self, include_iframes: bool) -> str:
        async with self.lock:
            page = await self.active_page()
            snapshot = await page.snapshot(include_iframes=include_iframes)
            self.snapshots[page.page_id] = _Snapshot(
                url=await page.url(),
                xpath_by_id=dict(snapshot.xpath_map),
            )
            return snapshot.formatted_tree

    async def screenshot(
        self,
        *,
        full_page: bool | None,
        image_type: Literal["png", "jpeg"],
        quality: int | None,
    ) -> tuple[bytes, str]:
        async with self.lock:
            page = await self.active_page()
            image = await page.screenshot(
                full_page=full_page,
                type=image_type,
                quality=quality,
            )
            return image, "image/jpeg" if image_type == "jpeg" else "image/png"

    async def _execute_code(self, page: Page, code: str) -> object:
        facade = _facade_source()
        source = f"""async (batchStagehand, input) => {{
  "use strict";
  const __stagehandCompatIdentity = (target) => target;
  for (let index = 0; index <= 32; index += 1) {{
    globalThis[index === 0 ? "__name" : "__name" + index] = __stagehandCompatIdentity;
  }}
  {facade}
  const runtime = await createPlaywrightCompatRuntime(batchStagehand);
  const page = runtime.page;
  const context = runtime.context;
  const browser = runtime.browser;
  return await (async () => {{
    {code}
  }})();
}}"""
        return await self.stagehand.experimental_batch(
            source,
            {},
            page=page,
            timeout=_positive_int_env("STAGEHAND_RUN_TIMEOUT_MS", 60_000),
        )

    async def _execute_actions(self, page: Page, actions: list[Action]) -> object:
        if not actions:
            raise ValueError("run actions must not be empty")
        snapshot = self.snapshots.get(page.page_id)
        if snapshot is None or snapshot.url != await page.url():
            raise ValueError("No current hydrated snapshot exists; call snapshot again")
        hydrated: list[dict[str, object]] = []
        for action in actions:
            value = action.model_dump(exclude_none=True)
            xpath = snapshot.xpath_by_id.get(action.id)
            if xpath is None:
                raise ValueError(f'Snapshot ID "{action.id}" is stale or not actionable')
            value["selector"] = f"xpath={_TEXT_NODE_SUFFIX.sub('', xpath)}"
            hydrated.append(value)
        result = await self.stagehand.experimental_batch(
            _ACTION_SOURCE,
            {"actions": hydrated},
            page=page,
            timeout=_positive_int_env("STAGEHAND_RUN_TIMEOUT_MS", 60_000),
        )
        return {"result": result, "url": await page.url()}


@dataclass
class _RegistryEntry:
    browser: _BrowserRuntime
    last_used: float


class _BrowserRegistry:
    def __init__(self) -> None:
        self.entries: dict[str, _RegistryEntry] = {}
        self.lock = asyncio.Lock()

    async def get(self, thread_id: str) -> _BrowserRuntime:
        now = time.monotonic()
        async with self.lock:
            await self._close_expired(now)
            entry = self.entries.get(thread_id)
            if entry is not None and not await _runtime_alive(entry.browser):
                # The handle is dead (socket drop, resume elsewhere) — closed
                # never flips on its own, so probe liveness. The keep-alive
                # session may still be running; reconnect by id before falling
                # back to a fresh launch.
                stale = self.entries.pop(thread_id)
                entry = None
                if stale.browser.session_id is not None:
                    try:
                        revived = await _BrowserRuntime.start(session_id=stale.browser.session_id)
                        entry = _RegistryEntry(revived, now)
                        self.entries[thread_id] = entry
                    except Exception:
                        # Reconnect failed — release the old keep-alive session
                        # best-effort so the replacement doesn't strand it.
                        if stale.browser.browserbase_api_key:
                            await _release_session(
                                stale.browser.browserbase_api_key,
                                stale.browser.browserbase_api_url,
                                stale.browser.session_id,
                            )
                        entry = None
            if entry is None:
                entry = _RegistryEntry(await _BrowserRuntime.start(), now)
                self.entries[thread_id] = entry
            entry.last_used = now
            return entry.browser

    async def _close_expired(self, now: float) -> None:
        ttl = _positive_int_env("STAGEHAND_SESSION_TTL_SECONDS", 1800)
        expired = [
            thread_id for thread_id, entry in self.entries.items() if now - entry.last_used >= ttl
        ]
        for thread_id in expired:
            # A failing close must not surface in the unrelated tool call that
            # triggered the sweep, nor abort cleanup of other expired entries.
            try:
                await self.entries.pop(thread_id).browser.close()
            except Exception:
                pass


async def _runtime_alive(runtime: _BrowserRuntime) -> bool:
    if runtime.browser.closed:
        return False
    try:
        await runtime.browser.context.pages()
    except Exception:
        return False
    return True


_REGISTRY = _BrowserRegistry()


def _thread_id(runtime: ToolRuntime) -> str:
    info = runtime.execution_info
    identifier = info.thread_id or info.run_id
    if identifier is None:
        raise RuntimeError("Stagehand tools require a managed thread or run ID")
    return str(identifier)


@tool
async def run(
    code: str | None = None,
    actions: list[Action] | None = None,
    *,
    runtime: ToolRuntime,
) -> str:
    """Execute JavaScript or snapshot-ID actions in the persistent Stagehand browser.

    Provide exactly one of code or actions. JavaScript receives Playwright-shaped page, context,
    and browser objects. Actions use IDs from the latest snapshot.
    """
    browser = await _REGISTRY.get(_thread_id(runtime))
    try:
        result = await browser.execute(code=code, actions=actions)
    except Exception as error:
        raise RuntimeError(_sanitize_error(str(error))) from None
    return json.dumps(result, default=str, separators=(",", ":"))


@tool
async def snapshot(includeIframes: bool = True, *, runtime: ToolRuntime) -> str:
    """Capture the active page tree and hydrate IDs for subsequent run actions."""
    browser = await _REGISTRY.get(_thread_id(runtime))
    try:
        return await browser.snapshot(includeIframes)
    except Exception as error:
        raise RuntimeError(_sanitize_error(str(error))) from None


@tool
async def screenshot(
    fullPage: bool | None = None,
    type: Literal["png", "jpeg"] = "png",
    quality: int | None = None,
    *,
    runtime: ToolRuntime,
) -> list[dict[str, object]]:
    """Capture a screenshot of the active page."""
    if quality is not None and not 0 <= quality <= 100:
        raise ValueError("quality must be between 0 and 100")
    if type == "png" and quality is not None:
        raise ValueError("quality is only valid for jpeg screenshots")
    browser = await _REGISTRY.get(_thread_id(runtime))
    try:
        image, mime_type = await browser.screenshot(
            full_page=fullPage,
            image_type=type,
            quality=quality,
        )
    except Exception as error:
        raise RuntimeError(_sanitize_error(str(error))) from None
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {
            "type": "image",
            "url": f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}",
        },
    ]
