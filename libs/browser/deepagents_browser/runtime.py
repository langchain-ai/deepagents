"""Lazy, bounded Playwright runtime management."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_browser.errors import BrowserError, BrowserErrorCode, BrowserRuntimeError
from deepagents_browser.network import NetworkPolicy, RouteLike

if TYPE_CHECKING:
    from deepagents_browser.schemas import BrowserAction

RuntimeFactory = Callable[[], Awaitable[tuple[Any, Any]]]
_MAX_THREAD_ID_CHARS = 512
_ACTIONABLE_SELECTOR = (
    "a,button,input,textarea,select,summary,[role=button],[role=link],"
    "[role=checkbox],[role=menuitem],[role=option],[role=radio],[role=switch],[tabindex]"
)
_STALE_ELEMENT_REFERENCE = BrowserErrorCode.STALE_ELEMENT_REFERENCE
_ELEMENT_IDENTITY_UNAVAILABLE = BrowserErrorCode.ELEMENT_IDENTITY_UNAVAILABLE
_ELEMENT_CHANGED = BrowserErrorCode.ELEMENT_CHANGED
_SENSITIVE_CONTROL_BLOCKED = BrowserErrorCode.SENSITIVE_CONTROL_BLOCKED
_NAVIGATION_INVALIDATED_REFERENCE = BrowserErrorCode.NAVIGATION_INVALIDATED_REFERENCE
_SENSITIVE_AUTOCOMPLETE_TOKENS = frozenset(
    {
        "cc-additional-name",
        "cc-csc",
        "cc-exp",
        "cc-exp-month",
        "cc-exp-year",
        "cc-family-name",
        "cc-given-name",
        "cc-name",
        "cc-number",
        "cc-type",
        "current-password",
        "new-password",
        "one-time-code",
        "transaction-amount",
        "transaction-currency",
    }
)
_PAYMENT_FIELD_MARKERS = (
    "cardexpiration",
    "cardexpiry",
    "cardholder",
    "cardnumber",
    "cardsecuritycode",
    "ccnumber",
    "creditcard",
    "cvv",
    "cvc",
    "debitcard",
    "expirationdate",
    "expirydate",
    "nameoncard",
)


@dataclass(frozen=True, slots=True)
class BrowserLimits:
    """Resource bounds enforced by the browser runtime.

    Args:
        max_contexts: Maximum concurrent thread-isolated contexts.
        max_tabs_per_context: Maximum tabs in one thread context.
        max_snapshot_nodes: Maximum actionable nodes inspected per snapshot.
        max_snapshot_chars: Maximum serialized snapshot characters returned.
        max_screenshot_bytes: Maximum raw PNG bytes accepted.
        max_screenshot_output_chars: Maximum serialized screenshot response characters.
        max_requests_per_context: Maximum intercepted requests over a context lifetime.
        action_timeout_ms: Playwright timeout applied to user-driven operations.
        startup_timeout_seconds: Maximum lazy browser startup time.
    """

    max_contexts: int = 8
    max_tabs_per_context: int = 4
    max_snapshot_nodes: int = 200
    max_snapshot_chars: int = 32_000
    max_screenshot_bytes: int = 2_000_000
    max_screenshot_output_chars: int = 3_000_000
    max_requests_per_context: int = 1_000
    action_timeout_ms: int = 15_000
    startup_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Reject invalid or effectively unbounded resource limits."""
        values_and_hard_caps = {
            "max_contexts": (self.max_contexts, 64),
            "max_tabs_per_context": (self.max_tabs_per_context, 16),
            "max_snapshot_nodes": (self.max_snapshot_nodes, 1_000),
            "max_snapshot_chars": (self.max_snapshot_chars, 1_000_000),
            "max_screenshot_bytes": (self.max_screenshot_bytes, 20_000_000),
            "max_screenshot_output_chars": (self.max_screenshot_output_chars, 30_000_000),
            "max_requests_per_context": (self.max_requests_per_context, 10_000),
            "action_timeout_ms": (self.action_timeout_ms, 120_000),
        }
        for name, (value, hard_cap) in values_and_hard_caps.items():
            if value < 1 or value > hard_cap:
                msg = f"{name} must be between 1 and {hard_cap}"
                raise ValueError(msg)
        if not 0 < self.startup_timeout_seconds <= 120:  # noqa: PLR2004
            msg = "startup_timeout_seconds must be greater than 0 and at most 120"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _ElementDescription:
    role: str | None
    aria_label: str | None
    name: str | None
    element_type: str | None
    href: str | None
    autocomplete: str | None
    element_id: str | None
    placeholder: str | None
    inputmode: str | None
    text: str

    @property
    def is_sensitive(self) -> bool:
        """Return whether this is a password or payment-related control."""
        if (self.element_type or "").strip().lower() == "password":
            return True
        autocomplete_tokens = {token.lower() for token in (self.autocomplete or "").split()}
        if autocomplete_tokens & _SENSITIVE_AUTOCOMPLETE_TOKENS:
            return True
        metadata_fields = (
            "".join(character for character in (value or "").lower() if character.isalnum())
            for value in (self.aria_label, self.name, self.element_id, self.placeholder)
        )
        return any(
            marker in metadata for metadata in metadata_fields for marker in _PAYMENT_FIELD_MARKERS
        )


@dataclass(slots=True)
class _ElementReference:
    page: Any
    handle: Any
    generation: int
    description: _ElementDescription


@dataclass(slots=True)
class BrowserSession:
    """One isolated browser context owned by a LangGraph thread."""

    context: Any
    policy: NetworkPolicy
    limits: BrowserLimits
    _active_page: Any | None = None
    _page_refs: dict[str, Any] = field(default_factory=dict)
    _observed_pages: dict[int, Any] = field(default_factory=dict)
    _element_refs: dict[str, _ElementReference] = field(default_factory=dict)
    _invalidated_element_codes: dict[str, BrowserErrorCode] = field(default_factory=dict)
    _generation: int = 0
    _request_count: int = 0
    _closed: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _route_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def initialize(self) -> None:
        """Install request interception before any page is used."""

        async def handle_route(route: RouteLike) -> None:
            await self._handle_route(route)

        await self.context.route("**/*", handle_route)
        pages = list(self.context.pages)
        if pages:
            self._active_page = pages[0]
            self._register_page(pages[0])

    async def _handle_route(self, route: RouteLike) -> None:
        async with self._route_lock:
            self._request_count += 1
            over_limit = self._request_count > self.limits.max_requests_per_context
        if over_limit:
            await route.abort("blockedbyclient")
            return
        await self.policy.handle_route(route)

    def _register_page(self, page: Any) -> str:  # noqa: ANN401
        page_id = id(page)
        if self._observed_pages.get(page_id) is not page:
            page.on(
                "framenavigated",
                lambda frame, observed_page=page: self._handle_frame_navigated(
                    observed_page, frame
                ),
            )
            self._observed_pages[page_id] = page
        for page_ref, existing in self._page_refs.items():
            if existing is page:
                return page_ref
        page_ref = f"p_{secrets.token_urlsafe(18)}"
        self._page_refs[page_ref] = page
        return page_ref

    def _handle_frame_navigated(self, page: Any, frame: Any) -> None:  # noqa: ANN401
        if frame is page.main_frame:
            self._invalidate_elements(code=_NAVIGATION_INVALIDATED_REFERENCE)

    async def _page(
        self,
        page_ref: str | None = None,
        *,
        create: bool = True,
    ) -> Any:  # noqa: ANN401
        if self._closed:
            msg = "Browser session is closed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
        if page_ref is not None:
            try:
                page = self._page_refs[page_ref]
            except KeyError as exc:
                msg = "Unknown or stale page reference"
                raise BrowserRuntimeError(
                    msg,
                    code=BrowserErrorCode.STALE_PAGE_REFERENCE,
                ) from exc
            if page not in self.context.pages:
                self._page_refs.pop(page_ref, None)
                msg = "Unknown or stale page reference"
                raise BrowserRuntimeError(
                    msg,
                    code=BrowserErrorCode.STALE_PAGE_REFERENCE,
                )
            self._active_page = page
            return page
        if self._active_page in self.context.pages:
            return self._active_page
        pages = list(self.context.pages)
        if pages:
            self._active_page = pages[0]
            self._register_page(pages[0])
            return pages[0]
        if not create:
            msg = "No browser tab is open"
            raise BrowserRuntimeError(msg)
        return await self._new_page()

    async def _new_page(self) -> Any:  # noqa: ANN401
        if len(self.context.pages) >= self.limits.max_tabs_per_context:
            msg = "Tab limit reached"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.TAB_LIMIT_REACHED)
        page = await self.context.new_page()
        page.set_default_timeout(self.limits.action_timeout_ms)
        self._active_page = page
        self._register_page(page)
        self._invalidate_elements()
        return page

    def _invalidate_elements(
        self,
        *,
        code: BrowserErrorCode = _STALE_ELEMENT_REFERENCE,
    ) -> None:
        self._generation += 1
        if self._element_refs:
            self._invalidated_element_codes = dict.fromkeys(self._element_refs, code)
            self._element_refs.clear()

    async def _enforce_page_limit(self) -> None:
        pages = list(self.context.pages)
        if len(pages) <= self.limits.max_tabs_per_context:
            return
        for page in pages[self.limits.max_tabs_per_context :]:
            await page.close()
        live_pages = set(self.context.pages)
        self._page_refs = {
            page_ref: page for page_ref, page in self._page_refs.items() if page in live_pages
        }
        self._observed_pages = {
            page_id: page for page_id, page in self._observed_pages.items() if page in live_pages
        }
        self._invalidate_elements()
        msg = "Page attempted to exceed the configured tab limit"
        raise BrowserRuntimeError(msg, code=BrowserErrorCode.TAB_LIMIT_REACHED)

    async def navigate(self, url: str, page_ref: str | None = None) -> str:
        """Navigate a selected tab after validating the destination."""
        await self.policy.validate_url(url)
        async with self._lock:
            page = await self._page(page_ref)
            await page.goto(url, wait_until="domcontentloaded")
            await self._enforce_page_limit()
            self._invalidate_elements()
            return json.dumps(
                {
                    "page_ref": self._register_page(page),
                    "title": (await page.title())[:300],
                    "url": page.url[:8_192],
                },
                separators=(",", ":"),
            )

    @staticmethod
    async def _describe_element(handle: Any) -> _ElementDescription:  # noqa: ANN401
        async def bounded_attribute(name: str, limit: int = 300) -> str | None:
            value = await handle.get_attribute(name)
            return None if value is None else value[:limit]

        return _ElementDescription(
            role=await bounded_attribute("role"),
            aria_label=await bounded_attribute("aria-label"),
            name=await bounded_attribute("name"),
            element_type=await bounded_attribute("type"),
            href=await bounded_attribute("href", 2_048),
            autocomplete=await bounded_attribute("autocomplete"),
            element_id=await bounded_attribute("id"),
            placeholder=await bounded_attribute("placeholder"),
            inputmode=await bounded_attribute("inputmode"),
            text=(await handle.text_content() or "").strip()[:300],
        )

    async def snapshot(self, page_ref: str | None = None) -> str:
        """Return a bounded snapshot with opaque, generation-scoped references."""
        async with self._lock:
            await self._enforce_page_limit()
            page = await self._page(page_ref)
            self._invalidate_elements()
            generation = self._generation
            locator = page.locator(_ACTIONABLE_SELECTOR)
            total = await locator.count()
            count = min(total, self.limits.max_snapshot_nodes)
            nodes: list[dict[str, str | None]] = []
            for index in range(count):
                item = locator.nth(index)
                if not await item.is_visible():
                    continue
                handle = await item.element_handle()
                if handle is None or not await handle.is_visible():
                    continue
                description = await self._describe_element(handle)
                if self._generation != generation:
                    self._invalidate_elements(code=BrowserErrorCode.PAGE_NAVIGATED)
                    msg = "Page navigated while the snapshot was being captured"
                    raise BrowserRuntimeError(msg, code=BrowserErrorCode.PAGE_NAVIGATED)
                if description.is_sensitive:
                    continue
                node_ref = f"e_{generation}_{secrets.token_urlsafe(18)}"
                candidate = {
                    "ref": node_ref,
                    "role": description.role,
                    "label": description.aria_label or description.name,
                    "type": description.element_type,
                    "href": description.href,
                    "text": description.text,
                }
                prospective = json.dumps(
                    {"page_ref": self._register_page(page), "nodes": [*nodes, candidate]},
                    separators=(",", ":"),
                )
                if len(prospective) > self.limits.max_snapshot_chars:
                    break
                self._element_refs[node_ref] = _ElementReference(
                    page=page,
                    handle=handle,
                    generation=generation,
                    description=description,
                )
                nodes.append(candidate)
            if self._generation != generation:
                self._invalidate_elements(code=BrowserErrorCode.PAGE_NAVIGATED)
                msg = "Page navigated while the snapshot was being captured"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.PAGE_NAVIGATED)
            result = json.dumps(
                {
                    "page_ref": self._register_page(page),
                    "generation": generation,
                    "nodes": nodes,
                    "truncated": count < total or len(nodes) < count,
                },
                separators=(",", ":"),
            )
            if len(result) > self.limits.max_snapshot_chars:
                msg = "Snapshot metadata exceeds configured output limit"
                raise BrowserRuntimeError(msg)
            return result

    async def _resolve_action_reference(self, element_ref: str) -> _ElementReference:
        """Resolve and revalidate one identity-pinned actionable element."""
        try:
            reference = self._element_refs[element_ref]
        except KeyError as exc:
            code = self._invalidated_element_codes.get(element_ref, _STALE_ELEMENT_REFERENCE)
            if code == _NAVIGATION_INVALIDATED_REFERENCE:
                msg = "Element reference was invalidated by top-level navigation"
            else:
                msg = "Unknown or stale element reference; take a new snapshot"
            raise BrowserRuntimeError(msg, code=code) from exc
        if reference.generation != self._generation:
            msg = "Unknown or stale element reference; take a new snapshot"
            raise BrowserRuntimeError(msg, code=_STALE_ELEMENT_REFERENCE)
        if reference.page not in self.context.pages:
            msg = "Element identity is no longer available"
            raise BrowserRuntimeError(msg, code=_ELEMENT_IDENTITY_UNAVAILABLE)
        try:
            is_visible = await reference.handle.is_visible()
            current_description = await self._describe_element(reference.handle)
        except Exception as exc:
            msg = "Element identity is no longer available"
            raise BrowserRuntimeError(msg, code=_ELEMENT_IDENTITY_UNAVAILABLE) from exc
        if not is_visible:
            msg = "Element identity is detached or no longer visible"
            raise BrowserRuntimeError(msg, code=_ELEMENT_IDENTITY_UNAVAILABLE)
        if current_description.is_sensitive:
            msg = "Actions on password or payment controls are blocked"
            raise BrowserRuntimeError(msg, code=_SENSITIVE_CONTROL_BLOCKED)
        if current_description != reference.description:
            msg = "Element changed since the snapshot; take a new snapshot"
            raise BrowserRuntimeError(msg, code=_ELEMENT_CHANGED)
        return reference

    async def act(self, action: BrowserAction) -> str:
        """Execute one allowlisted action against a fresh opaque reference."""
        async with self._lock:
            reference = await self._resolve_action_reference(action.ref)
            try:
                match action.kind:
                    case "click":
                        await reference.handle.click()
                    case "type":
                        await reference.handle.fill(action.text)
                    case "press":
                        await reference.handle.press(action.key)
                    case "select":
                        await reference.handle.select_option(action.value)
            except Exception as exc:
                msg = "Browser action failed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.ACTION_FAILED) from exc
            await self._enforce_page_limit()
            self._invalidate_elements()
            return json.dumps({"ok": True, "page_ref": self._register_page(reference.page)})

    async def screenshot(self, page_ref: str | None = None) -> str:
        """Return a bounded base64-encoded PNG of the fixed viewport."""
        async with self._lock:
            await self._enforce_page_limit()
            page = await self._page(page_ref)
            payload = await page.screenshot(type="png", full_page=False)
            page_ref_value = self._register_page(page)
            encoded_chars = 4 * ((len(payload) + 2) // 3)
            empty_response = json.dumps(
                {
                    "page_ref": page_ref_value,
                    "media_type": "image/png",
                    "base64": "",
                },
                separators=(",", ":"),
            )
            projected_chars = len(empty_response) + encoded_chars
            if (
                len(payload) > self.limits.max_screenshot_bytes
                or projected_chars > self.limits.max_screenshot_output_chars
            ):
                msg = "Screenshot exceeds configured output limit"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.SCREENSHOT_TOO_LARGE)
            result = json.dumps(
                {
                    "page_ref": page_ref_value,
                    "media_type": "image/png",
                    "base64": base64.b64encode(payload).decode("ascii"),
                },
                separators=(",", ":"),
            )
            if len(result) > self.limits.max_screenshot_output_chars:
                msg = "Screenshot exceeds configured output limit"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.SCREENSHOT_TOO_LARGE)
            return result

    async def tabs(self, operation: str, page_ref: str | None = None) -> str:
        """List, create, select, or close tabs without exposing raw page objects."""
        async with self._lock:
            await self._enforce_page_limit()
            if operation == "new":
                await self._new_page()
            elif operation == "select":
                if page_ref is None:
                    msg = "page_ref is required when selecting a tab"
                    raise BrowserRuntimeError(msg)
                await self._page(page_ref, create=False)
            elif operation == "close":
                page = await self._page(page_ref, create=False)
                await page.close()
                stale = [ref for ref, candidate in self._page_refs.items() if candidate is page]
                for ref in stale:
                    self._page_refs.pop(ref, None)
                self._observed_pages.pop(id(page), None)
                self._active_page = None
                self._invalidate_elements()
            elif operation != "list":
                msg = "Unsupported tab operation"
                raise BrowserRuntimeError(msg)
            tabs = [
                {
                    "page_ref": self._register_page(page),
                    "active": page is self._active_page,
                    "title": (await page.title())[:300],
                    "url": page.url[:8_192],
                }
                for page in list(self.context.pages)[: self.limits.max_tabs_per_context]
            ]
            return json.dumps({"tabs": tabs}, separators=(",", ":"))

    async def aclose(self) -> None:
        """Close this context once; repeated calls are harmless."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._element_refs.clear()
            self._invalidated_element_codes.clear()
            self._page_refs.clear()
            self._observed_pages.clear()
            await self.context.close()


def _require_provisioned_chromium(playwright: Any) -> None:  # noqa: ANN401
    """Fail before launch when Playwright's Chromium executable is absent."""
    executable_path = getattr(playwright.chromium, "executable_path", None)
    if not isinstance(executable_path, str) or not Path(executable_path).is_file():
        msg = (
            "Chromium is not provisioned; run `dcode --install browser` before using browser tools"
        )
        raise BrowserRuntimeError(
            msg,
            code=BrowserErrorCode.BROWSER_NOT_PROVISIONED,
        )


async def _default_runtime_factory() -> tuple[Any, Any]:
    """Import and launch Playwright only when browser access is first used."""
    from playwright.async_api import async_playwright  # noqa: PLC0415

    playwright = await async_playwright().start()
    try:
        _require_provisioned_chromium(playwright)
        browser = await playwright.chromium.launch(headless=False)
    except BaseException:
        await playwright.stop()
        raise
    return playwright, browser


class BrowserRuntimeManager:
    """Lazily own one browser and bounded, thread-isolated contexts.

    Args:
        limits: Browser resource limits.
        network_policy: Policy used for pre-navigation and request validation.
        runtime_factory: Internal dependency injection point for deterministic tests.
    """

    def __init__(
        self,
        *,
        limits: BrowserLimits | None = None,
        network_policy: NetworkPolicy | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        """Configure the manager without importing, starting, or launching Playwright."""
        self.limits = limits or BrowserLimits()
        self.network_policy = network_policy or NetworkPolicy()
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._sessions: dict[str, BrowserSession] = {}
        self._startup_lock = asyncio.Lock()
        self._sessions_lock = asyncio.Lock()
        self._closed = False

    async def validate_url(self, url: str) -> None:
        """Validate a URL without creating or accessing a browser runtime."""
        await self.network_policy.validate_url(url)

    @staticmethod
    async def _validate_runtime_factory_result(result: object) -> tuple[Any, Any]:
        if not isinstance(result, tuple) or len(result) != 2:  # noqa: PLR2004
            msg = "Browser runtime factory must return (playwright, browser)"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.INVALID_RUNTIME_FACTORY)
        playwright: Any = result[0]
        browser: Any = result[1]
        if not callable(getattr(playwright, "stop", None)):
            msg = "Browser runtime factory returned an invalid Playwright runtime"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.INVALID_RUNTIME_FACTORY)
        if not callable(getattr(browser, "new_context", None)) or not callable(
            getattr(browser, "close", None)
        ):
            try:
                await playwright.stop()
            except Exception as exc:
                msg = "Invalid browser runtime could not be cleaned up"
                raise BrowserRuntimeError(
                    msg,
                    code=BrowserErrorCode.STARTUP_FAILED,
                ) from exc
            msg = "Browser runtime factory returned an invalid browser"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.INVALID_RUNTIME_FACTORY)
        return playwright, browser

    async def _ensure_browser(self) -> Any:  # noqa: ANN401
        if self._closed:
            msg = "Browser runtime manager is closed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
        if self._browser is not None:
            return self._browser
        async with self._startup_lock:
            if self._closed:
                msg = "Browser runtime manager is closed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
            if self._browser is None:
                try:
                    async with asyncio.timeout(self.limits.startup_timeout_seconds):
                        result = await self._runtime_factory()
                    playwright, browser = await self._validate_runtime_factory_result(result)
                except TimeoutError as exc:
                    msg = "Browser runtime startup timed out"
                    raise BrowserRuntimeError(
                        msg,
                        code=BrowserErrorCode.STARTUP_TIMEOUT,
                    ) from exc
                except BrowserError:
                    raise
                except Exception as exc:
                    msg = "Browser runtime startup failed"
                    raise BrowserRuntimeError(
                        msg,
                        code=BrowserErrorCode.STARTUP_FAILED,
                    ) from exc
                self._playwright = playwright
                self._browser = browser
            return self._browser

    async def get_session(self, thread_id: str) -> BrowserSession:
        """Return or create the isolated context for one stable thread ID."""
        if not thread_id or len(thread_id) > _MAX_THREAD_ID_CHARS:
            msg = "Thread ID must contain between 1 and 512 characters"
            raise BrowserRuntimeError(msg)
        async with self._sessions_lock:
            if self._closed:
                msg = "Browser runtime manager is closed"
                raise BrowserRuntimeError(msg)
            existing = self._sessions.get(thread_id)
            if existing is not None:
                return existing
            if len(self._sessions) >= self.limits.max_contexts:
                msg = "Browser context limit reached"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.CONTEXT_LIMIT_REACHED)
            browser = await self._ensure_browser()
            context = await browser.new_context(
                accept_downloads=False,
                service_workers="block",
                viewport={"width": 1280, "height": 720},
            )
            session = BrowserSession(
                context=context, policy=self.network_policy, limits=self.limits
            )
            try:
                await session.initialize()
            except BaseException:
                with contextlib.suppress(Exception):
                    await context.close()
                raise
            self._sessions[thread_id] = session
            return session

    async def aclose(self) -> None:
        """Idempotently close every context, browser, and Playwright driver."""
        async with self._sessions_lock:
            if self._closed:
                return
            self._closed = True
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                with contextlib.suppress(Exception):
                    await session.aclose()
            async with self._startup_lock:
                browser, self._browser = self._browser, None
                playwright, self._playwright = self._playwright, None
                if browser is not None:
                    with contextlib.suppress(Exception):
                        await browser.close()
                if playwright is not None:
                    with contextlib.suppress(Exception):
                        await playwright.stop()
