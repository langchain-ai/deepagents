"""Lazy, bounded Playwright runtime management."""

from __future__ import annotations

import asyncio
import json
import math
import secrets
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
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
_PAGE_SCROLL_SCRIPT = """({direction, distance}) => {
  const before = {x: Math.round(window.scrollX), y: Math.round(window.scrollY)};
  const horizontal = direction === "left" || direction === "right";
  const viewport = horizontal ? window.innerWidth : window.innerHeight;
  const magnitude = Math.round(viewport * (distance === "page" ? 1 : 0.5));
  const sign = direction === "up" || direction === "left" ? -1 : 1;
  window.scrollBy(horizontal ? sign * magnitude : 0, horizontal ? 0 : sign * magnitude);
  const after = {x: Math.round(window.scrollX), y: Math.round(window.scrollY)};
  return {before, after};
}"""
_ELEMENT_SEMANTICS_SCRIPT = """element => {
  const bounded = value => typeof value === "string" ? value.trim().slice(0, 300) : null;
  const tag = element.tagName ? element.tagName.toLowerCase() : null;
  const type = (element.getAttribute("type") || "").toLowerCase();
  const explicitRole = bounded(element.getAttribute("role"));
  const implicitRoles = {
    a: element.hasAttribute("href") ? "link" : null,
    button: "button",
    select: element.multiple ? "listbox" : "combobox",
    textarea: "textbox",
    summary: "button"
  };
  let role = explicitRole || implicitRoles[tag] || null;
  if (!role && tag === "input") {
    role = {button: "button", checkbox: "checkbox", radio: "radio", range: "slider",
      search: "searchbox", submit: "button"}[type] || "textbox";
  }
  const labelledBy = (element.getAttribute("aria-labelledby") || "").split(/\\s+/)
    .filter(Boolean).map(id => document.getElementById(id)?.textContent || "").join(" ");
  const labelText = element.labels ? Array.from(element.labels)
    .map(label => label.textContent || "").join(" ") : "";
  const name = bounded(element.getAttribute("aria-label")) || bounded(labelledBy) ||
    bounded(labelText) || bounded(element.getAttribute("alt")) ||
    bounded(element.getAttribute("title")) || bounded(element.getAttribute("placeholder")) ||
    bounded(element.getAttribute("name")) || bounded(element.textContent) || null;
  const ariaBoolean = attribute => {
    const value = element.getAttribute(attribute);
    return value === null ? null : value.toLowerCase() === "true";
  };
  return {
    tag,
    role,
    accessibleName: name,
    disabled: Boolean(element.disabled) || ariaBoolean("aria-disabled") === true,
    checked: "checked" in element ? Boolean(element.checked) : ariaBoolean("aria-checked"),
    selected: "selected" in element ? Boolean(element.selected) : ariaBoolean("aria-selected"),
    expanded: ariaBoolean("aria-expanded"),
    readonly: Boolean(element.readOnly) || ariaBoolean("aria-readonly") === true,
    required: Boolean(element.required) || ariaBoolean("aria-required") === true,
    focused: element.ownerDocument?.activeElement === element,
    editable: !element.disabled && !element.readOnly &&
      (element.isContentEditable || tag === "textarea" ||
       (tag === "input" && !["button", "checkbox", "file", "hidden", "image", "radio",
         "range", "reset", "submit"].includes(type)))
  };
}"""


async def _element_semantics(
    handle: Any,  # noqa: ANN401
    *,
    timeout_ms: int,
) -> Mapping[str, object]:
    """Collect optional package semantics within a strict wall-clock bound."""
    evaluate = getattr(handle, "evaluate", None)
    if not callable(evaluate):
        return {}
    try:
        async with asyncio.timeout(timeout_ms / 1_000):
            value = await evaluate(_ELEMENT_SEMANTICS_SCRIPT)
    except Exception:  # noqa: BLE001  # optional metadata must not fail or stall snapshots
        return {}
    return value if isinstance(value, Mapping) else {}


def _semantic_text(
    semantics: Mapping[str, object],
    name: str,
    fallback: str | None = None,
) -> str | None:
    value = semantics.get(name)
    return value[:300] if isinstance(value, str) else fallback


def _semantic_bool(
    semantics: Mapping[str, object],
    name: str,
    *,
    fallback: bool,
) -> bool:
    value = semantics.get(name)
    return value if isinstance(value, bool) else fallback


def _semantic_optional_bool(
    semantics: Mapping[str, object],
    name: str,
    *,
    fallback: bool | None,
) -> bool | None:
    value = semantics.get(name)
    return value if isinstance(value, bool) else fallback


def _scroll_position(value: object) -> dict[str, int]:
    """Validate one browser-produced, JSON-safe scroll position."""
    if not isinstance(value, Mapping):
        msg = "Browser returned invalid scroll diagnostics"
        raise TypeError(msg)
    position: dict[str, int] = {}
    for axis in ("x", "y"):
        coordinate = value.get(axis)
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            or abs(coordinate) > 2**53 - 1
        ):
            msg = "Browser returned invalid scroll diagnostics"
            raise TypeError(msg)
        position[axis] = round(coordinate)
    return position


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
        semantic_timeout_ms: Wall-clock timeout for optional element semantic evaluation.
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
    semantic_timeout_ms: int = 1_000
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
            "semantic_timeout_ms": (self.semantic_timeout_ms, 10_000),
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
    tag: str | None
    accessible_name: str | None
    disabled: bool
    checked: bool | None
    selected: bool | None
    expanded: bool | None
    readonly: bool
    required: bool
    focused: bool
    editable: bool

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
            for value in (
                self.aria_label,
                self.accessible_name,
                self.name,
                self.element_id,
                self.placeholder,
            )
        )
        return any(
            marker in metadata for metadata in metadata_fields for marker in _PAYMENT_FIELD_MARKERS
        )

    @property
    def identity(self) -> tuple[str | None, ...]:
        """Return stable semantic fields used to detect exact-handle retargeting."""
        return (
            self.tag,
            self.role,
            self.aria_label,
            self.name,
            self.element_type,
            self.href,
            self.autocomplete,
            self.element_id,
            self.placeholder,
            self.inputmode,
            self.text,
            self.accessible_name,
        )

    def diagnostics(self) -> dict[str, str | bool | None]:
        """Return bounded action diagnostics without selectors or entered values."""
        return {
            "tag": self.tag,
            "role": self.role,
            "name": self.accessible_name,
            "disabled": self.disabled,
            "checked": self.checked,
            "selected": self.selected,
            "expanded": self.expanded,
            "readonly": self.readonly,
            "required": self.required,
            "focused": self.focused,
            "editable": self.editable,
        }


@dataclass(frozen=True, slots=True)
class ScreenshotResult:
    """Bounded fixed-viewport screenshot data and safe textual metadata."""

    page_ref: str
    media_type: str
    data: bytes

    def metadata(self) -> dict[str, str | int]:
        """Return metadata that intentionally excludes encoded image data."""
        return {
            "page_ref": self.page_ref,
            "media_type": self.media_type,
            "bytes": len(self.data),
        }

    def projected_output_chars(self) -> int:
        """Project serialized content plus artifact size before base64 allocation."""
        metadata = self.metadata()
        image = {"type": "image", "base64": "", "mime_type": self.media_type}
        content = [
            {"type": "text", "text": json.dumps(metadata, separators=(",", ":"))},
            image,
        ]
        artifact = {**metadata, "image": image}
        encoded_chars = 4 * ((len(self.data) + 2) // 3)
        return (
            len(json.dumps(content, separators=(",", ":")))
            + len(json.dumps(artifact, separators=(",", ":")))
            + 2 * encoded_chars
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

    async def _describe_element(self, handle: Any) -> _ElementDescription:  # noqa: ANN401
        async def bounded_attribute(name: str, limit: int = 300) -> str | None:
            value = await handle.get_attribute(name)
            return None if value is None else value[:limit]

        attributes = {
            name: await bounded_attribute(name, 2_048 if name == "href" else 300)
            for name in (
                "role",
                "aria-label",
                "name",
                "type",
                "href",
                "autocomplete",
                "id",
                "placeholder",
                "inputmode",
                "disabled",
                "checked",
                "selected",
                "aria-expanded",
                "readonly",
                "required",
            )
        }
        semantics = await _element_semantics(
            handle,
            timeout_ms=min(self.limits.semantic_timeout_ms, self.limits.action_timeout_ms),
        )

        def aria_state(name: str) -> bool | None:
            value = attributes[name]
            if value is None:
                return None
            return value.strip().lower() == "true" if name.startswith("aria-") else True

        text = (await handle.text_content() or "").strip()[:300]
        aria_label = attributes["aria-label"]
        name = attributes["name"]
        placeholder = attributes["placeholder"]
        role = _semantic_text(semantics, "role", attributes["role"])
        accessible_name = _semantic_text(
            semantics,
            "accessibleName",
            aria_label or name or placeholder or text or None,
        )
        element_type = attributes["type"]
        tag = _semantic_text(semantics, "tag")
        fallback_editable = tag in {"input", "textarea"} and (element_type or "").lower() not in {
            "button",
            "checkbox",
            "file",
            "hidden",
            "image",
            "radio",
            "range",
            "reset",
            "submit",
        }
        return _ElementDescription(
            role=role,
            aria_label=aria_label,
            name=name,
            element_type=element_type,
            href=attributes["href"],
            autocomplete=attributes["autocomplete"],
            element_id=attributes["id"],
            placeholder=placeholder,
            inputmode=attributes["inputmode"],
            text=text,
            tag=tag,
            accessible_name=accessible_name,
            disabled=_semantic_bool(semantics, "disabled", fallback=aria_state("disabled") is True),
            checked=_semantic_optional_bool(semantics, "checked", fallback=aria_state("checked")),
            selected=_semantic_optional_bool(
                semantics, "selected", fallback=aria_state("selected")
            ),
            expanded=_semantic_optional_bool(
                semantics, "expanded", fallback=aria_state("aria-expanded")
            ),
            readonly=_semantic_bool(semantics, "readonly", fallback=aria_state("readonly") is True),
            required=_semantic_bool(semantics, "required", fallback=aria_state("required") is True),
            focused=_semantic_bool(semantics, "focused", fallback=False),
            editable=_semantic_bool(semantics, "editable", fallback=fallback_editable),
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
            nodes: list[dict[str, str | bool | None]] = []
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
                    "tag": description.tag,
                    "name": description.accessible_name,
                    "disabled": description.disabled,
                    "checked": description.checked,
                    "selected": description.selected,
                    "expanded": description.expanded,
                    "readonly": description.readonly,
                    "required": description.required,
                    "focused": description.focused,
                    "editable": description.editable,
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

    async def _resolve_action_reference(
        self, element_ref: str
    ) -> tuple[_ElementReference, _ElementDescription]:
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
        if current_description.identity != reference.description.identity:
            msg = "Element changed since the snapshot; take a new snapshot"
            raise BrowserRuntimeError(msg, code=_ELEMENT_CHANGED)
        return reference, current_description

    def _validate_action_target(
        self,
        action: BrowserAction,
        description: _ElementDescription,
    ) -> None:
        if action.kind != "scroll_into_view" and description.disabled:
            msg = "Element is disabled; take a new snapshot"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.ELEMENT_DISABLED)
        if action.kind == "type" and not description.editable:
            msg = "Element is not editable; take a new snapshot"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.ELEMENT_NOT_EDITABLE)
        if action.kind == "select" and description.tag != "select":
            msg = "Action does not match the referenced element; take a new snapshot"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.ACTION_TARGET_MISMATCH)

    async def _perform_element_action(
        self,
        action: BrowserAction,
        handle: Any,  # noqa: ANN401
    ) -> None:
        timeout = self.limits.action_timeout_ms
        match action.kind:
            case "click":
                await handle.click(timeout=timeout)
            case "type":
                await handle.fill(action.text, timeout=timeout)
            case "press":
                await handle.press(action.key, timeout=timeout)
            case "select":
                await handle.select_option(action.value, timeout=timeout)
            case "scroll_into_view":
                await handle.scroll_into_view_if_needed(timeout=timeout)
            case "scroll":
                msg = "Page scroll does not have an element target"
                raise TypeError(msg)

    @staticmethod
    async def _perform_page_scroll(
        action: BrowserAction,
        page: Any,  # noqa: ANN401
    ) -> tuple[dict[str, int], dict[str, int]]:
        if action.kind != "scroll":
            msg = "Element action cannot be used for page scrolling"
            raise TypeError(msg)
        result = await page.evaluate(
            _PAGE_SCROLL_SCRIPT,
            {"direction": action.direction, "distance": action.distance},
        )
        if not isinstance(result, Mapping):
            msg = "Browser returned invalid scroll diagnostics"
            raise TypeError(msg)
        return _scroll_position(result.get("before")), _scroll_position(result.get("after"))

    @staticmethod
    def _action_failure_code(action_kind: str, exc: Exception) -> BrowserErrorCode:
        if isinstance(exc, TimeoutError) or (
            type(exc).__name__ == "TimeoutError" and type(exc).__module__.startswith("playwright.")
        ):
            return BrowserErrorCode.ACTION_TIMEOUT
        if action_kind in {"scroll", "scroll_into_view"}:
            return BrowserErrorCode.SCROLL_FAILED
        return BrowserErrorCode.ACTION_FAILED

    @classmethod
    def _action_failure(cls, action_kind: str, exc: Exception) -> BrowserRuntimeError:
        code = cls._action_failure_code(action_kind, exc)
        if code == BrowserErrorCode.ACTION_TIMEOUT:
            msg = "Browser action timed out"
        elif code == BrowserErrorCode.SCROLL_FAILED:
            msg = "Browser scroll failed"
        else:
            msg = "Browser action failed"
        return BrowserRuntimeError(msg, code=code)

    async def act(self, action: BrowserAction) -> str:
        """Execute one bounded page action or exact-handle element action."""
        async with self._lock:
            try:
                if action.kind == "scroll":
                    page = await self._page()
                    try:
                        async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                            before, after = await self._perform_page_scroll(action, page)
                    except Exception as exc:
                        raise self._action_failure(action.kind, exc) from exc
                    await self._enforce_page_limit()
                    return json.dumps(
                        {
                            "ok": True,
                            "page_ref": self._register_page(page),
                            "action": action.kind,
                            "direction": action.direction,
                            "distance": action.distance,
                            "before": before,
                            "after": after,
                            "moved": before != after,
                        },
                        separators=(",", ":"),
                    )

                reference, description = await self._resolve_action_reference(action.ref)
                self._validate_action_target(action, description)
                try:
                    async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                        await self._perform_element_action(action, reference.handle)
                except Exception as exc:
                    raise self._action_failure(action.kind, exc) from exc
                await self._enforce_page_limit()
                return json.dumps(
                    {
                        "ok": True,
                        "page_ref": self._register_page(reference.page),
                        "action": action.kind,
                        "target": description.diagnostics(),
                    },
                    separators=(",", ":"),
                )
            finally:
                self._invalidate_elements()

    async def screenshot(self, page_ref: str | None = None) -> ScreenshotResult:
        """Return bounded raw PNG data with metadata that excludes base64."""
        async with self._lock:
            await self._enforce_page_limit()
            page = await self._page(page_ref)
            payload = await page.screenshot(type="png", full_page=False)
            result = ScreenshotResult(
                page_ref=self._register_page(page),
                media_type="image/png",
                data=payload,
            )
            if (
                len(payload) > self.limits.max_screenshot_bytes
                or result.projected_output_chars() > self.limits.max_screenshot_output_chars
            ):
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
        """Close this context once while preserving retryability after failure."""
        async with self._lock:
            if self._closed:
                return
            try:
                async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                    await self.context.close()
            except Exception as exc:
                msg = "Browser context cleanup failed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED) from exc
            self._closed = True
            self._element_refs.clear()
            self._invalidated_element_codes.clear()
            self._page_refs.clear()
            self._observed_pages.clear()


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
        self._sessions: OrderedDict[str, BrowserSession] = OrderedDict()
        self._leases: dict[str, int] = {}
        self._closing_sessions: set[str] = set()
        self._startup_lock = asyncio.Lock()
        self._sessions_lock = asyncio.Lock()
        self._lease_condition = asyncio.Condition(self._sessions_lock)
        self._capacity_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_complete = False

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

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        if not thread_id or len(thread_id) > _MAX_THREAD_ID_CHARS:
            msg = "Thread ID must contain between 1 and 512 characters"
            raise BrowserRuntimeError(msg)

    @staticmethod
    def _context_limit_error() -> BrowserRuntimeError:
        msg = "Browser context limit reached; all contexts are currently in use"
        return BrowserRuntimeError(msg, code=BrowserErrorCode.CONTEXT_LIMIT_REACHED)

    async def _create_session(self) -> BrowserSession:
        browser = await self._ensure_browser()
        context = await browser.new_context(
            accept_downloads=False,
            service_workers="block",
            viewport={"width": 1280, "height": 720},
        )
        session = BrowserSession(context=context, policy=self.network_policy, limits=self.limits)
        try:
            await session.initialize()
        except BaseException:
            try:
                async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                    await context.close()
            except Exception as cleanup_error:
                msg = "Failed browser context initialization could not be cleaned up"
                raise BrowserRuntimeError(
                    msg, code=BrowserErrorCode.CLEANUP_FAILED
                ) from cleanup_error
            raise
        return session

    async def _get_or_create_session(self, thread_id: str, *, leased: bool) -> BrowserSession:
        self._validate_thread_id(thread_id)
        async with self._capacity_lock:
            async with self._lease_condition:
                if self._closed:
                    msg = "Browser runtime manager is closed"
                    raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
                existing = self._sessions.get(thread_id)
                if existing is not None and thread_id not in self._closing_sessions:
                    self._sessions.move_to_end(thread_id)
                    if leased:
                        self._leases[thread_id] += 1
                    return existing
                eviction = next(
                    (
                        (key, session)
                        for key, session in self._sessions.items()
                        if self._leases.get(key, 0) == 0 and key not in self._closing_sessions
                    ),
                    None,
                )
                if len(self._sessions) >= self.limits.max_contexts:
                    if eviction is None:
                        raise self._context_limit_error()
                    eviction_key, eviction_session = eviction
                    self._closing_sessions.add(eviction_key)
                else:
                    eviction_key = None
                    eviction_session = None

            if eviction_session is not None and eviction_key is not None:
                try:
                    await eviction_session.aclose()
                except BaseException:
                    async with self._lease_condition:
                        self._closing_sessions.discard(eviction_key)
                        self._lease_condition.notify_all()
                    raise
                async with self._lease_condition:
                    if self._sessions.get(eviction_key) is eviction_session:
                        self._sessions.pop(eviction_key)
                        self._leases.pop(eviction_key, None)
                    self._closing_sessions.discard(eviction_key)

            session = await self._create_session()
            async with self._lease_condition:
                self._sessions[thread_id] = session
                self._leases[thread_id] = int(leased)
                return session

    async def get_session(self, thread_id: str) -> BrowserSession:
        """Return an unleased session for advanced direct usage.

        Unleased sessions remain eligible for LRU eviction immediately after this
        method returns. Production callers that need retrieval and operation to be
        atomic with respect to eviction must use `lease_session`.
        """
        return await self._get_or_create_session(thread_id, leased=False)

    @asynccontextmanager
    async def lease_session(self, thread_id: str) -> AsyncIterator[BrowserSession]:
        """Lease one session so it cannot be evicted during an operation."""
        session = await self._get_or_create_session(thread_id, leased=True)
        try:
            yield session
        finally:
            async with self._lease_condition:
                if self._sessions.get(thread_id) is session:
                    leases = self._leases.get(thread_id, 0)
                    if leases > 0:
                        self._leases[thread_id] = leases - 1
                self._lease_condition.notify_all()

    async def aclose_session(self, thread_id: str) -> None:
        """Close one idle thread-isolated session, retaining it if cleanup fails."""
        self._validate_thread_id(thread_id)
        async with self._capacity_lock:
            async with self._lease_condition:
                session = self._sessions.get(thread_id)
                if session is None:
                    return
                if self._leases.get(thread_id, 0) > 0:
                    raise self._context_limit_error()
                self._closing_sessions.add(thread_id)
            try:
                await session.aclose()
            except BaseException:
                async with self._lease_condition:
                    self._closing_sessions.discard(thread_id)
                    self._lease_condition.notify_all()
                raise
            async with self._lease_condition:
                if self._sessions.get(thread_id) is session:
                    self._sessions.pop(thread_id)
                    self._leases.pop(thread_id, None)
                self._closing_sessions.discard(thread_id)
                self._lease_condition.notify_all()

    async def _wait_for_leases(self) -> None:
        try:
            async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                async with self._lease_condition:
                    await self._lease_condition.wait_for(lambda: not any(self._leases.values()))
        except TimeoutError as exc:
            msg = "Browser shutdown timed out waiting for active operations"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED) from exc

    async def aclose(self) -> None:
        """Close all resources after active leases finish, with retryable cleanup."""
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            async with self._capacity_lock, self._lease_condition:
                self._closed = True
            await self._wait_for_leases()

            async with self._capacity_lock:
                for thread_id, session in tuple(self._sessions.items()):
                    async with self._lease_condition:
                        self._closing_sessions.add(thread_id)
                    try:
                        await session.aclose()
                    except BaseException:
                        async with self._lease_condition:
                            self._closing_sessions.discard(thread_id)
                            self._lease_condition.notify_all()
                        raise
                    async with self._lease_condition:
                        if self._sessions.get(thread_id) is session:
                            self._sessions.pop(thread_id)
                            self._leases.pop(thread_id, None)
                        self._closing_sessions.discard(thread_id)
                        self._lease_condition.notify_all()

                async with self._startup_lock:
                    if self._browser is not None:
                        try:
                            async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                                await self._browser.close()
                        except Exception as exc:
                            msg = "Browser cleanup failed"
                            raise BrowserRuntimeError(
                                msg, code=BrowserErrorCode.CLEANUP_FAILED
                            ) from exc
                        self._browser = None
                    if self._playwright is not None:
                        try:
                            async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                                await self._playwright.stop()
                        except Exception as exc:
                            msg = "Playwright cleanup failed"
                            raise BrowserRuntimeError(
                                msg, code=BrowserErrorCode.CLEANUP_FAILED
                            ) from exc
                        self._playwright = None
                self._shutdown_complete = True
