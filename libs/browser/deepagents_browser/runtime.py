"""Lazy, bounded Playwright runtime management."""

from __future__ import annotations

import asyncio
import atexit
import json
import math
import queue
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import Future
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents_browser.errors import BrowserError, BrowserErrorCode, BrowserRuntimeError
from deepagents_browser.network import NetworkPolicy

RuntimeFactory = Callable[[], tuple[Any, Any]]

if TYPE_CHECKING:
    from deepagents_browser.schemas import BrowserAction

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


def _element_semantics(
    handle: Any,  # noqa: ANN401
    *,
    timeout_ms: int,
) -> Mapping[str, object]:
    """Collect optional semantics on the worker under Playwright page defaults."""
    evaluate = getattr(handle, "evaluate", None)
    if not callable(evaluate):
        return {}
    # Sync evaluate has no per-call timeout; the page default applies where supported.
    _ = timeout_ms
    try:
        value = evaluate(_ELEMENT_SEMANTICS_SCRIPT)
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
        semantic_timeout_ms: Requested bound for optional semantic evaluation where supported.
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
class _SyncBrowserSession:
    """One isolated browser context owned by a LangGraph thread."""

    context: Any
    validate_request: Callable[[str], bool]
    limits: BrowserLimits
    _active_page: Any | None = None
    _page_refs: dict[str, Any] = field(default_factory=dict)
    _observed_pages: dict[int, Any] = field(default_factory=dict)
    _element_refs: dict[str, _ElementReference] = field(default_factory=dict)
    _invalidated_element_codes: dict[str, BrowserErrorCode] = field(default_factory=dict)
    _generation: int = 0
    _request_count: int = 0
    _closed: bool = False

    def initialize(self) -> None:
        """Install request interception before any page is used."""

        def handle_route(route: Any) -> None:  # noqa: ANN401
            self._handle_route(route)

        self.context.route("**/*", handle_route)
        pages = list(self.context.pages)
        if pages:
            self._active_page = pages[0]
            self._register_page(pages[0])

    def _handle_route(self, route: Any) -> None:  # noqa: ANN401
        self._request_count += 1
        over_limit = self._request_count > self.limits.max_requests_per_context
        if over_limit:
            route.abort("blockedbyclient")
            return
        if self.validate_request(route.request.url):
            route.continue_()
        else:
            route.abort("blockedbyclient")

    def _register_page(self, page: Any) -> str:  # noqa: ANN401
        page_id = id(page)
        if self._observed_pages.get(page_id) is not page:
            page.set_default_timeout(self.limits.action_timeout_ms)
            page.set_default_navigation_timeout(self.limits.action_timeout_ms)
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

    def _page(
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
        return self._new_page()

    def _new_page(self) -> Any:  # noqa: ANN401
        if len(self.context.pages) >= self.limits.max_tabs_per_context:
            msg = "Tab limit reached"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.TAB_LIMIT_REACHED)
        page = self.context.new_page()
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

    def _enforce_page_limit(self) -> None:
        pages = list(self.context.pages)
        if len(pages) <= self.limits.max_tabs_per_context:
            return
        for page in pages[self.limits.max_tabs_per_context :]:
            page.close()
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

    def navigate(self, url: str, page_ref: str | None = None) -> str:
        """Navigate a selected tab after validating the destination."""
        page = self._page(page_ref)
        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=self.limits.action_timeout_ms,
        )
        self._enforce_page_limit()
        self._invalidate_elements()
        return json.dumps(
            {
                "page_ref": self._register_page(page),
                "title": (page.title())[:300],
                "url": page.url[:8_192],
            },
            separators=(",", ":"),
        )

    def _describe_element(self, handle: Any) -> _ElementDescription:  # noqa: ANN401
        def bounded_attribute(name: str, limit: int = 300) -> str | None:
            value = handle.get_attribute(name)
            return None if value is None else value[:limit]

        attributes = {
            name: bounded_attribute(name, 2_048 if name == "href" else 300)
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
        semantics = _element_semantics(
            handle,
            timeout_ms=min(self.limits.semantic_timeout_ms, self.limits.action_timeout_ms),
        )

        def aria_state(name: str) -> bool | None:
            value = attributes[name]
            if value is None:
                return None
            return value.strip().lower() == "true" if name.startswith("aria-") else True

        text = (handle.text_content() or "").strip()[:300]
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

    def snapshot(self, page_ref: str | None = None) -> str:
        """Return a bounded snapshot with opaque, generation-scoped references."""
        self._enforce_page_limit()
        page = self._page(page_ref)
        self._invalidate_elements()
        generation = self._generation
        locator = page.locator(_ACTIONABLE_SELECTOR)
        total = locator.count()
        count = min(total, self.limits.max_snapshot_nodes)
        nodes: list[dict[str, str | bool | None]] = []
        for index in range(count):
            item = locator.nth(index)
            if not item.is_visible():
                continue
            handle = item.element_handle()
            if handle is None or not handle.is_visible():
                continue
            description = self._describe_element(handle)
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

    def _resolve_action_reference(
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
            is_visible = reference.handle.is_visible()
            current_description = self._describe_element(reference.handle)
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

    def _perform_element_action(
        self,
        action: BrowserAction,
        handle: Any,  # noqa: ANN401
    ) -> None:
        timeout = self.limits.action_timeout_ms
        match action.kind:
            case "click":
                handle.click(timeout=timeout)
            case "type":
                handle.fill(action.text, timeout=timeout)
            case "press":
                handle.press(action.key, timeout=timeout)
            case "select":
                handle.select_option(action.value, timeout=timeout)
            case "scroll_into_view":
                handle.scroll_into_view_if_needed(timeout=timeout)
            case "scroll":
                msg = "Page scroll does not have an element target"
                raise TypeError(msg)

    @staticmethod
    def _perform_page_scroll(
        action: BrowserAction,
        page: Any,  # noqa: ANN401
    ) -> tuple[dict[str, int], dict[str, int]]:
        if action.kind != "scroll":
            msg = "Element action cannot be used for page scrolling"
            raise TypeError(msg)
        result = page.evaluate(
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

    def act(self, action: BrowserAction) -> str:
        """Execute one bounded page action or exact-handle element action."""
        try:
            if action.kind == "scroll":
                page = self._page()
                try:
                    before, after = self._perform_page_scroll(action, page)
                except Exception as exc:
                    raise self._action_failure(action.kind, exc) from exc
                self._enforce_page_limit()
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

            reference, description = self._resolve_action_reference(action.ref)
            self._validate_action_target(action, description)
            try:
                self._perform_element_action(action, reference.handle)
            except Exception as exc:
                raise self._action_failure(action.kind, exc) from exc
            self._enforce_page_limit()
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

    def screenshot(self, page_ref: str | None = None) -> ScreenshotResult:
        """Return bounded raw PNG data with metadata that excludes base64."""
        self._enforce_page_limit()
        page = self._page(page_ref)
        payload = page.screenshot(
            type="png",
            full_page=False,
            timeout=self.limits.action_timeout_ms,
        )
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

    def tabs(self, operation: str, page_ref: str | None = None) -> str:
        """List, create, select, or close tabs without exposing raw page objects."""
        self._enforce_page_limit()
        if operation == "new":
            self._new_page()
        elif operation == "select":
            if page_ref is None:
                msg = "page_ref is required when selecting a tab"
                raise BrowserRuntimeError(msg)
            self._page(page_ref, create=False)
        elif operation == "close":
            page = self._page(page_ref, create=False)
            page.close()
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
                "title": (page.title())[:300],
                "url": page.url[:8_192],
            }
            for page in list(self.context.pages)[: self.limits.max_tabs_per_context]
        ]
        return json.dumps({"tabs": tabs}, separators=(",", ":"))

    def aclose(self) -> None:
        """Close this context once while preserving retryability after failure."""
        if self._closed:
            return
        try:
            self.context.close()
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


def _default_runtime_factory() -> tuple[Any, Any]:
    """Import and launch Playwright only on the dedicated worker thread."""
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    playwright = sync_playwright().start()
    try:
        _require_provisioned_chromium(playwright)
        browser = playwright.chromium.launch(headless=False)
    except BaseException:
        playwright.stop()
        raise
    return playwright, browser


class _BrowserWorkerState:
    """All raw Playwright state, accessed only by one dedicated worker thread."""

    def __init__(self) -> None:
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.sessions: dict[str, _SyncBrowserSession] = {}
        self.thread_id: int | None = None
        self._validate_request: Callable[[str], bool] | None = None

    def start(
        self,
        factory: RuntimeFactory,
        validate_request: Callable[[str], bool],
    ) -> None:
        self._assert_worker_thread()
        if self.browser is not None:
            return
        try:
            result = factory()
        except BrowserError:
            raise
        except Exception as exc:
            msg = "Browser runtime startup failed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_FAILED) from exc
        playwright, browser = self._validate_factory_result(result)
        self.playwright = playwright
        self.browser = browser
        self._validate_request = validate_request

    def _assert_worker_thread(self) -> None:
        current = threading.get_ident()
        if self.thread_id is None:
            self.thread_id = current
        elif current != self.thread_id:
            msg = "Playwright state accessed outside its dedicated worker thread"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)

    @staticmethod
    def _validate_factory_result(result: object) -> tuple[Any, Any]:
        if not isinstance(result, tuple) or len(result) != 2:  # noqa: PLR2004
            msg = "Browser runtime factory must return (playwright, browser)"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.INVALID_RUNTIME_FACTORY)
        playwright, browser = cast("tuple[Any, Any]", result)
        if not callable(getattr(playwright, "stop", None)):
            msg = "Browser runtime factory returned an invalid Playwright runtime"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.INVALID_RUNTIME_FACTORY)
        if not callable(getattr(browser, "new_context", None)) or not callable(
            getattr(browser, "close", None)
        ):
            try:
                playwright.stop()
            except Exception as exc:
                msg = "Invalid browser runtime could not be cleaned up"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_FAILED) from exc
            msg = "Browser runtime factory returned an invalid browser"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.INVALID_RUNTIME_FACTORY)
        return playwright, browser

    def create_session(self, session_id: str, limits: BrowserLimits) -> None:
        self._assert_worker_thread()
        if self.browser is None or self._validate_request is None:
            msg = "Browser runtime is not started"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
        try:
            context = self.browser.new_context(
                accept_downloads=False,
                service_workers="block",
                viewport={"width": 1280, "height": 720},
            )
        except Exception as exc:
            msg = "Browser context initialization failed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_FAILED) from exc
        session = _SyncBrowserSession(
            context=context,
            validate_request=self._validate_request,
            limits=limits,
        )
        try:
            session.initialize()
        except BaseException as exc:
            try:
                context.close()
            except Exception as cleanup_error:
                msg = "Failed browser context initialization could not be cleaned up"
                raise BrowserRuntimeError(
                    msg,
                    code=BrowserErrorCode.CLEANUP_FAILED,
                ) from cleanup_error
            if isinstance(exc, BrowserError):
                raise
            if isinstance(exc, Exception):
                msg = "Browser context initialization failed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_FAILED) from exc
            raise
        self.sessions[session_id] = session

    def call_session(self, session_id: str, method: str, *args: object) -> object:
        self._assert_worker_thread()
        try:
            session = self.sessions[session_id]
        except KeyError as exc:
            msg = "Browser session is closed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED) from exc
        try:
            return getattr(session, method)(*args)
        except BrowserError:
            raise
        except Exception as exc:
            msg = "Browser operation failed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.ACTION_FAILED) from exc

    def close_session(self, session_id: str) -> None:
        self._assert_worker_thread()
        session = self.sessions.get(session_id)
        if session is None:
            return
        session.aclose()
        self.sessions.pop(session_id, None)

    def close_all(self) -> None:
        """Synchronously close all resources; failures remain retryable."""
        self._assert_worker_thread()
        for session_id in tuple(self.sessions):
            self.close_session(session_id)
        if self.browser is not None:
            try:
                self.browser.close()
            except Exception as exc:
                msg = "Browser cleanup failed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED) from exc
            self.browser = None
        if self.playwright is not None:
            try:
                self.playwright.stop()
            except Exception as exc:
                msg = "Playwright cleanup failed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED) from exc
            self.playwright = None


class _BrowserWorker:
    """Single daemon command thread that exclusively owns Playwright state."""

    def __init__(self) -> None:
        self._commands: queue.Queue[tuple[str, tuple[object, ...], Future[object]] | None] = (
            queue.Queue()
        )
        self._shutdown = False
        self._lifecycle_lock = threading.Lock()
        self._stopped: Future[None] = Future()
        self._thread = threading.Thread(
            target=self._run,
            name="deepagents-browser",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        state = _BrowserWorkerState()
        try:
            while (command := self._commands.get()) is not None:
                method, args, future = command
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = getattr(state, method)(*args)
                except BaseException as exc:  # noqa: BLE001  # preserve worker future semantics
                    future.set_exception(exc)
                else:
                    future.set_result(result)
        finally:
            self._stopped.set_result(None)

    def submit(self, method: str, *args: object) -> Future[object]:
        with self._lifecycle_lock:
            if self._shutdown:
                msg = "Browser worker is closed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
            future: Future[object] = Future()
            self._commands.put((method, args, future))
            return future

    def shutdown(self, *, wait: bool, timeout: float | None = None) -> None:
        with self._lifecycle_lock:
            if not self._shutdown:
                self._shutdown = True
                self._commands.put(None)
        if wait:
            self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        """Return whether the dedicated worker thread is still running."""
        return self._thread.is_alive()

    @property
    def stopped(self) -> Future[None]:
        """Return a future completed after the worker command loop exits."""
        return self._stopped


@dataclass(slots=True)
class BrowserSession:
    """Async facade for one worker-owned isolated browser context.

    Cancellation of an awaiting caller cannot preempt a synchronous Playwright call
    that has already started on the worker. Native Playwright timeouts bound methods
    that expose one. Manager action deadlines can return while a non-preemptible sync
    evaluate is still running, and the worker finishes it before the next command.
    """

    _manager: BrowserRuntimeManager
    _session_id: str
    policy: NetworkPolicy
    limits: BrowserLimits
    _closed: bool = False

    async def _call(self, method: str, *args: object) -> object:
        if self._closed:
            msg = "Browser session is closed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
        return await self._manager._call_session(  # noqa: SLF001
            self._session_id, method, *args
        )

    async def navigate(self, url: str, page_ref: str | None = None) -> str:
        """Validate and navigate a selected tab."""
        await self.policy.validate_url(url)
        return cast("str", await self._call("navigate", url, page_ref))

    async def snapshot(self, page_ref: str | None = None) -> str:
        """Return a bounded snapshot with opaque references."""
        return cast("str", await self._call("snapshot", page_ref))

    async def act(self, action: BrowserAction) -> str:
        """Execute one validated action."""
        return cast("str", await self._call("act", action))

    async def screenshot(self, page_ref: str | None = None) -> ScreenshotResult:
        """Return bounded screenshot bytes and metadata."""
        return cast("ScreenshotResult", await self._call("screenshot", page_ref))

    async def tabs(self, operation: str, page_ref: str | None = None) -> str:
        """List, create, select, or close tabs."""
        return cast("str", await self._call("tabs", operation, page_ref))

    async def aclose(self) -> None:
        """Close this worker-owned context once, preserving retryability."""
        if self._closed:
            return
        await self._manager._aclose_facade(self)  # noqa: SLF001


class BrowserRuntimeManager:
    """Lazily own a dedicated Playwright worker and bounded contexts."""

    def __init__(
        self,
        *,
        limits: BrowserLimits | None = None,
        network_policy: NetworkPolicy | None = None,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        """Configure the manager without starting a thread or registering atexit."""
        self.limits = limits or BrowserLimits()
        self.network_policy = network_policy or NetworkPolicy()
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._worker: _BrowserWorker | None = None
        self._retired_workers: set[_BrowserWorker] = set()
        self._runtime_started = False
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._sessions: OrderedDict[str, BrowserSession] = OrderedDict()
        self._leases: dict[str, int] = {}
        self._closing_sessions: set[str] = set()
        self._orphaned_session_ids: set[str] = set()
        self._startup_lock = asyncio.Lock()
        self._sessions_lock = asyncio.Lock()
        self._lease_condition = asyncio.Condition(self._sessions_lock)
        self._capacity_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._closed = False
        self._shutdown_complete = False
        self._atexit_registered = False
        self._atexit_complete = False
        self._lifecycle_lock = threading.Lock()

    async def validate_url(self, url: str) -> None:
        """Validate a URL without creating or accessing a browser runtime."""
        await self.network_policy.validate_url(url)

    async def _validate_request_url(self, url: str) -> bool:
        try:
            await self.network_policy.validate_url(url)
        except Exception:  # noqa: BLE001  # interception must fail closed
            return False
        return True

    def _worker_validate_request(self, url: str) -> bool:
        loop = self._owner_loop
        if loop is None or loop.is_closed():
            return False
        future = asyncio.run_coroutine_threadsafe(self._validate_request_url(url), loop)
        try:
            return future.result(timeout=self.limits.action_timeout_ms / 1_000)
        except Exception:  # noqa: BLE001  # timeout, cancellation, and policy errors fail closed
            future.cancel()
            return False

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self._atexit_close)
        self._atexit_registered = True

    def _retire_worker(self, worker: _BrowserWorker) -> None:
        """Queue cleanup after an unpreemptible timed-out startup and retire its thread."""
        with suppress(BrowserRuntimeError):
            worker.submit("close_all")
        worker.shutdown(wait=False)
        self._retired_workers.add(worker)
        if self._worker is worker:
            self._worker = None
        self._runtime_started = False

    def _atexit_close(self) -> None:
        """Best-effort bounded synchronous fallback; explicit close is preferred."""
        with self._lifecycle_lock:
            if self._atexit_complete or self._shutdown_complete:
                return
            deadline = time.monotonic() + self.limits.action_timeout_ms / 1_000
            worker = self._worker
            close_succeeded = worker is None
            if worker is not None:
                try:
                    future = worker.submit("close_all")
                except Exception:  # noqa: BLE001  # interpreter shutdown cannot report
                    future = None
                worker.shutdown(wait=False)
                if future is not None:
                    try:
                        future.result(timeout=max(0.0, deadline - time.monotonic()))
                    except Exception as exc:  # noqa: BLE001  # interpreter shutdown cannot report
                        _ = exc
                    else:
                        close_succeeded = True
                worker.shutdown(
                    wait=True,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            all_stopped = worker is None or not worker.is_alive
            for retired in self._retired_workers:
                retired.shutdown(
                    wait=True,
                    timeout=max(0.0, deadline - time.monotonic()),
                )
                all_stopped = all_stopped and not retired.is_alive
            if close_succeeded and all_stopped:
                self._atexit_complete = True

    async def _await_future(
        self,
        future: Future[object],
        *,
        wait_seconds: float | None = None,
    ) -> object:
        """Await a worker command without pretending cancellation can preempt it."""
        wrapped = asyncio.wrap_future(future)
        if wait_seconds is None:
            return await asyncio.shield(wrapped)
        async with asyncio.timeout(wait_seconds):
            return await asyncio.shield(wrapped)

    @staticmethod
    async def _reconcile_future(
        future: Future[object],
        *,
        cancel_if_queued: bool,
    ) -> tuple[object | None, BaseException | None, asyncio.CancelledError | None, bool]:
        """Wait through cancellation and report the exact worker command outcome."""
        wrapped = asyncio.wrap_future(future)
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                if cancel_if_queued and future.cancel():
                    return None, None, cancellation, False
            except BaseException as exc:  # noqa: BLE001  # lifecycle state needs exact outcome
                return None, exc, cancellation, True
            else:
                return result, None, cancellation, True

    async def _ensure_worker(self) -> _BrowserWorker:
        if self._closed:
            msg = "Browser runtime manager is closed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
        if self._worker is not None and self._runtime_started:
            return self._worker
        async with self._startup_lock:
            if self._closed:
                msg = "Browser runtime manager is closed"
                raise BrowserRuntimeError(msg, code=BrowserErrorCode.RUNTIME_CLOSED)
            if self._worker is None:
                self._owner_loop = asyncio.get_running_loop()
                self._worker = _BrowserWorker()
                self._register_atexit()
            worker = self._worker
            if not self._runtime_started:
                try:
                    future = worker.submit(
                        "start", self._runtime_factory, self._worker_validate_request
                    )
                    await self._await_future(
                        future,
                        wait_seconds=self.limits.startup_timeout_seconds,
                    )
                    self._runtime_started = True
                except TimeoutError as exc:
                    self._retire_worker(worker)
                    msg = "Browser runtime startup timed out"
                    raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_TIMEOUT) from exc
                except BrowserError:
                    raise
                except Exception as exc:
                    msg = "Browser runtime startup failed"
                    raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_FAILED) from exc
        return worker

    async def _call_session(self, session_id: str, method: str, *args: object) -> object:
        worker = await self._ensure_worker()
        future = worker.submit("call_session", session_id, method, *args)
        try:
            return await self._await_future(
                future,
                wait_seconds=(self.limits.action_timeout_ms / 1_000 if method == "act" else None),
            )
        except TimeoutError as exc:
            msg = "Browser action timed out"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.ACTION_TIMEOUT) from exc

    async def _close_worker_session(self, session: BrowserSession) -> None:
        worker = self._worker
        if worker is None:
            session._closed = True  # noqa: SLF001
            return
        future = worker.submit("close_session", session._session_id)  # noqa: SLF001
        _, error, cancellation, ran = await self._reconcile_future(
            future,
            cancel_if_queued=True,
        )
        if error is None and ran:
            session._closed = True  # noqa: SLF001
        if cancellation is not None:
            raise cancellation
        if error is not None:
            raise error

    @staticmethod
    def _validate_thread_id(thread_id: str) -> None:
        if not thread_id or len(thread_id) > _MAX_THREAD_ID_CHARS:
            msg = "Thread ID must contain between 1 and 512 characters"
            raise BrowserRuntimeError(msg)

    @staticmethod
    def _context_limit_error() -> BrowserRuntimeError:
        msg = "Browser context limit reached; all contexts are currently in use"
        return BrowserRuntimeError(msg, code=BrowserErrorCode.CONTEXT_LIMIT_REACHED)

    async def _close_orphaned_sessions(self, worker: _BrowserWorker) -> None:
        for session_id in tuple(self._orphaned_session_ids):
            future = worker.submit("close_session", session_id)
            _, error, cancellation, ran = await self._reconcile_future(
                future,
                cancel_if_queued=True,
            )
            if error is None and ran:
                self._orphaned_session_ids.discard(session_id)
            if cancellation is not None:
                raise cancellation
            if error is not None:
                raise error

    async def _create_session(self) -> BrowserSession:
        worker = await self._ensure_worker()
        await self._close_orphaned_sessions(worker)
        session_id = secrets.token_urlsafe(24)
        future = worker.submit("create_session", session_id, self.limits)
        try:
            await self._await_future(future)
        except asyncio.CancelledError:
            _, create_error, _, ran = await self._reconcile_future(
                future,
                cancel_if_queued=True,
            )
            if create_error is None and ran:
                try:
                    close_future = worker.submit("close_session", session_id)
                    _, close_error, _, close_ran = await self._reconcile_future(
                        close_future,
                        cancel_if_queued=False,
                    )
                except BaseException:  # noqa: BLE001  # preserve cancellation and ownership
                    self._orphaned_session_ids.add(session_id)
                else:
                    if close_error is not None or not close_ran:
                        self._orphaned_session_ids.add(session_id)
            raise
        except BrowserError:
            raise
        except Exception as exc:
            msg = "Browser context initialization failed"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.STARTUP_FAILED) from exc
        return BrowserSession(
            _manager=self,
            _session_id=session_id,
            policy=self.network_policy,
            limits=self.limits,
        )

    async def _finish_session_close(
        self,
        thread_id: str,
        session: BrowserSession,
    ) -> None:
        """Reconcile an exact facade mapping after its worker close settles."""
        try:
            await self._close_worker_session(session)
        except BaseException:
            async with self._lease_condition:
                if session._closed and self._sessions.get(thread_id) is session:  # noqa: SLF001
                    self._sessions.pop(thread_id)
                    self._leases.pop(thread_id, None)
                self._closing_sessions.discard(thread_id)
                self._lease_condition.notify_all()
            raise
        async with self._lease_condition:
            if self._sessions.get(thread_id) is session:
                self._sessions.pop(thread_id)
                self._leases.pop(thread_id, None)
            self._closing_sessions.discard(thread_id)
            self._lease_condition.notify_all()

    async def _aclose_facade(self, session: BrowserSession) -> None:
        """Close one facade and remove only its exact manager mapping."""
        async with self._capacity_lock:
            async with self._lease_condition:
                mapping = next(
                    (
                        thread_id
                        for thread_id, candidate in self._sessions.items()
                        if candidate is session
                    ),
                    None,
                )
                if mapping is not None:
                    self._closing_sessions.add(mapping)
            if mapping is None:
                await self._close_worker_session(session)
                return
            await self._finish_session_close(mapping, session)

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
                await self._finish_session_close(eviction_key, eviction_session)

            session = await self._create_session()
            async with self._lease_condition:
                self._sessions[thread_id] = session
                self._leases[thread_id] = int(leased)
                return session

    async def get_session(self, thread_id: str) -> BrowserSession:
        """Return an unleased session eligible for immediate LRU eviction."""
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
            await self._finish_session_close(thread_id, session)

    async def _wait_for_leases(self) -> None:
        try:
            async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                async with self._lease_condition:
                    await self._lease_condition.wait_for(lambda: not any(self._leases.values()))
        except TimeoutError as exc:
            msg = "Browser shutdown timed out waiting for active operations"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED) from exc

    async def _wait_for_retired_workers(self) -> None:
        if not self._retired_workers:
            return
        stopped = [asyncio.wrap_future(worker.stopped) for worker in self._retired_workers]
        try:
            async with asyncio.timeout(self.limits.action_timeout_ms / 1_000):
                await asyncio.shield(asyncio.gather(*stopped))
        except TimeoutError as exc:
            msg = "Browser cleanup timed out waiting for a retired worker"
            raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED) from exc
        self._retired_workers.clear()

    async def aclose(self) -> None:
        """Deterministically close resources and stop the dedicated worker."""
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            async with self._capacity_lock, self._lease_condition:
                self._closed = True
            await self._wait_for_leases()
            async with self._capacity_lock:
                worker = self._worker
                if worker is not None:
                    try:
                        await self._await_future(
                            worker.submit("close_all"),
                            wait_seconds=self.limits.action_timeout_ms / 1_000,
                        )
                    except BrowserError:
                        raise
                    except Exception as exc:
                        msg = "Browser cleanup failed"
                        raise BrowserRuntimeError(
                            msg,
                            code=BrowserErrorCode.CLEANUP_FAILED,
                        ) from exc
                    worker.shutdown(wait=True, timeout=self.limits.action_timeout_ms / 1_000)
                    if worker.is_alive:
                        msg = "Browser worker shutdown timed out"
                        raise BrowserRuntimeError(msg, code=BrowserErrorCode.CLEANUP_FAILED)
                await self._wait_for_retired_workers()
                self._sessions.clear()
                self._leases.clear()
                self._closing_sessions.clear()
                self._orphaned_session_ids.clear()
                self._shutdown_complete = True
                self._atexit_complete = True
                self._worker = None
