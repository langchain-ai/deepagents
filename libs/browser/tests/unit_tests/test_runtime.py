import asyncio
import inspect
import json
import threading
import time

import pytest

from deepagents_browser.middleware import BrowserMiddleware
from deepagents_browser.network import NetworkPolicy
from deepagents_browser.runtime import (
    BrowserLimits,
    BrowserRuntimeError,
    BrowserRuntimeManager,
    ScreenshotResult,
)
from deepagents_browser.schemas import (
    ClickAction,
    PageScrollAction,
    PressAction,
    ScrollIntoViewAction,
    SelectAction,
    TypeAction,
)


async def _public_resolver(host: str, port: int):
    return ["93.184.216.34"]


class FakeElement:
    def __init__(  # noqa: PLR0913  # configurable fake models Playwright element states
        self,
        *,
        text="Button",
        attrs=None,
        tag="button",
        disabled=False,
        checked=None,
        selected=None,
        expanded=None,
        readonly=False,
        required=False,
        focused=False,
        editable=False,
        label_text=None,
        labelledby_text=None,
    ) -> None:
        self.text = text
        self.attrs = {"role": "button", "aria-label": "Go", **(attrs or {})}
        self.tag = tag
        self.disabled = disabled
        self.checked = checked
        self.selected = selected
        self.expanded = expanded
        self.readonly = readonly
        self.required = required
        self.focused = focused
        self.editable = editable
        self.label_text = label_text
        self.labelledby_text = labelledby_text
        self.semantic_failure = None
        self.semantic_wait = None
        self.visible = True
        self.attached = True
        self.actions = []
        self.timeouts = []
        self.failures = {}

    def get_attribute(self, name):
        return self.attrs.get(name)

    def text_content(self):
        return self.text

    def evaluate(self, script):
        assert "accessibleName" in script
        assert "aria-labelledby" in script
        assert "element.labels" in script
        assert 'getAttribute("title")' in script
        if self.semantic_failure is not None:
            raise self.semantic_failure
        if self.semantic_wait is not None:
            self.semantic_wait.wait()
        accessible_name = (
            self.attrs.get("aria-label")
            or self.labelledby_text
            or self.label_text
            or self.attrs.get("title")
            or self.text
        )
        return {
            "tag": self.tag,
            "role": self.attrs.get("role"),
            "accessibleName": accessible_name,
            "disabled": self.disabled,
            "checked": self.checked,
            "selected": self.selected,
            "expanded": self.expanded,
            "readonly": self.readonly,
            "required": self.required,
            "focused": self.focused,
            "editable": self.editable,
        }

    def is_visible(self):
        return self.visible and self.attached

    def element_handle(self):
        return self if self.attached else None

    def count(self):
        return int(self.attached)

    def _record(self, kind, value, timeout_ms):
        self.timeouts.append((kind, timeout_ms))
        failure = self.failures.get(kind)
        if failure is not None:
            raise failure
        self.actions.append((kind, value))

    def click(self, *, timeout):  # Mirrors Playwright's synchronous API.
        self._record("click", None, timeout)

    def fill(self, text, *, timeout):  # Mirrors Playwright's synchronous API.
        self._record("fill", text, timeout)

    def press(self, key, *, timeout):  # Mirrors Playwright's synchronous API.
        self._record("press", key, timeout)

    def select_option(self, value, *, timeout):  # Mirrors Playwright's synchronous API.
        self._record("select", value, timeout)

    def scroll_into_view_if_needed(self, *, timeout):  # Mirrors Playwright's synchronous API.
        self._record("scroll", None, timeout)


class FakeLocator:
    def __init__(self, elements) -> None:
        self.elements = elements

    def count(self):
        return len(self.elements)

    def nth(self, index):
        return self.elements[index]


class FakePage:
    def __init__(self, context, elements=None, screenshot=b"png") -> None:
        self.context = context
        self.url = "about:blank"
        self.elements = elements or []
        self.screenshot_payload = screenshot
        self.timeout = None
        self.main_frame = object()
        self.event_handlers = {}
        self.evaluate_calls = []
        self.evaluate_failure = None
        self.evaluate_started = None
        self.evaluate_wait = None
        self.scroll_result = {"before": {"x": 0, "y": 0}, "after": {"x": 0, "y": 360}}

    def set_default_timeout(self, timeout):
        self.timeout = timeout

    def set_default_navigation_timeout(self, timeout):
        self.navigation_timeout = timeout

    def on(self, event, handler):
        self.event_handlers.setdefault(event, []).append(handler)

    def emit_frame_navigated(self, frame):
        for handler in self.event_handlers.get("framenavigated", []):
            handler(frame)

    def goto(self, url, *, wait_until, timeout):
        assert timeout == self.timeout
        for _, handler in self.context.routes:
            route = FakeRoute(url)
            handler(route)
            self.context.requests.append(route)
        self.url = url
        self.emit_frame_navigated(self.main_frame)

    def title(self):
        return "Fake title"

    def locator(self, selector):
        assert "button" in selector
        return FakeLocator(self.elements)

    def screenshot(self, *, type, full_page, timeout):
        assert type == "png"
        assert full_page is False
        assert timeout == self.timeout
        return self.screenshot_payload

    def evaluate(self, script, arg):
        assert "window.scrollBy" in script
        assert "window.innerWidth" in script
        assert "window.innerHeight" in script
        self.evaluate_calls.append((script, arg))
        if self.evaluate_failure is not None:
            raise self.evaluate_failure
        if self.evaluate_started is not None:
            self.evaluate_started.set()
        if self.evaluate_wait is not None:
            self.evaluate_wait.wait()
        return self.scroll_result

    def close(self):
        self.context.pages.remove(self)


class FakeRequest:
    def __init__(self, url) -> None:
        self.url = url


class FakeRoute:
    def __init__(self, url) -> None:
        self.request = FakeRequest(url)
        self.continued = 0
        self.aborted = 0

    def continue_(self):
        self.continued += 1

    def abort(self, error_code="failed"):
        assert error_code == "blockedbyclient"
        self.aborted += 1


class FakeContext:
    def __init__(self, *, page_elements=None, screenshot=b"png") -> None:
        self.pages = []
        self.page_elements = page_elements
        self.screenshot = screenshot
        self.routes = []
        self.requests = []
        self.closed = 0
        self.close_failure = None
        self.close_started = None
        self.close_wait = None

    def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    def new_page(self):
        page = FakePage(self, self.page_elements, self.screenshot)
        self.pages.append(page)
        return page

    def close(self):
        self.closed += 1
        if self.close_started is not None:
            self.close_started.set()
        if self.close_wait is not None:
            self.close_wait.wait()
        if self.close_failure is not None:
            raise self.close_failure
        self.pages.clear()


class PlaywrightWrappingContext(FakeContext):
    def route(self, pattern, handler):
        if inspect.ismethod(handler):
            setattr(handler.__self__, f"_pw_impl_instance_{handler.__name__}", handler)
        else:
            handler._pw_impl_instance = handler
        super().route(pattern, handler)


class FakeBrowser:
    def __init__(self, context_factory=None) -> None:
        self.context_factory = context_factory or FakeContext
        self.contexts = []
        self.kwargs = []
        self.closed = 0

    def new_context(self, **kwargs):
        self.kwargs.append(kwargs)
        context = self.context_factory()
        self.contexts.append(context)
        return context

    def close(self):
        self.closed += 1


class FakePlaywright:
    def __init__(self) -> None:
        self.stopped = 0

    def stop(self):
        self.stopped += 1


def make_manager(*, browser=None, limits=None):
    playwright = FakePlaywright()
    fake_browser = browser or FakeBrowser()
    starts = []

    def factory():
        starts.append("start")
        return playwright, fake_browser

    manager = BrowserRuntimeManager(
        limits=limits,
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    return manager, playwright, fake_browser, starts


async def test_route_handler_supports_playwright_callback_wrapping():
    browser = FakeBrowser(context_factory=PlaywrightWrappingContext)
    manager, _, _, _ = make_manager(browser=browser)

    await manager.get_session("thread")

    assert len(browser.contexts[0].routes) == 1


async def test_runtime_is_lazy_single_flight_isolated_and_bounded():
    manager, _, browser, starts = make_manager(limits=BrowserLimits(max_contexts=2))
    assert starts == []
    sessions = await asyncio.gather(*(manager.get_session("one") for _ in range(10)))
    assert len(starts) == 1
    assert len(browser.contexts) == 1
    assert len({id(session) for session in sessions}) == 1
    assert browser.kwargs[0]["service_workers"] == "block"
    assert browser.kwargs[0]["accept_downloads"] is False
    second = await manager.get_session("two")
    assert second is not sessions[0]
    assert await manager.get_session("one") is sessions[0]

    third = await manager.get_session("three")
    assert third is not sessions[0]
    assert third is not second
    assert browser.contexts[1].closed == 1
    assert browser.contexts[0].closed == 0
    assert list(manager._sessions) == ["one", "three"]


async def test_in_flight_lease_is_not_evicted():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_contexts=2))

    async with manager.lease_session("one") as first:
        second = await manager.get_session("two")
        third = await manager.get_session("three")
        assert third is not first
        assert third is not second
        assert browser.contexts[0].closed == 0
        assert browser.contexts[1].closed == 1
        assert list(manager._sessions) == ["one", "three"]


async def test_all_leased_capacity_fails_bounded_without_eviction():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_contexts=2))

    async with manager.lease_session("one"), manager.lease_session("two"):
        with pytest.raises(BrowserRuntimeError) as exc_info:
            await manager.get_session("three")

    assert exc_info.value.code == "context_limit_reached"
    assert len(browser.contexts) == 2
    assert all(context.closed == 0 for context in browser.contexts)


async def test_cancelled_create_is_closed_before_capacity_can_be_reused():
    create_started = threading.Event()
    allow_create = threading.Event()

    class BlockingCreateBrowser(FakeBrowser):
        def new_context(self, **kwargs):
            create_started.set()
            allow_create.wait()
            return super().new_context(**kwargs)

    browser = BlockingCreateBrowser()
    manager, _, _, _ = make_manager(
        browser=browser,
        limits=BrowserLimits(max_contexts=1),
    )

    create_task = asyncio.create_task(manager.get_session("cancelled"))
    while not create_started.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0)
    create_task.cancel()
    await asyncio.sleep(0)
    assert not create_task.done()

    allow_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert len(browser.contexts) == 1
    assert browser.contexts[0].closed == 1
    replacement = await manager.get_session("replacement")
    assert replacement is manager._sessions["replacement"]
    assert list(manager._sessions) == ["replacement"]
    assert len(browser.contexts) == 2
    assert sum(context.closed == 0 for context in browser.contexts) == 1


async def test_cancelled_eviction_reconciles_close_and_removes_exact_facade():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_contexts=1))
    stale = await manager.get_session("stale")
    context = browser.contexts[0]
    context.close_started = threading.Event()
    context.close_wait = threading.Event()

    eviction_task = asyncio.create_task(manager.get_session("replacement"))
    while not context.close_started.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0)
    eviction_task.cancel()
    await asyncio.sleep(0)
    assert not eviction_task.done()

    context.close_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await eviction_task

    assert stale._closed is True
    assert "stale" not in manager._sessions
    replacement = await manager.get_session("replacement")
    assert replacement is not stale
    assert await manager.get_session("replacement") is replacement
    assert list(manager._sessions) == ["replacement"]


async def test_eviction_cleanup_failure_is_surfaced_and_ownership_is_retained():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_contexts=1))
    first = await manager.get_session("one")
    context = browser.contexts[0]
    context.close_failure = RuntimeError("raw cleanup failure")

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("two")

    assert exc_info.value.code == "cleanup_failed"
    assert "raw cleanup failure" not in str(exc_info.value)
    assert manager._sessions == {"one": first}
    assert len(browser.contexts) == 1
    assert first._closed is False

    context.close_failure = None
    second = await manager.get_session("two")
    assert second is not first
    assert len(browser.contexts) == 2
    assert context.closed == 2


def test_limits_enforce_hard_caps():
    with pytest.raises(ValueError, match="max_contexts"):
        BrowserLimits(max_contexts=65)
    with pytest.raises(ValueError, match="max_requests_per_context"):
        BrowserLimits(max_requests_per_context=10_001)
    with pytest.raises(ValueError, match="semantic_timeout_ms"):
        BrowserLimits(semantic_timeout_ms=10_001)


async def test_navigation_and_tabs_are_bounded():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_tabs_per_context=1))
    session = await manager.get_session("thread")
    result = json.loads(await session.navigate("https://example.com"))
    assert result["url"] == "https://example.com"
    assert browser.contexts[0].pages[0].timeout == 15_000
    with pytest.raises(BrowserRuntimeError, match="Tab limit"):
        await session.tabs("new")


async def test_request_budget_fails_closed_on_the_worker():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(max_requests_per_context=1))
    session = await manager.get_session("thread")

    await session.navigate("https://example.com")
    await session.navigate("https://example.com")

    allowed, denied = browser.contexts[0].requests
    assert allowed.continued == 1
    assert denied.aborted == 1


async def test_snapshot_refs_are_opaque_stale_and_fingerprint_checked():
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    await session.navigate("https://example.com")
    snapshot = json.loads(await session.snapshot())
    ref = snapshot["nodes"][0]["ref"]
    assert ref.startswith(f"e_{snapshot['generation']}_")
    await session.act(ClickAction(kind="click", ref=ref))
    with pytest.raises(BrowserRuntimeError, match="stale"):
        await session.act(ClickAction(kind="click", ref=ref))

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    element.text = "changed"
    with pytest.raises(BrowserRuntimeError, match="changed"):
        await session.act(ClickAction(kind="click", ref=ref))

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    element.attached = False
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "element_identity_unavailable"


async def test_element_refs_keep_identity_across_reorder_and_reject_replacement():
    first = FakeElement(text="same")
    second = FakeElement(text="same")
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[first, second]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    snapshot = json.loads(await session.snapshot())
    first_ref = snapshot["nodes"][0]["ref"]
    page = browser.contexts[0].pages[0]

    page.elements[:] = [second, first]
    await session.act(ClickAction(kind="click", ref=first_ref))
    assert first.actions == [("click", None)]
    assert second.actions == []

    replacement_ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    replaced = page.elements[0]
    replaced.attached = False
    page.elements[0] = FakeElement(text="same")
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=replacement_ref))
    assert exc_info.value.code == "element_identity_unavailable"
    assert page.elements[0].actions == []


async def test_sensitive_controls_are_omitted_and_blocked_at_action_time():
    password = FakeElement(attrs={"type": "password", "name": "password"})
    card = FakeElement(attrs={"autocomplete": "section-checkout cc-number"})
    card_without_autocomplete = FakeElement(attrs={"id": "credit-card-number"})
    safe = FakeElement(attrs={"type": "text", "name": "search"})
    browser = FakeBrowser(
        context_factory=lambda: FakeContext(
            page_elements=[password, card, card_without_autocomplete, safe]
        )
    )
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    snapshot = json.loads(await session.snapshot())
    assert len(snapshot["nodes"]) == 1
    assert snapshot["nodes"][0]["label"] == "Go"

    ref = snapshot["nodes"][0]["ref"]
    safe.attrs["type"] = "password"
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(TypeAction(kind="type", ref=ref, text="secret"))
    assert exc_info.value.code == "sensitive_control_blocked"
    assert safe.actions == []


@pytest.mark.parametrize(
    "source_kwargs",
    [
        {"attrs": {"aria-label": None}, "label_text": "Credit card number"},
        {
            "attrs": {"aria-label": None, "aria-labelledby": "payment-label"},
            "labelledby_text": "Credit card number",
        },
        {"attrs": {"aria-label": None, "title": "Credit card number"}},
    ],
    ids=["label", "aria-labelledby", "title"],
)
async def test_accessible_name_sources_are_sensitive_in_snapshot_and_action_revalidation(
    source_kwargs,
):
    sensitive = FakeElement(text="", **source_kwargs)
    safe = FakeElement(text="Safe", attrs={"aria-label": None})
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[sensitive, safe]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    snapshot = json.loads(await session.snapshot())
    assert [node["name"] for node in snapshot["nodes"]] == ["Safe"]

    ref = snapshot["nodes"][0]["ref"]
    if "label_text" in source_kwargs:
        safe.label_text = "Credit card number"
    elif "labelledby_text" in source_kwargs:
        safe.attrs["aria-labelledby"] = "payment-label"
        safe.labelledby_text = "Credit card number"
    else:
        safe.attrs["title"] = "Credit card number"
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "sensitive_control_blocked"
    assert safe.actions == []


async def test_optional_snapshot_semantics_fail_closed_to_bounded_fallback():
    element = FakeElement(text="Fallback", attrs={"aria-label": None})
    element.semantic_failure = RuntimeError("raw semantic failure")
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(
        browser=browser,
        limits=BrowserLimits(semantic_timeout_ms=1),
    )
    session = await manager.get_session("thread")

    snapshot = await asyncio.wait_for(session.snapshot(), timeout=0.1)

    node = json.loads(snapshot)["nodes"][0]
    assert node["name"] == "Fallback"
    assert "raw semantic failure" not in snapshot


async def test_only_top_level_navigation_invalidates_element_refs():
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    page = browser.contexts[0].pages[0]
    page.emit_frame_navigated(object())
    await session.act(ClickAction(kind="click", ref=ref))

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    page.emit_frame_navigated(page.main_frame)
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "navigation_invalidated_reference"


async def test_new_element_failure_paths_have_stable_codes():
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    element.text = "changed"
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "element_changed"

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref="e_unknown_reference"))
    assert exc_info.value.code == "stale_element_reference"


async def test_snapshot_includes_bounded_semantics_from_exact_handle():
    element = FakeElement(
        tag="input",
        attrs={"role": "checkbox", "aria-label": "Subscribe", "type": "checkbox"},
        checked=True,
        expanded=False,
        required=True,
        focused=True,
        editable=False,
    )
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")

    node = json.loads(await session.snapshot())["nodes"][0]

    assert node == {
        "ref": node["ref"],
        "role": "checkbox",
        "label": "Subscribe",
        "type": "checkbox",
        "href": None,
        "text": "Button",
        "tag": "input",
        "name": "Subscribe",
        "disabled": False,
        "checked": True,
        "selected": None,
        "expanded": False,
        "readonly": False,
        "required": True,
        "focused": True,
        "editable": False,
    }


async def test_snapshot_semantics_fall_back_for_fakes_without_evaluate():
    class AttributeOnlyElement(FakeElement):
        evaluate = None

    element = AttributeOnlyElement(attrs={"disabled": "", "required": ""})
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)

    node = json.loads(await (await manager.get_session("thread")).snapshot())["nodes"][0]

    assert node["tag"] is None
    assert node["name"] == "Go"
    assert node["disabled"] is True
    assert node["required"] is True


@pytest.mark.parametrize(
    ("element", "action_factory", "recorded_kind"),
    [
        (FakeElement(), lambda ref: ClickAction(kind="click", ref=ref), "click"),
        (
            FakeElement(tag="input", attrs={"type": "text"}, editable=True),
            lambda ref: TypeAction(kind="type", ref=ref, text="bounded"),
            "fill",
        ),
        (
            FakeElement(),
            lambda ref: PressAction(kind="press", ref=ref, key="Enter"),
            "press",
        ),
        (
            FakeElement(tag="select"),
            lambda ref: SelectAction(kind="select", ref=ref, value="one"),
            "select",
        ),
    ],
)
async def test_each_action_passes_the_configured_operation_timeout(
    element, action_factory, recorded_kind
):
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]

    await session.act(action_factory(ref))

    assert element.timeouts == [(recorded_kind, 15_000)]


async def test_scroll_into_view_uses_exact_handle_timeout_and_bounded_diagnostics():
    element = FakeElement(expanded=True)
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]

    result = json.loads(await session.act(ScrollIntoViewAction(kind="scroll_into_view", ref=ref)))

    assert element.actions == [("scroll", None)]
    assert element.timeouts == [("scroll", 15_000)]
    assert result["action"] == "scroll_into_view"
    assert result["target"]["tag"] == "button"
    assert result["target"]["name"] == "Go"
    assert result["target"]["expanded"] is True
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ScrollIntoViewAction(kind="scroll_into_view", ref=ref))
    assert exc_info.value.code == "stale_element_reference"


@pytest.mark.parametrize(
    ("direction", "distance"),
    [
        ("up", "half_page"),
        ("down", "page"),
        ("left", "half_page"),
        ("right", "page"),
    ],
)
async def test_page_scroll_uses_only_validated_literals_and_reports_movement(direction, distance):
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    page = browser.contexts[0].pages[0]
    page.scroll_result = {"before": {"x": 10, "y": 20}, "after": {"x": 10, "y": 380}}

    result = json.loads(
        await session.act(PageScrollAction(kind="scroll", direction=direction, distance=distance))
    )

    assert len(page.evaluate_calls) == 1
    _, argument = page.evaluate_calls[0]
    assert argument == {"direction": direction, "distance": distance}
    assert result == {
        "ok": True,
        "page_ref": result["page_ref"],
        "action": "scroll",
        "direction": direction,
        "distance": distance,
        "before": {"x": 10, "y": 20},
        "after": {"x": 10, "y": 380},
        "moved": True,
    }
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert exc_info.value.code == "stale_element_reference"


async def test_page_scroll_reports_when_page_boundary_prevents_movement():
    manager, _, browser, _ = make_manager()
    session = await manager.get_session("thread")
    await session.tabs("new")
    page = browser.contexts[0].pages[0]
    page.scroll_result = {"before": {"x": 0, "y": 0}, "after": {"x": 0, "y": 0}}

    result = json.loads(await session.act(PageScrollAction(kind="scroll", direction="up")))

    assert result["moved"] is False
    assert result["before"] == result["after"]


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (TimeoutError(), "action_timeout"),
        (RuntimeError("raw playwright detail"), "scroll_failed"),
    ],
)
async def test_page_scroll_failures_are_bounded_and_invalidate_refs(failure, expected_code):
    element = FakeElement()
    browser = FakeBrowser(context_factory=lambda: FakeContext(page_elements=[element]))
    manager, _, _, _ = make_manager(browser=browser)
    session = await manager.get_session("thread")
    ref = json.loads(await session.snapshot())["nodes"][0]["ref"]
    browser.contexts[0].pages[0].evaluate_failure = failure

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(PageScrollAction(kind="scroll", direction="down"))

    assert exc_info.value.code == expected_code
    assert "raw playwright detail" not in str(exc_info.value)
    with pytest.raises(BrowserRuntimeError) as stale_info:
        await session.act(ClickAction(kind="click", ref=ref))
    assert stale_info.value.code == "stale_element_reference"


async def test_page_scroll_rejects_unstructured_browser_diagnostics():
    manager, _, browser, _ = make_manager()
    session = await manager.get_session("thread")
    await session.tabs("new")
    browser.contexts[0].pages[0].scroll_result = {"before": {"x": 0, "y": 0}, "after": None}

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.act(PageScrollAction(kind="scroll", direction="down"))

    assert exc_info.value.code == "scroll_failed"


async def test_caller_cancellation_does_not_preempt_running_sync_playwright_call():
    manager, _, browser, _ = make_manager()
    session = await manager.get_session("thread")
    await session.tabs("new")
    page = browser.contexts[0].pages[0]
    page.evaluate_started = threading.Event()
    page.evaluate_wait = threading.Event()

    task = asyncio.create_task(session.act(PageScrollAction(kind="scroll", direction="down")))
    while not page.evaluate_started.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    queued = asyncio.create_task(session.tabs("list"))
    await asyncio.sleep(0)
    assert not queued.done()
    page.evaluate_wait.set()
    await queued


async def test_sync_evaluate_returns_stable_deadline_without_preempting_worker():
    manager, _, browser, _ = make_manager(limits=BrowserLimits(action_timeout_ms=10))
    session = await manager.get_session("thread")
    await session.tabs("new")
    page = browser.contexts[0].pages[0]
    page.evaluate_started = threading.Event()
    page.evaluate_wait = threading.Event()

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await asyncio.wait_for(
            session.act(PageScrollAction(kind="scroll", direction="down")),
            timeout=0.2,
        )

    assert exc_info.value.code == "action_timeout"
    assert page.evaluate_started.is_set()
    queued = asyncio.create_task(session.tabs("list"))
    await asyncio.sleep(0)
    assert not queued.done()
    page.evaluate_wait.set()
    await queued


async def test_action_preconditions_and_failures_have_stable_codes_and_invalidate_refs():
    cases = [
        (
            FakeElement(disabled=True),
            lambda ref: ClickAction(kind="click", ref=ref),
            "element_disabled",
        ),
        (
            FakeElement(tag="input", editable=False),
            lambda ref: TypeAction(kind="type", ref=ref, text="value"),
            "element_not_editable",
        ),
        (
            FakeElement(tag="button"),
            lambda ref: SelectAction(kind="select", ref=ref, value="value"),
            "action_target_mismatch",
        ),
    ]
    timeout_element = FakeElement()
    timeout_element.failures["click"] = TimeoutError()
    cases.append(
        (
            timeout_element,
            lambda ref: ClickAction(kind="click", ref=ref),
            "action_timeout",
        )
    )
    scroll_element = FakeElement()
    scroll_element.failures["scroll"] = RuntimeError("raw playwright detail")
    cases.append(
        (
            scroll_element,
            lambda ref: ScrollIntoViewAction(kind="scroll_into_view", ref=ref),
            "scroll_failed",
        )
    )

    for index, (element, action_factory, expected_code) in enumerate(cases):
        browser = FakeBrowser(
            context_factory=lambda element=element: FakeContext(page_elements=[element])
        )
        manager, _, _, _ = make_manager(browser=browser)
        session = await manager.get_session(f"thread-{index}")
        ref = json.loads(await session.snapshot())["nodes"][0]["ref"]

        with pytest.raises(BrowserRuntimeError) as exc_info:
            await session.act(action_factory(ref))
        assert exc_info.value.code == expected_code
        assert "raw playwright detail" not in str(exc_info.value)
        with pytest.raises(BrowserRuntimeError) as stale_info:
            await session.act(ClickAction(kind="click", ref=ref))
        assert stale_info.value.code == "stale_element_reference"


async def test_tab_refs_are_opaque_and_stale_refs_fail():
    manager, _, _, _ = make_manager()
    session = await manager.get_session("thread")
    tabs = json.loads(await session.tabs("new"))["tabs"]
    page_ref = tabs[0]["page_ref"]
    assert page_ref.startswith("p_")
    await session.tabs("close", page_ref)
    with pytest.raises(BrowserRuntimeError, match="stale"):
        await session.tabs("select", page_ref)


def _oversized_context():
    return FakeContext(
        page_elements=[FakeElement(text="x" * 300) for _ in range(20)],
        screenshot=b"x" * 11,
    )


async def test_snapshot_and_screenshot_outputs_are_bounded():
    browser = FakeBrowser(context_factory=_oversized_context)
    limits = BrowserLimits(
        max_snapshot_nodes=20,
        max_snapshot_chars=600,
        max_screenshot_bytes=10,
    )
    manager, _, _, _ = make_manager(browser=browser, limits=limits)
    session = await manager.get_session("thread")
    snapshot = await session.snapshot()
    assert len(snapshot) <= 600
    assert json.loads(snapshot)["truncated"] is True
    with pytest.raises(BrowserRuntimeError, match="Screenshot exceeds"):
        await session.screenshot()


async def test_screenshot_rejects_duplicated_response_before_base64_encoding(monkeypatch):
    payload = b"x" * 100
    projected = ScreenshotResult(
        page_ref=f"p_{'x' * 24}",
        media_type="image/png",
        data=payload,
    )
    metadata = projected.metadata()
    empty_image = {"type": "image", "base64": "", "mime_type": "image/png"}
    empty_content = [
        {"type": "text", "text": json.dumps(metadata, separators=(",", ":"))},
        empty_image,
    ]
    empty_artifact = {**metadata, "image": empty_image}
    encoded_chars = 4 * ((len(payload) + 2) // 3)
    content_only_chars = len(json.dumps(empty_content, separators=(",", ":"))) + encoded_chars
    total_chars = (
        content_only_chars + len(json.dumps(empty_artifact, separators=(",", ":"))) + encoded_chars
    )
    assert projected.projected_output_chars() == total_chars
    browser = FakeBrowser(context_factory=lambda: FakeContext(screenshot=payload))
    manager, _, _, _ = make_manager(
        browser=browser,
        limits=BrowserLimits(
            max_screenshot_bytes=len(payload),
            max_screenshot_output_chars=content_only_chars,
        ),
    )
    session = await manager.get_session("thread")
    page_ref = json.loads(await session.tabs("new"))["tabs"][0]["page_ref"]
    middleware = BrowserMiddleware(runtime_manager=manager)
    middleware._fallback_thread_id = "thread"

    def unexpected_encode(_payload):
        pytest.fail("base64 allocation should not occur for an oversized total response")

    monkeypatch.setattr("deepagents_browser.middleware.base64.b64encode", unexpected_encode)

    class Runtime:
        def __init__(self) -> None:
            self.state = {"_browser_enabled": True}

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await middleware.tools[3].coroutine(page_ref=page_ref, runtime=Runtime())
    assert exc_info.value.code == "screenshot_too_large"


async def test_runtime_startup_is_bounded_retryable_and_coded():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(1)
        return FakePlaywright(), FakeBrowser()

    manager = BrowserRuntimeManager(
        limits=BrowserLimits(startup_timeout_seconds=0.01),
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("first")
    assert exc_info.value.code == "startup_timeout"

    session = await manager.get_session("second")
    assert session is not None
    assert calls == 2


async def test_invalid_runtime_factory_result_is_rejected_and_cleaned_up():
    playwright = FakePlaywright()

    def factory():
        return playwright, object()

    manager = BrowserRuntimeManager(
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("thread")
    assert exc_info.value.code == "invalid_runtime_factory"
    assert playwright.stopped == 1
    assert manager._runtime_started is False


async def test_runtime_factory_failure_preserves_cause_and_code():
    failure = OSError("launch failed")

    def factory():
        raise failure

    manager = BrowserRuntimeManager(
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.get_session("thread")
    assert exc_info.value.code == "startup_failed"
    assert exc_info.value.__cause__ is failure


async def test_aclose_session_is_idempotent_and_allows_fresh_context():
    manager, _, browser, _ = make_manager()
    first = await manager.get_session("thread")

    await manager.aclose_session("thread")
    await manager.aclose_session("thread")
    second = await manager.get_session("thread")

    assert first is not second
    assert browser.contexts[0].closed == 1
    assert browser.contexts[1].closed == 0


async def test_cancelled_explicit_close_reconciles_facade_and_manager_mapping():
    manager, _, browser, _ = make_manager()
    stale = await manager.get_session("thread")
    context = browser.contexts[0]
    context.close_started = threading.Event()
    context.close_wait = threading.Event()

    close_task = asyncio.create_task(stale.aclose())
    while not context.close_started.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0)
    close_task.cancel()
    await asyncio.sleep(0)
    assert not close_task.done()

    context.close_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert stale._closed is True
    assert "thread" not in manager._sessions
    replacement = await manager.get_session("thread")
    assert replacement is not stale
    assert await manager.get_session("thread") is replacement


async def test_browser_session_close_failure_remains_retryable():
    manager, _, browser, _ = make_manager()
    session = await manager.get_session("thread")
    context = browser.contexts[0]
    context.close_failure = RuntimeError("raw cleanup failure")

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await session.aclose()
    assert exc_info.value.code == "cleanup_failed"
    assert session._closed is False

    context.close_failure = None
    await session.aclose()
    assert session._closed is True
    assert context.closed == 2


async def test_cancelled_manager_close_reconciles_mapping_before_reraising():
    manager, _, browser, _ = make_manager()
    stale = await manager.get_session("thread")
    context = browser.contexts[0]
    context.close_started = threading.Event()
    context.close_wait = threading.Event()

    close_task = asyncio.create_task(manager.aclose_session("thread"))
    while not context.close_started.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0)
    close_task.cancel()
    context.close_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert stale._closed is True
    assert "thread" not in manager._sessions
    assert await manager.get_session("thread") is not stale


async def test_aclose_session_rejects_active_lease_without_closing_it():
    manager, _, browser, _ = make_manager()

    async with manager.lease_session("thread"):
        with pytest.raises(BrowserRuntimeError) as exc_info:
            await manager.aclose_session("thread")
        assert exc_info.value.code == "context_limit_reached"
        assert browser.contexts[0].closed == 0

    await manager.aclose_session("thread")
    assert browser.contexts[0].closed == 1


async def test_shutdown_waits_for_active_lease_and_concurrent_callers():
    manager, playwright, browser, _ = make_manager()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def operation():
        async with manager.lease_session("thread"):
            entered.set()
            await release.wait()

    operation_task = asyncio.create_task(operation())
    await entered.wait()
    close_tasks = [asyncio.create_task(manager.aclose()) for _ in range(2)]
    await asyncio.sleep(0)
    assert browser.contexts[0].closed == 0

    release.set()
    await asyncio.gather(operation_task, *close_tasks)

    assert browser.contexts[0].closed == 1
    assert browser.closed == 1
    assert playwright.stopped == 1


async def test_failed_explicit_close_is_not_marked_complete_and_remains_retryable():
    manager, _, browser, _ = make_manager()
    await manager.get_session("thread")
    context = browser.contexts[0]
    context.close_failure = RuntimeError("raw cleanup failure")

    with pytest.raises(BrowserRuntimeError) as exc_info:
        await manager.aclose()

    assert exc_info.value.code == "cleanup_failed"
    assert manager._shutdown_complete is False
    assert manager._atexit_complete is False
    context.close_failure = None
    await manager.aclose()
    assert manager._shutdown_complete is True
    assert manager._atexit_complete is True


async def test_cleanup_is_idempotent():
    manager, playwright, browser, _ = make_manager()
    await manager.get_session("thread")
    context = browser.contexts[0]
    await asyncio.gather(manager.aclose(), manager.aclose())
    await manager.aclose()
    assert context.closed == 1
    assert browser.closed == 1
    assert playwright.stopped == 1


class _AffinityProbe:
    def __init__(self) -> None:
        self.worker_thread_id = None
        self.access_thread_ids = []

    def establish_worker(self):
        self.worker_thread_id = threading.get_ident()
        self.access_thread_ids.append(self.worker_thread_id)

    def check(self):
        thread_id = threading.get_ident()
        self.access_thread_ids.append(thread_id)
        assert thread_id == self.worker_thread_id


class _ThreadAffine:
    probe: _AffinityProbe

    def __getattribute__(self, name) -> object:
        if not name.startswith("_") and name != "probe":
            probe = object.__getattribute__(self, "probe")
            if probe.worker_thread_id is not None:
                probe.check()
        return super().__getattribute__(name)


async def test_every_raw_playwright_object_stays_on_one_dedicated_thread():
    probe = _AffinityProbe()

    class AffineElement(_ThreadAffine, FakeElement):
        pass

    class AffineLocator(_ThreadAffine, FakeLocator):
        pass

    class AffineRequest(_ThreadAffine, FakeRequest):
        pass

    class AffineRoute(_ThreadAffine, FakeRoute):
        def __init__(self, url) -> None:
            self.request = AffineRequest(url)
            self.continued = 0
            self.aborted = 0

    class AffinePage(_ThreadAffine, FakePage):
        def goto(self, url, *, wait_until, timeout):
            assert wait_until == "domcontentloaded"
            assert timeout == self.timeout
            for _, handler in self.context.routes:
                route = AffineRoute(url)
                handler(route)
                self.context.requests.append(route)
            self.url = url
            self.emit_frame_navigated(self.main_frame)

        def locator(self, selector):
            assert "button" in selector
            return AffineLocator(self.elements)

    class AffineContext(_ThreadAffine, FakeContext):
        def new_page(self):
            page = AffinePage(self, self.page_elements, self.screenshot)
            self.pages.append(page)
            return page

    class AffineBrowser(_ThreadAffine, FakeBrowser):
        pass

    class AffinePlaywright(_ThreadAffine, FakePlaywright):
        pass

    for cls in (
        AffineElement,
        AffineLocator,
        AffineRequest,
        AffineRoute,
        AffinePage,
        AffineContext,
        AffineBrowser,
        AffinePlaywright,
    ):
        cls.probe = probe

    element = AffineElement()
    browser = AffineBrowser(context_factory=lambda: AffineContext(page_elements=[element]))
    playwright = AffinePlaywright()

    def factory():
        probe.establish_worker()
        return playwright, browser

    manager = BrowserRuntimeManager(
        network_policy=NetworkPolicy(resolver=_public_resolver),
        runtime_factory=factory,
    )
    session = await manager.get_session("thread")
    await session.navigate("https://example.com")
    snapshot = json.loads(await session.snapshot())
    await session.act(ClickAction(kind="click", ref=snapshot["nodes"][0]["ref"]))
    await session.screenshot()
    await session.tabs("list")
    await manager.aclose()

    assert probe.worker_thread_id != threading.get_ident()
    assert set(probe.access_thread_ids) == {probe.worker_thread_id}


async def test_close_before_start_creates_no_thread_or_atexit_registration(monkeypatch):
    callbacks = []
    monkeypatch.setattr("deepagents_browser.runtime.atexit.register", callbacks.append)
    manager, _, _, starts = make_manager()

    await manager.aclose()
    await manager.aclose()

    assert starts == []
    assert callbacks == []
    assert manager._worker is None


async def test_runtime_thread_and_atexit_registration_are_lazy_and_single_flight(monkeypatch):
    callbacks = []
    monkeypatch.setattr("deepagents_browser.runtime.atexit.register", callbacks.append)
    manager, _, _, starts = make_manager()

    assert manager._worker is None
    assert manager._atexit_registered is False
    assert callbacks == []

    session = await manager.get_session("thread")
    await session.tabs("list")

    assert starts == ["start"]
    assert manager._worker is not None
    assert len(callbacks) == 1

    await manager.aclose()
    callbacks[0]()

    assert len(callbacks) == 1
    assert manager._shutdown_complete is True
    assert manager._atexit_complete is True


async def test_atexit_fallback_closes_on_owner_worker_and_stops_it(monkeypatch):
    callbacks = []
    monkeypatch.setattr("deepagents_browser.runtime.atexit.register", callbacks.append)
    manager, playwright, browser, _ = make_manager()
    await manager.get_session("thread")
    worker = manager._worker
    assert worker is not None

    callbacks[0]()

    assert worker.stopped.done()
    assert worker.is_alive is False
    assert browser.contexts[0].closed == 1
    assert browser.closed == 1
    assert playwright.stopped == 1
    assert manager._atexit_complete is True


async def test_atexit_timeout_is_bounded_and_not_marked_complete(monkeypatch):
    callbacks = []
    monkeypatch.setattr("deepagents_browser.runtime.atexit.register", callbacks.append)
    manager, _, browser, _ = make_manager(limits=BrowserLimits(action_timeout_ms=10))
    await manager.get_session("thread")
    worker = manager._worker
    assert worker is not None
    context = browser.contexts[0]
    context.close_started = threading.Event()
    context.close_wait = threading.Event()

    started = time.monotonic()
    callbacks[0]()
    elapsed = time.monotonic() - started

    assert context.close_started.is_set()
    assert elapsed < 0.2
    assert manager._atexit_complete is False
    assert worker.is_alive is True
    context.close_wait.set()
    await asyncio.wrap_future(worker.stopped)


async def test_route_policy_bridge_times_out_and_fails_closed():
    calls = 0

    async def resolver(host, port):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ["93.184.216.34"]
        await asyncio.Event().wait()
        return []

    browser = FakeBrowser()
    manager = BrowserRuntimeManager(
        limits=BrowserLimits(action_timeout_ms=10),
        network_policy=NetworkPolicy(resolver=resolver),
        runtime_factory=lambda: (FakePlaywright(), browser),
    )
    session = await manager.get_session("thread")

    await session.navigate("https://example.com")

    assert browser.contexts[0].requests[0].aborted == 1
    assert browser.contexts[0].requests[0].continued == 0
    await manager.aclose()
